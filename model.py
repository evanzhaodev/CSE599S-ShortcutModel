import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import segmentation_models_pytorch as smp

def modulate(x, shift, scale):
    # Original JAX code: x * (1 + scale[:, None]) + shift[:, None]
    # scale = torch.clamp(scale, -1, 1)
    return x * (1 + scale[:, None]) + shift[:, None]

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float, device=pos.device)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)
    
    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product
    
    emb_sin = torch.sin(out) # (M, D/2)
    emb_cos = torch.cos(out) # (M, D/2)
    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb

def get_1d_sincos_pos_embed(embed_dim, length, device=None):
    emb = get_1d_sincos_pos_embed_from_grid(embed_dim, torch.arange(length, dtype=torch.float, device=device))
    return emb.unsqueeze(0)

def get_2d_sincos_pos_embed(embed_dim, length, device=None):
    # example: embed_dim = 256, length = 16*16
    grid_size = int(length ** 0.5)
    assert grid_size * grid_size == length
    
    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0
        # use half of dimensions to encode grid_h
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
        emb = torch.cat([emb_h, emb_w], dim=1) # (H*W, D)
        return emb
    
    grid_h = torch.arange(grid_size, dtype=torch.float, device=device)
    grid_w = torch.arange(grid_size, dtype=torch.float, device=device)
    grid = torch.meshgrid(grid_w, grid_h, indexing='ij')  # here w goes first
    grid = torch.stack(grid, dim=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed.unsqueeze(0) # (1, H*W, D)

class TrainConfig:
    def __init__(self, dtype=torch.float32):
        self.dtype = dtype
    
    def kern_init(self, name='default', zero=False):
        def init_func(module):
            if zero or 'bias' in name:
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
                if hasattr(module, 'weight'):
                    nn.init.zeros_(module.weight)
            else:
                if hasattr(module, 'weight'):
                    nn.init.xavier_uniform_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
        return init_func

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, tc, frequency_embedding_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.tc = tc
        self.frequency_embedding_size = frequency_embedding_size
        
        self.mlp1 = nn.Linear(frequency_embedding_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, hidden_size)
        
        # Initialize with normal instead of xavier for time embedder
        nn.init.normal_(self.mlp1.weight, std=0.02)
        nn.init.normal_(self.mlp2.weight, std=0.02)
        
        # Apply initializers for biases
        self.tc.kern_init('time_bias', zero=True)(self.mlp1)
        self.tc.kern_init('time_bias')(self.mlp2)
        
    def forward(self, t):
        x = self.timestep_embedding(t)
        x = self.mlp1(x)
        x = F.silu(x)
        x = self.mlp2(x)
        return x
    
    # t is between [0, 1]
    def timestep_embedding(self, t, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        """
        t = t.float()
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float, device=t.device) / half
        )
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        embedding = embedding.to(self.tc.dtype)
        return embedding

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations.
    """
    def __init__(self, num_classes, hidden_size, tc):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.tc = tc
        
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        nn.init.normal_(self.embedding_table.weight, std=0.02)
        
    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings

class ImageEmbedder(nn.Module):
    """
    Embeds 2D images into vector representations using features extracted through CNN networks
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.feature_extractor = smp.Unet('tu-mobilenetv4_conv_small', classes=hidden_size, in_channels=4)
    
    def forward(self, imgs):
        return self.feature_extractor(imgs)

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding """
    def __init__(self, patch_size, hidden_size, tc, bias=True):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.tc = tc
        self.bias = bias
        
        self.proj = nn.Conv2d(
            in_channels=4,  # Latent space channels
            out_channels=hidden_size,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            bias=bias
        )
        
        # Apply initializers
        self.tc.kern_init('patch')(self.proj)
        if bias:
            self.tc.kern_init('patch_bias', zero=True)(self.proj)
        
    def forward(self, x):
        B, H, W, C = x.shape
        num_patches = H // self.patch_size
        
        # Transpose for Conv2d which expects [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)
        
        # Reshape to match JAX output format [B, num_patches*num_patches, hidden_size]
        x = x.permute(0, 2, 3, 1)  # [B, num_patches, num_patches, hidden_size]
        x = rearrange(x, 'b h w c -> b (h w) c', h=num_patches, w=num_patches)
        return x

class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""
    def __init__(self, mlp_dim, tc, out_dim=None, dropout_rate=None):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.tc = tc
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        
        # Pre-initialize the layers with default sizes to match the saved state dict
        self.fc1 = nn.Linear(768, mlp_dim)  # Default hidden size is typically 768
        self.fc2 = nn.Linear(mlp_dim, 768)  # Default output is same as input
        
        # Initialize weights
        self.tc.kern_init()(self.fc1)
        self.tc.kern_init()(self.fc2)
        
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else nn.Identity()
    
    def _update_fc1_if_needed(self, in_dim, device):
        # If dimensions don't match, recreate the layer
        if self.fc1.in_features != in_dim:
            old_fc1 = self.fc1
            self.fc1 = nn.Linear(in_dim, self.mlp_dim).to(device)
            self.tc.kern_init()(self.fc1)
            
    def _update_fc2_if_needed(self, out_dim, device):
        # If dimensions don't match, recreate the layer
        if self.fc2.out_features != out_dim:
            old_fc2 = self.fc2
            self.fc2 = nn.Linear(self.mlp_dim, out_dim).to(device)
            self.tc.kern_init()(self.fc2)
        
    def forward(self, inputs, train=False):
        device = inputs.device
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        
        # Update layers if dimensions don't match
        self._update_fc1_if_needed(inputs.shape[-1], device)
        self._update_fc2_if_needed(actual_out_dim, device)
        
        # Move layers to the correct device if needed
        if self.fc1.weight.device != device:
            self.fc1 = self.fc1.to(device)
        if self.fc2.weight.device != device:
            self.fc2 = self.fc2.to(device)
        
        # Forward pass
        x = self.fc1(inputs)
        x = F.gelu(x)
        x = self.dropout(x) if train else x
        x = self.fc2(x)
        x = self.dropout(x) if train else x
        return x

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, tc, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.tc = tc
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        
        # Layer norms (without affine parameters)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        
        # AdaLN modulation via conditioning
        self.adaLN_modulation = nn.Linear(hidden_size, 6 * hidden_size)
        self.tc.kern_init()(self.adaLN_modulation)
        
        # Attention components
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_output = nn.Linear(hidden_size, hidden_size)
        
        # Initialize attention components
        for layer in [self.q_proj, self.k_proj, self.v_proj, self.attn_output]:
            self.tc.kern_init()(layer)
        
        # MLP block
        self.mlp = MlpBlock(
            mlp_dim=int(hidden_size * mlp_ratio),
            tc=tc,
            dropout_rate=dropout
        )
        
    def forward(self, x, c, train=False):
        # Process conditioning to get modulation parameters
        c = F.silu(c)
        adaLN_params = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.split(
            adaLN_params, self.hidden_size, dim=-1
        )
        
        # Attention block
        x_norm = self.norm1(x)
        x_norm = modulate(x_norm, shift_msa, scale_msa)
        
        # Multi-head attention
        B, N, C = x_norm.shape
        channels_per_head = C // self.num_heads
        
        # Query, key, value projections
        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)
        
        # Reshape for multi-head attention
        q = q.reshape(B, N, self.num_heads, channels_per_head)
        k = k.reshape(B, N, self.num_heads, channels_per_head)
        v = v.reshape(B, N, self.num_heads, channels_per_head)
        
        # Scale query
        q = q / math.sqrt(3)
        
        # Compute attention
        attn = torch.einsum('bqhc,bkhc->bhqk', q, k)
        attn = attn.float()  # For numerical stability
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        y = torch.einsum('bhqk,bkhc->bqhc', attn, v)
        y = y.reshape(B, N, C)
        
        # Output projection
        attn_output = self.attn_output(y)
        
        # Apply gate and residual
        x = x + gate_msa[:, None] * attn_output
        
        # MLP block
        x_norm2 = self.norm2(x)
        x_norm2 = modulate(x_norm2, shift_mlp, scale_mlp)
        mlp_output = self.mlp(x_norm2, train=train)
        x = x + gate_mlp[:, None] * mlp_output
        
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, patch_size, out_channels, hidden_size, tc):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.tc = tc
        
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.adaLN_modulation = nn.Linear(hidden_size, 2 * hidden_size)
        self.output_proj = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        
        # Initialize with zeros for final layer
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        
    def forward(self, x, c):
        # Process conditioning
        c = F.silu(c)
        adaLN_params = self.adaLN_modulation(c)
        shift, scale = torch.split(adaLN_params, self.hidden_size, dim=-1)
        
        # Normalize and modulate
        x = self.norm(x)
        x = modulate(x, shift, scale)
        
        # Project to output dimensions
        x = self.output_proj(x)
        return x

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        patch_size,
        hidden_size,
        depth,
        num_heads,
        mlp_ratio,
        out_channels,
        class_dropout_prob,
        num_classes,
        ignore_dt=False,
        dropout=0.0,
        is_image=True,
        dtype=torch.float32
    ):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.out_channels = out_channels
        self.class_dropout_prob = class_dropout_prob
        self.num_classes = num_classes
        self.ignore_dt = ignore_dt
        self.dropout = dropout
        self.is_image = is_image
        self.dtype = dtype
        
        # Create training config
        self.tc = TrainConfig(dtype=dtype)
        
        # Patch embedding
        self.patch_embed = PatchEmbed(patch_size, hidden_size, tc=self.tc)
        
        # Time, dt, and label embedders
        self.timestep_embedder = TimestepEmbedder(hidden_size, tc=self.tc)
        self.dt_embedder = TimestepEmbedder(hidden_size, tc=self.tc)
        self.label_embedder = LabelEmbedder(num_classes, hidden_size, tc=self.tc)

        # image embedder
        self.image_embedder = ImageEmbedder(hidden_size)
        
        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, tc=self.tc, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        
        # Final layer
        self.final_layer = FinalLayer(patch_size, out_channels, hidden_size, tc=self.tc)
        
        # Logvar embedding
        self.logvar_embed = nn.Embedding(256, 1)
        nn.init.zeros_(self.logvar_embed.weight)
        
    def forward(self, x, t, dt, y, train=False, return_activations=False):
        # (x = (B, H, W, C) image, t = (B,) timesteps, y = (B,) class labels)
        # print(f"DiT: Input of shape {x.shape} dtype {x.dtype}")
        activations = {}
        
        batch_size = x.shape[0]
        input_size = x.shape[1]
        in_channels = x.shape[-1]
        num_patches = (input_size // self.patch_size) ** 2
        num_patches_side = input_size // self.patch_size
        
        if self.ignore_dt:
            dt = torch.zeros_like(t)
        
        # Get positional embedding
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, num_patches, device=x.device)
        pos_embed = pos_embed.to(self.dtype)
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, hidden_size)
        # print(f"DiT: After patch embed, shape is {x.shape} dtype {x.dtype}")
        activations['patch_embed'] = x
        
        # Add positional embedding
        x = x + pos_embed
        x = x.to(self.dtype)
        
        # Time, dt, and label/image embeddings
        te = self.timestep_embedder(t)  # (B, hidden_size)
        dte = self.dt_embedder(dt)  # (B, hidden_size)

        if self.is_image:
            ye = self.image_embedder(y)  # (B, hidden_size)
        else:
            ye = self.label_embedder(y)  # (B, hidden_size)

        c = te + ye + dte
        
        # Store activations
        activations['pos_embed'] = pos_embed
        activations['time_embed'] = te
        activations['dt_embed'] = dte
        activations['label_embed'] = ye
        activations['conditioning'] = c
        
        # print(f"DiT: Patch Embed of shape {x.shape} dtype {x.dtype}")
        # print(f"DiT: Conditioning of shape {c.shape} dtype {c.dtype}")
        
        # Apply transformer blocks
        for i in range(self.depth):
            x = self.blocks[i](x, c, train=train)
            activations[f'dit_block_{i}'] = x
            
        # Apply final layer
        x = self.final_layer(x, c)  # (B, num_patches, p*p*c)
        activations['final_layer'] = x
        
        # Reshape output
        x = x.reshape(batch_size, num_patches_side, num_patches_side, 
                      self.patch_size, self.patch_size, self.out_channels)
        x = torch.einsum('bhwpqc->bhpwqc', x)
        x = rearrange(x, 'B H P W Q C -> B (H P) (W Q) C', H=int(num_patches_side), W=int(num_patches_side))
        assert x.shape == (batch_size, input_size, input_size, self.out_channels)
        
        # Calculate logvars
        t_discrete = torch.floor(t * 256).long()
        logvars = self.logvar_embed(t_discrete) * 100
        
        if return_activations:
            return x, logvars, activations
        return x