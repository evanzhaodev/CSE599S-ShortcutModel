import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

def modulate(x, shift, scale):
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
    grid_size = int(length ** 0.5)
    assert grid_size * grid_size == length
    
    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
        emb = torch.cat([emb_h, emb_w], dim=1) # (H*W, D)
        return emb
    
    grid_h = torch.arange(grid_size, dtype=torch.float, device=device)
    grid_w = torch.arange(grid_size, dtype=torch.float, device=device)
    grid = torch.meshgrid(grid_w, grid_h, indexing='ij')
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
        
        nn.init.normal_(self.mlp1.weight, std=0.02)
        nn.init.normal_(self.mlp2.weight, std=0.02)
        
        self.tc.kern_init('time_bias', zero=True)(self.mlp1)
        self.tc.kern_init('time_bias')(self.mlp2)
        
    def forward(self, t):
        x = self.timestep_embedding(t)
        x = self.mlp1(x)
        x = F.silu(x)
        x = self.mlp2(x)
        return x
    
    def timestep_embedding(self, t, max_period=10000):
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
        
        self.tc.kern_init('patch')(self.proj)
        if bias:
            self.tc.kern_init('patch_bias', zero=True)(self.proj)
        
    def forward(self, x):
        B, H, W, C = x.shape
        num_patches = H // self.patch_size
        
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)
        
        x = x.permute(0, 2, 3, 1)
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
        
        self.fc1 = nn.Linear(1152, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, 1152)
        
        self.tc.kern_init()(self.fc1)
        self.tc.kern_init()(self.fc2)
        
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else nn.Identity()
    
    def _update_fc1_if_needed(self, in_dim, device):
        if self.fc1.in_features != in_dim:
            old_fc1 = self.fc1
            self.fc1 = nn.Linear(in_dim, self.mlp_dim).to(device)
            self.tc.kern_init()(self.fc1)
            
    def _update_fc2_if_needed(self, out_dim, device):
        if self.fc2.out_features != out_dim:
            old_fc2 = self.fc2
            self.fc2 = nn.Linear(self.mlp_dim, out_dim).to(device)
            self.tc.kern_init()(self.fc2)
        
    def forward(self, inputs, train=False):
        device = inputs.device
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        
        self._update_fc1_if_needed(inputs.shape[-1], device)
        self._update_fc2_if_needed(actual_out_dim, device)
        
        if self.fc1.weight.device != device:
            self.fc1 = self.fc1.to(device)
        if self.fc2.weight.device != device:
            self.fc2 = self.fc2.to(device)
        
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
        
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        
        self.adaLN_modulation = nn.Linear(hidden_size, 6 * hidden_size)
        self.tc.kern_init()(self.adaLN_modulation)
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_output = nn.Linear(hidden_size, hidden_size)
        
        for layer in [self.q_proj, self.k_proj, self.v_proj, self.attn_output]:
            self.tc.kern_init()(layer)
        
        self.mlp = MlpBlock(
            mlp_dim=int(hidden_size * mlp_ratio),
            tc=tc,
            dropout_rate=dropout
        )
        
    def forward(self, x, c, train=False):
        c = F.silu(c)
        adaLN_params = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.split(
            adaLN_params, self.hidden_size, dim=-1
        )
        
        x_norm = self.norm1(x)
        x_norm = modulate(x_norm, shift_msa, scale_msa)
        
        B, N, C = x_norm.shape
        channels_per_head = C // self.num_heads
        
        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)
        
        q = q.reshape(B, N, self.num_heads, channels_per_head)
        k = k.reshape(B, N, self.num_heads, channels_per_head)
        v = v.reshape(B, N, self.num_heads, channels_per_head)
        
        q = q / math.sqrt(3)
        
        attn = torch.einsum('bqhc,bkhc->bhqk', q, k)
        attn = attn.float()
        attn = F.softmax(attn, dim=-1)
        
        y = torch.einsum('bhqk,bkhc->bqhc', attn, v)
        y = y.reshape(B, N, C)
        
        attn_output = self.attn_output(y)
        
        x = x + gate_msa[:, None] * attn_output
        
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
        
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        
    def forward(self, x, c):
        c = F.silu(c)
        adaLN_params = self.adaLN_modulation(c)
        shift, scale = torch.split(adaLN_params, self.hidden_size, dim=-1)
        
        x = self.norm(x)
        x = modulate(x, shift, scale)
        
        x = self.output_proj(x)
        return x

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    Modified for HOMO (High-Order Matching) with separate velocity and acceleration models.
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
        use_low_res_cond=True,
        ignore_dt=False,
        dropout=0.0,
        dtype=torch.float32,
        order='first'  # 'first' for velocity only, 'second' for acceleration
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
        self.use_low_res_cond = use_low_res_cond
        self.ignore_dt = ignore_dt
        self.dropout = dropout
        self.dtype = dtype
        self.order = order
        
        self.tc = TrainConfig(dtype=dtype)
        
        self.patch_embed = PatchEmbed(patch_size, hidden_size, tc=self.tc)
        
        if self.use_low_res_cond:
            self.low_res_embed = PatchEmbed(patch_size, hidden_size, tc=self.tc)
            self.concat_proj = nn.Linear(hidden_size * 2, hidden_size)
            self.tc.kern_init()(self.concat_proj)
        
        # For second-order model, we need to embed the velocity input
        if self.order == 'second':
            self.velocity_embed = PatchEmbed(patch_size, hidden_size, tc=self.tc)
        
        self.timestep_embedder = TimestepEmbedder(hidden_size, tc=self.tc)
        self.dt_embedder = TimestepEmbedder(hidden_size, tc=self.tc)
        self.label_embedder = LabelEmbedder(num_classes, hidden_size, tc=self.tc)
        
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, tc=self.tc, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        
        self.final_layer = FinalLayer(patch_size, out_channels, hidden_size, tc=self.tc)
        
        self.logvar_embed = nn.Embedding(256, 1)
        nn.init.zeros_(self.logvar_embed.weight)
        
    def forward(self, x, low_res=None, t=None, dt=None, y=None, v=None, train=False, return_activations=False, cfg_scale=0.0):
        """
        Args:
            x: (B, H, W, C) noise state
            low_res: (B, H, W, C) low-res conditioning
            t: (B,) timesteps
            dt: (B,) step sizes
            y: (B,) class labels
            v: (B, H, W, C) velocity (only for second-order model)
        """
        activations = {}
        
        batch_size = x.shape[0]
        input_size = x.shape[1]
        in_channels = x.shape[-1]
        num_patches = (input_size // self.patch_size) ** 2
        num_patches_side = input_size // self.patch_size
        
        if self.ignore_dt:
            dt = torch.zeros_like(t)
        
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, num_patches, device=x.device)
        pos_embed = pos_embed.to(self.dtype)
        
        x_embed = self.patch_embed(x)
        activations['patch_embed'] = x_embed
        
        x_embed = x_embed + pos_embed
        x_embed = x_embed.to(self.dtype)
        
        # For second-order model, also embed velocity
        if self.order == 'second' and v is not None:
            v_embed = self.velocity_embed(v)
            v_embed = v_embed + pos_embed
            v_embed = v_embed.to(self.dtype)
            # Concatenate noise and velocity embeddings
            x_embed = torch.cat([x_embed, v_embed], dim=-1)
            x_embed = self.concat_proj(x_embed)
        
        # Process low-res conditioning if provided
        if self.use_low_res_cond and low_res is not None:
            low_res_embed = self.low_res_embed(low_res)
            low_res_embed = low_res_embed + pos_embed
            low_res_embed = low_res_embed.to(self.dtype)
            activations['low_res_embed'] = low_res_embed
            
            x = torch.cat([x_embed, low_res_embed], dim=-1)
            x = self.concat_proj(x)
            activations['concat_proj'] = x
        else:
            x = x_embed
        
        te = self.timestep_embedder(t)
        dte = self.dt_embedder(dt)
        
        if cfg_scale > 0 and y is not None:
            # Similar CFG handling as before
            x_cond = x[:batch_size//2].clone()
            t_cond = t[:batch_size//2].clone()
            dt_cond = dt[:batch_size//2].clone()
            y_cond = y[:batch_size//2].clone()
            
            ye_cond = self.label_embedder(y_cond)
            c_cond = te[:batch_size//2] + ye_cond + dte[:batch_size//2]
            
            null_labels = torch.zeros_like(y_cond)
            ye_uncond = self.label_embedder(null_labels)
            c_uncond = te[:batch_size//2] + ye_uncond + dte[:batch_size//2]
            
            x_cond_out = x_cond.clone()
            for i in range(self.depth):
                x_cond_out = self.blocks[i](x_cond_out, c_cond, train=train)
            
            x_uncond_out = x_cond.clone()
            for i in range(self.depth):
                x_uncond_out = self.blocks[i](x_uncond_out, c_uncond, train=train)
            
            x_cond_out = self.final_layer(x_cond_out, c_cond)
            x_uncond_out = self.final_layer(x_uncond_out, c_uncond)
            
            guided_output = x_uncond_out + cfg_scale * (x_cond_out - x_uncond_out)
            
            if batch_size > batch_size//2:
                ye = self.label_embedder(y[batch_size//2:])
                c = te[batch_size//2:] + ye + dte[batch_size//2:]
                
                x_remain = x[batch_size//2:]
                for i in range(self.depth):
                    x_remain = self.blocks[i](x_remain, c, train=train)
                x_remain = self.final_layer(x_remain, c)
                
                x = torch.cat([guided_output, x_remain], dim=0)
            else:
                x = guided_output
        else:
            ye = self.label_embedder(y)
            c = te + ye + dte
            
            activations['time_embed'] = te
            activations['dt_embed'] = dte
            activations['label_embed'] = ye
            activations['conditioning'] = c
            
            for i in range(self.depth):
                x = self.blocks[i](x, c, train=train)
                activations[f'dit_block_{i}'] = x
                
            x = self.final_layer(x, c)
            activations['final_layer'] = x
        
        x = x.reshape(batch_size, num_patches_side, num_patches_side, 
                    self.patch_size, self.patch_size, self.out_channels)
        x = torch.einsum('bhwpqc->bhpwqc', x)
        x = rearrange(x, 'B H P W Q C -> B (H P) (W Q) C', H=int(num_patches_side), W=int(num_patches_side))
        assert x.shape == (batch_size, input_size, input_size, self.out_channels)
        
        t_discrete = torch.floor(t * 256).long()
        logvars = self.logvar_embed(t_discrete) * 100
        
        if return_activations:
            return x, logvars, activations
        return x, logvars


class HOMOModel(nn.Module):
    """
    HOMO (High-Order Matching for One-step shortcut) Model
    Combines first-order (velocity) and second-order (acceleration) models
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
        use_low_res_cond=True,
        ignore_dt=False,
        dropout=0.0,
        dtype=torch.float32
    ):
        super().__init__()
        
        # First-order model (velocity)
        self.u1 = DiT(
            patch_size=patch_size,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            out_channels=out_channels,
            class_dropout_prob=class_dropout_prob,
            num_classes=num_classes,
            use_low_res_cond=use_low_res_cond,
            ignore_dt=ignore_dt,
            dropout=dropout,
            dtype=dtype,
            order='first'
        )
        
        # Second-order model (acceleration)
        self.u2 = DiT(
            patch_size=patch_size,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            out_channels=out_channels,
            class_dropout_prob=class_dropout_prob,
            num_classes=num_classes,
            use_low_res_cond=use_low_res_cond,
            ignore_dt=ignore_dt,
            dropout=dropout,
            dtype=dtype,
            order='second'
        )
    
    def forward_velocity(self, x, low_res=None, t=None, dt=None, y=None, train=False, cfg_scale=0.0):
        """Forward pass for velocity (first-order) model"""
        v, logvars = self.u1(x, low_res, t, dt, y, train=train, cfg_scale=cfg_scale)
        return v
    
    def forward_acceleration(self, x, v, low_res=None, t=None, dt=None, y=None, train=False, cfg_scale=0.0):
        """Forward pass for acceleration (second-order) model"""
        a, logvars = self.u2(x, low_res, t, dt, y, v=v, train=train, cfg_scale=cfg_scale)
        return a
    
    def forward(self, x, low_res=None, t=None, dt=None, y=None, train=False, cfg_scale=0.0):
        """Forward pass for both velocity and acceleration"""
        v = self.forward_velocity(x, low_res, t, dt, y, train, cfg_scale)
        a = self.forward_acceleration(x, v, low_res, t, dt, y, train, cfg_scale)
        return v, a