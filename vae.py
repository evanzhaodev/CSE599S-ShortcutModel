import torch
import torch.nn as nn
from functools import partial, cached_property
from diffusers import AutoencoderKL
from einops import rearrange
from typing import Dict, Any, Optional

class StableVAE(nn.Module):
    """PyTorch implementation of the StableVAE used in the original code."""
    
    def __init__(self, model_id: str = "stabilityai/sd-vae-ft-mse"):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(model_id)
        
    def encode(self, images: torch.Tensor, scale: bool = True) -> torch.Tensor:
        """
        Encode images to latent space.
        
        Args:
            images: [B, H, W, 3] tensor in range [-1, 1]
            scale: Whether to scale the latents by the VAE scaling factor
            
        Returns:
            [B, h, w, 4] latent representation
        """
        # Convert from [B, H, W, C] to [B, C, H, W]
        images = rearrange(images, "b h w c -> b c h w")
        
        # Encode the images to latent space
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            
        # Scale the latents
        if scale:
            latents = latents * self.vae.config.scaling_factor
            
        # Convert back to [B, h, w, C]
        latents = rearrange(latents, "b c h w -> b h w c")
        
        return latents
    
    def decode(self, latents: torch.Tensor, scale: bool = True) -> torch.Tensor:
        """
        Decode latents to images.
        
        Args:
            latents: [B, h, w, 4] latent representation
            scale: Whether to scale the latents by the VAE scaling factor
            
        Returns:
            [B, H, W, 3] images in range [-1, 1]
        """
        # Convert from [B, h, w, C] to [B, C, h, w]
        latents = rearrange(latents, "b h w c -> b c h w")
        
        # Scale the latents
        if scale:
            latents = latents / self.vae.config.scaling_factor
            
        # Decode the latents to images
        with torch.no_grad():
            images = self.vae.decode(latents).sample
            
        # Convert back to [B, H, W, C]
        images = rearrange(images, "b c h w -> b h w c")
        
        return images
    
    @property
    def downscale_factor(self) -> int:
        """Returns the downscale factor of the VAE."""
        return 2 ** (len(self.vae.config.block_out_channels) - 1)