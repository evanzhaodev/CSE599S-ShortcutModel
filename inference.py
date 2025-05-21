import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.utils import save_image
import glob

from model import DiT
from vae import StableVAE
from utils import load_json, process_image

def parse_args():
    parser = argparse.ArgumentParser(description='Inference with shortcut model for super-resolution')
    
    # Input/output arguments
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with low-resolution images')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for high-resolution images')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config_path', type=str, required=True, help='Path to model config file')
    
    # Inference arguments
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--steps', type=int, default=1, help='Number of denoising steps')
    parser.add_argument('--use_ema', action='store_true', help='Use EMA model weights')
    parser.add_argument('--cfg_scale', type=float, default=0.0, help='Classifier-free guidance scale')
    
    return parser.parse_args()

def load_model(config, checkpoint_path, device, use_ema=False):
    """
    Load model from checkpoint.
    
    Args:
        config: Model configuration
        checkpoint_path: Path to checkpoint
        device: Device to load model on
        use_ema: Whether to use EMA weights
        
    Returns:
        Loaded model
    """
    # Create model
    model = DiT(
        patch_size=config['patch_size'],
        hidden_size=config['hidden_size'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=config['mlp_ratio'],
        out_channels=config.get('image_channels', 4 if config.get('use_stable_vae', False) else 3),
        class_dropout_prob=0.1,  # Not used but keep as in original
        num_classes=1,  # Unconditional
        dropout=config.get('dropout', 0.0),
        use_low_res_cond=True,  # Enable low-res conditioning
        ignore_dt=False
    )
    
    try:
        # First try loading with weights_only=True (safer)
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        print(f"Warning: Could not load checkpoint with weights_only=True. Trying with weights_only=False: {e}")
        # Fall back to weights_only=False if needed
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load weights
    if use_ema and isinstance(checkpoint, dict) and 'ema_model' in checkpoint:
        model.load_state_dict(checkpoint['ema_model'])
        print("Loaded EMA model weights")
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        print("Loaded model weights")
    else:
        # Assume checkpoint is just the model state dict
        model.load_state_dict(checkpoint)
        print("Loaded model weights directly")
    
    model.to(device)
    model.eval()
    
    return model

def prepare_image(image_path, image_size, low_res_factor):
    """
    Load and prepare image for inference.
    
    Args:
        image_path: Path to image
        image_size: Target image size
        low_res_factor: Factor for low-resolution
        
    Returns:
        Low-resolution image tensor [1, H, W, C]
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Scale to [-1, 1]
    ])
    
    low_res_transform = transforms.Compose([
        transforms.Resize((image_size // low_res_factor, image_size // low_res_factor), 
                         interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.Resize((image_size, image_size), 
                         interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Scale to [-1, 1]
    ])
    
    # Load image
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Create high-res and low-res versions for comparison
        high_res = transform(img)
        low_res = low_res_transform(img)
    
    # Convert to [H, W, C] format expected by the model
    high_res = high_res.permute(1, 2, 0)
    low_res = low_res.permute(1, 2, 0)
    
    # Add batch dimension
    high_res = high_res.unsqueeze(0)
    low_res = low_res.unsqueeze(0)
    
    return low_res, high_res

def generate_sample(model, x_low_res, steps=1, device=None, cfg_scale=0.0):
    """
    Generate sample using the shortcut model.
    
    Args:
        model: Shortcut model
        x_low_res: Low-resolution conditioning input [B, H, W, C]
        steps: Number of denoising steps
        device: Device to use
        cfg_scale: Classifier-free guidance scale
        
    Returns:
        Generated high-resolution image [B, H, W, C]
    """
    if device is None:
        device = next(model.parameters()).device
    
    batch_size = x_low_res.shape[0]
    
    # Start from pure noise
    x = torch.randn_like(x_low_res, device=device)
    
    # Calculate step size
    d = 1.0 / steps
    
    # Initialize current time
    t = torch.zeros(batch_size, device=device)
    
    # For timestep conditioning
    dt_base = torch.ones(batch_size, dtype=torch.int64, device=device) * int(np.log2(steps))
    
    # Zero labels for unconditional
    labels = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    # Denoising loop
    for step in range(steps):
        # Forward pass to get velocity with low-res conditioning
        with torch.no_grad():
            # If using CFG, we need to do two forward passes
            if cfg_scale > 0:
                # Conditional pass
                v_cond = model(x, x_low_res, t, dt_base, labels)
                
                # Unconditional pass (using null labels or special technique for unconditional generation)
                # Here we use the same inputs but with null labels
                v_uncond = model(x, x_low_res, t, dt_base, torch.ones_like(labels) * model.num_classes)
                
                # Apply classifier-free guidance
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = model(x, x_low_res, t, dt_base, labels)
        
        # Update x using Euler method
        x = x + v * d
        
        # Update time
        t = t + d
    
    return x

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    config = load_json(args.config_path)
    
    # Load model
    model = load_model(config, args.model_path, device, args.use_ema)
    
    # Create VAE if needed
    if config.get('use_stable_vae', False):
        vae = StableVAE()
        vae.to(device)
        vae.eval()
        print("Using StableVAE")
    else:
        vae = None
    
    # Get all image files in input directory
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPEG']:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, "**", ext), recursive=True))
    
    print(f"Found {len(image_paths)} images")
    
    # Process images in batches
    for i in tqdm(range(0, len(image_paths), args.batch_size)):
        batch_paths = image_paths[i:i + args.batch_size]
        batch_size = len(batch_paths)
        
        # Prepare batch of images
        batch_low_res = []
        original_names = []
        
        for path in batch_paths:
            low_res, _ = prepare_image(
                path,
                config.get('image_size', 256),
                config.get('low_res_factor', 4)
            )
            batch_low_res.append(low_res)
            original_names.append(os.path.basename(path))
        
        # Concatenate batch
        batch_low_res = torch.cat(batch_low_res, dim=0).to(device)
        
        # Encode with VAE if needed
        if vae is not None:
            with torch.no_grad():
                batch_low_res = vae.encode(batch_low_res)
        
        # Generate high-resolution samples from noise conditioned on low-res
        batch_high_res = generate_sample(
            model=model,
            x_low_res=batch_low_res, 
            steps=args.steps,
            device=device,
            cfg_scale=args.cfg_scale
        )
        
        # Decode with VAE if needed
        if vae is not None:
            with torch.no_grad():
                batch_low_res = vae.decode(batch_low_res)
                batch_high_res = vae.decode(batch_high_res)
        
        # Save images
        for j in range(batch_size):
            output_name = f"{os.path.splitext(original_names[j])[0]}_sr.png"
            output_path = os.path.join(args.output_dir, output_name)
            
            # Convert to [C, H, W] for saving
            high_res_img = batch_high_res[j].permute(2, 0, 1)
            
            # Normalize and save
            save_image(
                high_res_img,
                output_path,
                normalize=True,
                value_range=(-1, 1)
            )
            
            # For comparison, also save the low-res input
            low_res_output_path = os.path.join(args.output_dir, f"{os.path.splitext(original_names[j])[0]}_lr.png")
            low_res_img = batch_low_res[j].permute(2, 0, 1)
            save_image(
                low_res_img,
                low_res_output_path,
                normalize=True,
                value_range=(-1, 1)
            )
    
    print("Inference completed!")

if __name__ == '__main__':
    main()