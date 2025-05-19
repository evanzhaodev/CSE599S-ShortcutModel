import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.utils import save_image
import glob
from transformers import CLIPProcessor, CLIPVisionModel
import torch.nn.functional as F

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
    
    return parser.parse_args()

def create_clip_embedder():
    """Create a CLIP vision model for image embeddings"""
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()  # Set to evaluation mode
    
    # Freeze the model parameters
    for param in model.parameters():
        param.requires_grad = False
        
    return processor, model

def get_clip_embeddings(images, processor, clip_model, device):
    """
    Get CLIP embeddings for images
    
    Args:
        images: Images tensor in range [-1, 1], shape [B, H, W, C]
        processor: CLIP processor
        clip_model: CLIP vision model
        device: Device to use
        
    Returns:
        Embeddings tensor, shape [B, 768]
    """
    # Convert from [-1, 1] to [0, 1] range
    images = (images + 1) / 2
    
    # Convert to correct format for CLIP
    images = images.permute(0, 3, 1, 2)  # [B, C, H, W]
    images = images * 255
    
    # Convert to PIL images for the processor
    processed_images = []
    for img in images:
        # Ensure we have 3 channels (RGB)
        if img.shape[0] == 4:  # If 4 channels, take first 3
            img = img[:3]
        
        # Resize to CLIP expected size
        img = F.interpolate(img.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
        processed_images.append(img.cpu())
    
    # Process images with CLIP processor
    pixel_values = processor(images=processed_images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    
    # Get embeddings from CLIP model
    with torch.no_grad():
        outputs = clip_model(pixel_values)
        embeddings = outputs.pooler_output  # [B, 768]
    
    return embeddings

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
        num_classes=1,  # We'll use CLIP embeddings instead
        dropout=config.get('dropout', 0.0),
        ignore_dt=False,
        is_image=False  # We're using CLIP embeddings instead of images for conditioning
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

def generate_sample(model, x_0, clip_embeddings, steps=1, device=None):
    """
    Generate sample using the shortcut model.
    
    Args:
        model: Shortcut model
        x_0: Low-resolution input [B, H, W, C]
        clip_embeddings: CLIP embeddings for conditioning [B, 768]
        steps: Number of denoising steps
        device: Device to use
        
    Returns:
        Generated high-resolution image [B, H, W, C]
    """
    if device is None:
        device = next(model.parameters()).device
    
    batch_size = x_0.shape[0]
    
    # Start from low-resolution image
    x = x_0.clone()
    
    # Calculate step size
    d = 1.0 / steps
    
    # Initialize current time
    t = torch.zeros(batch_size, device=device)
    
    # For timestep conditioning
    dt_base = torch.ones(batch_size, dtype=torch.int64, device=device) * int(np.log2(steps))
    
    # Use CLIP embeddings for conditioning
    labels = clip_embeddings
    
    # Denoising loop
    for step in range(steps):
        # Forward pass to get velocity
        with torch.no_grad():
            v = model(x, t, dt_base, labels)
        
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
    
    # Create CLIP embedder
    clip_processor, clip_model = create_clip_embedder()
    clip_model.to(device)
    print("Created CLIP embedder")
    
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
        
        # Get CLIP embeddings from original low-res images before VAE encoding
        clip_embeddings = get_clip_embeddings(batch_low_res, clip_processor, clip_model, device)
        
        # Encode with VAE if needed
        if vae is not None:
            with torch.no_grad():
                batch_low_res_vae = vae.encode(batch_low_res)
        else:
            batch_low_res_vae = batch_low_res
        
        # Generate high-resolution samples
        batch_high_res = generate_sample(
            model, 
            batch_low_res_vae, 
            clip_embeddings, 
            steps=args.steps, 
            device=device
        )
        
        # Decode with VAE if needed
        if vae is not None:
            with torch.no_grad():
                batch_low_res_out = vae.decode(batch_low_res_vae)
                batch_high_res = vae.decode(batch_high_res)
        else:
            batch_low_res_out = batch_low_res
        
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
            low_res_img = batch_low_res_out[j].permute(2, 0, 1)
            save_image(
                low_res_img,
                low_res_output_path,
                normalize=True,
                value_range=(-1, 1)
            )
    
    print("Inference completed!")

if __name__ == '__main__':
    main()