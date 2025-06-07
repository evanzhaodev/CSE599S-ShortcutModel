import os
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Optional, Tuple
import logging
import json
from torchvision.transforms.functional import to_pil_image
from torch.optim.lr_scheduler import LambdaLR

def setup_logger(name, log_file, level=logging.INFO):
    """Set up logger."""
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    
    return logger

def save_checkpoint(model, optimizer, scheduler, epoch, step, config, metrics, filename):
    """Save model checkpoint."""
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'epoch': epoch,
        'step': step,
        'config': config,
        'metrics': metrics
    }
    
    # If the model has an EMA module, save that too
    if hasattr(model, 'ema_model'):
        checkpoint['ema_model'] = model.ema_model.state_dict()
    
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, scheduler, filename, device):
    """Load model checkpoint."""
    try:
        # First try loading with weights_only=True (safer)
        checkpoint = torch.load(filename, map_location=device)
    except Exception as e:
        print(f"Warning: Could not load checkpoint with weights_only=True. Trying with weights_only=False: {e}")
        # Fall back to weights_only=False if needed
        checkpoint = torch.load(filename, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model'])
    
    # If the model has an EMA module and the checkpoint has it, load that too
    if hasattr(model, 'ema_model') and 'ema_model' in checkpoint:
        model.ema_model.load_state_dict(checkpoint['ema_model'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    if scheduler is not None and checkpoint['scheduler'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    return checkpoint['epoch'], checkpoint['step'], checkpoint['metrics']

def plot_image_grid(images, nrow=8, title=None, save_path=None, normalize=True):
    """Plot a grid of images."""
    # Convert images from [-1, 1] to [0, 1] if normalize is True
    if normalize:
        images = images * 0.5 + 0.5
    
    # Convert to [B, C, H, W] if needed
    if images.shape[-1] == 3:  # [B, H, W, C]
        images = images.permute(0, 3, 1, 2)
    
    # Create grid
    grid = torchvision.utils.make_grid(images, nrow=nrow, padding=2, normalize=not normalize)
    
    # Convert to numpy for plotting
    grid_np = grid.cpu().numpy().transpose(1, 2, 0)
    
    # Plot
    plt.figure(figsize=(15, 15))
    plt.imshow(grid_np)
    
    if title:
        plt.title(title)
    
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_warmup_cosine_lr_scheduler(
    optimizer, 
    warmup_steps: int, 
    max_steps: int, 
    min_lr: float = 0.0
):
    """
    Create a learning rate scheduler with warmup and cosine decay.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        max_steps: Total number of training steps
        min_lr: Minimum learning rate at the end of training
        
    Returns:
        PyTorch learning rate scheduler
    """
    def lr_lambda(step):
        # Linear warmup
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        
        # Cosine decay
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return min_lr + 0.5 * (1.0 - min_lr) * (1.0 + np.cos(np.pi * progress))
    
    return LambdaLR(optimizer, lr_lambda)

def update_ema(model, ema_model, decay: float = 0.999):
    """
    Update exponential moving average (EMA) of model weights.
    
    Args:
        model: Source model
        ema_model: EMA model to update
        decay: EMA decay rate (higher = slower updates)
    """
    with torch.no_grad():
        model_params = dict(model.named_parameters())
        ema_params = dict(ema_model.named_parameters())
        
        for name, param in model_params.items():
            # Skip if not in EMA model
            if name not in ema_params:
                continue
                
            ema_params[name].data.mul_(decay).add_(param.data, alpha=1 - decay)

class EMAModel:
    """
    Maintains exponential moving average of model weights.
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.ema_model = self._create_ema_model(model)
        self.update(decay=0.0)  # Initialize with model weights
        
    def _create_ema_model(self, model):
        """Create a copy of the model for EMA."""
        ema_model = type(model)(*model._get_constructor_args())
        return ema_model
    
    def update(self, decay=None):
        """Update EMA weights."""
        decay = self.decay if decay is None else decay
        update_ema(self.model, self.ema_model, decay)
    
    def state_dict(self):
        """Get state dictionary."""
        return self.ema_model.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load state dictionary."""
        self.ema_model.load_state_dict(state_dict)
    
    def eval(self):
        """Set to evaluation mode."""
        self.ema_model.eval()
    
    def __call__(self, *args, **kwargs):
        """Forward pass."""
        return self.ema_model(*args, **kwargs)

def process_image(img, normalize=True):
    """
    Process image for visualization.
    
    Args:
        img: Tensor or numpy array of shape [H, W, C] or [C, H, W]
        normalize: Whether to normalize from [-1, 1] to [0, 1]
        
    Returns:
        PIL Image
    """
    # Convert to tensor if numpy
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    
    # Ensure [C, H, W] format
    if img.shape[-1] == 3:  # [H, W, C]
        img = img.permute(2, 0, 1)
    
    # Normalize to [0, 1]
    if normalize:
        img = img * 0.5 + 0.5
    
    # Clamp values
    img = torch.clamp(img, 0, 1)
    
    # Convert to PIL
    img_pil = to_pil_image(img)
    
    return img_pil

def save_json(data, filename):
    """Save data to JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(filename):
    """Load data from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)