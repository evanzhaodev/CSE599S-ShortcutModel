import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from typing import Tuple, Optional, List
import glob

class SuperResolutionDataset(Dataset):
    """Dataset for super-resolution with shortcut models."""
    
    def __init__(
        self, 
        high_res_dir: str,
        image_size: int = 256,
        low_res_factor: int = 4,
        return_paths: bool = False,
        max_images: Optional[int] = None,
        image_extensions: List[str] = ["*.jpg", "*.jpeg", "*.png", "*.JPEG"]
    ):
        """
        Initialize the dataset.
        
        Args:
            high_res_dir: Directory with high-resolution images
            image_size: Size of the high-resolution images (square)
            low_res_factor: Factor by which to reduce the low-resolution images
            return_paths: Whether to return the paths of the images
            max_images: Maximum number of images to load
            image_extensions: List of image extensions to search for
        """
        self.high_res_dir = high_res_dir
        self.image_size = image_size
        self.low_res_factor = low_res_factor
        self.return_paths = return_paths
        
        # Find all image files
        self.image_paths = []
        for ext in image_extensions:
            self.image_paths.extend(glob.glob(os.path.join(high_res_dir, "**", ext), recursive=True))
        
        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]
        
        # Set up transformations
        self.high_res_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Scale to [-1, 1]
        ])
        
        self.low_res_transform = transforms.Compose([
            transforms.Resize((image_size // low_res_factor, image_size // low_res_factor), 
                             interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.Resize((image_size, image_size), 
                             interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Scale to [-1, 1]
        ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Get a pair of low-res and high-res images.
        
        Returns:
            Tuple of (low_res_image, high_res_image, path) if return_paths is True
            otherwise (low_res_image, high_res_image)
        """
        path = self.image_paths[idx]
        
        # Load the image
        with Image.open(path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Create high-res and low-res versions
            high_res = self.high_res_transform(img)
            low_res = self.low_res_transform(img)
        
        # Convert to [H, W, C] format expected by the model
        high_res = high_res.permute(1, 2, 0)
        low_res = low_res.permute(1, 2, 0)
        
        if self.return_paths:
            return low_res, high_res, path
        else:
            return low_res, high_res

def create_dataloaders(
    train_dir: str, 
    val_dir: str, 
    batch_size: int, 
    image_size: int = 256,
    low_res_factor: int = 4,
    num_workers: int = 16,
    max_train_images: Optional[int] = None,
    max_val_images: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        train_dir: Directory with training images
        val_dir: Directory with validation images
        batch_size: Batch size for training
        image_size: Size of the high-resolution images
        low_res_factor: Factor by which to reduce the low-resolution images
        num_workers: Number of dataloader workers
        max_train_images: Maximum number of training images
        max_val_images: Maximum number of validation images
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    train_dataset = SuperResolutionDataset(
        train_dir, 
        image_size=image_size,
        low_res_factor=low_res_factor,
        max_images=max_train_images
    )
    
    val_dataset = SuperResolutionDataset(
        val_dir, 
        image_size=image_size,
        low_res_factor=low_res_factor,
        max_images=max_val_images
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_dataloader, val_dataloader