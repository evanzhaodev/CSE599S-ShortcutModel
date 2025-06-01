import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
import time
import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import random
import torchvision

from model import DiT, HOMOModel
from dataset import create_dataloaders
from targets import get_targets
from vae import StableVAE
from utils import (
    setup_logger, 
    save_checkpoint, 
    load_checkpoint, 
    plot_image_grid, 
    create_warmup_cosine_lr_scheduler,
    update_ema,
    save_json,
    load_json,
    process_image
)

def parse_args():
    parser = argparse.ArgumentParser(description='Train HOMO model for super-resolution')
    
    # Dataset arguments
    parser.add_argument('--train_dir', type=str, required=True, help='Directory with training images')
    parser.add_argument('--val_dir', type=str, required=True, help='Directory with validation images')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--image_size', type=int, default=256, help='Size of high-resolution images')
    parser.add_argument('--low_res_factor', type=int, default=4, help='Factor for low-resolution images')
    
    # Model arguments
    parser.add_argument('--hidden_size', type=int, default=768, help='Hidden size of the model')
    parser.add_argument('--patch_size', type=int, default=2, help='Patch size for DiT')
    parser.add_argument('--depth', type=int, default=12, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--mlp_ratio', type=int, default=4, help='MLP ratio')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--use_stable_vae', action='store_true', help='Use StableVAE')
    
    # HOMO arguments
    parser.add_argument('--use_homo', action='store_true', help='Use HOMO (High-Order Matching)')
    parser.add_argument('--homo_lambda_v', type=float, default=1.0, help='Weight for velocity loss')
    parser.add_argument('--homo_lambda_a', type=float, default=1.0, help='Weight for acceleration loss')
    parser.add_argument('--homo_lambda_sc', type=float, default=1.0, help='Weight for self-consistency loss')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--warmup', type=int, default=0, help='Warmup steps')
    parser.add_argument('--max_steps', type=int, default=1000000, help='Maximum number of training steps')
    parser.add_argument('--save_interval', type=int, default=10000, help='Save interval')
    parser.add_argument('--eval_interval', type=int, default=1000, help='Evaluation interval')
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval')
    parser.add_argument('--use_cosine', action='store_true', help='Use cosine learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloader')
    
    # Shortcut model arguments
    parser.add_argument('--denoise_timesteps', type=int, default=128, help='Number of denoising timesteps')
    parser.add_argument('--target_update_rate', type=float, default=0.999, help='EMA update rate')
    parser.add_argument('--bootstrap_every', type=int, default=8, help='Bootstrap every N samples')
    parser.add_argument('--bootstrap_ema', type=int, default=1, help='Use EMA for bootstrap')
    parser.add_argument('--bootstrap_dt_bias', type=int, default=0, help='Bias for dt sampling')
    parser.add_argument('--cfg_scale', type=float, default=0.0, help='Classifier-free guidance scale')
    parser.add_argument('--bootstrap_cfg', type=float, default=0.0, help='CFG scale for bootstrap')
    
    # Checkpoint arguments
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments
    save_json(vars(args), os.path.join(args.output_dir, 'args.json'))
    
    # Set up logger
    logger = setup_logger('train', os.path.join(args.output_dir, 'train.log'))
    logger.info(f"Arguments: {args}")
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(
        args.train_dir,
        args.val_dir,
        args.batch_size,
        args.image_size,
        args.low_res_factor,
        args.num_workers
    )
    logger.info(f"Created dataloaders. Train: {len(train_dataloader)}, Val: {len(val_dataloader)}")
    
    # Create VAE
    if args.use_stable_vae:
        vae = StableVAE()
        vae.to(device)
        vae.eval()
        logger.info("Using StableVAE")
    else:
        vae = None
    
    # Get sample image shape
    x_lr, x_hr = next(iter(train_dataloader))
    x_lr, x_hr = x_lr.to(device), x_hr.to(device)
    
    if vae is not None:
        with torch.no_grad():
            x_hr = vae.encode(x_hr)
            x_lr = vae.encode(x_lr)
    
    input_shape = x_hr.shape[1:]  # [H, W, C]
    image_channels = input_shape[-1]
    
    logger.info(f"Input shape: {input_shape}")
    
    # Create model
    if args.use_homo:
        logger.info("Using HOMO model with high-order matching")
        model = HOMOModel(
            patch_size=args.patch_size,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            out_channels=image_channels,
            class_dropout_prob=0.1,
            num_classes=1,
            dropout=args.dropout,
            use_low_res_cond=True,
            ignore_dt=False
        )
    else:
        logger.info("Using standard DiT model")
        model = DiT(
            patch_size=args.patch_size,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            out_channels=image_channels,
            class_dropout_prob=0.1,
            num_classes=1,
            dropout=args.dropout,
            use_low_res_cond=True,
            ignore_dt=False
        )
    model.to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create EMA model
    if args.use_homo:
        ema_model = HOMOModel(
            patch_size=args.patch_size,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            out_channels=image_channels,
            class_dropout_prob=0.1,
            num_classes=1,
            dropout=args.dropout,
            use_low_res_cond=True,
            ignore_dt=False
        )
    else:
        ema_model = DiT(
            patch_size=args.patch_size,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            out_channels=image_channels,
            class_dropout_prob=0.1,
            num_classes=1,
            dropout=args.dropout,
            use_low_res_cond=True,
            ignore_dt=False
        )
    ema_model.to(device)
    # Initialize EMA model with model weights
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        ema_param.data.copy_(param.data)
    ema_model.eval()
    
    # Create learning rate scheduler
    if args.use_cosine:
        scheduler = create_warmup_cosine_lr_scheduler(
            optimizer,
            args.warmup,
            args.max_steps
        )
    elif args.warmup > 0:
        def lr_lambda(step):
            if step < args.warmup:
                return step / args.warmup
            return 1.0
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None
    
    # Set up tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))
    
    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    best_metrics = {}
    if args.resume is not None:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch, global_step, best_metrics = load_checkpoint(
            model, optimizer, scheduler, args.resume, device
        )
        # Update EMA model with loaded weights
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            ema_param.data.copy_(param.data)
    
    # Training loop
    logger.info("Starting training...")
    model.train()
    
    for epoch in range(start_epoch, args.max_steps // len(train_dataloader) + 1):
        for i, (x_lr, x_hr) in enumerate(train_dataloader):
            if global_step >= args.max_steps:
                break
                
            x_lr, x_hr = x_lr.to(device), x_hr.to(device)
            
            if vae is not None:
                with torch.no_grad():
                    x_hr = vae.encode(x_hr)
                    x_lr = vae.encode(x_lr)
            
            # Get targets
            if args.use_homo:
                x_t, v_t, a_t, x_noise, x_low_res, t, dt_base, labels, info = get_targets(
                    batch_size=args.batch_size,
                    x_1=x_hr,
                    x_0=x_lr,
                    model=ema_model if global_step > 0 else None,
                    use_ema=args.bootstrap_ema,
                    bootstrap_every=args.bootstrap_every,
                    bootstrap_ema=args.bootstrap_ema,
                    bootstrap_cfg=args.bootstrap_cfg,
                    bootstrap_dt_bias=args.bootstrap_dt_bias,
                    denoise_timesteps=args.denoise_timesteps,
                    cfg_scale=args.cfg_scale,
                    num_classes=1,
                    device=device,
                    use_homo=True
                )
                
                # Forward pass
                v_pred = model.forward_velocity(x_t, x_low_res, t, dt_base, labels, train=True)
                a_pred = model.forward_acceleration(x_t, v_pred, x_low_res, t, dt_base, labels, train=True)
                
                # Self-consistency target for velocity
                with torch.no_grad():
                    model.eval()  # Ensure model is in eval mode
                    # Take a small step
                    d = 1.0 / args.denoise_timesteps
                    t_next = torch.clamp(t + d, 0.0, 1.0)  # Ensure t stays in [0, 1]
                    x_next = x_t + d * v_pred.detach() + (d**2 / 2) * a_pred.detach()
                    x_next = torch.clamp(x_next, -4, 4)
                    
                    # Get velocity at next step
                    v_next = model.forward_velocity(x_next, x_low_res, t_next, dt_base, labels, train=False)
                    v_sc_target = (v_pred.detach() + v_next.detach()) / 2
                    model.train()  # Set back to train mode
                
                # Compute losses
                loss_v = torch.mean((v_pred - v_t) ** 2)
                loss_a = torch.mean((a_pred - a_t) ** 2)
                loss_sc = torch.mean((v_pred - v_sc_target.detach()) ** 2)
                
                # Total loss with weights
                loss = args.homo_lambda_v * loss_v + \
                       args.homo_lambda_a * loss_a + \
                       args.homo_lambda_sc * loss_sc
                
            else:
                # Original training without HOMO
                x_t, v_t, _, x_noise, x_low_res, t, dt_base, labels, info = get_targets(
                    batch_size=args.batch_size,
                    x_1=x_hr,
                    x_0=x_lr,
                    model=ema_model if global_step > 0 else None,
                    use_ema=args.bootstrap_ema,
                    bootstrap_every=args.bootstrap_every,
                    bootstrap_ema=args.bootstrap_ema,
                    bootstrap_cfg=args.bootstrap_cfg,
                    bootstrap_dt_bias=args.bootstrap_dt_bias,
                    denoise_timesteps=args.denoise_timesteps,
                    cfg_scale=args.cfg_scale,
                    num_classes=1,
                    device=device,
                    use_homo=False
                )
                
                # Forward pass
                v_pred = model(x_t, x_low_res, t, dt_base, labels)
                
                # Compute loss
                mse_v = torch.mean((v_pred - v_t) ** 2, dim=(1, 2, 3))
                loss = torch.mean(mse_v)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient norm for monitoring
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            
            # Optimizer step
            optimizer.step()
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step()
                
            # Update EMA model
            update_ema(model, ema_model, args.target_update_rate)
            
            # Log
            if global_step % args.log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                
                log_dict = {
                    'train/loss': loss.item(),
                    'train/grad_norm': grad_norm,
                    'train/lr': lr,
                }
                
                if args.use_homo:
                    log_dict['train/loss_velocity'] = loss_v.item()
                    log_dict['train/loss_acceleration'] = loss_a.item()
                    log_dict['train/loss_self_consistency'] = loss_sc.item()
                
                # Add info metrics
                for k, v in info.items():
                    if isinstance(v, torch.Tensor):
                        log_dict[f'train/{k}'] = v.item()
                
                # Add to tensorboard
                for k, v in log_dict.items():
                    writer.add_scalar(k, v, global_step)
                
                # Log to console
                if args.use_homo:
                    logger.info(
                        f"Epoch: {epoch}, Step: {global_step}, Loss: {loss.item():.4f}, "
                        f"Loss_v: {loss_v.item():.4f}, Loss_a: {loss_a.item():.4f}, "
                        f"Loss_sc: {loss_sc.item():.4f}, Grad Norm: {grad_norm:.4f}, LR: {lr:.6f}"
                    )
                else:
                    logger.info(
                        f"Epoch: {epoch}, Step: {global_step}, Loss: {loss.item():.4f}, "
                        f"Grad Norm: {grad_norm:.4f}, LR: {lr:.6f}"
                    )
            
            # Evaluation
            if global_step % args.eval_interval == 0:
                model.eval()
                val_losses = []
                val_losses_v = []
                val_losses_a = []
                
                with torch.no_grad():
                    for x_lr_val, x_hr_val in val_dataloader:
                        x_lr_val, x_hr_val = x_lr_val.to(device), x_hr_val.to(device)
                        
                        if vae is not None:
                            x_hr_val = vae.encode(x_hr_val)
                            x_lr_val = vae.encode(x_lr_val)
                        
                        # Generate random noise
                        batch_size_val = x_lr_val.shape[0]
                        x_noise_val = torch.randn_like(x_hr_val)
                        
                        # Sample t uniformly
                        t_val = torch.rand(batch_size_val, device=device)
                        t_full_val = t_val.view(-1, 1, 1, 1)
                        
                        # Interpolate
                        x_t_val = (1 - t_full_val) * x_noise_val + t_full_val * x_hr_val
                        
                        # Compute target velocity and acceleration
                        v_t_val = x_hr_val - x_noise_val
                        a_t_val = torch.zeros_like(v_t_val)
                        
                        # Default to flow-matching dt
                        dt_base_val = torch.ones(batch_size_val, dtype=torch.int64, device=device) * int(np.log2(args.denoise_timesteps))
                        
                        # Zero labels for unconditional
                        labels_val = torch.zeros(batch_size_val, dtype=torch.long, device=device)
                        
                        if args.use_homo:
                            # Forward pass
                            v_pred_val = model.forward_velocity(x_t_val, x_lr_val, t_val, dt_base_val, labels_val, train=False)
                            a_pred_val = model.forward_acceleration(x_t_val, v_pred_val, x_lr_val, t_val, dt_base_val, labels_val, train=False)
                            
                            # Compute losses
                            loss_v_val = torch.mean((v_pred_val - v_t_val) ** 2)
                            loss_a_val = torch.mean((a_pred_val - a_t_val) ** 2)
                            val_loss = loss_v_val + loss_a_val
                            
                            val_losses_v.append(loss_v_val.item())
                            val_losses_a.append(loss_a_val.item())
                        else:
                            # Forward pass
                            v_pred_val = model(x_t_val, x_lr_val, t_val, dt_base_val, labels_val)
                            
                            # Compute loss
                            mse_v_val = torch.mean((v_pred_val - v_t_val) ** 2, dim=(1, 2, 3))
                            val_loss = torch.mean(mse_v_val)
                        
                        val_losses.append(val_loss.item())
                
                # Compute average validation loss
                avg_val_loss = np.mean(val_losses)
                
                # Log validation loss
                writer.add_scalar('val/loss', avg_val_loss, global_step)
                
                if args.use_homo:
                    avg_val_loss_v = np.mean(val_losses_v)
                    avg_val_loss_a = np.mean(val_losses_a)
                    writer.add_scalar('val/loss_velocity', avg_val_loss_v, global_step)
                    writer.add_scalar('val/loss_acceleration', avg_val_loss_a, global_step)
                    logger.info(f"Validation Loss: {avg_val_loss:.4f}, Loss_v: {avg_val_loss_v:.4f}, Loss_a: {avg_val_loss_a:.4f}")
                else:
                    logger.info(f"Validation Loss: {avg_val_loss:.4f}")
                
                # Generate samples for visualization
                with torch.no_grad():
                    # Choose a fixed low-res image for visualization
                    x_lr_sample = x_lr_val[:8].clone()
                    
                    # Generate one-step sample
                    one_step_images = generate_sample(
                        model=ema_model,
                        x_low_res=x_lr_sample,
                        steps=1,
                        device=device,
                        cfg_scale=args.cfg_scale,
                        use_homo=args.use_homo
                    )
                    
                    # Generate multi-step sample (e.g., 8 steps)
                    multi_step_images = generate_sample(
                        model=ema_model,
                        x_low_res=x_lr_sample,
                        steps=8,
                        device=device,
                        cfg_scale=args.cfg_scale,
                        use_homo=args.use_homo
                    )
                    
                    # Decode if using VAE
                    if vae is not None:
                        x_lr_sample = vae.decode(x_lr_sample)
                        one_step_images = vae.decode(one_step_images)
                        multi_step_images = vae.decode(multi_step_images)
                    
                    # Visualize
                    all_images = torch.cat([x_lr_sample, one_step_images, multi_step_images], dim=0)
                    grid = torchvision.utils.make_grid(
                        all_images.permute(0, 3, 1, 2), 
                        nrow=8, 
                        normalize=True,
                        value_range=(-1, 1)
                    )
                    writer.add_image('samples', grid, global_step)
                    
                    # Save images
                    os.makedirs(os.path.join(args.output_dir, 'samples'), exist_ok=True)
                    torchvision.utils.save_image(
                        all_images.permute(0, 3, 1, 2),
                        os.path.join(args.output_dir, 'samples', f'step_{global_step}.png'),
                        nrow=8,
                        normalize=True,
                        value_range=(-1, 1)
                    )
                
                model.train()
            
            # Save checkpoint
            if global_step % args.save_interval == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    step=global_step,
                    config=vars(args),
                    metrics={'val_loss': avg_val_loss if 'avg_val_loss' in locals() else 0.0},
                    filename=os.path.join(args.output_dir, f'model_step_{global_step}.pt')
                )
                
                # Save EMA model
                torch.save(
                    ema_model.state_dict(),
                    os.path.join(args.output_dir, f'ema_model_step_{global_step}.pt')
                )
                
                logger.info(f"Saved checkpoint at step {global_step}")
            
            global_step += 1
    
    # Save final model
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        step=global_step,
        config=vars(args),
        metrics={'val_loss': avg_val_loss if 'avg_val_loss' in locals() else 0.0},
        filename=os.path.join(args.output_dir, 'model_final.pt')
    )
    
    # Save final EMA model
    torch.save(
        ema_model.state_dict(),
        os.path.join(args.output_dir, 'ema_model_final.pt')
    )
    
    logger.info("Training completed!")

def generate_sample(model, x_low_res, steps=1, device=None, cfg_scale=0.0, use_homo=False):
    """
    Generate sample using the model.
    
    Args:
        model: Model (DiT or HOMOModel)
        x_low_res: Low-resolution conditioning input [B, H, W, C]
        steps: Number of denoising steps
        device: Device to use
        cfg_scale: Classifier-free guidance scale
        use_homo: Whether to use HOMO sampling
        
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
        with torch.no_grad():
            if use_homo:
                # Get velocity and acceleration
                v = model.forward_velocity(x, x_low_res, t, dt_base, labels, train=False)
                a = model.forward_acceleration(x, v, x_low_res, t, dt_base, labels, train=False)
                
                # Update using second-order integration
                x = x + v * d + a * (d**2 / 2)
                x = torch.clamp(x, -4, 4)  # Clamp to prevent instability
            else:
                # Original first-order update
                if cfg_scale > 0:
                    # Conditional pass
                    v_cond = model(x, x_low_res, t, dt_base, labels)
                    
                    # Unconditional pass
                    v_uncond = model(x, x_low_res, t, dt_base, torch.ones_like(labels) * model.num_classes)
                    
                    # Apply classifier-free guidance
                    v = v_uncond + cfg_scale * (v_cond - v_uncond)
                else:
                    v = model(x, x_low_res, t, dt_base, labels)
                
                # Update x using Euler method
                x = x + v * d
                x = torch.clamp(x, -4, 4)  # Clamp to prevent instability
        
        # Update time
        t = torch.clamp(t + d, 0.0, 1.0)
    
    return x

if __name__ == '__main__':
    main()