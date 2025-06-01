import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional

def get_targets(
    batch_size: int,
    x_1: torch.Tensor,  # High-res images (target)
    x_0: torch.Tensor,  # Low-res images (conditioning)
    model: torch.nn.Module,
    use_ema: bool = True,
    bootstrap_every: int = 8,
    bootstrap_ema: int = 1,
    bootstrap_cfg: float = 0.0,
    bootstrap_dt_bias: int = 0,
    denoise_timesteps: int = 128,
    cfg_scale: float = 0.0,
    num_classes: int = 1000,
    force_t: float = -1,
    force_dt: float = -1,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    use_homo: bool = True  # Whether to use high-order matching
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Generate targets for the HOMO model with first and second order terms.
    
    Returns:
        Tuple of (x_t, v_t, a_t, x_noise, x_low_res, t, dt_base, labels, info)
        where v_t is velocity target and a_t is acceleration target
    """
    info = {}
    
    # 1) =========== Sample dt. ============
    bootstrap_batchsize = batch_size // bootstrap_every
    log2_sections = np.log2(denoise_timesteps).astype(np.int32)
    if bootstrap_dt_bias == 0:
        dt_base = torch.repeat_interleave(
            torch.tensor(log2_sections - 1 - torch.arange(log2_sections), device=device),
            bootstrap_batchsize // log2_sections
        )
        dt_base = torch.cat([dt_base, torch.zeros(bootstrap_batchsize - dt_base.shape[0], device=device)])
        num_dt_cfg = bootstrap_batchsize // log2_sections
    else:
        dt_base = torch.repeat_interleave(
            torch.tensor(log2_sections - 1 - torch.arange(log2_sections - 2), device=device),
            (bootstrap_batchsize // 2) // log2_sections
        )
        dt_base = torch.cat([
            dt_base, 
            torch.ones(bootstrap_batchsize // 4, device=device), 
            torch.zeros(bootstrap_batchsize // 4, device=device)
        ])
        dt_base = torch.cat([dt_base, torch.zeros(bootstrap_batchsize - dt_base.shape[0], device=device)])
        num_dt_cfg = (bootstrap_batchsize // 2) // log2_sections
    
    force_dt_vec = torch.ones(bootstrap_batchsize, device=device) * force_dt
    dt_base = torch.where(force_dt_vec != -1, force_dt_vec, dt_base)
    dt = 1 / (2 ** dt_base)
    dt_base_bootstrap = dt_base + 1
    dt_bootstrap = dt / 2
    
    # 2) =========== Sample t. ============
    dt_sections = torch.pow(2, dt_base)
    
    t_rand = torch.rand(bootstrap_batchsize, device=device)
    t = (t_rand * dt_sections).floor()
    
    t = t / dt_sections
    force_t_vec = torch.ones(bootstrap_batchsize, device=device) * force_t
    t = torch.where(force_t_vec != -1, force_t_vec, t)
    t_full = t[:, None, None, None]
    
    # 3) =========== Generate Bootstrap Targets ============
    bst_x1 = x_1[:bootstrap_batchsize]
    bst_x0 = x_0[:bootstrap_batchsize]
    
    x_noise = torch.randn_like(bst_x1, device=device)
    
    x_t = (1 - t_full) * x_noise + t_full * bst_x1
    
    bst_labels = torch.zeros(bootstrap_batchsize, dtype=torch.long, device=device)
    
    # Initialize default targets
    v_target = bst_x1 - x_noise  # Default velocity target
    a_target = torch.zeros_like(v_target)  # Default acceleration target
    
    if model is not None and use_homo:
        call_model_fn = model if not bootstrap_ema else model.ema_model if hasattr(model, 'ema_model') else model
        
        if not bootstrap_cfg:
            # First step - get velocity
            v_b1 = call_model_fn.forward_velocity(x_t, bst_x0, t, dt_base_bootstrap, bst_labels)
            
            # Calculate intermediate point using velocity
            t2 = t + dt_bootstrap
            x_t2 = x_t + dt_bootstrap[:, None, None, None] * v_b1
            x_t2 = torch.clamp(x_t2, -4, 4)
            
            # Get acceleration at first point
            a_b1 = call_model_fn.forward_acceleration(x_t, v_b1, bst_x0, t, dt_base_bootstrap, bst_labels)
            
            # Second step - get velocity at intermediate point
            v_b2 = call_model_fn.forward_velocity(x_t2, bst_x0, t2, dt_base_bootstrap, bst_labels)
            
            # Get acceleration at second point
            a_b2 = call_model_fn.forward_acceleration(x_t2, v_b2, bst_x0, t2, dt_base_bootstrap, bst_labels)
            
            # Target is average of two steps (self-consistency)
            v_target = (v_b1 + v_b2) / 2
            a_target = (a_b1 + a_b2) / 2
        else:
            # With CFG - similar process but with conditional/unconditional
            x_t_extra = torch.cat([x_t, x_t[:num_dt_cfg]], dim=0)
            low_res_extra = torch.cat([bst_x0, bst_x0[:num_dt_cfg]], dim=0) 
            t_extra = torch.cat([t, t[:num_dt_cfg]], dim=0)
            dt_base_extra = torch.cat([dt_base_bootstrap, dt_base_bootstrap[:num_dt_cfg]], dim=0)
            
            labels_extra = torch.cat([
                bst_labels, 
                torch.ones(num_dt_cfg, dtype=torch.long, device=device) * num_classes
            ], dim=0)
            
            # First step
            v_b1_raw = call_model_fn.forward_velocity(x_t_extra, low_res_extra, t_extra, dt_base_extra, labels_extra)
            v_b_cond = v_b1_raw[:bst_x1.shape[0]]
            v_b_uncond = v_b1_raw[bst_x1.shape[0]:]
            
            v_cfg = v_b_uncond + cfg_scale * (v_b_cond[:num_dt_cfg] - v_b_uncond)
            v_b1 = torch.cat([v_cfg, v_b_cond[num_dt_cfg:]], dim=0)
            
            # Get acceleration with CFG
            a_b1_raw = call_model_fn.forward_acceleration(x_t_extra, v_b1_raw, low_res_extra, t_extra, dt_base_extra, labels_extra)
            a_b_cond = a_b1_raw[:bst_x1.shape[0]]
            a_b_uncond = a_b1_raw[bst_x1.shape[0]:]
            
            a_cfg = a_b_uncond + cfg_scale * (a_b_cond[:num_dt_cfg] - a_b_uncond)
            a_b1 = torch.cat([a_cfg, a_b_cond[num_dt_cfg:]], dim=0)
            
            # Second step
            t2 = t + dt_bootstrap
            x_t2 = x_t + dt_bootstrap[:, None, None, None] * v_b1 + \
                   (dt_bootstrap[:, None, None, None] ** 2) / 2 * a_b1
            x_t2 = torch.clamp(x_t2, -4, 4)
            
            x_t2_extra = torch.cat([x_t2, x_t2[:num_dt_cfg]], dim=0)
            t2_extra = torch.cat([t2, t2[:num_dt_cfg]], dim=0)
            
            v_b2_raw = call_model_fn.forward_velocity(x_t2_extra, low_res_extra, t2_extra, dt_base_extra, labels_extra)
            v_b2_cond = v_b2_raw[:bst_x1.shape[0]]
            v_b2_uncond = v_b2_raw[bst_x1.shape[0]:]
            
            v_b2_cfg = v_b2_uncond + cfg_scale * (v_b2_cond[:num_dt_cfg] - v_b2_uncond)
            v_b2 = torch.cat([v_b2_cfg, v_b2_cond[num_dt_cfg:]], dim=0)
            
            a_b2_raw = call_model_fn.forward_acceleration(x_t2_extra, v_b2_raw, low_res_extra, t2_extra, dt_base_extra, labels_extra)
            a_b2_cond = a_b2_raw[:bst_x1.shape[0]]
            a_b2_uncond = a_b2_raw[bst_x1.shape[0]:]
            
            a_b2_cfg = a_b2_uncond + cfg_scale * (a_b2_cond[:num_dt_cfg] - a_b2_uncond)
            a_b2 = torch.cat([a_b2_cfg, a_b2_cond[num_dt_cfg:]], dim=0)
            
            v_target = (v_b1 + v_b2) / 2
            a_target = (a_b1 + a_b2) / 2
        
        v_target = torch.clamp(v_target, -4, 4)
        a_target = torch.clamp(a_target, -4, 4)
    elif model is not None and not use_homo:
        # Fallback to original first-order only method
        call_model_fn = model if not bootstrap_ema else model.ema_model if hasattr(model, 'ema_model') else model
        
        if not bootstrap_cfg:
            v_b1 = call_model_fn(x_t, bst_x0, t, dt_base_bootstrap, bst_labels)
            
            t2 = t + dt_bootstrap
            x_t2 = x_t + dt_bootstrap[:, None, None, None] * v_b1
            x_t2 = torch.clamp(x_t2, -4, 4)
            
            v_b2 = call_model_fn(x_t2, bst_x0, t2, dt_base_bootstrap, bst_labels)
            
            v_target = (v_b1 + v_b2) / 2
        else:
            # CFG handling for first-order only
            x_t_extra = torch.cat([x_t, x_t[:num_dt_cfg]], dim=0)
            low_res_extra = torch.cat([bst_x0, bst_x0[:num_dt_cfg]], dim=0) 
            t_extra = torch.cat([t, t[:num_dt_cfg]], dim=0)
            dt_base_extra = torch.cat([dt_base_bootstrap, dt_base_bootstrap[:num_dt_cfg]], dim=0)
            
            labels_extra = torch.cat([
                bst_labels, 
                torch.ones(num_dt_cfg, dtype=torch.long, device=device) * num_classes
            ], dim=0)
            
            v_b1_raw = call_model_fn(x_t_extra, low_res_extra, t_extra, dt_base_extra, labels_extra)
            v_b_cond = v_b1_raw[:bst_x1.shape[0]]
            v_b_uncond = v_b1_raw[bst_x1.shape[0]:]
            
            v_cfg = v_b_uncond + cfg_scale * (v_b_cond[:num_dt_cfg] - v_b_uncond)
            v_b1 = torch.cat([v_cfg, v_b_cond[num_dt_cfg:]], dim=0)
            
            t2 = t + dt_bootstrap
            x_t2 = x_t + dt_bootstrap[:, None, None, None] * v_b1
            x_t2 = torch.clamp(x_t2, -4, 4)
            
            x_t2_extra = torch.cat([x_t2, x_t2[:num_dt_cfg]], dim=0)
            t2_extra = torch.cat([t2, t2[:num_dt_cfg]], dim=0)
            
            v_b2_raw = call_model_fn(x_t2_extra, low_res_extra, t2_extra, dt_base_extra, labels_extra)
            v_b2_cond = v_b2_raw[:bst_x1.shape[0]]
            v_b2_uncond = v_b2_raw[bst_x1.shape[0]:]
            
            v_b2_cfg = v_b2_uncond + cfg_scale * (v_b2_cond[:num_dt_cfg] - v_b2_uncond)
            v_b2 = torch.cat([v_b2_cfg, v_b2_cond[num_dt_cfg:]], dim=0)
            
            v_target = (v_b1 + v_b2) / 2
        
        v_target = torch.clamp(v_target, -4, 4)
    
    bst_v = v_target
    bst_a = a_target
    bst_dt = dt_base
    bst_t = t
    bst_xt = x_t
    bst_noise = x_noise
    bst_l = bst_labels
    bst_low_res = bst_x0
    
    # 4) =========== Generate Flow-Matching Targets ============
    labels_dropped = torch.zeros(x_1.shape[0], dtype=torch.long, device=device)
    
    t = torch.randint(0, denoise_timesteps, (x_1.shape[0],), device=device).float()
    t = t / denoise_timesteps
    force_t_vec = torch.ones(x_1.shape[0], device=device) * force_t
    t = torch.where(force_t_vec != -1, force_t_vec, t)
    t_full = t[:, None, None, None]
    
    x_noise_flow = torch.randn_like(x_1, device=device)
    
    x_t_flow = (1 - t_full) * x_noise_flow + t_full * x_1
    v_t_flow = x_1 - x_noise_flow
    a_t_flow = torch.zeros_like(v_t_flow)  # Zero acceleration for flow matching
    
    dt_flow = np.log2(denoise_timesteps).astype(np.int32)
    dt_base_flow = torch.ones(x_1.shape[0], dtype=torch.int64, device=device) * dt_flow
    
    # ==== 5) Merge Flow+Bootstrap ====
    bst_size = batch_size // bootstrap_every
    bst_size_data = batch_size - bst_size
    
    x_t_combined = torch.cat([bst_xt, x_t_flow[:bst_size_data]], dim=0)
    x_noise_combined = torch.cat([bst_noise, x_noise_flow[:bst_size_data]], dim=0)
    x_low_res_combined = torch.cat([bst_low_res, x_0[:bst_size_data]], dim=0)
    t_combined = torch.cat([bst_t, t[:bst_size_data]], dim=0)
    dt_base_combined = torch.cat([bst_dt, dt_base_flow[:bst_size_data]], dim=0)
    v_t_combined = torch.cat([bst_v, v_t_flow[:bst_size_data]], dim=0)
    a_t_combined = torch.cat([bst_a, a_t_flow[:bst_size_data]], dim=0)
    labels_combined = torch.cat([bst_l, labels_dropped[:bst_size_data]], dim=0)
    
    info['bootstrap_ratio'] = torch.mean((dt_base_combined != dt_flow).float())
    
    if model is not None and use_homo:
        info['v_magnitude_bootstrap'] = torch.sqrt(torch.mean(torch.square(bst_v)))
        info['a_magnitude_bootstrap'] = torch.sqrt(torch.mean(torch.square(bst_a)))
        if 'v_b1' in locals():
            info['v_magnitude_b1'] = torch.sqrt(torch.mean(torch.square(v_b1)))
            info['v_magnitude_b2'] = torch.sqrt(torch.mean(torch.square(v_b2)))
        if 'a_b1' in locals():
            info['a_magnitude_b1'] = torch.sqrt(torch.mean(torch.square(a_b1)))
            info['a_magnitude_b2'] = torch.sqrt(torch.mean(torch.square(a_b2)))
    
    return x_t_combined, v_t_combined, a_t_combined, x_noise_combined, x_low_res_combined, t_combined, dt_base_combined, labels_combined, info