import torch
import numpy as np
import math

def compute_plucker_from_spherical(x, y, z):
    """
    Compute Plucker coordinates from spherical angles for inference.
    
    Args:
        x: polar angle (degrees)
        y: azimuth angle (degrees) 
        z: radius/distance
        
    Returns:
        torch.Tensor: 6D Plucker coordinates [d_x, d_y, d_z, m_x, m_y, m_z]
    """
    azimuth_rad = math.radians(y)
    polar_rad = math.radians(x)
    
    # Condition camera at origin
    center_cond = np.array([0., 0., 0.])
    
    # Target camera position
    center_target = np.array([
        z * np.cos(azimuth_rad) * np.cos(polar_rad),
        z * np.sin(azimuth_rad) * np.cos(polar_rad),
        z * np.sin(polar_rad)
    ])
    
    # Direction vector
    direction = center_target - center_cond
    if np.linalg.norm(direction) < 1e-8:
        direction = np.array([1.0, 0.0, 0.0])
    d_norm = direction / np.linalg.norm(direction)
    
    # Moment vector
    moment = np.cross(center_cond, d_norm)
    if np.linalg.norm(moment) > 1e-8:
        m_norm = moment / np.linalg.norm(moment)
    else:
        m_norm = np.zeros(3)
    
    plucker_coords = np.concatenate([d_norm, m_norm])
    return torch.tensor(plucker_coords, dtype=torch.float32)

def load_model_with_mode_detection(config, ckpt, device, verbose=False):
    """Load model and auto-detect if it uses plucker coordinates"""
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    
    # Detect mode from checkpoint
    cc_weight_key = "cc_projection.weight"
    use_plucker = False
    if cc_weight_key in sd:
        input_dim = sd[cc_weight_key].shape[1]
        if input_dim == 778:
            use_plucker = True
            print("Detected Plucker mode from checkpoint")
        elif input_dim == 772:
            use_plucker = False
            print("Detected vanilla mode from checkpoint")
        else:
            print(f"Warning: Unknown cc_projection dimension {input_dim}, assuming vanilla")
    
    # Override config with detected mode
    config.model.params.use_plucker = use_plucker
    
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.to(device)
    model.eval()
    return model 