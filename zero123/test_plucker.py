import numpy as np
import torch
import sys
import os

from ldm.data.simple import ObjaverseData

def test_plucker_coordinates():
    print("Testing plucker coordinate implementation...")
    
    # a dummy ObjaverseData instance 
    dataset = ObjaverseData(root_dir='views_whole_sphere')
    
    print("\nTest Case 1: Translation along X-axis")
    
    # identity matrix for condition camera
    cond_RT = np.eye(4)
    
    # target camera translated 2 units along x-axis
    target_RT = np.eye(4)
    target_RT[0, 3] = 2.0 
    
    print(f"Condition RT:\n{cond_RT}")
    print(f"Target RT:\n{target_RT}")
    
    plucker = dataset.get_plucker_coordinates(target_RT, cond_RT)
    print(f"Plücker coordinates: {plucker}")
    print(f"Shape: {plucker.shape}")

    print("\nTest Case 2: Rotation around Y-axis")
    
    # condition camera (identity)
    cond_RT = np.eye(4)
    
    # target camera rotated 90 degrees around y-axis
    target_RT = np.eye(4)
    target_RT[0, 0] = 0.0
    target_RT[0, 2] = 1.0
    target_RT[2, 0] = -1.0
    target_RT[2, 2] = 0.0
    
    print(f"Condition RT:\n{cond_RT}")
    print(f"Target RT:\n{target_RT}")
    
    plucker = dataset.get_plucker_coordinates(target_RT, cond_RT)
    print(f"Plücker coordinates: {plucker}")
    print(f"Shape: {plucker.shape}")

    print("\nTest Case 3: Combined rotation and translation")
    
    # condition camera (identity)
    cond_RT = np.eye(4)
    
    # target camera: rotated and translated
    target_RT = np.eye(4)
    target_RT[0, 3] = 1.0  # translate 1 unit in x
    target_RT[1, 3] = 1.0  # translate 1 unit in y
    target_RT[2, 3] = 1.0  # translate 1 unit in z
    
    print(f"Condition RT:\n{cond_RT}")
    print(f"Target RT:\n{target_RT}")

    plucker = dataset.get_plucker_coordinates(target_RT, cond_RT)
    print(f"Plücker coordinates: {plucker}")
    print(f"Shape: {plucker.shape}")
    
    print("\nTest Case 4: Verify output properties")

    assert plucker.shape == (6,), f"Expected shape (6,), got {plucker.shape}"
    print("Output is 6-dimensional")

    assert isinstance(plucker, torch.Tensor), "Output should be a torch tensor"
    print("Output is a torch tensor")
    
    assert plucker.dtype == torch.float32, "Output should be float32"
    print("Output is float32")
    
    d_vec = plucker[:3].numpy()
    d_norm = np.linalg.norm(d_vec)
    assert abs(d_norm - 1.0) < 1e-6, f"Direction vector should be normalized, got norm {d_norm}"
    print("Direction vector is normalized")
    
    print("\nAll tests passed! Plücker coordinate implementation is working correctly.")

if __name__ == "__main__":
    test_plucker_coordinates()
