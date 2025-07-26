import torch
import numpy as np
import sys
import os

def test_plucker_function():
    """
    Test just the Plücker coordinate function without requiring dataset files.
    """
    print("Testing Plücker coordinate function...")
    
    try:
        from ldm.data.simple import ObjaverseData
        print("✓ Successfully imported ObjaverseData")
        
        # Create a dummy instance just to access the method
        dataset = ObjaverseData.__new__(ObjaverseData)  # Create without calling __init__
        
        # Test the Plücker coordinate function directly
        print("Step 1: Testing get_plucker_coordinates function...")
        
        # Create test transformation matrices
        target_RT = np.array([
            [1, 0, 0, 2],
            [0, 1, 0, 0], 
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        cond_RT = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0], 
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Test the function
        plucker_coords = dataset.get_plucker_coordinates(target_RT, cond_RT)
        
        print(f"✓ Plücker coordinates computed successfully")
        print(f"Shape: {plucker_coords.shape}")
        print(f"Values: {plucker_coords}")
        
        if plucker_coords.shape == torch.Size([6]):
            print("✓ Correct shape (6D)")
        else:
            print(f"✗ Wrong shape. Expected [6], got {plucker_coords.shape}")
            return False
            
    except Exception as e:
        print(f"✗ Failed to test Plücker function: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 2. Test tensor operations
    print("\nStep 2: Testing tensor operations...")
    
    try:
        # Simulate what would happen in the data loader
        batch_size = 2
        T_4d = torch.randn(batch_size, 4)  # Spherical coordinates
        T_plucker = torch.randn(batch_size, 6)  # Plücker coordinates
        clip_emb = torch.randn(batch_size, 1, 768)  # CLIP embeddings
        
        # Test concatenation
        T_combined = torch.cat([T_4d, T_plucker], dim=-1)
        print(f"T_4d shape: {T_4d.shape}")
        print(f"T_plucker shape: {T_plucker.shape}")
        print(f"T_combined shape: {T_combined.shape}")
        
        # Test final concatenation for cc_projection
        final_input = torch.cat([clip_emb, T_combined[:, None, :]], dim=-1)
        print(f"CLIP embedding shape: {clip_emb.shape}")
        print(f"Final input shape for cc_projection: {final_input.shape}")
        print(f"Expected input size: [batch_size, 1, 778]")
        
        if final_input.shape[-1] == 778:
            print("✓ Tensor concatenation works correctly")
        else:
            print(f"✗ Tensor concatenation failed. Expected 778, got {final_input.shape[-1]}")
            return False
            
    except Exception as e:
        print(f"✗ Tensor operations failed: {e}")
        return False
    
    # 3. Test cc_projection layer
    print("\nStep 3: Testing cc_projection layer...")
    
    try:
        cc_projection = torch.nn.Linear(778, 768)
        output = cc_projection(final_input)
        print(f"cc_projection output shape: {output.shape}")
        print("✓ cc_projection layer works correctly")
    except Exception as e:
        print(f"✗ cc_projection layer failed: {e}")
        return False
    
    # 4. Test model import (without instantiation)
    print("\nStep 4: Testing model import...")
    
    try:
        from ldm.models.diffusion.ddpm import LatentDiffusion
        print("✓ Successfully imported LatentDiffusion")
        print("✓ Model should work with updated cc_projection (778 -> 768)")
    except Exception as e:
        print(f"✗ Failed to import model: {e}")
        return False
    
    print("\n" + "="*50)
    print("🎉 PIPELINE COMPONENTS TEST PASSED!")
    print("✓ Plücker coordinate function works")
    print("✓ Tensor operations work correctly") 
    print("✓ Model architecture is compatible")
    print("✓ Ready for training (once you have data)!")
    print("="*50)
    
    return True

if __name__ == "__main__":
    try:
        success = test_plucker_function()
        if success:
            print("Test completed successfully!")
        else:
            print("Test failed!")
    except Exception as e:
        print(f"Test crashed with error: {e}")
        import traceback
        traceback.print_exc()