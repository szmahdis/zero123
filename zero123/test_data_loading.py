#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

from simple_data_module import SimpleObjaverseData, SimpleObjaverseDataModule

def test_data_loading():
    """Test the data loading to identify issues with noisy images."""
    
    # Initialize the dataset
    print("=== Testing Data Loading ===")
    
    # Test with the views_release directory
    root_dir = "views_release"
    
    if not os.path.exists(root_dir):
        print(f"Error: {root_dir} directory does not exist!")
        return
    
    print(f"Root directory: {root_dir}")
    print(f"Root directory exists: {os.path.exists(root_dir)}")
    
    # Initialize the dataset
    dataset = SimpleObjaverseData(
        root_dir=root_dir,
        total_view=12,
        validation=False,
        image_size=256
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test loading the first item
    if len(dataset) > 0:
        print("\n=== Testing first dataset item ===")
        try:
            first_item = dataset[0]
            print("Successfully loaded first item!")
            print(f"Keys in item: {list(first_item.keys())}")
            
            # Check image properties
            if "image_target" in first_item:
                target_img = first_item["image_target"]
                print(f"Target image shape: {target_img.shape}")
                print(f"Target image min/max: {target_img.min():.3f}, {target_img.max():.3f}")
            
            if "image_cond" in first_item:
                cond_img = first_item["image_cond"]
                print(f"Condition image shape: {cond_img.shape}")
                print(f"Condition image min/max: {cond_img.min():.3f}, {cond_img.max():.3f}")
            
            if "control" in first_item:
                control_img = first_item["control"]
                print(f"Control image shape: {control_img.shape}")
                print(f"Control image min/max: {control_img.min():.3f}, {control_img.max():.3f}")
            
        except Exception as e:
            print(f"Error loading first item: {e}")
            import traceback
            traceback.print_exc()
    
    # Test specific object ID
    print("\n=== Testing specific object ID ===")
    target_object_id = "aa15f4e92fcc42a49cdd15e7c94d323f"
    
    # Check if this object exists in the dataset
    if target_object_id in dataset.paths:
        print(f"Object {target_object_id} is in the dataset!")
        object_index = dataset.paths.index(target_object_id)
        print(f"Object index: {object_index}")
        
        # Test loading this specific object
        try:
            item = dataset[object_index]
            print("Successfully loaded target object!")
            
            # Check image properties
            if "image_target" in item:
                target_img = item["image_target"]
                print(f"Target image shape: {target_img.shape}")
                print(f"Target image min/max: {target_img.min():.3f}, {target_img.max():.3f}")
            
            if "image_cond" in item:
                cond_img = item["image_cond"]
                print(f"Condition image shape: {cond_img.shape}")
                print(f"Condition image min/max: {cond_img.min():.3f}, {cond_img.max():.3f}")
            
            if "control" in item:
                control_img = item["control"]
                print(f"Control image shape: {control_img.shape}")
                print(f"Control image min/max: {control_img.min():.3f}, {control_img.max():.3f}")
                
        except Exception as e:
            print(f"Error loading target object: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Object {target_object_id} is NOT in the dataset!")
        print(f"First 5 objects in dataset: {dataset.paths[:5]}")
    
    # Test control image loading directly
    print("\n=== Testing control image loading directly ===")
    dataset.test_load_control_image(target_object_id, 0)

if __name__ == "__main__":
    test_data_loading() 