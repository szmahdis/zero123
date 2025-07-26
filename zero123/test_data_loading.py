#!/usr/bin/env python3

import sys
import os
sys.path.append(os.getcwd())

from simple_data_module import SimpleObjaverseDataModule, SimpleObjaverseData

def test_data_loading():
    """Test if the data loading works correctly."""
    
    print("Testing SimpleObjaverseData...")
    
    # Test the dataset directly
    try:
        dataset = SimpleObjaverseData(
            root_dir='views_release',
            total_view=2,
            validation=False,
            image_size=256
        )
        print(f"✅ Dataset created successfully with {len(dataset)} samples")
        
        # Test loading a sample
        sample = dataset[0]
        print(f"✅ Sample loaded successfully")
        print(f"   - image_target shape: {sample['image_target'].shape}")
        print(f"   - image_cond shape: {sample['image_cond'].shape}")
        print(f"   - T shape: {sample['T'].shape}")
        
    except Exception as e:
        print(f"❌ Failed to create dataset: {e}")
        return False
    
    print("\nTesting SimpleObjaverseDataModule...")
    
    # Test the data module
    try:
        data_module = SimpleObjaverseDataModule(
            root_dir='views_release',
            batch_size=2,
            total_view=2,
            num_workers=0,
            image_size=256
        )
        print("✅ Data module created successfully")
        
        # Test train dataloader
        train_loader = data_module.train_dataloader()
        print(f"✅ Train dataloader created with {len(train_loader)} batches")
        
        # Test loading a batch
        batch = next(iter(train_loader))
        print(f"✅ Batch loaded successfully")
        print(f"   - batch size: {len(batch['image_target'])}")
        print(f"   - image_target shape: {batch['image_target'].shape}")
        print(f"   - image_cond shape: {batch['image_cond'].shape}")
        print(f"   - T shape: {batch['T'].shape}")
        
    except Exception as e:
        print(f"❌ Failed to create data module: {e}")
        return False
    
    print("\n✅ All data loading tests passed!")
    return True

if __name__ == "__main__":
    success = test_data_loading()
    sys.exit(0 if success else 1) 