#!/usr/bin/env python3

import sys
import os
sys.path.append(os.getcwd())

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

def test_checkpoint_callback():
    """Test if the checkpoint callback can be instantiated correctly."""
    
    # Load the configuration
    config_path = "configs/sd-objaverse-finetune-c_concat-256.yaml"
    
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        return False
    
    try:
        config = OmegaConf.load(config_path)
        print("Configuration loaded successfully")
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return False
    
    # Test the simple checkpoint callback
    try:
        from simple_checkpoint_callback import SimpleCheckpointCallback
        callback = SimpleCheckpointCallback(save_every_n_steps=1000)
        print("SimpleCheckpointCallback instantiated successfully")
    except Exception as e:
        print(f"Failed to instantiate SimpleCheckpointCallback: {e}")
        return False
    
    # Test the lightning callbacks
    lightning_config = config.lightning
    if "callbacks" in lightning_config:
        callbacks = lightning_config.callbacks
        print(f"Found {len(callbacks)} callbacks in configuration")
        
        for name, callback_config in callbacks.items():
            try:
                callback = instantiate_from_config(callback_config)
                print(f"Successfully instantiated callback: {name}")
            except Exception as e:
                print(f"Failed to instantiate callback {name}: {e}")
                return False
    
    # Test modelcheckpoint configuration
    if "modelcheckpoint" in lightning_config:
        try:
            modelcheckpoint = instantiate_from_config(lightning_config.modelcheckpoint)
            print("ModelCheckpoint instantiated successfully")
        except Exception as e:
            print(f"Failed to instantiate ModelCheckpoint: {e}")
            return False
    
    print("\nAll checkpoint configurations test passed!")
    return True

if __name__ == "__main__":
    success = test_checkpoint_callback()
    sys.exit(0 if success else 1) 