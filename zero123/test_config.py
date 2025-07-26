#!/usr/bin/env python3

import os
import sys
import yaml
from omegaconf import OmegaConf

def test_config():
    """Test if the configuration loads correctly and checkpoints are configured properly."""
    
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
    
    # Check if lightning section exists
    if "lightning" not in config:
        print("No 'lightning' section in configuration")
        return False
    
    lightning_config = config.lightning
    print("Lightning configuration found")
    
    # Check if callbacks section exists
    if "callbacks" not in lightning_config:
        print("No 'callbacks' section in lightning configuration")
        return False
    
    callbacks = lightning_config.callbacks
    print("Callbacks section found")
    
    # Check for checkpoint callback
    if "checkpoint_callback" not in callbacks:
        print("No 'checkpoint_callback' in callbacks")
        return False
    
    checkpoint_cfg = callbacks.checkpoint_callback
    print("Checkpoint callback found")
    
    # Check checkpoint parameters
    required_params = ["dirpath", "filename", "every_n_train_steps", "save_last", "monitor"]
    for param in required_params:
        if param not in checkpoint_cfg.params:
            print(f"❌ Missing required parameter: {param}")
            return False
    
    print("All required checkpoint parameters found")
    
    # Print checkpoint configuration
    print("\nCheckpoint Configuration:")
    print(f"  Directory: {checkpoint_cfg.params.dirpath}")
    print(f"  Filename pattern: {checkpoint_cfg.params.filename}")
    print(f"  Save every N steps: {checkpoint_cfg.params.every_n_train_steps}")
    print(f"  Monitor metric: {checkpoint_cfg.params.monitor}")
    print(f"  Save top K: {checkpoint_cfg.params.save_top_k}")
    print(f"  Save last: {checkpoint_cfg.params.save_last}")
    
    # Check for backup checkpoint
    if "backup_checkpoint_callback" in callbacks:
        backup_cfg = callbacks.backup_checkpoint_callback
        print(f"\n Backup Checkpoint Configuration:")
        print(f"  Directory: {backup_cfg.params.dirpath}")
        print(f"  Filename pattern: {backup_cfg.params.filename}")
        print(f"  Save every N steps: {backup_cfg.params.every_n_train_steps}")
    
    # Check model configuration
    if "model" in config:
        model_config = config.model
        if "monitor" in model_config.params:
            print(f"\nModel Monitor: {model_config.params.monitor}")
    
    # Check trainer configuration
    if "trainer" in lightning_config:
        trainer_config = lightning_config.trainer
        print(f"\nTrainer Configuration:")
        print(f"  Validation check interval: {trainer_config.val_check_interval}")
        print(f"  Sanity validation steps: {trainer_config.num_sanity_val_steps}")
    
    print("\n Configuration test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_config()
    sys.exit(0 if success else 1) 