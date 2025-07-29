#!/usr/bin/env python3

import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
sys.path.append('.')

from simple_data_module import SimpleObjaverseData

def visualize_images():
    """Visualize the images being loaded to check for noise."""
    
    # Initialize dataset
    dataset = SimpleObjaverseData(
        root_dir="views_release",
        total_view=12,
        validation=False,
        image_size=256
    )
    
    if len(dataset) == 0:
        print("No data found!")
        return
    
    # Load first item
    item = dataset[0]
    
    # Convert tensors back to images for visualization
    def tensor_to_image(tensor):
        """Convert tensor back to PIL image for visualization."""
        # tensor is in [H, W, C] format with values in [-1, 1]
        # Convert to [0, 1] range
        img_array = (tensor + 1) / 2
        # Convert to [0, 255] range
        img_array = (img_array * 255).clamp(0, 255).numpy().astype(np.uint8)
        return Image.fromarray(img_array)
    
    # Get images
    target_img = tensor_to_image(item["image_target"])
    cond_img = tensor_to_image(item["image_cond"])
    
    # Convert control image
    control_tensor = item["control"]  # Shape: [H, W, 1]
    control_array = control_tensor.squeeze().numpy()  # Remove channel dimension
    control_array = (control_array * 255).clamp(0, 255).astype(np.uint8)
    control_img = Image.fromarray(control_array, mode='L')
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(target_img)
    axes[0].set_title("Target Image")
    axes[0].axis('off')
    
    axes[1].imshow(cond_img)
    axes[1].set_title("Condition Image")
    axes[1].axis('off')
    
    axes[2].imshow(control_img, cmap='gray')
    axes[2].set_title("Control Image")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('loaded_images.png', dpi=150, bbox_inches='tight')
    print("Images saved to 'loaded_images.png'")
    
    # Print statistics
    print(f"\nImage Statistics:")
    print(f"Target image - min: {item['image_target'].min():.3f}, max: {item['image_target'].max():.3f}")
    print(f"Condition image - min: {item['image_cond'].min():.3f}, max: {item['image_cond'].max():.3f}")
    print(f"Control image - min: {item['control'].min():.3f}, max: {item['control'].max():.3f}")
    
    # Check for any extreme values that might indicate noise
    target_std = item['image_target'].std()
    cond_std = item['image_cond'].std()
    control_std = item['control'].std()
    
    print(f"\nStandard Deviations:")
    print(f"Target image std: {target_std:.3f}")
    print(f"Condition image std: {cond_std:.3f}")
    print(f"Control image std: {control_std:.3f}")
    
    # If standard deviation is very high, it might indicate noise
    if target_std > 0.5 or cond_std > 0.5:
        print("⚠️  WARNING: High standard deviation detected - possible noise!")
    else:
        print("✅ Standard deviations look normal - no obvious noise detected")

if __name__ == "__main__":
    visualize_images() 