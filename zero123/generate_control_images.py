#!/usr/bin/env python3
"""
Script to generate control images from existing test data.
This creates edge-based control images from the rendered images.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import json
from PIL import Image
import argparse

def create_edge_control(image_path, output_path):
    """Create edge-based control image from input image."""
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to load image: {image_path}")
        return False
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges using Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # Save edge image
    cv2.imwrite(str(output_path), edges)
    return True

def process_dataset(root_dir):
    """Process all images in the dataset to create control images."""
    root_path = Path(root_dir)
    
    # Load valid paths
    valid_path_file = root_path / 'valid_paths.json'
    if not valid_path_file.exists():
        print(f"valid_paths.json not found in {root_dir}")
        return
    
    with open(valid_path_file) as f:
        all_paths = json.load(f)
    
    print(f"Processing {len(all_paths)} object paths...")
    
    for i, path in enumerate(all_paths):
        full_path = root_path / path
        if not full_path.exists():
            print(f"Skipping {path}: directory does not exist")
            continue
        
        # Create control directory
        control_path = full_path / 'control'
        control_path.mkdir(exist_ok=True)
        
        # Process all PNG files in the directory
        png_files = list(full_path.glob('*.png'))
        if not png_files:
            print(f"No PNG files found in {path}")
            continue
        
        print(f"Processing {len(png_files)} images in {path} ({i+1}/{len(all_paths)})")
        
        for png_file in png_files:
            # Create corresponding control image path
            control_file = control_path / f"{png_file.stem}_control.png"
            
            # Generate control image
            success = create_edge_control(png_file, control_file)
            if success:
                print(f"  Created: {control_file.name}")
            else:
                print(f"  Failed: {png_file.name}")

def main():
    parser = argparse.ArgumentParser(description='Generate control images from dataset')
    parser.add_argument('--root_dir', type=str, default='views_release',
                       help='Root directory containing the dataset')
    
    args = parser.parse_args()
    
    print(f"Generating control images for dataset in: {args.root_dir}")
    process_dataset(args.root_dir)
    print("Control image generation completed!")

if __name__ == "__main__":
    main() 