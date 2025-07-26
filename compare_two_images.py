#!/usr/bin/env python3
"""
Simple script to compare two images from an Objaverse object
Calculates PSNR, SSIM, LPIPS, and feature distance (FID-like) metrics between them
"""

import argparse
import sys
import numpy as np
import torch
from pathlib import Path
from PIL import Image
# import cv2  # Not needed

# Add zero123 to path for metric functions
sys.path.append('zero123')

try:
    from ldm.modules.evaluate.evaluate_perceptualsim import ssim_metric, psnr
    ZERO123_METRICS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import Zero-1-to-3 PSNR/SSIM: {e}")
    ZERO123_METRICS_AVAILABLE = False

try:
    from taming.modules.losses.lpips import LPIPS
    LPIPS_AVAILABLE = True
except ImportError:
    print("Warning: LPIPS not available")
    LPIPS_AVAILABLE = False

try:
    from torchvision.models import inception_v3
    from scipy.linalg import sqrtm
    FID_AVAILABLE = True
except ImportError:
    print("Warning: FID dependencies not available (need torchvision and scipy)")
    FID_AVAILABLE = False

def load_and_resize_image(image_path, size=(256, 256)):
    """Load and resize image with proper transparency handling"""
    image = Image.open(image_path)
    
    # Handle transparency properly - convert to white background instead of black
    if image.mode in ('RGBA', 'LA'):
        # Create white background
        white_bg = Image.new('RGB', image.size, (255, 255, 255))
        if image.mode == 'RGBA':
            white_bg.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
        else:
            white_bg.paste(image, mask=image.split()[-1])
        image = white_bg
    else:
        image = image.convert('RGB')
    
    image = image.resize(size, Image.Resampling.LANCZOS)
    return np.array(image) / 255.0

def calculate_psnr_basic(img1, img2):
    """Calculate PSNR using basic implementation"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim_basic(img1, img2):
    """Calculate SSIM using basic implementation"""
    try:
        from skimage.metrics import structural_similarity as ssim
        
        # Calculate SSIM for each channel
        ssim_values = []
        for i in range(3):  # RGB channels
            ssim_val = ssim(img1[:,:,i], img2[:,:,i], data_range=1.0)
            ssim_values.append(ssim_val)
        
        return np.mean(ssim_values)
    except ImportError:
        # Fallback: simple correlation coefficient
        img1_flat = img1.flatten()
        img2_flat = img2.flatten()
        correlation = np.corrcoef(img1_flat, img2_flat)[0, 1]
        return max(0, correlation)  # Clamp to positive

def calculate_lpips_basic(img1, img2):
    """Calculate LPIPS using Zero-1-to-3 implementation"""
    if not LPIPS_AVAILABLE:
        return None
        
    try:
        # Convert to tensors
        img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().contiguous()
        img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float().contiguous()
        
        # Initialize LPIPS
        lpips_fn = LPIPS().eval()
        
        # Calculate LPIPS
        with torch.no_grad():
            lpips_val = lpips_fn(img1_tensor * 2 - 1, img2_tensor * 2 - 1)
        
        return lpips_val.item()
    except Exception as e:
        print(f"LPIPS calculation failed: {e}")
        return None

def get_inception_features(img, model):
    """Extract features from Inception-v3 model"""
    # Preprocess image for Inception
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    
    # Resize to 299x299 for Inception-v3
    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(299, 299), mode='bilinear', align_corners=False)
    
    # Normalize to [-1, 1] range
    img_tensor = img_tensor * 2 - 1
    
    with torch.no_grad():
        features = model(img_tensor)
    
    return features.squeeze().cpu().numpy()

def calculate_fid_basic(img1, img2):
    """Calculate simplified FID-like distance between two images"""
    if not FID_AVAILABLE:
        return None
        
    try:
        # Load Inception-v3 model
        model = inception_v3(pretrained=True, transform_input=False)
        model.eval()
        
        # Remove final classification layer to get features
        model.fc = torch.nn.Identity()
        
        # Get features for both images
        features1 = get_inception_features(img1, model)
        features2 = get_inception_features(img2, model)
        
        # For single images, calculate Euclidean distance in feature space
        # This gives us a FID-like measure without requiring distributions
        feature_distance = np.linalg.norm(features1 - features2)
        
        # Scale to make values more interpretable (similar to typical FID ranges)
        fid_like_score = feature_distance / 10.0
        
        return float(fid_like_score)
        
    except Exception as e:
        print(f"FID calculation failed: {e}")
        return None

def calculate_all_metrics(img1, img2, use_zero123_metrics=True):
    """Calculate all available metrics"""
    results = {}
    
    # Try Zero-1-to-3 implementations first
    if use_zero123_metrics and ZERO123_METRICS_AVAILABLE:
        try:
            # Convert to tensors for Zero-1-to-3 metrics
            img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().contiguous()
            img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float().contiguous()
            
            # PSNR
            psnr_val = psnr(img1_tensor, img2_tensor)
            results['PSNR'] = float(psnr_val.item() if torch.is_tensor(psnr_val) else psnr_val)
            
            # SSIM  
            ssim_val = ssim_metric(img1_tensor, img2_tensor)
            results['SSIM'] = float(ssim_val.item() if torch.is_tensor(ssim_val) else ssim_val)
            
            print("✅ Using Zero-1-to-3 metric implementations")
            
        except Exception as e:
            print(f"⚠️  Zero-1-to-3 metrics failed: {e}")
            use_zero123_metrics = False
    else:
        use_zero123_metrics = False
    
    # Fallback to basic implementations
    if not use_zero123_metrics:
        print("📊 Using basic metric implementations")
        results['PSNR'] = calculate_psnr_basic(img1, img2)
        results['SSIM'] = calculate_ssim_basic(img1, img2)
    
    # LPIPS (try if available)
    lpips_val = calculate_lpips_basic(img1, img2)
    if lpips_val is not None:
        results['LPIPS'] = lpips_val
    
    # FID (try if available) - Note: This is feature distance, not true FID
    fid_val = calculate_fid_basic(img1, img2)
    if fid_val is not None:
        results['FID_like'] = fid_val
    
    return results

def compare_images(object_id, view1_idx, view2_idx, views_dir, save_comparison=False, output_dir="evaluation_results/comparisons"):
    """Compare two images from an object"""
    
    # Construct paths
    object_path = Path(views_dir) / object_id
    
    if not object_path.exists():
        raise FileNotFoundError(f"Object directory not found: {object_path}")
    
    image1_path = object_path / f"{view1_idx:03d}.png"
    image2_path = object_path / f"{view2_idx:03d}.png"
    
    if not image1_path.exists():
        raise FileNotFoundError(f"Image not found: {image1_path}")
    if not image2_path.exists():
        raise FileNotFoundError(f"Image not found: {image2_path}")
    
    print(f"📸 Comparing images:")
    print(f"  Image 1: {image1_path}")
    print(f"  Image 2: {image2_path}")
    
    # Load images
    img1 = load_and_resize_image(image1_path)
    img2 = load_and_resize_image(image2_path)
    
    print(f"  Resolution: {img1.shape}")
    
    # Calculate metrics
    print(f"\n🔍 Calculating metrics...")
    metrics = calculate_all_metrics(img1, img2)
    
    # Display results
    print(f"\n📊 RESULTS:")
    print(f"=" * 40)
    for metric, value in metrics.items():
        if isinstance(value, float):
            if metric == 'PSNR':
                print(f"{metric:>8}: {value:>8.2f} dB")
            elif metric in ['SSIM', 'LPIPS']:
                print(f"{metric:>8}: {value:>8.4f}")
            elif metric == 'FID_like':
                print(f"FID-like: {value:>8.2f}")
            else:
                print(f"{metric:>8}: {value:>8.4f}")
        else:
            print(f"{metric:>8}: {value}")
    print(f"=" * 40)
    
    # Interpretation
    print(f"\n💡 INTERPRETATION:")
    if 'PSNR' in metrics:
        psnr_val = metrics['PSNR']
        if psnr_val > 30:
            psnr_quality = "Excellent"
        elif psnr_val > 25:
            psnr_quality = "Good"
        elif psnr_val > 20:
            psnr_quality = "Fair"
        else:
            psnr_quality = "Poor"
        print(f"  PSNR: {psnr_quality} similarity")
    
    if 'SSIM' in metrics:
        ssim_val = metrics['SSIM']
        if ssim_val > 0.9:
            ssim_quality = "Very high"
        elif ssim_val > 0.8:
            ssim_quality = "High"
        elif ssim_val > 0.6:
            ssim_quality = "Moderate"
        else:
            ssim_quality = "Low"
        print(f"  SSIM: {ssim_quality} structural similarity")
    
    if 'LPIPS' in metrics:
        lpips_val = metrics['LPIPS']
        if lpips_val < 0.1:
            lpips_quality = "Very similar"
        elif lpips_val < 0.3:
            lpips_quality = "Similar"
        elif lpips_val < 0.5:
            lpips_quality = "Somewhat different"
        else:
            lpips_quality = "Very different"
        print(f"  LPIPS: {lpips_quality} (perceptually)")
    
    if 'FID_like' in metrics:
        fid_val = metrics['FID_like']
        if fid_val < 5:
            fid_quality = "Excellent"
        elif fid_val < 15:
            fid_quality = "Good"
        elif fid_val < 30:
            fid_quality = "Fair"
        else:
            fid_quality = "Poor"
        print(f"  FID-like: {fid_quality} feature similarity")
        print(f"  Note: For true FID (like in papers), use batch evaluation with multiple views")
    
    # Save comparison image if requested
    if save_comparison:
        # Create output directory structure
        comparisons_dir = Path(output_dir)
        comparisons_dir.mkdir(parents=True, exist_ok=True)
        
        comparison_path = comparisons_dir / f"comparison_{object_id}_{view1_idx:03d}_vs_{view2_idx:03d}.png"
        
        # Create side-by-side comparison
        img1_pil = Image.fromarray((img1 * 255).astype(np.uint8))
        img2_pil = Image.fromarray((img2 * 255).astype(np.uint8))
        
        # Create combined image with labels
        gap = 10
        label_height = 30
        combined_width = img1_pil.width + img2_pil.width + gap
        combined_height = max(img1_pil.height, img2_pil.height) + label_height
        combined = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))
        
        # Add labels (simplified - just paste images with space for labels)
        combined.paste(img1_pil, (0, label_height))
        combined.paste(img2_pil, (img1_pil.width + gap, label_height))
        
        # Save with object subdirectory
        object_comparisons_dir = comparisons_dir / object_id
        object_comparisons_dir.mkdir(exist_ok=True)
        final_comparison_path = object_comparisons_dir / f"view_{view1_idx:03d}_vs_{view2_idx:03d}.png"
        
        combined.save(final_comparison_path)
        print(f"\n💾 Comparison saved: {final_comparison_path}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Compare two images from an Objaverse object")
    parser.add_argument("object_id", help="Object ID (folder name)")
    parser.add_argument("view1", type=int, help="First view index (0-11)")
    parser.add_argument("view2", type=int, help="Second view index (0-11)")
    parser.add_argument("--views_dir", default="/cluster/51/ecekarasu/views_release",
                       help="Path to views_release directory")
    parser.add_argument("--output_dir", default="evaluation_results/comparisons",
                       help="Output directory for comparison results")
    parser.add_argument("--save_comparison", action="store_true",
                       help="Save side-by-side comparison image")
    
    args = parser.parse_args()
    
    # Validate view indices
    if not (0 <= args.view1 <= 11):
        print("Error: view1 must be between 0 and 11")
        return
    if not (0 <= args.view2 <= 11):
        print("Error: view2 must be between 0 and 11")
        return
    
    if args.view1 == args.view2:
        print("Warning: Comparing the same view with itself!")
    
    try:
        print(f"🎯 Object: {args.object_id}")
        print(f"📍 Views directory: {args.views_dir}")
        
        metrics = compare_images(
            args.object_id, 
            args.view1, 
            args.view2, 
            args.views_dir, 
            args.save_comparison,
            args.output_dir
        )
        
        print(f"\n✅ Comparison completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 