#!/usr/bin/env python3
"""
Prepare evaluation results for pytorch-fid calculation
Organizes generated and ground truth images into separate directories
"""

import argparse
import shutil
from pathlib import Path
import json

def organize_images_for_fid(results_dir: str, output_dir: str = "fid_evaluation"):
    """Organize evaluation results for pytorch-fid"""
    
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    generated_dir = output_path / "generated"
    gt_dir = output_path / "ground_truth"
    
    generated_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Organizing images for FID calculation...")
    print(f"📂 Output directory: {output_path}")
    
    # Find all object directories
    object_dirs = [d for d in results_path.iterdir() if d.is_dir() and d.name != "__pycache__"]
    
    generated_count = 0
    gt_count = 0
    
    for obj_dir in object_dirs:
        object_id = obj_dir.name
        print(f"📦 Processing {object_id}...")
        
        # Copy generated images (use best sample - sample_00)
        for generated_file in obj_dir.glob("generated_*_sample_00.png"):
            view_num = generated_file.name.split('_')[1]  # Extract view number
            dest_name = f"{object_id}_view_{view_num}_generated.png"
            shutil.copy2(generated_file, generated_dir / dest_name)
            generated_count += 1
            
        # Copy ground truth images
        for gt_file in obj_dir.glob("gt_*.png"):
            view_num = gt_file.name.split('_')[1].split('.')[0]  # Extract view number
            dest_name = f"{object_id}_view_{view_num}_gt.png"
            shutil.copy2(gt_file, gt_dir / dest_name)
            gt_count += 1
    
    print(f"✅ Organized {generated_count} generated images")
    print(f"✅ Organized {gt_count} ground truth images")
    print(f"📊 Ready for FID calculation!")
    print()
    print(f"🚀 Run FID calculation:")
    print(f"python -m pytorch_fid {gt_dir} {generated_dir}")
    print()
    print(f"🔧 Advanced options:")
    print(f"python -m pytorch_fid {gt_dir} {generated_dir} --device cuda --batch-size 50")
    
    return generated_dir, gt_dir

def run_fid_calculation(gt_dir: Path, generated_dir: Path, device: str = "cuda"):
    """Run pytorch-fid calculation"""
    import subprocess
    import sys
    
    print(f"🔥 Running FID calculation...")
    
    cmd = [
        sys.executable, "-m", "pytorch_fid",
        str(gt_dir), str(generated_dir),
        "--device", device,
        "--batch-size", "50"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        fid_value = float(result.stdout.strip().split()[-1])
        
        print(f"🎯 FID Result: {fid_value:.3f}")
        
        # Save result
        result_file = Path("fid_results.json")
        fid_results = {
            "FID": fid_value,
            "method": "pytorch-fid",
            "num_generated": len(list(generated_dir.glob("*.png"))),
            "num_ground_truth": len(list(gt_dir.glob("*.png"))),
            "command": " ".join(cmd)
        }
        
        with open(result_file, 'w') as f:
            json.dump(fid_results, f, indent=2)
            
        print(f"💾 Results saved to: {result_file}")
        return fid_value
        
    except subprocess.CalledProcessError as e:
        print(f"❌ FID calculation failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Prepare and run FID evaluation")
    parser.add_argument("results_dir", help="Path to batch evaluation results directory")
    parser.add_argument("--output_dir", default="fid_evaluation", help="Output directory for organized images")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device for FID calculation")
    parser.add_argument("--run_fid", action="store_true", help="Automatically run FID calculation")
    
    args = parser.parse_args()
    
    # Organize images
    generated_dir, gt_dir = organize_images_for_fid(args.results_dir, args.output_dir)
    
    # Run FID calculation if requested
    if args.run_fid:
        fid_value = run_fid_calculation(gt_dir, generated_dir, args.device)
        
        if fid_value is not None:
            print(f"\n🎉 Final FID: {fid_value:.3f}")
            
            # Compare with paper
            paper_fid = 0.319  # Best result from paper
            if fid_value < paper_fid:
                improvement = ((paper_fid - fid_value) / paper_fid) * 100
                print(f"🔥 {improvement:.1f}% better than paper's best FID ({paper_fid})!")
            elif fid_value < 10:
                print(f"✅ Good FID result (paper range: 0.319-10.202)")
            else:
                print(f"⚠️  FID higher than paper range (0.319-10.202)")

if __name__ == "__main__":
    main() 