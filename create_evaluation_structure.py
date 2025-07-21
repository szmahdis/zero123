#!/usr/bin/env python3
"""
Script to create and organize evaluation_results directory structure
"""

import os
from pathlib import Path
import json
from datetime import datetime

def create_evaluation_structure():
    """Create organized directory structure for evaluation results"""
    
    print("📁 Setting up evaluation_results directory structure...")
    
    # Main evaluation results directory
    eval_dir = Path("evaluation_results")
    eval_dir.mkdir(exist_ok=True)
    
    # Subdirectories
    subdirs = {
        "batch_evaluation": "Full Zero-1-to-3 batch evaluation results",
        "comparisons": "Simple 2-image comparison results", 
        "camera_analysis": "Camera data inspection results",
        "summaries": "Summary reports and aggregated metrics",
        "visualizations": "Generated plots and charts"
    }
    
    for subdir, description in subdirs.items():
        (eval_dir / subdir).mkdir(exist_ok=True)
        
        # Create README in each subdirectory
        readme_path = eval_dir / subdir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(f"# {subdir.replace('_', ' ').title()}\n\n")
            f.write(f"{description}\n\n")
            f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Create main structure info
    structure_info = {
        "created": datetime.now().isoformat(),
        "structure": {
            "evaluation_results/": {
                "description": "Main directory for all evaluation outputs",
                "subdirectories": subdirs
            }
        },
        "usage": {
            "batch_evaluation": "./run_evaluation.sh OBJECT_ID",
            "comparisons": "python compare_two_images.py OBJECT_ID VIEW1 VIEW2 --save_comparison",
            "camera_analysis": "python inspect_camera_data.py OBJECT_ID"
        }
    }
    
    # Save structure info
    with open(eval_dir / "structure_info.json", 'w') as f:
        json.dump(structure_info, f, indent=2)
    
    # Create main README
    main_readme = eval_dir / "README.md"
    with open(main_readme, 'w') as f:
        f.write("# Zero-1-to-3 Evaluation Results\n\n")
        f.write("This directory contains all evaluation results and analysis outputs.\n\n")
        
        f.write("## Directory Structure\n\n")
        for subdir, desc in subdirs.items():
            f.write(f"- **`{subdir}/`**: {desc}\n")
        
        f.write("\n## Quick Usage\n\n")
        f.write("```bash\n")
        f.write("# Full evaluation\n")
        f.write("./run_evaluation.sh fffc6d07590b4872be191dc7820de94f\n\n")
        f.write("# Simple comparison\n") 
        f.write("python compare_two_images.py fffc6d07590b4872be191dc7820de94f 0 1 --save_comparison\n\n")
        f.write("# Camera analysis\n")
        f.write("python inspect_camera_data.py fffc6d07590b4872be191dc7820de94f\n")
        f.write("```\n\n")
        
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print("✅ Directory structure created!")
    print(f"📂 Main directory: {eval_dir.absolute()}")
    
    # Show structure
    print("\n📋 Structure:")
    for subdir in subdirs.keys():
        subpath = eval_dir / subdir
        print(f"  📁 {subpath}")
        
    return eval_dir

def show_current_structure():
    """Show current evaluation_results directory contents"""
    eval_dir = Path("evaluation_results")
    
    if not eval_dir.exists():
        print("❌ evaluation_results directory does not exist")
        return
    
    print(f"📂 Current structure of {eval_dir.absolute()}:")
    print()
    
    def show_tree(path, prefix=""):
        items = sorted(list(path.iterdir()))
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item.name}")
            
            if item.is_dir() and len(list(item.iterdir())) > 0:
                next_prefix = prefix + ("    " if is_last else "│   ")
                show_tree(item, next_prefix)
    
    show_tree(eval_dir)

def main():
    print("🎯 Zero-1-to-3 Evaluation Results Organizer")
    print("=" * 50)
    
    # Create structure
    eval_dir = create_evaluation_structure()
    
    print(f"\n📊 All results will be saved under:")
    print(f"   {eval_dir.absolute()}")
    
    print(f"\n🔧 Updated scripts to use this structure:")
    print(f"   - Batch evaluation → evaluation_results/batch_evaluation/")
    print(f"   - Image comparisons → evaluation_results/comparisons/")
    print(f"   - Camera analysis → evaluation_results/camera_analysis/")
    
    print(f"\n🚀 Ready to run evaluations!")

if __name__ == "__main__":
    main() 