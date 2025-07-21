# 📁 Organized Zero-1-to-3 Evaluation System

All results are now **perfectly organized** under `evaluation_results/` directory!

## 🏗️ **Directory Structure**

```
evaluation_results/
├── README.md                    # Main documentation
├── structure_info.json          # Structure metadata
├── batch_evaluation/            # Full Zero-1-to-3 evaluation results
│   ├── fffc6d07590b4872be191dc7820de94f/
│   │   ├── generated_001_sample_00.png
│   │   ├── gt_001.png
│   │   └── ...
│   └── batch_evaluation_results.json
├── comparisons/                 # Simple 2-image comparisons
│   └── fffc6d07590b4872be191dc7820de94f/
│       ├── view_000_vs_001.png
│       ├── view_000_vs_006.png
│       └── ...
├── camera_analysis/             # Camera data inspection
│   └── camera_analysis_fffc6d07590b4872be191dc7820de94f.txt
├── summaries/                   # Aggregated reports
└── visualizations/              # Plots and charts
```

## 🚀 **Quick Commands**

### **1. Complete Workflow (Recommended)**
```bash
./example_organized_evaluation.sh
```
Does everything: camera analysis → comparisons → full evaluation

### **2. Individual Components**

**Full Zero-1-to-3 Evaluation:**
```bash
./run_evaluation.sh fffc6d07590b4872be191dc7820de94f
# Results → evaluation_results/batch_evaluation/
```

**Simple Image Comparison:**
```bash
python compare_two_images.py fffc6d07590b4872be191dc7820de94f 0 1 \
    --save_comparison --output_dir evaluation_results/comparisons
# Results → evaluation_results/comparisons/
```

**Camera Analysis:**
```bash
python inspect_camera_data.py fffc6d07590b4872be191dc7820de94f \
    > evaluation_results/camera_analysis/camera_report.txt
# Results → evaluation_results/camera_analysis/
```

## 📊 **What You Get**

### **1. Batch Evaluation Results**
- `evaluation_results/batch_evaluation/OBJECT_ID/`
  - **Generated images**: `generated_XXX_sample_XX.png`
  - **Ground truth copies**: `gt_XXX.png`
  - **Input image**: `input_000.png`
- `batch_evaluation_results.json` - Complete metrics (PSNR, SSIM, LPIPS, FID) & metadata

### **2. Image Comparisons**
- `evaluation_results/comparisons/OBJECT_ID/`
  - **Side-by-side images**: `view_XXX_vs_YYY.png`
  - **Organized by object ID**
  - **Visual quality assessment**

### **3. Camera Analysis**
- `evaluation_results/camera_analysis/`
  - **Camera parameter breakdown**
  - **Coordinate system explanations**
  - **Viewpoint positions**

## 🔧 **Setup (One-time)**

```bash
# Create organized structure
python create_evaluation_structure.py

# Make scripts executable
chmod +x *.py *.sh
```

## 📈 **Results Analysis**

After running evaluations, check:

1. **Overall Performance**: `evaluation_results/batch_evaluation/batch_evaluation_results.json`
2. **Visual Comparisons**: Browse `evaluation_results/comparisons/OBJECT_ID/`
3. **Individual Metrics**: Object-specific folders in `batch_evaluation/`

## 🎯 **Typical Workflow**

```bash
# 1. Setup (once)
python create_evaluation_structure.py

# 2. Quick test with comparisons
python compare_two_images.py fffc6d07590b4872be191dc7820de94f 0 1 --save_comparison --output_dir evaluation_results/comparisons

# 3. Full evaluation
./run_evaluation.sh fffc6d07590b4872be191dc7820de94f

# 4. Results are automatically organized!
ls evaluation_results/batch_evaluation/
ls evaluation_results/comparisons/
```

## 📋 **Benefits**

✅ **Organized**: Everything in logical subdirectories  
✅ **Documented**: README files in each directory  
✅ **Traceable**: Timestamped structure info  
✅ **Scalable**: Easy to add more objects  
✅ **Professional**: Publication-ready organization  

## 🎉 **Ready to Use!**

All your evaluation results will now be beautifully organized and easy to navigate! Perfect for research, presentations, and publication! 🌟 