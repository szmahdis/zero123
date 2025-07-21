# Zero-1-to-3 Batch Evaluation System

This system evaluates Zero-1-to-3 model performance on your Objaverse objects using object IDs as input.

## 🚀 Quick Start

### 1. Basic Usage (Single Object)
```bash
# Make scripts executable
chmod +x run_evaluation.sh

# Run evaluation on one object
./run_evaluation.sh fffc6d07590b4872be191dc7820de94f
```

### 2. Multiple Objects
```bash
# Run evaluation on multiple objects
./run_evaluation.sh \
    fffc6d07590b4872be191dc7820de94f \
    object_id_2 \
    object_id_3 \
    object_id_4
```

### 3. Run Example
```bash
# Use the provided example
chmod +x example_run.sh
./example_run.sh
```

## 📁 What You Need

1. **Your object IDs** - Just the folder names from views_release
2. **Views directory** - Path to your views_release folder (already configured)
3. **GPU** - CUDA-capable GPU with sufficient VRAM

## 🎯 What It Does

For each object ID, the system:

1. **Takes view 000.png as input**
2. **Generates novel views** using Zero-1-to-3 model
3. **Compares against ground truth** (001.png, 002.png, etc.)
4. **Calculates metrics**: PSNR, SSIM, LPIPS, FID
5. **Saves all results** with organized output

## 📊 Output Structure

```
evaluation_results/
├── fffc6d07590b4872be191dc7820de94f/
│   ├── input_000.png                    # Input image (copy)
│   ├── generated_001_sample_00.png      # Generated view 1, sample 0
│   ├── generated_001_sample_01.png      # Generated view 1, sample 1
│   ├── generated_002_sample_00.png      # Generated view 2, sample 0
│   ├── gt_001.png                       # Ground truth view 1
│   ├── gt_002.png                       # Ground truth view 2
│   └── ...
├── object_id_2/
│   └── ...
└── batch_evaluation_results.json        # Complete metrics & metadata
```

## 📈 Results Format

The `batch_evaluation_results.json` contains:

```json
{
  "objects": {
    "fffc6d07590b4872be191dc7820de94f": {
      "metrics": {
        "summary": {
          "PSNR": {"mean": 25.4, "std": 2.1},
          "SSIM": {"mean": 0.85, "std": 0.05},
          "LPIPS": {"mean": 0.12, "std": 0.03},
          "FID": {"mean": 18.5, "std": 4.2}
        }
      }
    }
  },
  "summary": {
    "overall_metrics": {
      "PSNR": {"mean": 24.8, "std": 3.2},
      "SSIM": {"mean": 0.82, "std": 0.08},
      "LPIPS": {"mean": 0.15, "std": 0.04},
      "FID": {"mean": 22.3, "std": 6.1}
    }
  }
}
```

## ⚙️ Configuration

Edit `run_evaluation.sh` to change:

- `VIEWS_DIR`: Path to your views_release directory
- `OUTPUT_DIR`: Where to save results
- GPU device, sampling steps, etc.

## 🔧 Parameters

Default settings:
- **Input view**: 000.png (first view)
- **Samples per view**: 4
- **DDIM steps**: 50
- **Guidance scale**: 3.0

## 🎮 Advanced Usage

```bash
# Custom parameters
python batch_eval_zero123.py \
    --object_ids fffc6d07590b4872be191dc7820de94f another_id \
    --views_dir /your/views/directory \
    --output_dir ./custom_results \
    --input_view 2 \
    --n_samples 8 \
    --ddim_steps 100 \
    --guidance_scale 5.0
```

## 🐛 Troubleshooting

### Error: Views directory not found
- Update `VIEWS_DIR` in `run_evaluation.sh`
- Make sure path points to your views_release folder

### Error: Object not found
- Check that object ID folder exists in views_release
- Verify folder contains *.png and *.npy files

### CUDA out of memory
- Reduce `--n_samples` (default: 4)
- Reduce `--ddim_steps` (default: 50)
- Use smaller batch of objects

### Requirements issues
- Run from zero123-main directory
- Ensure Python environment has required packages

## 📝 Example Output

```
🚀 Zero-1-to-3 Batch Evaluation Script
======================================
📁 Views directory: /cluster/51/ecekarasu/views_release
📁 Output directory: ./evaluation_results
🎯 Object IDs: fffc6d07590b4872be191dc7820de94f

Loading Zero-1-to-3 model...
Model loaded successfully on cuda:0

Evaluating object fffc6d07590b4872be191dc7820de94f (12 views)
  Generating view 001...
    PSNR: 25.43, SSIM: 0.8521, LPIPS: 0.1234, FID: 16.78
  Generating view 002...
    PSNR: 24.87, SSIM: 0.8334, LPIPS: 0.1456, FID: 19.45
  Summary - PSNR: 25.15±0.28, SSIM: 0.8428±0.0094, LPIPS: 0.1345±0.0111, FID: 18.12±1.34

=== BATCH SUMMARY ===
Successfully processed: 1/1 objects
Overall PSNR: 25.15 ± 0.28
Overall SSIM: 0.8428 ± 0.0094
Overall LPIPS: 0.1345 ± 0.0111
Overall FID: 18.12 ± 1.34

✅ Evaluation completed!
```

## 🎉 Ready to Use!

Just run:
```bash
./run_evaluation.sh your_object_id_1 your_object_id_2 ...
``` 