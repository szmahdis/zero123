# Zero-1-to-3 Evaluation Results

This directory contains all evaluation results and analysis outputs.

## Directory Structure

- **`batch_evaluation/`**: Full Zero-1-to-3 batch evaluation results
- **`comparisons/`**: Simple 2-image comparison results
- **`camera_analysis/`**: Camera data inspection results
- **`summaries/`**: Summary reports and aggregated metrics
- **`visualizations/`**: Generated plots and charts

## Quick Usage

```bash
# Full evaluation
./run_evaluation.sh fffc6d07590b4872be191dc7820de94f

# Simple comparison
python compare_two_images.py fffc6d07590b4872be191dc7820de94f 0 1 --save_comparison

# Camera analysis
python inspect_camera_data.py fffc6d07590b4872be191dc7820de94f
```

Created: 2025-07-20 23:50:39
