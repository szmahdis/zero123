# 📸 Simple Image Comparison Tool

Quick and easy tool to compare any two images from your Objaverse objects and get instant metrics.

## 🚀 Quick Usage

```bash
# Basic comparison
python compare_two_images.py OBJECT_ID VIEW1 VIEW2

# With your example object
python compare_two_images.py fffc6d07590b4872be191dc7820de94f 0 1 --output_dir evaluation_results/comparisons

# Save side-by-side comparison image
python compare_two_images.py fffc6d07590b4872be191dc7820de94f 0 6 --save_comparison --output_dir evaluation_results/comparisons
```

## 📊 What You Get

- **PSNR** (Peak Signal-to-Noise Ratio) - Higher is better
- **SSIM** (Structural Similarity Index) - Closer to 1.0 is better  
- **LPIPS** (Learned Perceptual Image Patch Similarity) - Lower is better
- **FID** (Feature Distance using Inception-v3) - Lower is better
- **Quality interpretation** (Excellent/Good/Fair/Poor)
- **Optional side-by-side comparison image**

## 🎯 Examples

### Example 1: Adjacent Views
```bash
python compare_two_images.py fffc6d07590b4872be191dc7820de94f 0 1
```
Expected: High similarity (same object, slight rotation)

### Example 2: Opposite Views  
```bash
python compare_two_images.py fffc6d07590b4872be191dc7820de94f 0 6
```
Expected: Lower similarity (different viewpoints)

### Example 3: With Comparison Image
```bash
python compare_two_images.py fffc6d07590b4872be191dc7820de94f 2 8 --save_comparison
```
Creates: `comparison_fffc6d07590b4872be191dc7820de94f_002_vs_008.png`

## 📋 Parameters

- **object_id**: Your object folder name (e.g., `fffc6d07590b4872be191dc7820de94f`)
- **view1**: First view index (0-11)
- **view2**: Second view index (0-11)
- **--views_dir**: Path to views_release (default: `/cluster/51/ecekarasu/views_release`)
- **--save_comparison**: Save side-by-side image

## 📝 Example Output

```
🎯 Object: fffc6d07590b4872be191dc7820de94f
📍 Views directory: /cluster/51/ecekarasu/views_release

📸 Comparing images:
  Image 1: /cluster/51/ecekarasu/views_release/fffc6d07590b4872be191dc7820de94f/000.png
  Image 2: /cluster/51/ecekarasu/views_release/fffc6d07590b4872be191dc7820de94f/001.png
  Resolution: (256, 256, 3)

🔍 Calculating metrics...
✅ Using Zero-1-to-3 metric implementations

📊 RESULTS:
========================================
    PSNR:    24.56 dB
    SSIM:    0.8234
   LPIPS:    0.1456
     FID:    15.43
========================================

💡 INTERPRETATION:
  PSNR: Good similarity
  SSIM: High structural similarity
  LPIPS: Very similar (perceptually)
  FID: Good feature similarity

✅ Comparison completed!
```

## 🏃‍♂️ Run Examples

```bash
# Run all examples
./example_image_comparison.sh

# Or try your own combinations
python compare_two_images.py fffc6d07590b4872be191dc7820de94f 0 3
python compare_two_images.py fffc6d07590b4872be191dc7820de94f 5 11
```

## 🔧 Use Cases

- **Quick quality check** between any two views
- **Validate object consistency** across viewpoints  
- **Compare similar viewpoints** to check rendering quality
- **Test metric calculations** before running full batch evaluation
- **Generate comparison images** for presentations

## 💡 Tips

- **Views 0-11**: Each object has 12 views (000.png to 011.png)
- **Adjacent views** (0↔1, 1↔2) should have high similarity
- **Opposite views** (0↔6) will have lower similarity 
- **Save comparisons** to visually verify the metrics
- **PSNR > 25**: Generally good quality
- **SSIM > 0.8**: High structural similarity
- **LPIPS < 0.2**: Perceptually very similar
- **FID < 15**: Good feature similarity

Perfect for quick testing before running the full batch evaluation! 🎯 