#!/bin/bash

# Example script showing how to compare two images from an Objaverse object

echo "🎯 Image Comparison Examples"
echo "============================"

# Make the script executable
chmod +x compare_two_images.py

echo ""
echo "📸 Example 1: Compare view 000 vs view 001 (adjacent views)"
python compare_two_images.py fffc6d07590b4872be191dc7820de94f 0 1 --output_dir evaluation_results/comparisons

echo ""
echo "📸 Example 2: Compare view 000 vs view 006 (opposite views)"
python compare_two_images.py fffc6d07590b4872be191dc7820de94f 0 6 --save_comparison --output_dir evaluation_results/comparisons

echo ""
echo "📸 Example 3: Compare distant views (should have lower similarity)"
python compare_two_images.py fffc6d07590b4872be191dc7820de94f 2 8 --save_comparison --output_dir evaluation_results/comparisons

echo ""
echo "✅ Comparison completed!"
echo "💡 Try different view combinations:"
echo "   python compare_two_images.py OBJECT_ID VIEW1 VIEW2"
echo "   Views range from 0 to 11" 