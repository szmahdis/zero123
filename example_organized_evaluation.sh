#!/bin/bash

# Example: Complete organized evaluation workflow
# All results will be saved under evaluation_results/

echo "🎯 Complete Zero-1-to-3 Evaluation Workflow"
echo "============================================="
echo "📁 All results will be saved in evaluation_results/"
echo ""

# Setup
chmod +x *.py *.sh

# Create organized directory structure
echo "📂 Setting up directory structure..."
python create_evaluation_structure.py

echo ""
echo "🔍 Step 1: Camera Analysis"
echo "-------------------------"
echo "Analyzing camera data for your object..."
python inspect_camera_data.py fffc6d07590b4872be191dc7820de94f > evaluation_results/camera_analysis/camera_analysis_fffc6d07590b4872be191dc7820de94f.txt

echo ""
echo "📊 Step 2: Simple Image Comparisons"  
echo "-----------------------------------"
echo "Comparing different viewpoints..."

# Adjacent views
python compare_two_images.py fffc6d07590b4872be191dc7820de94f 0 1 \
    --save_comparison --output_dir evaluation_results/comparisons

# Opposite views
python compare_two_images.py fffc6d07590b4872be191dc7820de94f 0 6 \
    --save_comparison --output_dir evaluation_results/comparisons

# Distant views
python compare_two_images.py fffc6d07590b4872be191dc7820de94f 2 8 \
    --save_comparison --output_dir evaluation_results/comparisons

echo ""
echo "🚀 Step 3: Full Zero-1-to-3 Evaluation"
echo "--------------------------------------"
echo "Running complete model evaluation..."
./run_evaluation.sh fffc6d07590b4872be191dc7820de94f

echo ""
echo "📋 Step 4: Results Summary"
echo "-------------------------"

# Show final structure
echo "📂 Final results structure:"
ls -la evaluation_results/
echo ""

echo "📊 Batch evaluation results:"
if [ -d "evaluation_results/batch_evaluation" ]; then
    ls -la evaluation_results/batch_evaluation/
fi

echo ""
echo "🖼️  Comparison images:"
if [ -d "evaluation_results/comparisons" ]; then
    find evaluation_results/comparisons -name "*.png" | head -5
fi

echo ""
echo "✅ Complete evaluation finished!"
echo ""
echo "📁 Check your results in:"
echo "   evaluation_results/batch_evaluation/    # Zero-1-to-3 model results"
echo "   evaluation_results/comparisons/         # Image comparison results"  
echo "   evaluation_results/camera_analysis/     # Camera data analysis"
echo ""
echo "🎉 All organized and ready for analysis!" 