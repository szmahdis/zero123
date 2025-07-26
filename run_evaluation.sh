#!/bin/bash

# Zero-1-to-3 Batch Evaluation Runner
# Usage: ./run_evaluation.sh object_id1 object_id2 object_id3 ...

set -e

echo "🚀 Zero-1-to-3 Batch Evaluation Script"
echo "======================================"

# Check if object IDs provided
if [ $# -eq 0 ]; then
    echo "❌ Error: Please provide object IDs as arguments"
    echo ""
    echo "Usage: $0 <object_id1> <object_id2> ..."
    echo "Example: $0 fffc6d07590b4872be191dc7820de94f a1b2c3d4e5f6..."
    exit 1
fi

# Configuration
VIEWS_DIR="/cluster/51/ecekarasu/views_release"
OUTPUT_DIR="./evaluation_results/batch_evaluation"
CONFIG_PATH="zero123/configs/sd-objaverse-finetune-c_concat-256.yaml"
CHECKPOINT_PATH="zero123/105000.ckpt"

# Print configuration
echo "📁 Views directory: $VIEWS_DIR"
echo "📁 Output directory: $OUTPUT_DIR"
echo "🎯 Object IDs: $@"
echo ""

# Check if views directory exists
if [ ! -d "$VIEWS_DIR" ]; then
    echo "❌ Error: Views directory not found: $VIEWS_DIR"
    echo "Please update VIEWS_DIR in this script to point to your views_release directory"
    exit 1
fi

# Create zero123 directory if needed and download checkpoint
if [ ! -d "zero123" ]; then
    echo "📦 Setting up Zero-1-to-3..."
    # If zero123 directory doesn't exist, we're probably in the wrong location
    if [ ! -f "README.md" ] || ! grep -q "Zero-1-to-3" README.md 2>/dev/null; then
        echo "❌ Error: Please run this script from the zero123-main directory"
        exit 1
    fi
fi

# Download model checkpoint if not exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "📥 Downloading Zero-1-to-3 checkpoint..."
    cd zero123
    wget -q https://cv.cs.columbia.edu/zero123/assets/105000.ckpt
    cd ..
    echo "✅ Checkpoint downloaded"
fi

# Check if config exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ Error: Config file not found: $CONFIG_PATH"
    exit 1
fi

# Install requirements if needed
if [ ! -f "zero123/requirements_installed.flag" ]; then
    echo "📦 Installing requirements..."
    cd zero123
    pip install -r requirements.txt
    touch requirements_installed.flag
    cd ..
    echo "✅ Requirements installed"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation
echo "🔥 Starting evaluation..."
echo ""

python batch_eval_zero123.py \
    --object_ids "$@" \
    --views_dir "$VIEWS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --config "$CONFIG_PATH" \
    --ckpt "$CHECKPOINT_PATH" \
    --device "cuda:0" \
    --input_view 0 \
    --n_samples 4 \
    --ddim_steps 50 \
    --guidance_scale 3.0

echo ""
echo "✅ Evaluation completed!"
echo "📊 Results saved to: $OUTPUT_DIR"
echo ""
echo "📋 Summary of outputs:"
echo "  - Individual object results: $OUTPUT_DIR/[object_id]/"
echo "  - Generated images: $OUTPUT_DIR/[object_id]/generated_*.png"
echo "  - Ground truth copies: $OUTPUT_DIR/[object_id]/gt_*.png"
echo "  - Complete results: $OUTPUT_DIR/batch_evaluation_results.json"
echo ""
echo "🎉 Done!" 