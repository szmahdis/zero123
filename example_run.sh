#!/bin/bash

# Example: How to run Zero-1-to-3 evaluation on your objects
# Replace the object IDs below with your actual ones

# Make the evaluation script executable
chmod +x run_evaluation.sh

echo "🎯 Running Zero-1-to-3 evaluation on sample objects..."

# Example with the object ID you showed me
./run_evaluation.sh fffc6d07590b4872be191dc7820de94f

# Example with multiple object IDs (replace with your actual IDs)
# ./run_evaluation.sh \
#     fffc6d07590b4872be191dc7820de94f \
#     a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6 \
#     another_object_id_here \
#     yet_another_object_id

echo "Done! Check evaluation_results/ directory for outputs" 