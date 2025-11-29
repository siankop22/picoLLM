#!/bin/bash

# Step 1: Collect training and testing loss for TinyStories weight 0.3
# Note: The text generation samples (greedy, 0.95, 1.0) are just for display
# and don't affect the training/testing loss metrics

echo "=========================================="
echo "Collecting metrics for TinyStories weight: 0.3"
echo "=========================================="

OMP_NUM_THREADS=1 python3 pico-llm.py \
    --tinystories_weight 0.3 \
    --test_split_ratio 0.4 \
    --record_custom_sweep \
    --custom_sweep_log artifacts/training_budget_sweep.json \
    --enable_transformer_variants gptoss \
    --num_epochs 3 \
    --batch_size 16 \
    --max_steps_per_epoch 50 \
    --device_id cpu

if [ $? -eq 0 ]; then
    echo "✓ Successfully completed training for weight 0.3"
    echo ""
    echo "Training and testing loss metrics have been saved to:"
    echo "  artifacts/training_budget_sweep.json"
else
    echo "✗ Failed training for weight 0.3"
    exit 1
fi

