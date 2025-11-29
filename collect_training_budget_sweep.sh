#!/bin/bash

# Script to collect training budget sweep metrics for different TinyStories weights
# This will run training for weights: 0.3, 0.5, 0.7, 0.9, 1.0

WEIGHTS=(0.3 0.5 0.7 0.9 1.0)
LOG_FILE="artifacts/training_budget_sweep.json"

# Create artifacts directory if it doesn't exist
mkdir -p artifacts

# Clear existing log file if you want a fresh start (comment out if you want to append)
# rm -f "$LOG_FILE"

echo "Starting training budget sweep collection..."
echo "This will run training for ${#WEIGHTS[@]} different TinyStories weights"
echo "Log file: $LOG_FILE"
echo ""

for weight in "${WEIGHTS[@]}"; do
    echo "=========================================="
    echo "Running training with TinyStories weight: $weight"
    echo "=========================================="
    
    OMP_NUM_THREADS=1 python3 pico-llm.py \
        --tinystories_weight "$weight" \
        --test_split_ratio 0.4 \
        --record_custom_sweep \
        --custom_sweep_log "$LOG_FILE" \
        --enable_transformer_variants gptoss \
        --num_epochs 3 \
        --batch_size 16 \
        --device_id cpu
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed training for weight $weight"
    else
        echo "✗ Failed training for weight $weight"
        exit 1
    fi
    echo ""
done

echo "=========================================="
echo "All training runs completed!"
echo "Results saved to: $LOG_FILE"
echo "=========================================="

