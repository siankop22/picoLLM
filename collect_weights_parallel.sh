#!/bin/bash

# Script to collect training budget sweep metrics for different TinyStories weights in parallel
# This will run training for weights: 0.3, 0.5, 0.7, 0.9, 1.0 simultaneously

WEIGHTS=(0.3 0.5 0.7 0.9 1.0)
LOG_FILE="artifacts/training_budget_sweep.json"
TEMP_DIR="artifacts/temp_sweep_logs"

# Create artifacts directory if it doesn't exist
mkdir -p artifacts
mkdir -p "$TEMP_DIR"

echo "Starting parallel training budget sweep collection..."
echo "This will run training for ${#WEIGHTS[@]} different TinyStories weights in parallel"
echo "Log file: $LOG_FILE"
echo "Temporary logs: $TEMP_DIR"
echo ""

# Clear the main log file and temp directory before starting new collection
> "$LOG_FILE"
rm -f "$TEMP_DIR"/*.json

# Function to run training for a single weight
run_weight() {
    local weight=$1
    local temp_log="$TEMP_DIR/weight_${weight}.json"
    
    echo "[Weight $weight] Starting training..."
    
    OMP_NUM_THREADS=1 python3 pico-llm.py \
        --tinystories_weight "$weight" \
        --test_split_ratio 0.4 \
        --tinystories_eval_ratio 0.2 \
        --record_custom_sweep \
        --custom_sweep_log "$temp_log" \
        --enable_transformer_variants gptoss \
        --num_epochs 3 \
        --batch_size 16 \
        --max_steps_per_epoch 50 \
        --save_checkpoints \
        --checkpoint_dir artifacts/checkpoints \
        --device_id cpu
    
    if [ $? -eq 0 ]; then
        echo "[Weight $weight] ✓ Successfully completed training"
    else
        echo "[Weight $weight] ✗ Failed training"
        return 1
    fi
}

# Run all weights in parallel using background processes
PIDS=()
for weight in "${WEIGHTS[@]}"; do
    run_weight "$weight" &
    PIDS+=($!)
done

# Wait for all background processes to complete
echo ""
echo "Waiting for all training runs to complete..."
FAILED=0
for i in "${!WEIGHTS[@]}"; do
    weight=${WEIGHTS[$i]}
    pid=${PIDS[$i]}
    wait $pid
    if [ $? -ne 0 ]; then
        echo "[Weight $weight] Process failed!"
        FAILED=1
    else
        echo "[Weight $weight] Process completed successfully"
    fi
done

# Merge all temporary log files into the main log file
echo ""
echo "Merging results from all weights..."
python3 << EOF
import json
from pathlib import Path

log_file = Path("$LOG_FILE")
temp_dir = Path("$TEMP_DIR")

all_entries = []
for temp_file in sorted(temp_dir.glob("weight_*.json")):
    try:
        with open(temp_file) as f:
            data = json.load(f)
            if isinstance(data, list):
                all_entries.extend(data)
            else:
                all_entries.append(data)
        print(f"  Merged {temp_file.name}")
    except Exception as e:
        print(f"  Error reading {temp_file.name}: {e}")

# Sort by timestamp
all_entries.sort(key=lambda x: x.get("timestamp", 0))

# Write merged results
with open(log_file, "w") as f:
    json.dump(all_entries, f, indent=2)

print(f"\n✓ Merged {len(all_entries)} entries into {log_file}")
EOF

if [ $FAILED -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "All training runs completed successfully!"
    echo "Results saved to: $LOG_FILE"
    echo "=================================================="
else
    echo ""
    echo "=================================================="
    echo "Some training runs failed. Check the output above."
    echo "Partial results saved to: $LOG_FILE"
    echo "=================================================="
    exit 1
fi

