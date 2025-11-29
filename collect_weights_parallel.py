#!/usr/bin/env python3
"""
Script to collect training budget sweep metrics for different TinyStories weights in parallel.
This will run training for weights: 0.3, 0.5, 0.7, 0.9, 1.0 simultaneously.
"""

import subprocess
import sys
import os
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

WEIGHTS = [0.3, 0.5, 0.7, 0.9, 1.0]
LOG_FILE = "artifacts/training_budget_sweep.json"
TEMP_DIR = Path("artifacts/temp_sweep_logs")

# Create artifacts directory if it doesn't exist
Path("artifacts").mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

print("Starting parallel training budget sweep collection...")
print(f"This will run training for {len(WEIGHTS)} different TinyStories weights in parallel")
print(f"Log file: {LOG_FILE}")
print(f"Temporary logs: {TEMP_DIR}")
print()

# Clear the main log file and temp directory before starting new collection
if Path(LOG_FILE).exists():
    Path(LOG_FILE).unlink()
for temp_file in TEMP_DIR.glob("weight_*.json"):
    temp_file.unlink()


def run_weight(weight):
    """Run training for a single weight and return the result."""
    temp_log = TEMP_DIR / f"weight_{weight}.json"
    
    print(f"[Weight {weight}] Starting training...")
    start_time = time.time()
    
    cmd = [
        "python3", "pico-llm.py",
        "--tinystories_weight", str(weight),
        "--test_split_ratio", "0.4",
        "--tinystories_eval_ratio", "0.2",
        "--record_custom_sweep",
        "--custom_sweep_log", str(temp_log),
        "--enable_transformer_variants", "gptoss",
        "--num_epochs", "3",
        "--batch_size", "16",
        "--max_steps_per_epoch", "50",
        "--save_checkpoints",
        "--checkpoint_dir", "artifacts/checkpoints",
        "--device_id", "cpu"
    ]
    
    # Set OMP_NUM_THREADS environment variable
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"[Weight {weight}] ✓ Successfully completed training (took {elapsed:.1f}s)")
        return (weight, True, None)
    else:
        error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
        print(f"[Weight {weight}] ✗ Failed training (took {elapsed:.1f}s)")
        print(f"[Weight {weight}] Error: {error_msg}")
        return (weight, False, error_msg)


# Run all weights in parallel
print("Launching parallel training runs...")
print()

start_time = time.time()
failed_weights = []

# Use ProcessPoolExecutor to run in parallel
with ProcessPoolExecutor(max_workers=len(WEIGHTS)) as executor:
    # Submit all tasks
    future_to_weight = {executor.submit(run_weight, weight): weight for weight in WEIGHTS}
    
    # Wait for all to complete and collect results
    for future in as_completed(future_to_weight):
        weight, success, error = future.result()
        if not success:
            failed_weights.append((weight, error))

total_time = time.time() - start_time

# Merge all temporary log files into the main log file
print()
print("Merging results from all weights...")
all_entries = []

for temp_file in sorted(TEMP_DIR.glob("weight_*.json")):
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
with open(LOG_FILE, "w") as f:
    json.dump(all_entries, f, indent=2)

print(f"\n✓ Merged {len(all_entries)} entries into {LOG_FILE}")

# Summary
print()
print("=" * 50)
if not failed_weights:
    print("All training runs completed successfully!")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
else:
    print(f"Some training runs failed ({len(failed_weights)}/{len(WEIGHTS)})")
    for weight, error in failed_weights:
        print(f"  Weight {weight}: Failed")
    print("Partial results saved to:", LOG_FILE)
    sys.exit(1)
print(f"Results saved to: {LOG_FILE}")
print("=" * 50)

