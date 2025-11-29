#!/usr/bin/env python3
"""
Script to collect training budget sweep metrics for different TinyStories weights.
This will run training for weights: 0.3, 0.5, 0.7, 0.9, 1.0
"""

import subprocess
import sys
import os
from pathlib import Path

WEIGHTS = [0.3, 0.5, 0.7, 0.9, 1.0]
LOG_FILE = "artifacts/training_budget_sweep.json"

# Create artifacts directory if it doesn't exist
Path("artifacts").mkdir(parents=True, exist_ok=True)

print("Starting training budget sweep collection...")
print(f"This will run training for {len(WEIGHTS)} different TinyStories weights")
print(f"Log file: {LOG_FILE}")
print()

for weight in WEIGHTS:
    print("=" * 50)
    print(f"Running training with TinyStories weight: {weight}")
    print("=" * 50)
    
    cmd = [
        "python3", "pico-llm.py",
        "--tinystories_weight", str(weight),
        "--test_split_ratio", "0.4",
        "--record_custom_sweep",
        "--custom_sweep_log", LOG_FILE,
        "--enable_transformer_variants", "gptoss",
        "--num_epochs", "3",
        "--batch_size", "16",
        "--device_id", "cpu"
    ]
    
    # Set OMP_NUM_THREADS environment variable
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    
    result = subprocess.run(cmd, env=env)
    
    if result.returncode == 0:
        print(f"✓ Successfully completed training for weight {weight}")
    else:
        print(f"✗ Failed training for weight {weight}")
        sys.exit(1)
    print()

print("=" * 50)
print("All training runs completed!")
print(f"Results saved to: {LOG_FILE}")
print("=" * 50)

