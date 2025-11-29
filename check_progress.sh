#!/bin/bash

# Quick script to check progress of training runs

echo "=== Training Progress Check ==="
echo ""

# Check running processes
echo "Running processes:"
python3 << 'PROCESS_CHECK'
import subprocess
import re

result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
lines = result.stdout.split('\n')

weights_found = []
for line in lines:
    if 'pico-llm.py' in line and 'tinystories_weight' in line and 'grep' not in line:
        # Extract weight
        weight_match = re.search(r'--tinystories_weight\s+([0-9.]+)', line)
        if weight_match:
            weight = weight_match.group(1)
            # Extract log file
            log_match = re.search(r'--custom_sweep_log\s+([^\s]+)', line)
            log = log_match.group(1) if log_match else 'main log'
            weights_found.append((weight, log))

for weight, log in sorted(weights_found):
    print(f"  Weight {weight} -> {log}")

if not weights_found:
    print("  No processes found")
PROCESS_CHECK

echo ""
echo "Collected data:"
python3 << EOF
import json
from pathlib import Path

log_file = Path("artifacts/training_budget_sweep.json")
if log_file.exists():
    data = json.loads(log_file.read_text())
    weights = sorted(set([e.get('tinystories_weight', 0) for e in data]))
    print(f"  ✓ Weights in main log: {weights}")
    print(f"  Total entries: {len(data)}")
else:
    print("  ⏳ No data in main log yet")

# Check temp files
temp_dir = Path("artifacts/temp_sweep_logs")
if temp_dir.exists():
    temp_files = list(temp_dir.glob("weight_*.json"))
    if temp_files:
        print(f"  ✓ Temporary log files: {len(temp_files)}")
        for f in sorted(temp_files):
            print(f"    - {f.name}")
    else:
        print("  ⏳ No temporary log files yet")
else:
    print("  ⏳ Temp directory not created yet")
EOF

echo ""
echo "Target weights: 0.3, 0.5, 0.7, 0.9, 1.0"

