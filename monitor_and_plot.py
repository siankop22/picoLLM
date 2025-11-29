#!/usr/bin/env python3
"""
Monitor training progress and generate plot when all weights are collected.
"""

import json
import time
from pathlib import Path
import matplotlib.pyplot as plt

TARGET_WEIGHTS = [0.3, 0.5, 0.7, 0.9, 1.0]
LOG_FILE = "artifacts/training_budget_sweep.json"

print("Monitoring training progress...")
print(f"Target weights: {TARGET_WEIGHTS}")
print()

while True:
    if not Path(LOG_FILE).exists():
        print("Waiting for log file to be created...")
        time.sleep(10)
        continue
    
    try:
        with open(LOG_FILE, 'r') as f:
            sweep = json.load(f)
    except (json.JSONDecodeError, IOError):
        print("Log file is being written, waiting...")
        time.sleep(10)
        continue
    
    # Filter for TinyStories entries with target weights
    tinystories_entries = []
    for weight in TARGET_WEIGHTS:
        matching = [e for e in sweep if abs(e.get("tinystories_weight", 0.0) - weight) < 0.01]
        if matching:
            tinystories_entries.extend(matching)
    
    collected_weights = sorted(set([e.get("tinystories_weight", 0.0) for e in tinystories_entries]))
    missing = [w for w in TARGET_WEIGHTS if w not in collected_weights]
    
    print(f"Collected weights: {collected_weights}")
    if missing:
        print(f"Still missing: {missing}")
        print("Waiting 30 seconds before next check...")
        time.sleep(30)
    else:
        print("\n✓ All weights collected! Generating plot...")
        break

# Generate the plot
sweep_sorted = sorted(tinystories_entries, key=lambda e: e.get("tinystories_weight", 0.0))
weights = [e.get("tinystories_weight", 0.0) for e in sweep_sorted]
lstm_training = [e["model_metrics"]["lstm_seq"].get("final_loss") for e in sweep_sorted]
lstm_testing = [e["model_metrics"]["lstm_seq"].get("eval_loss") or e["model_metrics"]["lstm_seq"].get("final_eval_loss") for e in sweep_sorted]
tf_training = [e["model_metrics"]["transformer_gptoss"].get("final_loss") for e in sweep_sorted]
tf_testing = [e["model_metrics"]["transformer_gptoss"].get("eval_loss") or e["model_metrics"]["transformer_gptoss"].get("final_eval_loss") for e in sweep_sorted]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(weights, lstm_training, marker="o", color="#d62728", label="LSTM training loss", linewidth=2.5, markersize=8)
ax.plot(weights, lstm_testing, marker="o", color="#d62728", linestyle="--", label="LSTM testing loss", linewidth=2.5, markersize=8, alpha=0.7)
ax.plot(weights, tf_training, marker="s", color="#2ca02c", label="GPT-oss training loss", linewidth=2.5, markersize=8)
ax.plot(weights, tf_testing, marker="s", color="#2ca02c", linestyle="--", label="GPT-oss testing loss", linewidth=2.5, markersize=8, alpha=0.7)
ax.set_xlabel("TinyStories Weight", fontsize=12, fontweight='bold')
ax.set_ylabel("Average Loss", fontsize=12, fontweight='bold')
ax.set_title("Loss vs. TinyStories Weight (60/40 Train/Test Split)", fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(TARGET_WEIGHTS)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='best', fontsize=11, framealpha=0.9)
plt.tight_layout()
plt.savefig("artifacts/training_budget_sweep_plot.png", dpi=150, bbox_inches='tight')
print(f"Plot saved to artifacts/training_budget_sweep_plot.png")
plt.show()

print("\nSummary:")
for entry in sweep_sorted:
    weight = entry.get("tinystories_weight", 0.0)
    eval_loss_lstm = entry["model_metrics"]["lstm_seq"].get("eval_loss") or entry["model_metrics"]["lstm_seq"].get("final_eval_loss")
    eval_loss_tf = entry["model_metrics"]["transformer_gptoss"].get("eval_loss") or entry["model_metrics"]["transformer_gptoss"].get("final_eval_loss")
    testing_lstm_str = f"{eval_loss_lstm:.4f}" if eval_loss_lstm is not None else "N/A"
    testing_tf_str = f"{eval_loss_tf:.4f}" if eval_loss_tf is not None else "N/A"
    print(
        f"weight={weight:.2f}, training_lstm={entry['model_metrics']['lstm_seq']['final_loss']:.4f}, "
        f"testing_lstm={testing_lstm_str}, "
        f"training_gptoss={entry['model_metrics']['transformer_gptoss']['final_loss']:.4f}, "
        f"testing_gptoss={testing_tf_str}"
    )

