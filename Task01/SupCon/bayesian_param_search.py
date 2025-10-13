import os
import subprocess
import itertools
import pandas as pd
import numpy as np
from pathlib import Path

# Parameter grid
DETECTION_THRESHOLDS = [0.4, 0.5, 0.6, 0.7]
CONFIDENCE_THRESHOLDS = [0.4, 0.5, 0.6, 0.7]
ALPHAS = [0.4, 0.5, 0.6, 0.7, 0.8]

# Path to validation_bayesian.py (assume same directory)
SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "validation_bayesian.py")

# Output directory (should match default in validation_bayesian.py)
OUTPUT_BASE = "results_emd_augmented_online/results_bayesian_datadriven"

# Metrics to maximize
METRICS = ["f1_micro", "f1_macro", "f1_weighted"]

results = []

def run_one(detection, confidence, alpha):
    run_suffix = f"det{detection}_conf{confidence}_alpha{alpha}"
    cmd = [
        "python", SCRIPT_PATH,
        "--detection_threshold", str(detection),
        "--confidence_threshold", str(confidence),
        "--alpha", str(alpha),
        "--run_suffix", run_suffix
    ]
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Run failed: {e}")
        return None
    # Find metrics file
    metrics_path = os.path.join(
        OUTPUT_BASE, f"run_{run_suffix}", f"cv_adjusted_metrics_datadriven_{run_suffix}.csv"
    )
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found: {metrics_path}")
        return None
    try:
        df = pd.read_csv(metrics_path, index_col=0)
        # Extract "All" column for the metrics
        row = {"detection": detection, "confidence": confidence, "alpha": alpha}
        for metric in METRICS:
            val = df.loc[metric, "All"] if metric in df.index else np.nan
            row[metric] = val
        results.append(row)
        return row
    except Exception as e:
        print(f"Failed to parse metrics: {e}")
        return None

def main():
    param_grid = list(itertools.product(DETECTION_THRESHOLDS, CONFIDENCE_THRESHOLDS, ALPHAS))
    for detection, confidence, alpha in param_grid:
        run_one(detection, confidence, alpha)
    # Summarize results
    if not results:
        print("No successful runs.")
        return
    df = pd.DataFrame(results)
    print("\nBest parameter sets:")
    for metric in METRICS:
        best_idx = df[metric].idxmax()
        best_row = df.loc[best_idx]
        print(f"Best {metric}: {best_row[metric]:.4f} at detection={best_row['detection']}, confidence={best_row['confidence']}, alpha={best_row['alpha']}")
    print("\nTop 5 overall:")
    print(df.sort_values(METRICS, ascending=False).head())

    # Save summary to file
    summary_path = os.path.join(OUTPUT_BASE, "summary.txt")
    try:
        os.makedirs(OUTPUT_BASE, exist_ok=True)
        with open(summary_path, "w") as f:
            f.write("Best parameter sets:\n")
            for metric in METRICS:
                best_idx = df[metric].idxmax()
                best_row = df.loc[best_idx]
                f.write(f"Best {metric}: {best_row[metric]:.4f} at detection={best_row['detection']}, confidence={best_row['confidence']}, alpha={best_row['alpha']}\n")
            f.write("\nTop 5 overall:\n")
            f.write(df.sort_values(METRICS, ascending=False).head().to_string(index=False))
            f.write("\n")
        print(f"Summary saved to {summary_path}")
    except Exception as e:
        print(f"Failed to write summary: {e}")

if __name__ == "__main__":
    main() 