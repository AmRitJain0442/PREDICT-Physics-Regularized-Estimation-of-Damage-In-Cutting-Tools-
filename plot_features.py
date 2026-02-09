"""
Plot individual feature columns from signal_features.csv
Each feature column gets its own plot showing values across all entries, color-coded by case.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Paths
CSV_PATH = "results/eda/signal_features.csv"
OUT_DIR = "results/features"
os.makedirs(OUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(CSV_PATH)

# Feature columns (exclude case and run)
feature_cols = [c for c in df.columns if c not in ("case", "run")]

# Build a label for x-axis: "c{case}_r{run}"
df["label"] = "c" + df["case"].astype(int).astype(str) + "_r" + df["run"].astype(int).astype(str)

# Color map by case
cases = sorted(df["case"].unique())
cmap = plt.cm.tab10
case_colors = {c: cmap(i % 10) for i, c in enumerate(cases)}
colors = [case_colors[c] for c in df["case"]]

print(f"Plotting {len(feature_cols)} feature columns across {len(df)} entries...")

for idx, col in enumerate(feature_cols):
    fig, ax = plt.subplots(figsize=(16, 5))
    vals = df[col].values.astype(float)

    ax.bar(range(len(vals)), vals, color=colors, edgecolor='black', linewidth=0.3)
    ax.set_title(col, fontsize=14, fontweight='bold')
    ax.set_xlabel("Entry (case_run)")
    ax.set_ylabel("Value")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(df["label"], rotation=90, fontsize=5)
    ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')

    # Legend for cases
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=case_colors[c], edgecolor='black', label=f"Case {int(c)}") for c in cases]
    ax.legend(handles=legend_handles, fontsize=7, loc='upper right', ncol=len(cases))

    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f"{col}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [{idx+1}/{len(feature_cols)}] {fname}")

print(f"\nDone — {len(feature_cols)} plots saved to {OUT_DIR}/")
