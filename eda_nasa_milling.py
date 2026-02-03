"""
NASA Milling Dataset - Exploratory Data Analysis
=================================================
Generates overlapping signal plots, statistical metrics, and correlation
analysis to inform data science approach for tool wear prediction.

Outputs saved to: ./results/eda/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────
CSV_DIR = "./data/nasa_milling/csv"
SIGNALS_DIR = os.path.join(CSV_DIR, "signals")
RESULTS_DIR = "./results/eda"
os.makedirs(RESULTS_DIR, exist_ok=True)

SIGNAL_COLS = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
SIGNAL_LABELS = {
    "smcAC": "Spindle Motor Current (AC)",
    "smcDC": "Spindle Motor Current (DC)",
    "vib_table": "Table Vibration",
    "vib_spindle": "Spindle Vibration",
    "AE_table": "Acoustic Emission (Table)",
    "AE_spindle": "Acoustic Emission (Spindle)",
}


def load_all_data():
    """Load metadata and all signal files."""
    meta = pd.read_csv(os.path.join(CSV_DIR, "metadata.csv"))
    signals = {}
    for _, row in meta.iterrows():
        case_id = int(row["case"])
        run_id = int(row["run"])
        fname = f"case{case_id}_run{run_id:03d}.csv"
        fpath = os.path.join(SIGNALS_DIR, fname)
        if os.path.exists(fpath):
            signals[(case_id, run_id)] = pd.read_csv(fpath)
    return meta, signals


def compute_signal_features(signal_df):
    """Compute statistical features for a single run's signals."""
    feats = {}
    for col in SIGNAL_COLS:
        s = signal_df[col].values
        feats[f"{col}_mean"] = np.mean(s)
        feats[f"{col}_std"] = np.std(s)
        feats[f"{col}_rms"] = np.sqrt(np.mean(s ** 2))
        feats[f"{col}_kurtosis"] = stats.kurtosis(s)
        feats[f"{col}_skewness"] = stats.skew(s)
        feats[f"{col}_peak"] = np.max(np.abs(s))
        feats[f"{col}_p2p"] = np.ptp(s)
        feats[f"{col}_crest"] = np.max(np.abs(s)) / (np.sqrt(np.mean(s ** 2)) + 1e-10)
        # Frequency-domain
        fft_vals = np.abs(np.fft.rfft(s))
        feats[f"{col}_fft_energy"] = np.sum(fft_vals ** 2)
        feats[f"{col}_fft_peak_freq"] = np.argmax(fft_vals[1:]) + 1
        feats[f"{col}_fft_mean"] = np.mean(fft_vals)
    return feats


def plot_overlapping_signals_by_case(meta, signals):
    """Plot overlapping signals for each case, colored by wear level."""
    cases = sorted(meta["case"].unique())
    for col in SIGNAL_COLS:
        fig, axes = plt.subplots(4, 4, figsize=(20, 16), sharex=True)
        fig.suptitle(f"Overlapping {SIGNAL_LABELS[col]} Across Runs (colored by wear VB)",
                     fontsize=14, fontweight="bold")
        for idx, case_id in enumerate(cases):
            ax = axes[idx // 4, idx % 4]
            case_runs = meta[meta["case"] == case_id].sort_values("run")
            vb_vals = case_runs["VB"].values
            vb_max = meta["VB"].max()
            cmap = get_cmap("coolwarm")
            for _, row in case_runs.iterrows():
                key = (int(row["case"]), int(row["run"]))
                if key not in signals:
                    continue
                sig = signals[key][col].values
                vb = row["VB"]
                color = cmap(vb / vb_max) if pd.notna(vb) else "gray"
                alpha = 0.6 if pd.notna(vb) else 0.2
                # Downsample for plotting speed
                plot_sig = sig[::10]
                ax.plot(plot_sig, color=color, alpha=alpha, linewidth=0.5)
            ax.set_title(f"Case {case_id}", fontsize=9)
            ax.tick_params(labelsize=7)
        for ax in axes.flat:
            ax.set_xlabel("Sample (×10)", fontsize=7)
            ax.set_ylabel("Amplitude", fontsize=7)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(RESULTS_DIR, f"overlap_{col}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")


def plot_wear_progression(meta, feature_df):
    """Plot how sensor features evolve with wear (VB)."""
    df = feature_df.merge(meta[["case", "run", "VB", "DOC", "feed", "material"]],
                          on=["case", "run"])
    df_valid = df.dropna(subset=["VB"])

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Sensor Feature RMS vs Flank Wear (VB)", fontsize=14, fontweight="bold")
    for idx, col in enumerate(SIGNAL_COLS):
        ax = axes[idx // 3, idx % 3]
        feat_col = f"{col}_rms"
        for case_id in sorted(df_valid["case"].unique()):
            subset = df_valid[df_valid["case"] == case_id].sort_values("VB")
            ax.plot(subset["VB"], subset[feat_col], "o-", markersize=3,
                    alpha=0.6, label=f"Case {case_id}")
        ax.set_xlabel("Flank Wear VB (mm)")
        ax.set_ylabel("RMS")
        ax.set_title(SIGNAL_LABELS[col], fontsize=10)
        ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(RESULTS_DIR, "wear_vs_rms.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    # Correlation heatmap
    rms_cols = [f"{c}_rms" for c in SIGNAL_COLS]
    std_cols = [f"{c}_std" for c in SIGNAL_COLS]
    mean_cols = [f"{c}_mean" for c in SIGNAL_COLS]
    kurt_cols = [f"{c}_kurtosis" for c in SIGNAL_COLS]
    fft_cols = [f"{c}_fft_energy" for c in SIGNAL_COLS]
    all_feat_cols = rms_cols + std_cols + mean_cols + kurt_cols + fft_cols
    corr_with_vb = df_valid[all_feat_cols + ["VB"]].corr()["VB"].drop("VB").sort_values()

    fig, ax = plt.subplots(figsize=(10, 12))
    colors = ["#d73027" if v < 0 else "#4575b4" for v in corr_with_vb.values]
    ax.barh(range(len(corr_with_vb)), corr_with_vb.values, color=colors)
    ax.set_yticks(range(len(corr_with_vb)))
    ax.set_yticklabels(corr_with_vb.index, fontsize=7)
    ax.set_xlabel("Pearson Correlation with VB")
    ax.set_title("Feature Correlation with Flank Wear (VB)", fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "correlation_with_VB.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    return corr_with_vb


def plot_distributions(meta, feature_df):
    """Plot feature distributions grouped by wear severity."""
    df = feature_df.merge(meta[["case", "run", "VB"]], on=["case", "run"])
    df_valid = df.dropna(subset=["VB"])
    df_valid["wear_group"] = pd.cut(df_valid["VB"], bins=[0, 0.2, 0.4, 0.6, 2.0],
                                     labels=["Low (<0.2)", "Medium (0.2-0.4)",
                                             "High (0.4-0.6)", "Severe (>0.6)"])

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Signal RMS Distribution by Wear Severity", fontsize=14, fontweight="bold")
    colors = ["#2166ac", "#92c5de", "#f4a582", "#b2182b"]
    for idx, col in enumerate(SIGNAL_COLS):
        ax = axes[idx // 3, idx % 3]
        feat_col = f"{col}_rms"
        for i, grp in enumerate(["Low (<0.2)", "Medium (0.2-0.4)",
                                  "High (0.4-0.6)", "Severe (>0.6)"]):
            subset = df_valid[df_valid["wear_group"] == grp][feat_col]
            if len(subset) > 0:
                ax.hist(subset, bins=15, alpha=0.5, label=grp, color=colors[i])
        ax.set_title(SIGNAL_LABELS[col], fontsize=10)
        ax.set_xlabel("RMS")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(RESULTS_DIR, "distributions_by_wear.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_condition_comparison(meta, feature_df):
    """Compare signal behaviour across machining conditions."""
    df = feature_df.merge(meta[["case", "run", "VB", "DOC", "feed", "material"]],
                          on=["case", "run"])
    df_valid = df.dropna(subset=["VB"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Signal Response Under Different Machining Conditions",
                 fontsize=13, fontweight="bold")

    # By DOC
    ax = axes[0]
    for doc in sorted(df_valid["DOC"].unique()):
        sub = df_valid[df_valid["DOC"] == doc]
        ax.scatter(sub["VB"], sub["smcAC_rms"], alpha=0.5, s=20, label=f"DOC={doc}")
    ax.set_xlabel("VB (mm)"); ax.set_ylabel("smcAC RMS")
    ax.set_title("Depth of Cut"); ax.legend(); ax.grid(True, alpha=0.3)

    # By Feed
    ax = axes[1]
    for feed in sorted(df_valid["feed"].unique()):
        sub = df_valid[df_valid["feed"] == feed]
        ax.scatter(sub["VB"], sub["vib_table_rms"], alpha=0.5, s=20, label=f"Feed={feed}")
    ax.set_xlabel("VB (mm)"); ax.set_ylabel("vib_table RMS")
    ax.set_title("Feed Rate"); ax.legend(); ax.grid(True, alpha=0.3)

    # By Material
    ax = axes[2]
    for mat in sorted(df_valid["material"].unique()):
        sub = df_valid[df_valid["material"] == mat]
        lbl = "Cast Iron" if mat == 1 else "Steel"
        ax.scatter(sub["VB"], sub["AE_table_rms"], alpha=0.5, s=20, label=lbl)
    ax.set_xlabel("VB (mm)"); ax.set_ylabel("AE_table RMS")
    ax.set_title("Material"); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(RESULTS_DIR, "condition_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("=" * 60)
    print("  NASA Milling Dataset - Exploratory Data Analysis")
    print("=" * 60)

    # ── 1. Load data ───────────────────────────────────────────
    print("\n[1/6] Loading data ...")
    meta, signals = load_all_data()
    print(f"  Loaded metadata: {len(meta)} runs, {len(signals)} signal files")

    # ── 2. Compute features ────────────────────────────────────
    print("\n[2/6] Computing statistical features ...")
    rows = []
    for (case_id, run_id), sig_df in sorted(signals.items()):
        feats = compute_signal_features(sig_df)
        feats["case"] = case_id
        feats["run"] = run_id
        rows.append(feats)
    feature_df = pd.DataFrame(rows)
    feat_path = os.path.join(RESULTS_DIR, "signal_features.csv")
    feature_df.to_csv(feat_path, index=False)
    print(f"  Saved features: {feat_path}  ({len(feature_df)} runs × {len(feature_df.columns)} cols)")

    # ── 3. Overlapping signal plots ────────────────────────────
    print("\n[3/6] Plotting overlapping signals by case ...")
    plot_overlapping_signals_by_case(meta, signals)

    # ── 4. Wear progression ────────────────────────────────────
    print("\n[4/6] Plotting wear progression and correlations ...")
    corr_with_vb = plot_wear_progression(meta, feature_df)

    # ── 5. Distribution analysis ───────────────────────────────
    print("\n[5/6] Plotting distributions by wear severity ...")
    plot_distributions(meta, feature_df)

    # ── 6. Condition comparison ────────────────────────────────
    print("\n[6/6] Plotting machining condition comparisons ...")
    plot_condition_comparison(meta, feature_df)

    # ── Summary report ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Summary Metrics")
    print("=" * 60)

    meta_valid = meta.dropna(subset=["VB"])
    print(f"\n  Total runs: {len(meta)}")
    print(f"  Runs with VB measurement: {len(meta_valid)}")
    print(f"  Cases: {sorted(meta['case'].unique().tolist())}")
    print(f"  VB range: {meta_valid['VB'].min():.3f} - {meta_valid['VB'].max():.3f} mm")
    print(f"  VB mean ± std: {meta_valid['VB'].mean():.3f} ± {meta_valid['VB'].std():.3f} mm")

    print(f"\n  Top 5 features correlated with VB:")
    top5 = corr_with_vb.abs().sort_values(ascending=False).head(5)
    for feat, val in top5.items():
        sign = corr_with_vb[feat]
        print(f"    {feat:30s}  r = {sign:+.3f}")

    print(f"\n  Bottom 5 features (weakest correlation):")
    bot5 = corr_with_vb.abs().sort_values(ascending=True).head(5)
    for feat, val in bot5.items():
        sign = corr_with_vb[feat]
        print(f"    {feat:30s}  r = {sign:+.3f}")

    # Save summary text
    summary_path = os.path.join(RESULTS_DIR, "summary_report.txt")
    with open(summary_path, "w") as f:
        f.write("NASA Milling Dataset - EDA Summary Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total runs: {len(meta)}\n")
        f.write(f"Runs with VB: {len(meta_valid)}\n")
        f.write(f"Cases: {len(meta['case'].unique())}\n")
        f.write(f"VB range: {meta_valid['VB'].min():.3f} - {meta_valid['VB'].max():.3f} mm\n")
        f.write(f"VB mean +/- std: {meta_valid['VB'].mean():.3f} +/- {meta_valid['VB'].std():.3f} mm\n")
        f.write(f"DOC values: {sorted(meta['DOC'].unique().tolist())}\n")
        f.write(f"Feed values: {sorted(meta['feed'].unique().tolist())}\n")
        f.write(f"Materials: {sorted(meta['material'].unique().tolist())}\n\n")
        f.write("Feature Correlations with VB (sorted by |r|)\n")
        f.write("-" * 50 + "\n")
        for feat, val in corr_with_vb.abs().sort_values(ascending=False).items():
            f.write(f"  {feat:35s}  r = {corr_with_vb[feat]:+.4f}\n")
    print(f"\n  Saved summary: {summary_path}")

    print("\n" + "=" * 60)
    print("  All results saved to: ./results/eda/")
    print("=" * 60)


if __name__ == "__main__":
    main()
