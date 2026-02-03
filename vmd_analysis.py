"""
VMD (Variational Mode Decomposition) Analysis - NASA Milling Dataset
=====================================================================
Decomposes sensor signals into intrinsic mode functions (IMFs),
extracts mode-level features, and evaluates correlation with tool wear.

Outputs saved to: ./results/vmd/
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from vmdpy import VMD

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────
CSV_DIR = "./data/nasa_milling/csv"
SIGNALS_DIR = os.path.join(CSV_DIR, "signals")
RESULTS_DIR = "./results/vmd"
os.makedirs(RESULTS_DIR, exist_ok=True)

# VMD parameters
K = 5           # Number of modes
ALPHA = 2000    # Bandwidth constraint (moderate)
TAU = 0         # Noise tolerance (no strict fidelity)
DC = 0          # No DC part imposed
INIT = 1        # Uniform initialization of omegas
TOL = 1e-7      # Convergence tolerance

# Focus on the most physically relevant sensors
TARGET_SENSORS = ["vib_spindle", "vib_table", "AE_table", "AE_spindle", "smcAC", "smcDC"]
SENSOR_LABELS = {
    "vib_spindle": "Spindle Vibration",
    "vib_table": "Table Vibration",
    "AE_table": "Acoustic Emission (Table)",
    "AE_spindle": "Acoustic Emission (Spindle)",
    "smcAC": "Spindle Motor Current (AC)",
    "smcDC": "Spindle Motor Current (DC)",
}


def load_data():
    """Load metadata and all signal files."""
    meta = pd.read_csv(os.path.join(CSV_DIR, "metadata.csv"))
    signals = {}
    for _, row in meta.iterrows():
        c, r = int(row["case"]), int(row["run"])
        fpath = os.path.join(SIGNALS_DIR, f"case{c}_run{r:03d}.csv")
        if os.path.exists(fpath):
            signals[(c, r)] = pd.read_csv(fpath)
    return meta, signals


def apply_vmd(signal, K=K, alpha=ALPHA):
    """Apply VMD to a 1D signal. Returns modes (K x N) and center freqs."""
    # Downsample to 3000 points for computational efficiency
    if len(signal) > 3000:
        indices = np.linspace(0, len(signal) - 1, 3000, dtype=int)
        signal = signal[indices]
    u, u_hat, omega = VMD(signal, alpha, TAU, K, DC, INIT, TOL)
    return u, omega


def compute_mode_features(mode):
    """Extract features from a single VMD mode."""
    feats = {}
    feats["energy"] = np.sum(mode ** 2)
    feats["rms"] = np.sqrt(np.mean(mode ** 2))
    feats["std"] = np.std(mode)
    feats["mean"] = np.mean(mode)
    feats["kurtosis"] = stats.kurtosis(mode)
    feats["skewness"] = stats.skew(mode)
    feats["peak"] = np.max(np.abs(mode))
    feats["p2p"] = np.ptp(mode)
    feats["entropy"] = -np.sum((mode ** 2 / (np.sum(mode ** 2) + 1e-12))
                                * np.log(mode ** 2 / (np.sum(mode ** 2) + 1e-12) + 1e-12))
    # Frequency domain of mode
    fft_vals = np.abs(np.fft.rfft(mode))
    feats["fft_energy"] = np.sum(fft_vals ** 2)
    feats["fft_peak_freq"] = np.argmax(fft_vals[1:]) + 1
    feats["fft_mean"] = np.mean(fft_vals)
    return feats


def plot_vmd_decomposition_examples(meta, signals):
    """Plot VMD decomposition for a few example runs showing wear progression."""
    # Pick one case with good wear spread
    case_runs = meta[(meta["case"] == 1) & meta["VB"].notna()].sort_values("VB")
    # Select low, mid, high wear
    if len(case_runs) >= 3:
        examples = [case_runs.iloc[0], case_runs.iloc[len(case_runs) // 2], case_runs.iloc[-1]]
    else:
        examples = [case_runs.iloc[0], case_runs.iloc[-1]]

    sensor = "vib_spindle"
    fig, axes = plt.subplots(len(examples), K + 1, figsize=(22, 4 * len(examples)))
    fig.suptitle(f"VMD Decomposition of {SENSOR_LABELS[sensor]} (Case 1, K={K})",
                 fontsize=14, fontweight="bold")

    for row_idx, row in enumerate(examples):
        c, r = int(row["case"]), int(row["run"])
        vb = row["VB"]
        sig = signals[(c, r)][sensor].values
        modes, omega = apply_vmd(sig)

        # Plot original
        ax = axes[row_idx, 0] if len(examples) > 1 else axes[0]
        ds = sig[::3] if len(sig) > 3000 else sig
        ax.plot(ds, color="black", linewidth=0.5)
        ax.set_title(f"Original (VB={vb:.2f}mm)", fontsize=9)
        ax.set_ylabel(f"Run {r}", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)

        # Plot each mode
        for k in range(K):
            ax = axes[row_idx, k + 1] if len(examples) > 1 else axes[k + 1]
            ax.plot(modes[k], color=f"C{k}", linewidth=0.5)
            ax.set_title(f"Mode {k + 1} (f={omega[-1, k]:.2f})", fontsize=9)
            ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(RESULTS_DIR, "vmd_decomposition_example.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_mode_spectra(meta, signals):
    """Plot frequency spectra of VMD modes for low vs high wear."""
    meta_valid = meta.dropna(subset=["VB"])
    low_wear = meta_valid[meta_valid["VB"] < 0.15].sample(min(10, len(meta_valid[meta_valid["VB"] < 0.15])), random_state=42)
    high_wear = meta_valid[meta_valid["VB"] > 0.5].sample(min(10, len(meta_valid[meta_valid["VB"] > 0.5])), random_state=42)

    sensor = "vib_spindle"
    fig, axes = plt.subplots(2, K, figsize=(20, 8))
    fig.suptitle(f"VMD Mode Frequency Spectra: Low Wear vs High Wear ({SENSOR_LABELS[sensor]})",
                 fontsize=13, fontweight="bold")

    for k in range(K):
        for runs, color, label, row in [(low_wear, "#2166ac", "Low VB", 0),
                                         (high_wear, "#b2182b", "High VB", 1)]:
            ax = axes[row, k]
            for _, r in runs.iterrows():
                c, run = int(r["case"]), int(r["run"])
                if (c, run) not in signals:
                    continue
                sig = signals[(c, run)][sensor].values
                modes, _ = apply_vmd(sig)
                fft_vals = np.abs(np.fft.rfft(modes[k]))
                freqs = np.fft.rfftfreq(len(modes[k]))
                ax.plot(freqs[:200], fft_vals[:200], color=color, alpha=0.3, linewidth=0.7)
            ax.set_title(f"Mode {k + 1} - {label}", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            if k == 0:
                ax.set_ylabel("FFT Amplitude", fontsize=8)
        axes[1, k].set_xlabel("Normalized Freq", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(RESULTS_DIR, "vmd_mode_spectra.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def extract_all_vmd_features(meta, signals):
    """Extract VMD-based features for all runs and all target sensors."""
    all_rows = []
    n = len(signals)
    for idx, ((c, r), sig_df) in enumerate(sorted(signals.items())):
        row = {"case": c, "run": r}
        for sensor in TARGET_SENSORS:
            sig = sig_df[sensor].values
            try:
                modes, omega = apply_vmd(sig)
                for k in range(K):
                    feats = compute_mode_features(modes[k])
                    for fname, fval in feats.items():
                        row[f"{sensor}_m{k + 1}_{fname}"] = fval
                    row[f"{sensor}_m{k + 1}_center_freq"] = omega[-1, k]
            except Exception as e:
                pass  # Skip if VMD fails on a signal
        all_rows.append(row)
        if (idx + 1) % 20 == 0 or idx == n - 1:
            print(f"    VMD features: {idx + 1}/{n} runs processed")

    return pd.DataFrame(all_rows)


def plot_vmd_correlation(meta, vmd_feats):
    """Plot correlation of VMD features with VB and compare to raw features."""
    df = vmd_feats.merge(meta[["case", "run", "VB"]], on=["case", "run"])
    df_valid = df.dropna(subset=["VB"])

    feat_cols = [c for c in vmd_feats.columns if c not in ["case", "run"]]
    corr = df_valid[feat_cols + ["VB"]].corr()["VB"].drop("VB").dropna().sort_values()

    # Top 30 features
    top30 = corr.abs().sort_values(ascending=False).head(30)
    top30_vals = corr[top30.index]

    fig, ax = plt.subplots(figsize=(10, 12))
    colors = ["#d73027" if v < 0 else "#4575b4" for v in top30_vals.values]
    ax.barh(range(len(top30_vals)), top30_vals.values, color=colors)
    ax.set_yticks(range(len(top30_vals)))
    ax.set_yticklabels(top30_vals.index, fontsize=7)
    ax.set_xlabel("Pearson Correlation with VB")
    ax.set_title("Top 30 VMD Features Correlated with Flank Wear",
                 fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "vmd_correlation_top30.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    return corr


def plot_vmd_vs_raw_comparison(vmd_corr):
    """Compare VMD correlation improvement vs raw features."""
    # Load raw features for comparison
    raw_feat_path = "./results/eda/signal_features.csv"
    if not os.path.exists(raw_feat_path):
        print("  Skipped raw comparison (EDA features not found)")
        return

    raw_feats = pd.read_csv(raw_feat_path)
    meta = pd.read_csv(os.path.join(CSV_DIR, "metadata.csv"))
    df_raw = raw_feats.merge(meta[["case", "run", "VB"]], on=["case", "run"])
    df_raw_valid = df_raw.dropna(subset=["VB"])
    raw_cols = [c for c in raw_feats.columns if c not in ["case", "run"]]
    raw_corr = df_raw_valid[raw_cols + ["VB"]].corr()["VB"].drop("VB").dropna()

    # Compare top correlations
    raw_top = raw_corr.abs().max()
    vmd_top = vmd_corr.abs().max()
    raw_top10_mean = raw_corr.abs().sort_values(ascending=False).head(10).mean()
    vmd_top10_mean = vmd_corr.abs().sort_values(ascending=False).head(10).mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("VMD Features vs Raw Features: Correlation with VB",
                 fontsize=13, fontweight="bold")

    # Bar comparison
    ax = axes[0]
    labels = ["Best Single\nFeature", "Top-10\nAverage"]
    raw_vals = [raw_top, raw_top10_mean]
    vmd_vals = [vmd_top, vmd_top10_mean]
    x = np.arange(len(labels))
    ax.bar(x - 0.15, raw_vals, 0.3, label="Raw Features", color="#92c5de")
    ax.bar(x + 0.15, vmd_vals, 0.3, label="VMD Features", color="#b2182b")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("|Pearson r| with VB")
    ax.set_title("Correlation Magnitude Comparison")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Distribution of correlations
    ax = axes[1]
    ax.hist(raw_corr.abs().values, bins=20, alpha=0.6, label="Raw Features", color="#92c5de")
    ax.hist(vmd_corr.abs().values, bins=20, alpha=0.6, label="VMD Features", color="#b2182b")
    ax.set_xlabel("|Pearson r| with VB")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of |Correlation| Values")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(RESULTS_DIR, "vmd_vs_raw_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    return raw_top, vmd_top, raw_top10_mean, vmd_top10_mean


def plot_mode_energy_heatmap(meta, vmd_feats):
    """Heatmap of mode energy across runs for each sensor."""
    df = vmd_feats.merge(meta[["case", "run", "VB"]], on=["case", "run"])
    df_valid = df.dropna(subset=["VB"]).sort_values("VB")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("VMD Mode Energy Distribution vs Wear Progression",
                 fontsize=13, fontweight="bold")

    for idx, sensor in enumerate(TARGET_SENSORS):
        ax = axes[idx // 3, idx % 3]
        energy_cols = [f"{sensor}_m{k + 1}_energy" for k in range(K)]
        existing = [c for c in energy_cols if c in df_valid.columns]
        if not existing:
            continue
        energy_data = df_valid[existing].values
        # Normalize per row
        row_sums = energy_data.sum(axis=1, keepdims=True) + 1e-12
        energy_pct = energy_data / row_sums * 100

        im = ax.imshow(energy_pct.T, aspect="auto", cmap="YlOrRd",
                       extent=[0, len(df_valid), 0.5, K + 0.5])
        ax.set_xlabel("Runs (sorted by VB)")
        ax.set_ylabel("Mode")
        ax.set_yticks(range(1, K + 1))
        ax.set_title(SENSOR_LABELS[sensor], fontsize=10)
        plt.colorbar(im, ax=ax, label="Energy %")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(RESULTS_DIR, "vmd_mode_energy_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("=" * 60)
    print("  VMD Analysis - NASA Milling Dataset")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────
    print("\n[1/7] Loading data ...")
    meta, signals = load_data()
    print(f"  {len(signals)} runs loaded")

    # ── VMD decomposition examples ─────────────────────────────
    print("\n[2/7] Plotting VMD decomposition examples ...")
    plot_vmd_decomposition_examples(meta, signals)

    # ── Mode spectra comparison ────────────────────────────────
    print("\n[3/7] Plotting mode spectra (low vs high wear) ...")
    plot_mode_spectra(meta, signals)

    # ── Extract VMD features for all runs ──────────────────────
    print("\n[4/7] Extracting VMD features for all runs (this takes a few minutes) ...")
    vmd_feats = extract_all_vmd_features(meta, signals)
    feat_path = os.path.join(RESULTS_DIR, "vmd_features.csv")
    vmd_feats.to_csv(feat_path, index=False)
    print(f"  Saved: {feat_path}  ({vmd_feats.shape[0]} runs × {vmd_feats.shape[1]} cols)")

    # ── VMD correlation analysis ───────────────────────────────
    print("\n[5/7] Plotting VMD feature correlations ...")
    vmd_corr = plot_vmd_correlation(meta, vmd_feats)

    # ── VMD vs Raw comparison ──────────────────────────────────
    print("\n[6/7] Comparing VMD vs raw features ...")
    comparison = plot_vmd_vs_raw_comparison(vmd_corr)

    # ── Mode energy heatmap ────────────────────────────────────
    print("\n[7/7] Plotting mode energy heatmap ...")
    plot_mode_energy_heatmap(meta, vmd_feats)

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  VMD Analysis Summary")
    print("=" * 60)

    top10_vmd = vmd_corr.abs().sort_values(ascending=False).head(10)
    print(f"\n  VMD parameters: K={K}, alpha={ALPHA}, tau={TAU}")
    print(f"  Total VMD features: {len(vmd_corr)}")
    print(f"  Best single VMD feature correlation: |r| = {vmd_corr.abs().max():.4f}")
    print(f"  Top-10 VMD avg correlation: |r| = {top10_vmd.mean():.4f}")

    if comparison:
        raw_best, vmd_best, raw_mean, vmd_mean = comparison
        improvement = ((vmd_best - raw_best) / raw_best) * 100
        print(f"\n  Raw best: |r| = {raw_best:.4f}")
        print(f"  VMD best: |r| = {vmd_best:.4f}")
        print(f"  Improvement: {improvement:+.1f}%")

    print(f"\n  Top 10 VMD features correlated with VB:")
    for feat in top10_vmd.index:
        print(f"    {feat:45s}  r = {vmd_corr[feat]:+.4f}")

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "vmd_summary.txt")
    with open(summary_path, "w") as f:
        f.write("VMD Analysis Summary - NASA Milling Dataset\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"VMD parameters: K={K}, alpha={ALPHA}, tau={TAU}\n")
        f.write(f"Sensors analyzed: {', '.join(TARGET_SENSORS)}\n")
        f.write(f"Total VMD features: {len(vmd_corr)}\n")
        f.write(f"Best |r|: {vmd_corr.abs().max():.4f}\n")
        f.write(f"Top-10 avg |r|: {top10_vmd.mean():.4f}\n\n")
        if comparison:
            f.write(f"Raw best |r|: {comparison[0]:.4f}\n")
            f.write(f"VMD best |r|: {comparison[1]:.4f}\n")
            f.write(f"Improvement: {((comparison[1]-comparison[0])/comparison[0])*100:+.1f}%\n\n")
        f.write("All VMD Feature Correlations with VB (sorted by |r|)\n")
        f.write("-" * 55 + "\n")
        for feat in vmd_corr.abs().sort_values(ascending=False).index:
            f.write(f"  {feat:45s}  r = {vmd_corr[feat]:+.4f}\n")
    print(f"\n  Saved summary: {summary_path}")

    print("\n" + "=" * 60)
    print("  All VMD results saved to: ./results/vmd/")
    print("=" * 60)


if __name__ == "__main__":
    main()
