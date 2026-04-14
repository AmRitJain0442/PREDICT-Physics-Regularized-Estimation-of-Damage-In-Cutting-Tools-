from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from cross_dataset_research import (
    load_nasa_features,
    load_phm_features,
    load_uniwear_window_features,
)


sns.set_theme(style="whitegrid")

OUT_ROOT = Path("plots")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

FEATURE_SUFFIXES = [
    "mean",
    "std",
    "rms",
    "crest",
    "q10",
    "q50",
    "q90",
    "fft_energy",
    "fft_entropy",
    "fft_centroid",
]


def plot_feature_family(df, dataset_name: str, suffix: str, out_dir: Path):
    cols = [c for c in df.columns if c.endswith(f"_{suffix}")]
    if not cols:
        return None

    cols = sorted(cols)
    n = len(cols)
    n_cols = min(3, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    x = df["tool_wear_mm"].to_numpy()
    for i, col in enumerate(cols):
        ax = axes_flat[i]
        y = df[col].to_numpy()
        ax.scatter(x, y, s=12, alpha=0.45)
        if len(np.unique(x)) > 2 and np.std(y) > 1e-12:
            p = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 100)
            ax.plot(xx, p[0] * xx + p[1], "r--", linewidth=1)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("Tool wear (mm)")
        ax.set_ylabel("Feature value")

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"{dataset_name}: {suffix} features vs tool wear", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = out_dir / f"{dataset_name.lower()}_{suffix}.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def main():
    phm_df, _ = load_phm_features(downsample=25)
    nasa_df = load_nasa_features()
    uniwear_df = load_uniwear_window_features(window_size=120, stride=60)

    datasets = {
        "PHM2010": phm_df,
        "NASA": nasa_df,
        "UniWear": uniwear_df,
    }

    saved = []
    for dataset_name, df in datasets.items():
        out_dir = OUT_ROOT / dataset_name.lower()
        out_dir.mkdir(parents=True, exist_ok=True)
        for suffix in FEATURE_SUFFIXES:
            out_path = plot_feature_family(df, dataset_name, suffix, out_dir)
            if out_path is not None:
                saved.append(str(out_path))

    print("Saved plots:")
    for p in saved:
        print(p)


if __name__ == "__main__":
    main()
