"""
Feature Selection Pipeline
===========================
Combines three methods to select robust VMD features:
  1. Filter method (Pearson correlation threshold)
  2. Mutual Information (non-linear dependency)
  3. Recursive Feature Elimination (RFE) with tree model

Only features appearing in >= 2 methods are kept.

Outputs: ./results/feature_selection/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────
CSV_DIR = "./data/nasa_milling/csv"
VMD_FEAT_PATH = "./results/vmd/vmd_features.csv"
RESULTS_DIR = "./results/feature_selection"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data():
    meta = pd.read_csv(os.path.join(CSV_DIR, "metadata.csv"))
    vmd = pd.read_csv(VMD_FEAT_PATH)
    df = vmd.merge(meta[["case", "run", "VB"]], on=["case", "run"])
    df = df.dropna(subset=["VB"])
    feat_cols = [c for c in vmd.columns if c not in ["case", "run"]]
    # Drop columns with NaN or zero variance
    df_feats = df[feat_cols].copy()
    df_feats = df_feats.loc[:, df_feats.std() > 1e-10]
    df_feats = df_feats.dropna(axis=1)
    feat_cols = df_feats.columns.tolist()
    X = df_feats.values
    y = df["VB"].values
    return df, feat_cols, X, y


def filter_method(X, y, feat_cols, threshold=0.3):
    """Select features with |Pearson r| > threshold."""
    corrs = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
    mask = np.abs(corrs) > threshold
    selected = [feat_cols[i] for i in range(len(feat_cols)) if mask[i]]
    print(f"  Filter (|r| > {threshold}): {len(selected)} features")
    return set(selected), corrs


def mutual_info_method(X, y, feat_cols, top_n=40):
    """Select top-N features by mutual information."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    mi = mutual_info_regression(X_scaled, y, random_state=42, n_neighbors=5)
    top_idx = np.argsort(mi)[::-1][:top_n]
    selected = [feat_cols[i] for i in top_idx]
    print(f"  Mutual Information (top {top_n}): {len(selected)} features")
    return set(selected), mi


def rfe_method(X, y, feat_cols, n_select=30):
    """RFE with gradient boosting."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    estimator = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, random_state=42, subsample=0.8
    )
    rfe = RFE(estimator, n_features_to_select=n_select, step=10)
    rfe.fit(X_scaled, y)
    selected = [feat_cols[i] for i in range(len(feat_cols)) if rfe.support_[i]]
    print(f"  RFE (top {n_select}): {len(selected)} features")
    return set(selected), rfe.ranking_


def plot_selection_venn(filter_set, mi_set, rfe_set, final_set):
    """Visualize overlap between methods."""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = {"Filter": filter_set, "MI": mi_set, "RFE": rfe_set}
    all_feats = filter_set | mi_set | rfe_set

    # Count method membership for each feature
    membership = {}
    for f in all_feats:
        count = sum(f in s for s in [filter_set, mi_set, rfe_set])
        membership[f] = count

    counts = [0, 0, 0, 0]  # 0 methods, 1, 2, 3
    for f, c in membership.items():
        counts[c] += 1

    bars = ax.bar(["1 method\n(discarded)", "2 methods\n(selected)", "3 methods\n(selected)"],
                  counts[1:], color=["#fee090", "#4575b4", "#d73027"])
    ax.set_ylabel("Number of Features")
    ax.set_title("Feature Selection Consensus\n(Features kept if selected by >= 2 methods)",
                 fontsize=12, fontweight="bold")
    for bar, val in zip(bars, counts[1:]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    path = os.path.join(RESULTS_DIR, "selection_consensus.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_selected_features(feat_cols, corrs, mi_scores, final_features):
    """Bar chart of selected features ranked by |correlation|."""
    # Get corr and MI for final features
    feat_data = []
    for f in final_features:
        idx = feat_cols.index(f)
        feat_data.append({"feature": f, "corr": corrs[idx], "mi": mi_scores[idx]})
    fdf = pd.DataFrame(feat_data).sort_values("corr", key=abs, ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(8, len(fdf) * 0.35)))
    fig.suptitle(f"Selected Features ({len(fdf)} total)", fontsize=13, fontweight="bold")

    # Correlation
    ax = axes[0]
    colors = ["#d73027" if v < 0 else "#4575b4" for v in fdf["corr"].values]
    ax.barh(range(len(fdf)), fdf["corr"].values, color=colors)
    ax.set_yticks(range(len(fdf)))
    ax.set_yticklabels(fdf["feature"].values, fontsize=7)
    ax.set_xlabel("Pearson r with VB")
    ax.set_title("Correlation")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(True, axis="x", alpha=0.3)

    # Mutual Information
    ax = axes[1]
    ax.barh(range(len(fdf)), fdf["mi"].values, color="#2ca02c")
    ax.set_yticks(range(len(fdf)))
    ax.set_yticklabels(fdf["feature"].values, fontsize=7)
    ax.set_xlabel("Mutual Information")
    ax.set_title("Mutual Information")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(RESULTS_DIR, "selected_features.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("=" * 60)
    print("  Feature Selection Pipeline")
    print("=" * 60)

    print("\n[1/5] Loading data ...")
    df, feat_cols, X, y = load_data()
    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")

    print("\n[2/5] Running filter method ...")
    filter_set, corrs = filter_method(X, y, feat_cols, threshold=0.3)

    print("\n[3/5] Running mutual information ...")
    mi_set, mi_scores = mutual_info_method(X, y, feat_cols, top_n=40)

    print("\n[4/5] Running RFE ...")
    rfe_set, rfe_ranking = rfe_method(X, y, feat_cols, n_select=30)

    # Consensus: keep features in >= 2 methods
    all_feats = filter_set | mi_set | rfe_set
    final_features = []
    for f in all_feats:
        count = sum(f in s for s in [filter_set, mi_set, rfe_set])
        if count >= 2:
            final_features.append(f)

    # Sort by |correlation|
    final_features = sorted(final_features,
                            key=lambda f: abs(corrs[feat_cols.index(f)]), reverse=True)

    print(f"\n  Consensus features (>= 2 methods): {len(final_features)}")

    print("\n[5/5] Plotting results ...")
    plot_selection_venn(filter_set, mi_set, rfe_set, set(final_features))
    plot_selected_features(feat_cols, corrs, mi_scores, final_features)

    # Save selected features
    feat_list_path = os.path.join(RESULTS_DIR, "selected_features.txt")
    with open(feat_list_path, "w") as f:
        for feat in final_features:
            idx = feat_cols.index(feat)
            f.write(f"{feat},{corrs[idx]:.4f},{mi_scores[idx]:.4f}\n")
    print(f"  Saved: {feat_list_path}")

    # Save selected feature matrix
    X_sel = df[final_features + ["case", "run", "VB"]].copy()
    matrix_path = os.path.join(RESULTS_DIR, "selected_feature_matrix.csv")
    X_sel.to_csv(matrix_path, index=False)
    print(f"  Saved: {matrix_path}")

    # Summary
    print("\n" + "=" * 60)
    print("  Feature Selection Summary")
    print("=" * 60)
    print(f"  Initial features: {len(feat_cols)}")
    print(f"  Filter method: {len(filter_set)}")
    print(f"  Mutual Information: {len(mi_set)}")
    print(f"  RFE: {len(rfe_set)}")
    print(f"  Final consensus: {len(final_features)}")
    print(f"\n  Selected features:")
    for f in final_features:
        idx = feat_cols.index(f)
        print(f"    {f:45s}  r={corrs[idx]:+.4f}  MI={mi_scores[idx]:.4f}")

    print(f"\n  Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
