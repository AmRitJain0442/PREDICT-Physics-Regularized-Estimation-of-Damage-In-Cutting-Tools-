"""
Model Interpretation with SHAP + Ablation Study
=================================================
SHAP analysis on the best model to explain predictions.
Ablation: raw features only vs VMD features only vs combined.

Outputs: ./results/interpretation/
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import shap
import joblib

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
CSV_DIR = "./data/nasa_milling/csv"
VMD_FEAT_PATH = "./results/vmd/vmd_features.csv"
RAW_FEAT_PATH = "./results/eda/signal_features.csv"
SELECTED_FEAT_PATH = "./results/feature_selection/selected_feature_matrix.csv"
MODEL_PATH = "./results/modeling/xgboost_final.joblib"
SCALER_PATH = "./results/modeling/scaler_final.joblib"
FEAT_COLS_PATH = "./results/modeling/feature_cols.joblib"
RESULTS_DIR = "./results/interpretation"
os.makedirs(RESULTS_DIR, exist_ok=True)


def loco_cv_quick(df, feat_cols):
    """Quick LOCO-CV returning overall RMSE, MAE, R2."""
    cases = sorted(df["case"].unique())
    y_all, p_all = [], []
    for case_id in cases:
        test_mask = df["case"] == case_id
        train_mask = ~test_mask
        X_tr = df.loc[train_mask, feat_cols].values
        y_tr = df.loc[train_mask, "VB"].values
        X_te = df.loc[test_mask, feat_cols].values
        y_te = df.loc[test_mask, "VB"].values
        if len(y_te) < 2 or len(y_tr) < 10:
            continue
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            reg_lambda=1.0, random_state=42, verbosity=0
        )
        model.fit(X_tr_s, y_tr)
        y_all.extend(y_te)
        p_all.extend(model.predict(X_te_s))
    y_all, p_all = np.array(y_all), np.array(p_all)
    return {
        "rmse": np.sqrt(mean_squared_error(y_all, p_all)),
        "mae": mean_absolute_error(y_all, p_all),
        "r2": r2_score(y_all, p_all),
    }


def shap_analysis():
    """SHAP analysis on the final XGBoost model."""
    print("\n  Running SHAP ...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feat_cols = joblib.load(FEAT_COLS_PATH)

    df = pd.read_csv(SELECTED_FEAT_PATH)
    df = df.dropna(subset=["VB"])
    X = df[feat_cols].values
    X_s = scaler.transform(X)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_s)

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(feat_cols) * 0.4)))
    shap.summary_plot(shap_values, X_s, feature_names=feat_cols, show=False)
    plt.title("SHAP Feature Importance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "shap_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # Bar plot (mean |SHAP|)
    fig, ax = plt.subplots(figsize=(10, max(6, len(feat_cols) * 0.4)))
    shap.summary_plot(shap_values, X_s, feature_names=feat_cols,
                      plot_type="bar", show=False)
    plt.title("Mean |SHAP| Values", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "shap_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # Dependence plots for top 4 features
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top4_idx = np.argsort(mean_abs_shap)[::-1][:4]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SHAP Dependence Plots (Top 4 Features)", fontsize=13, fontweight="bold")
    for i, idx in enumerate(top4_idx):
        ax = axes[i // 2, i % 2]
        shap.dependence_plot(idx, shap_values, X_s, feature_names=feat_cols,
                             ax=ax, show=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(RESULTS_DIR, "shap_dependence.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    # Save SHAP importance ranking
    shap_importance = pd.DataFrame({
        "feature": feat_cols,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)
    shap_importance.to_csv(os.path.join(RESULTS_DIR, "shap_importance.csv"), index=False)

    return shap_importance


def ablation_study():
    """Compare raw-only vs VMD-only vs combined features."""
    print("\n  Running ablation study ...")
    meta = pd.read_csv(os.path.join(CSV_DIR, "metadata.csv"))

    # Load raw features
    raw = pd.read_csv(RAW_FEAT_PATH)
    raw_cols = [c for c in raw.columns if c not in ["case", "run"]]
    df_raw = raw.merge(meta[["case", "run", "VB"]], on=["case", "run"]).dropna(subset=["VB"])

    # Load VMD features
    vmd = pd.read_csv(VMD_FEAT_PATH)
    vmd_cols = [c for c in vmd.columns if c not in ["case", "run"]]
    df_vmd = vmd.merge(meta[["case", "run", "VB"]], on=["case", "run"]).dropna(subset=["VB"])

    # Combined
    df_combined = raw.merge(vmd, on=["case", "run"])
    df_combined = df_combined.merge(meta[["case", "run", "VB"]], on=["case", "run"]).dropna(subset=["VB"])
    combined_cols = raw_cols + vmd_cols

    # Selected VMD (our final)
    df_sel = pd.read_csv(SELECTED_FEAT_PATH).dropna(subset=["VB"])
    sel_cols = [c for c in df_sel.columns if c not in ["case", "run", "VB"]]

    configs = {
        "Raw Features Only": (df_raw, raw_cols),
        "All VMD Features": (df_vmd, vmd_cols),
        "Raw + VMD Combined": (df_combined, combined_cols),
        "Selected VMD (Ours)": (df_sel, sel_cols),
    }

    results = {}
    for name, (df, cols) in configs.items():
        # Drop NaN/zero-var columns
        valid_cols = [c for c in cols if c in df.columns and df[c].std() > 1e-10 and df[c].notna().all()]
        r = loco_cv_quick(df, valid_cols)
        r["n_features"] = len(valid_cols)
        results[name] = r
        print(f"    {name:25s} — RMSE={r['rmse']:.4f}, R2={r['r2']:.4f} ({len(valid_cols)} features)")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Ablation Study: Feature Set Comparison (LOCO-CV)",
                 fontsize=13, fontweight="bold")
    names = list(results.keys())
    colors = ["#92c5de", "#4575b4", "#d73027", "#2ca02c"]

    for idx, (metric, title) in enumerate([("rmse", "RMSE (mm)"), ("r2", "R-squared"), ("mae", "MAE (mm)")]):
        ax = axes[idx]
        vals = [results[n][metric] for n in names]
        bars = ax.bar(range(len(names)), vals, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8, rotation=20, ha="right")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(RESULTS_DIR, "ablation_study.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    return results


def main():
    print("=" * 60)
    print("  Model Interpretation + Ablation")
    print("=" * 60)

    # SHAP
    print("\n[1/2] SHAP Analysis ...")
    shap_imp = shap_analysis()

    # Ablation
    print("\n[2/2] Ablation Study ...")
    ablation_results = ablation_study()

    # Summary
    print("\n" + "=" * 60)
    print("  Interpretation Summary")
    print("=" * 60)

    print("\n  SHAP Top 5 Features:")
    for _, row in shap_imp.head(5).iterrows():
        print(f"    {row['feature']:45s}  |SHAP| = {row['mean_abs_shap']:.4f}")

    print("\n  Ablation Study:")
    print(f"  {'Config':<25} {'Features':>8} {'RMSE':>8} {'R2':>8}")
    print("  " + "-" * 55)
    for name, r in ablation_results.items():
        print(f"  {name:<25} {r['n_features']:>8d} {r['rmse']:>8.4f} {r['r2']:>8.4f}")

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "interpretation_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Model Interpretation Summary\n")
        f.write("=" * 55 + "\n\n")
        f.write("SHAP Feature Importance (Top 10):\n")
        for _, row in shap_imp.head(10).iterrows():
            f.write(f"  {row['feature']:45s}  |SHAP| = {row['mean_abs_shap']:.4f}\n")
        f.write(f"\nAblation Study (LOCO-CV):\n")
        f.write(f"{'Config':<25} {'Features':>8} {'RMSE':>8} {'R2':>8}\n")
        f.write("-" * 55 + "\n")
        for name, r in ablation_results.items():
            f.write(f"{name:<25} {r['n_features']:>8d} {r['rmse']:>8.4f} {r['r2']:>8.4f}\n")
    print(f"\n  Saved: {summary_path}")
    print(f"  All results in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
