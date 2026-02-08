"""
Model Training with Leave-One-Case-Out Cross-Validation (LOCO-CV)
==================================================================
Trains XGBoost, SVR, and Random Forest on selected VMD features.
Validates using LOCO-CV for realistic generalization assessment.

Outputs: ./results/modeling/
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
FEAT_MATRIX_PATH = "./results/feature_selection/selected_feature_matrix.csv"
RESULTS_DIR = "./results/modeling"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(FEAT_MATRIX_PATH)
    feat_cols = [c for c in df.columns if c not in ["case", "run", "VB"]]
    return df, feat_cols


def mape(y_true, y_pred):
    mask = y_true > 0.01  # Avoid division by near-zero
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def get_models():
    return {
        "XGBoost": xgb.XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            reg_lambda=1.0, random_state=42, verbosity=0
        ),
        "SVR": SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.01),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=3,
            random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42
        ),
    }


def loco_cv(df, feat_cols, model_factory, model_name):
    """Leave-One-Case-Out Cross-Validation."""
    cases = sorted(df["case"].unique())
    fold_results = []
    all_preds = []

    for case_id in cases:
        test_mask = df["case"] == case_id
        train_mask = ~test_mask

        X_train = df.loc[train_mask, feat_cols].values
        y_train = df.loc[train_mask, "VB"].values
        X_test = df.loc[test_mask, feat_cols].values
        y_test = df.loc[test_mask, "VB"].values

        if len(y_test) < 2 or len(y_train) < 10:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = model_factory()
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else np.nan
        mape_val = mape(y_test, y_pred)

        fold_results.append({
            "case": case_id, "rmse": rmse, "mae": mae,
            "r2": r2, "mape": mape_val, "n_test": len(y_test)
        })

        for i in range(len(y_test)):
            all_preds.append({
                "case": case_id,
                "run": df.loc[test_mask, "run"].values[i],
                "VB_actual": y_test[i],
                "VB_pred": y_pred[i],
            })

    return pd.DataFrame(fold_results), pd.DataFrame(all_preds)


def train_final_model(df, feat_cols):
    """Train final XGBoost on all data and save."""
    X = df[feat_cols].values
    y = df["VB"].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
        reg_lambda=1.0, random_state=42, verbosity=0
    )
    model.fit(X_s, y)

    joblib.dump(model, os.path.join(RESULTS_DIR, "xgboost_final.joblib"))
    joblib.dump(scaler, os.path.join(RESULTS_DIR, "scaler_final.joblib"))
    joblib.dump(feat_cols, os.path.join(RESULTS_DIR, "feature_cols.joblib"))
    print(f"  Saved final model + scaler to {RESULTS_DIR}")
    return model, scaler


def plot_model_comparison(all_results):
    """Bar chart comparing model performance."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Model Comparison (LOCO-CV)", fontsize=14, fontweight="bold")

    metrics = ["rmse", "mae", "r2", "mape"]
    titles = ["RMSE (mm)", "MAE (mm)", "R-squared", "MAPE (%)"]
    colors = ["#4575b4", "#d73027", "#2ca02c", "#ff7f00"]

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        names = list(all_results.keys())
        means = [all_results[n][metric].mean() for n in names]
        stds = [all_results[n][metric].std() for n in names]
        bars = ax.bar(range(len(names)), means, yerr=stds, capsize=5,
                      color=[colors[i] for i in range(len(names))], alpha=0.8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8, rotation=15)
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{m:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(RESULTS_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_predictions(all_preds_dict):
    """Scatter plot of actual vs predicted for each model."""
    n_models = len(all_preds_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    fig.suptitle("Actual vs Predicted Flank Wear (LOCO-CV)", fontsize=14, fontweight="bold")
    if n_models == 1:
        axes = [axes]

    for idx, (name, preds) in enumerate(all_preds_dict.items()):
        ax = axes[idx]
        for case_id in sorted(preds["case"].unique()):
            sub = preds[preds["case"] == case_id]
            ax.scatter(sub["VB_actual"], sub["VB_pred"], s=20, alpha=0.6,
                       label=f"Case {case_id}")
        lims = [0, max(preds["VB_actual"].max(), preds["VB_pred"].max()) * 1.1]
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Actual VB (mm)")
        ax.set_ylabel("Predicted VB (mm)")
        r2 = r2_score(preds["VB_actual"], preds["VB_pred"])
        rmse = np.sqrt(mean_squared_error(preds["VB_actual"], preds["VB_pred"]))
        ax.set_title(f"{name}\nR²={r2:.3f}, RMSE={rmse:.4f}", fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(RESULTS_DIR, "predictions_scatter.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_per_case_performance(all_results, best_model):
    """Per-case RMSE for the best model."""
    results = all_results[best_model]
    fig, ax = plt.subplots(figsize=(12, 5))
    cases = results["case"].values
    rmses = results["rmse"].values
    colors = ["#d73027" if r > results["rmse"].mean() else "#4575b4" for r in rmses]
    bars = ax.bar(range(len(cases)), rmses, color=colors)
    ax.axhline(results["rmse"].mean(), color="black", linewidth=1, linestyle="--",
               label=f"Mean RMSE = {results['rmse'].mean():.4f}")
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels([f"Case {int(c)}" for c in cases], rotation=45, fontsize=8)
    ax.set_ylabel("RMSE (mm)")
    ax.set_title(f"Per-Case RMSE — {best_model} (LOCO-CV)", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "per_case_rmse.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_wear_trajectories(preds, model_name):
    """Plot predicted vs actual wear trajectory for each case."""
    cases = sorted(preds["case"].unique())
    n_cols = 4
    n_rows = (len(cases) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    fig.suptitle(f"Wear Trajectories — {model_name} (LOCO-CV)",
                 fontsize=14, fontweight="bold")

    for idx, case_id in enumerate(cases):
        ax = axes[idx // n_cols, idx % n_cols] if n_rows > 1 else axes[idx % n_cols]
        sub = preds[preds["case"] == case_id].sort_values("run")
        ax.plot(sub["run"], sub["VB_actual"], "ko-", markersize=4, label="Actual", linewidth=1.5)
        ax.plot(sub["run"], sub["VB_pred"], "rs--", markersize=4, label="Predicted", linewidth=1.5)
        ax.set_title(f"Case {int(case_id)}", fontsize=9)
        ax.set_xlabel("Run")
        ax.set_ylabel("VB (mm)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(cases), n_rows * n_cols):
        ax = axes[idx // n_cols, idx % n_cols] if n_rows > 1 else axes[idx % n_cols]
        ax.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(RESULTS_DIR, "wear_trajectories.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("=" * 60)
    print("  Model Training — LOCO Cross-Validation")
    print("=" * 60)

    print("\n[1/6] Loading selected features ...")
    df, feat_cols = load_data()
    print(f"  Samples: {len(df)}, Features: {len(feat_cols)}")

    models = get_models()
    all_results = {}
    all_preds = {}

    print("\n[2/6] Running LOCO-CV for all models ...")
    for name, model in models.items():
        print(f"\n  --- {name} ---")
        factory = lambda m=model: type(m)(**m.get_params())
        fold_results, preds = loco_cv(df, feat_cols, factory, name)
        all_results[name] = fold_results
        all_preds[name] = preds
        print(f"  RMSE: {fold_results['rmse'].mean():.4f} +/- {fold_results['rmse'].std():.4f}")
        print(f"  MAE:  {fold_results['mae'].mean():.4f} +/- {fold_results['mae'].std():.4f}")
        print(f"  R2:   {fold_results['r2'].mean():.4f} +/- {fold_results['r2'].std():.4f}")

    # Find best model
    best_model = min(all_results, key=lambda n: all_results[n]["rmse"].mean())
    print(f"\n  Best model: {best_model}")

    print("\n[3/6] Plotting model comparison ...")
    plot_model_comparison(all_results)

    print("\n[4/6] Plotting predictions ...")
    plot_predictions(all_preds)

    print("\n[5/6] Plotting per-case performance and trajectories ...")
    plot_per_case_performance(all_results, best_model)
    plot_wear_trajectories(all_preds[best_model], best_model)

    print("\n[6/6] Training and saving final model ...")
    final_model, final_scaler = train_final_model(df, feat_cols)

    # Save detailed results
    for name, res in all_results.items():
        res.to_csv(os.path.join(RESULTS_DIR, f"loco_results_{name.replace(' ', '_')}.csv"), index=False)
    for name, preds in all_preds.items():
        preds.to_csv(os.path.join(RESULTS_DIR, f"predictions_{name.replace(' ', '_')}.csv"), index=False)

    # Summary
    print("\n" + "=" * 60)
    print("  LOCO-CV Results Summary")
    print("=" * 60)
    print(f"\n  {'Model':<20} {'RMSE':>10} {'MAE':>10} {'R2':>10} {'MAPE%':>10}")
    print("  " + "-" * 60)
    for name in all_results:
        r = all_results[name]
        print(f"  {name:<20} {r['rmse'].mean():>10.4f} {r['mae'].mean():>10.4f} "
              f"{r['r2'].mean():>10.4f} {r['mape'].mean():>10.1f}")
    print(f"\n  Best: {best_model} (RMSE = {all_results[best_model]['rmse'].mean():.4f})")

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "modeling_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Model Training Summary — LOCO Cross-Validation\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Samples: {len(df)}, Features: {len(feat_cols)}\n")
        f.write(f"Validation: Leave-One-Case-Out ({len(df['case'].unique())} folds)\n\n")
        f.write(f"{'Model':<20} {'RMSE':>10} {'MAE':>10} {'R2':>10} {'MAPE%':>10}\n")
        f.write("-" * 60 + "\n")
        for name in all_results:
            r = all_results[name]
            f.write(f"{name:<20} {r['rmse'].mean():>10.4f} {r['mae'].mean():>10.4f} "
                    f"{r['r2'].mean():>10.4f} {r['mape'].mean():>10.1f}\n")
        f.write(f"\nBest model: {best_model}\n")
    print(f"  Saved: {summary_path}")
    print(f"\n  All results in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
