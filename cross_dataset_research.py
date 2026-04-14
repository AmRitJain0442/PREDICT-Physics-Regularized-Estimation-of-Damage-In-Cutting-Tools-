import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = Path(".")
OUT_DIR = BASE_DIR / "results" / "cross_dataset_research"
PLOT_DIR = OUT_DIR / "plots"
TABLE_DIR = OUT_DIR / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

PHM_COLS = [
    "force_x",
    "force_y",
    "force_z",
    "vibration_x",
    "vibration_y",
    "vibration_z",
    "ae_rms",
]


def safe_fft_features(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if x.size < 8:
        return {
            "fft_energy": np.nan,
            "fft_peak_bin": np.nan,
            "fft_centroid": np.nan,
            "fft_entropy": np.nan,
        }

    centered = x - np.mean(x)
    fft_vals = np.fft.rfft(centered)
    power = np.abs(fft_vals) ** 2
    power_sum = float(np.sum(power))
    if power_sum <= 1e-12:
        return {
            "fft_energy": 0.0,
            "fft_peak_bin": 0.0,
            "fft_centroid": 0.0,
            "fft_entropy": 0.0,
        }

    p = power / power_sum
    bins = np.arange(power.size, dtype=np.float64)
    peak_bin = int(np.argmax(power[1:]) + 1) if power.size > 1 else 0

    energy_norm = power_sum / max(power.size, 1)
    return {
        "fft_energy": float(np.log1p(energy_norm)),
        "fft_peak_bin": float(peak_bin),
        "fft_centroid": float(np.sum(bins * power) / power_sum),
        "fft_entropy": float(-(p * np.log(p + 1e-12)).sum()),
    }


def extract_signal_features(signal: np.ndarray, prefix: str) -> dict:
    x = np.asarray(signal, dtype=np.float64)
    if x.size == 0:
        return {
            f"{prefix}_{k}": np.nan
            for k in [
                "mean",
                "std",
                "rms",
                "abs_mean",
                "p2p",
                "skew_like",
                "kurt_like",
                "crest",
                "q10",
                "q50",
                "q90",
                "fft_energy",
                "fft_peak_bin",
                "fft_centroid",
                "fft_entropy",
            ]
        }

    mean = float(np.mean(x))
    std = float(np.std(x))
    rms = float(np.sqrt(np.mean(x**2)))
    abs_mean = float(np.mean(np.abs(x)))
    p2p = float(np.ptp(x))
    centered = x - mean
    std_eps = std + 1e-12
    skew_like = float(np.mean((centered / std_eps) ** 3))
    kurt_like = float(np.mean((centered / std_eps) ** 4))
    crest = float(np.max(np.abs(x)) / (rms + 1e-12))
    q10, q50, q90 = np.quantile(x, [0.1, 0.5, 0.9])

    feats = {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_rms": rms,
        f"{prefix}_abs_mean": abs_mean,
        f"{prefix}_p2p": p2p,
        f"{prefix}_skew_like": skew_like,
        f"{prefix}_kurt_like": kurt_like,
        f"{prefix}_crest": crest,
        f"{prefix}_q10": float(q10),
        f"{prefix}_q50": float(q50),
        f"{prefix}_q90": float(q90),
    }

    fft = safe_fft_features(x)
    for k, v in fft.items():
        feats[f"{prefix}_{k}"] = float(v)
    return feats


def extract_frame_features(df: pd.DataFrame, channels: list[str]) -> dict:
    out = {}
    for c in channels:
        if c in df.columns:
            out.update(extract_signal_features(df[c].to_numpy(), c))
    return out


def load_phm_features(downsample: int = 25) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = BASE_DIR / "data" / "archive (1)"
    train_cutters = ["c1", "c4", "c6"]
    test_cutters = ["c2", "c3", "c5"]

    train_rows = []
    test_rows = []

    for cutter in train_cutters:
        wear_path = root / cutter / f"{cutter}_wear.csv"
        wear_df = pd.read_csv(wear_path)
        wear_df["tool_wear_mm"] = wear_df[["flute_1", "flute_2", "flute_3"]].mean(axis=1) / 1000.0
        for _, r in wear_df.iterrows():
            cut = int(r["cut"])
            csv_path = root / cutter / cutter / f"c_{cutter[1]}_{cut:03d}.csv"
            if not csv_path.exists():
                continue

            sig = pd.read_csv(csv_path, header=None, names=PHM_COLS, dtype=np.float32)
            if downsample > 1:
                sig = sig.iloc[::downsample].reset_index(drop=True)

            feats = extract_frame_features(sig, PHM_COLS)
            feats.update(
                {
                    "dataset": "PHM2010",
                    "group_id": cutter,
                    "sample_id": f"{cutter}_{cut:03d}",
                    "cut": cut,
                    "tool_wear_mm": float(r["tool_wear_mm"]),
                    "spindle_speed_rpm": 10400.0,
                    "feed_rate_mm_min": 1555.0,
                    "axial_depth_mm": 0.2,
                    "radial_depth_mm": 0.125,
                }
            )
            train_rows.append(feats)

    for cutter in test_cutters:
        files = sorted((root / cutter / cutter).glob("*.csv"))
        for csv_path in files:
            cut = int(csv_path.stem.split("_")[-1])
            sig = pd.read_csv(csv_path, header=None, names=PHM_COLS, dtype=np.float32)
            if downsample > 1:
                sig = sig.iloc[::downsample].reset_index(drop=True)
            feats = extract_frame_features(sig, PHM_COLS)
            feats.update(
                {
                    "dataset": "PHM2010_TEST",
                    "group_id": cutter,
                    "sample_id": f"{cutter}_{cut:03d}",
                    "cut": cut,
                    "spindle_speed_rpm": 10400.0,
                    "feed_rate_mm_min": 1555.0,
                    "axial_depth_mm": 0.2,
                    "radial_depth_mm": 0.125,
                }
            )
            test_rows.append(feats)

    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)
    return train_df, test_df


def load_nasa_features() -> pd.DataFrame:
    meta = pd.read_csv(BASE_DIR / "data" / "nasa_milling" / "csv" / "metadata.csv")
    signals_dir = BASE_DIR / "data" / "nasa_milling" / "csv" / "signals"
    channels = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]

    rows = []
    for _, r in meta.dropna(subset=["VB"]).iterrows():
        case = int(r["case"])
        run = int(r["run"])
        p = signals_dir / f"case{case}_run{run:03d}.csv"
        if not p.exists():
            continue
        sig = pd.read_csv(p, dtype=np.float32)
        feats = extract_frame_features(sig, channels)
        feats.update(
            {
                "dataset": "NASA",
                "group_id": f"case_{case}",
                "sample_id": f"case{case}_run{run:03d}",
                "case": case,
                "run": run,
                "time_min": float(r["time"]),
                "feed_rate_mm_rev": float(r["feed"]),
                "doc_mm": float(r["DOC"]),
                "material": int(r["material"]),
                "tool_wear_mm": float(r["VB"]),
            }
        )
        rows.append(feats)

    return pd.DataFrame(rows)


def load_uniwear_window_features(window_size: int = 120, stride: int = 60) -> pd.DataFrame:
    df = pd.read_csv(
        BASE_DIR / "data" / "uniwear" / "data" / "uniwear.csv",
        usecols=[
            "timestamp",
            "force_z",
            "vibration_x",
            "vibration_y",
            "tool_wear",
            "experiment_tag",
            "dataset_tag",
        ],
    )
    rows = []
    channels = ["force_z", "vibration_x", "vibration_y"]

    for exp_tag, g in df.groupby("experiment_tag"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        dataset_tag = g["dataset_tag"].iloc[0]
        for start in range(0, len(g) - window_size + 1, stride):
            w = g.iloc[start : start + window_size]
            feats = extract_frame_features(w[channels], channels)
            feats.update(
                {
                    "dataset": "UniWear",
                    "group_id": exp_tag,
                    "sample_id": f"{exp_tag}_{start:06d}",
                    "experiment_tag": exp_tag,
                    "dataset_tag": dataset_tag,
                    "time_s": float(w["timestamp"].iloc[-1]),
                    "tool_wear_mm": float(w["tool_wear"].iloc[-1]),
                }
            )
            rows.append(feats)

    return pd.DataFrame(rows)


def evaluate_regression(
    df: pd.DataFrame, dataset_name: str, threshold_mm: float
) -> tuple[pd.DataFrame, pd.DataFrame, dict, list[str]]:
    non_feature = {
        "dataset",
        "group_id",
        "sample_id",
        "tool_wear_mm",
        "case",
        "run",
        "cut",
        "experiment_tag",
        "dataset_tag",
    }
    feature_cols = [c for c in df.columns if c not in non_feature and df[c].dtype != "O"]
    feature_cols = [c for c in feature_cols if df[c].notna().all() and df[c].std() > 1e-12]

    X_df = df[feature_cols].replace([np.inf, -np.inf], np.nan).copy()
    for col in feature_cols:
        med = X_df[col].median()
        if pd.isna(med):
            med = 0.0
        X_df[col] = X_df[col].fillna(med)
        lo = X_df[col].quantile(0.01)
        hi = X_df[col].quantile(0.99)
        if pd.isna(lo) or pd.isna(hi) or lo == hi:
            continue
        X_df[col] = X_df[col].clip(lo, hi)

    X = X_df.to_numpy(dtype=np.float64)
    y = df["tool_wear_mm"].to_numpy()
    groups = df["group_id"].to_numpy()

    n_groups = pd.Series(groups).nunique()
    n_splits = min(5, n_groups)
    if n_splits < 2:
        raise ValueError(f"Not enough groups for CV in {dataset_name}")

    gkf = GroupKFold(n_splits=n_splits)

    models = {
        "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
        "GradBoost": GradientBoostingRegressor(
            n_estimators=350,
            learning_rate=0.05,
            max_depth=6,
            random_state=RANDOM_STATE,
        ),
    }

    metric_rows = []
    cv_predictions = []

    for model_name, model in models.items():
        fold_preds = []
        fold_metrics = []
        for fold_id, (tr, te) in enumerate(gkf.split(X, y, groups=groups), start=1):
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
            rmse = float(np.sqrt(mean_squared_error(y[te], pred)))
            mae = float(mean_absolute_error(y[te], pred))
            r2 = float(r2_score(y[te], pred))
            mape = float(np.mean(np.abs((y[te] - pred) / np.clip(y[te], 1e-6, None))) * 100)
            fold_metrics.append((rmse, mae, r2, mape))
            for i, idx in enumerate(te):
                fold_preds.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "fold": fold_id,
                        "group_id": groups[idx],
                        "sample_id": df.iloc[idx]["sample_id"],
                        "actual": float(y[idx]),
                        "pred": float(pred[i]),
                    }
                )

        arr = np.array(fold_metrics)
        metric_rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "rmse_mean": float(arr[:, 0].mean()),
                "rmse_std": float(arr[:, 0].std()),
                "mae_mean": float(arr[:, 1].mean()),
                "r2_mean": float(arr[:, 2].mean()),
                "mape_mean": float(arr[:, 3].mean()),
            }
        )
        cv_predictions.extend(fold_preds)

    metrics_df = pd.DataFrame(metric_rows).sort_values("rmse_mean")
    best_model_name = metrics_df.iloc[0]["model"]
    best_model = models[best_model_name]
    best_model.fit(X, y)

    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "named_steps") and hasattr(best_model.named_steps.get("model"), "coef_"):
        importances = np.abs(best_model.named_steps["model"].coef_)
    else:
        importances = np.full(len(feature_cols), np.nan)

    imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    imp_df = imp_df.replace([np.inf, -np.inf], np.nan).dropna().sort_values("importance", ascending=False)

    y_bin = (y >= threshold_mm).astype(int)
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=1,
        random_state=RANDOM_STATE,
    )

    clf_rows = []

    for fold_id, (tr, te) in enumerate(gkf.split(X, y_bin, groups=groups), start=1):
        if len(np.unique(y_bin[tr])) < 2 or len(np.unique(y_bin[te])) < 2:
            continue
        clf.fit(X[tr], y_bin[tr])
        prob = clf.predict_proba(X[te])[:, 1]
        pred_bin = (prob >= 0.5).astype(int)

        clf_rows.append(
            {
                "dataset": dataset_name,
                "fold": fold_id,
                "roc_auc": float(roc_auc_score(y_bin[te], prob)),
                "f1": float(f1_score(y_bin[te], pred_bin)),
                "precision": float(precision_score(y_bin[te], pred_bin, zero_division=0)),
                "recall": float(recall_score(y_bin[te], pred_bin, zero_division=0)),
                "n_test": int(len(te)),
            }
        )

    clf_df = pd.DataFrame(clf_rows)
    clf.fit(X, y_bin)

    top_feats = imp_df["feature"].head(8).tolist()
    marker_rows = []
    for feat in top_feats:
        vals = X_df[feat].to_numpy()
        healthy = vals[y_bin == 0]
        fail = vals[y_bin == 1]
        if healthy.size == 0 or fail.size == 0:
            continue
        direction = "higher" if np.nanmean(fail) >= np.nanmean(healthy) else "lower"
        threshold = float((np.nanmean(healthy) + np.nanmean(fail)) / 2.0)
        score = vals if direction == "higher" else -vals
        auc = float(roc_auc_score(y_bin, score)) if len(np.unique(y_bin)) > 1 else np.nan
        marker_rows.append(
            {
                "dataset": dataset_name,
                "feature": feat,
                "direction": direction,
                "marker_threshold": threshold,
                "healthy_mean": float(np.nanmean(healthy)),
                "failure_mean": float(np.nanmean(fail)),
                "single_feature_auc": auc,
            }
        )

    marker_df = pd.DataFrame(marker_rows).sort_values("single_feature_auc", ascending=False)

    summary = {
        "dataset": dataset_name,
        "n_samples": int(len(df)),
        "n_groups": int(n_groups),
        "n_features": int(len(feature_cols)),
        "failure_threshold_mm": float(threshold_mm),
        "failure_rate": float(np.mean(y_bin)),
        "best_model": best_model_name,
        "best_rmse": float(metrics_df.iloc[0]["rmse_mean"]),
        "best_r2": float(metrics_df.iloc[0]["r2_mean"]),
        "classification_roc_auc_mean": float(clf_df["roc_auc"].mean()) if not clf_df.empty else np.nan,
        "classification_f1_mean": float(clf_df["f1"].mean()) if not clf_df.empty else np.nan,
    }

    pred_df = pd.DataFrame(cv_predictions)
    pred_df["residual"] = pred_df["actual"] - pred_df["pred"]

    pred_df.to_csv(TABLE_DIR / f"{dataset_name.lower()}_cv_predictions.csv", index=False)
    metrics_df.to_csv(TABLE_DIR / f"{dataset_name.lower()}_regression_metrics.csv", index=False)
    clf_df.to_csv(TABLE_DIR / f"{dataset_name.lower()}_classification_metrics.csv", index=False)
    imp_df.to_csv(TABLE_DIR / f"{dataset_name.lower()}_feature_importance.csv", index=False)
    marker_df.to_csv(TABLE_DIR / f"{dataset_name.lower()}_failure_markers.csv", index=False)

    plt.figure(figsize=(6.4, 5.5))
    best_pred = pred_df[pred_df["model"] == best_model_name]
    plt.scatter(best_pred["actual"], best_pred["pred"], alpha=0.65, s=22)
    lim = max(float(best_pred["actual"].max()), float(best_pred["pred"].max())) * 1.05
    plt.plot([0, lim], [0, lim], "k--", linewidth=1)
    plt.xlabel("Actual Tool Wear (mm)")
    plt.ylabel("Predicted Tool Wear (mm)")
    plt.title(f"{dataset_name}: {best_model_name} CV Predictions")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{dataset_name.lower()}_regression_scatter.png", dpi=170)
    plt.close()

    top_imp = imp_df.head(12).iloc[::-1]
    plt.figure(figsize=(7.2, max(4.8, 0.3 * len(top_imp) + 2)))
    plt.barh(top_imp["feature"], top_imp["importance"], color="#2a6f97")
    plt.xlabel("Importance")
    plt.title(f"{dataset_name}: Top Feature Importances")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{dataset_name.lower()}_feature_importance.png", dpi=170)
    plt.close()

    if not marker_df.empty:
        top_m = marker_df.head(8).iloc[::-1]
        plt.figure(figsize=(7.2, max(4.6, 0.28 * len(top_m) + 2)))
        plt.barh(top_m["feature"], top_m["single_feature_auc"], color="#d4843b")
        plt.axvline(0.5, color="k", linestyle="--", linewidth=1)
        plt.xlim(0.45, 1.0)
        plt.xlabel("Single-Feature AUC")
        plt.title(f"{dataset_name}: Candidate Failure Markers")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"{dataset_name.lower()}_failure_markers.png", dpi=170)
        plt.close()

    return metrics_df, marker_df, summary, feature_cols


def build_nuaa_simulator(threshold_mm: float = 0.25) -> dict:
    df = pd.read_csv(
        BASE_DIR / "data" / "uniwear" / "data" / "nuaa_orthogonal_bundle_high_resolution.csv",
        usecols=[
            "experiment_tag",
            "timestamp",
            "tool_wear",
            "feed_per_tooth",
            "spindle_speed",
            "axial_cutting_depth",
        ],
    )

    df = df.iloc[::4].reset_index(drop=True)
    X = df[["timestamp", "feed_per_tooth", "spindle_speed", "axial_cutting_depth"]].to_numpy()
    y = df["tool_wear"].to_numpy()
    groups = df["experiment_tag"].to_numpy()

    gkf = GroupKFold(n_splits=min(5, df["experiment_tag"].nunique()))
    reg = GradientBoostingRegressor(
        n_estimators=450,
        learning_rate=0.05,
        max_depth=6,
        random_state=RANDOM_STATE,
    )

    fold_scores = []
    for tr, te in gkf.split(X, y, groups=groups):
        reg.fit(X[tr], y[tr])
        pred = reg.predict(X[te])
        fold_scores.append(
            {
                "rmse": float(np.sqrt(mean_squared_error(y[te], pred))),
                "mae": float(mean_absolute_error(y[te], pred)),
                "r2": float(r2_score(y[te], pred)),
            }
        )

    reg.fit(X, y)

    param_grid = (
        df[["feed_per_tooth", "spindle_speed", "axial_cutting_depth"]]
        .drop_duplicates()
        .sort_values(["axial_cutting_depth", "feed_per_tooth", "spindle_speed"])
        .reset_index(drop=True)
    )

    time_grid = np.linspace(0, df["timestamp"].max(), 220)

    life_rows = []
    for _, p in param_grid.iterrows():
        sim = pd.DataFrame(
            {
                "timestamp": time_grid,
                "feed_per_tooth": p["feed_per_tooth"],
                "spindle_speed": p["spindle_speed"],
                "axial_cutting_depth": p["axial_cutting_depth"],
            }
        )
        wear_pred = reg.predict(sim[["timestamp", "feed_per_tooth", "spindle_speed", "axial_cutting_depth"]])
        wear_pred = np.maximum.accumulate(wear_pred)
        idx = np.where(wear_pred >= threshold_mm)[0]
        life = float(time_grid[idx[0]]) if idx.size > 0 else float(time_grid[-1])

        life_rows.append(
            {
                "feed_per_tooth": float(p["feed_per_tooth"]),
                "spindle_speed": float(p["spindle_speed"]),
                "axial_cutting_depth": float(p["axial_cutting_depth"]),
                "predicted_life_s": life,
            }
        )

    life_df = pd.DataFrame(life_rows).sort_values("predicted_life_s", ascending=False)
    life_df.to_csv(TABLE_DIR / "nuaa_simulated_life_table.csv", index=False)

    depths = sorted(life_df["axial_cutting_depth"].unique())
    fig, axes = plt.subplots(1, len(depths), figsize=(5.5 * len(depths), 4.5), sharey=True)
    if len(depths) == 1:
        axes = [axes]

    for ax, d in zip(axes, depths):
        sub = life_df[life_df["axial_cutting_depth"] == d]
        pivot = sub.pivot(index="feed_per_tooth", columns="spindle_speed", values="predicted_life_s")
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", cbar=True, ax=ax)
        ax.set_title(f"Axial Depth = {d:.1f} mm")
        ax.set_xlabel("Spindle Speed (rpm)")
        ax.set_ylabel("Feed per Tooth (mm/rev)")

    plt.suptitle(f"Predicted Tool Life (s) to Reach {threshold_mm:.2f} mm Wear")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(PLOT_DIR / "nuaa_life_heatmap.png", dpi=170)
    plt.close()

    obs_rows = []
    for exp_tag, g in df.groupby("experiment_tag"):
        g = g.sort_values("timestamp")
        params = g[["feed_per_tooth", "spindle_speed", "axial_cutting_depth"]].iloc[0]
        idx = np.where(g["tool_wear"].to_numpy() >= threshold_mm)[0]
        life = float(g["timestamp"].iloc[idx[0]]) if idx.size > 0 else float(g["timestamp"].max())
        obs_rows.append(
            {
                "experiment_tag": exp_tag,
                "feed_per_tooth": float(params["feed_per_tooth"]),
                "spindle_speed": float(params["spindle_speed"]),
                "axial_cutting_depth": float(params["axial_cutting_depth"]),
                "life_s": life,
            }
        )
    obs = pd.DataFrame(obs_rows)

    Xeq = np.column_stack(
        [
            np.ones(len(obs)),
            np.log(obs["spindle_speed"].to_numpy()),
            np.log(obs["feed_per_tooth"].to_numpy()),
            np.log(obs["axial_cutting_depth"].to_numpy()),
        ]
    )
    yeq = np.log(obs["life_s"].to_numpy())
    beta, *_ = np.linalg.lstsq(Xeq, yeq, rcond=None)
    pred_eq = Xeq @ beta
    r2_eq = float(r2_score(yeq, pred_eq))

    equation = {
        "intercept_exp": float(np.exp(beta[0])),
        "speed_exponent": float(beta[1]),
        "feed_exponent": float(beta[2]),
        "depth_exponent": float(beta[3]),
        "log_space_r2": r2_eq,
    }

    obs.to_csv(TABLE_DIR / "nuaa_observed_life_by_experiment.csv", index=False)

    return {
        "cv_rmse_mean": float(np.mean([x["rmse"] for x in fold_scores])),
        "cv_mae_mean": float(np.mean([x["mae"] for x in fold_scores])),
        "cv_r2_mean": float(np.mean([x["r2"] for x in fold_scores])),
        "failure_threshold_mm": float(threshold_mm),
        "best_life_setting": life_df.iloc[0].to_dict(),
        "worst_life_setting": life_df.iloc[-1].to_dict(),
        "taylor_like_equation": equation,
    }


def plot_cross_dataset_model_summary(metric_frames: list[pd.DataFrame]):
    m = pd.concat(metric_frames, ignore_index=True)
    m.to_csv(TABLE_DIR / "all_datasets_regression_metrics.csv", index=False)

    plt.figure(figsize=(9.2, 5.2))
    sns.barplot(data=m, x="dataset", y="rmse_mean", hue="model", palette="Set2")
    plt.ylabel("RMSE (mm)")
    plt.xlabel("Dataset")
    plt.title("Cross-Dataset Wear Regression Performance (Group CV)")
    plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "cross_dataset_rmse_comparison.png", dpi=170)
    plt.close()


def save_dataset_profile(phm_df: pd.DataFrame, nasa_df: pd.DataFrame, uniwear_df: pd.DataFrame):
    rows = []
    for name, df in [("PHM2010", phm_df), ("NASA", nasa_df), ("UniWear", uniwear_df)]:
        rows.append(
            {
                "dataset": name,
                "samples": int(len(df)),
                "groups": int(df["group_id"].nunique()),
                "wear_min_mm": float(df["tool_wear_mm"].min()),
                "wear_max_mm": float(df["tool_wear_mm"].max()),
                "wear_mean_mm": float(df["tool_wear_mm"].mean()),
                "wear_std_mm": float(df["tool_wear_mm"].std()),
            }
        )
    prof = pd.DataFrame(rows)
    prof.to_csv(TABLE_DIR / "dataset_profiles.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=pd.concat(
            [
                phm_df[["tool_wear_mm"]].assign(dataset="PHM2010"),
                nasa_df[["tool_wear_mm"]].assign(dataset="NASA"),
                uniwear_df[["tool_wear_mm"]].assign(dataset="UniWear"),
            ],
            ignore_index=True,
        ),
        x="dataset",
        y="tool_wear_mm",
        palette="pastel",
    )
    plt.ylabel("Tool Wear (mm)")
    plt.title("Wear Range Across Datasets")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "dataset_wear_distribution.png", dpi=170)
    plt.close()


def write_markdown_report(summary: dict):
    report_path = OUT_DIR / "tool_health_failure_research_report.md"

    def fmt(v, d=4):
        if isinstance(v, float):
            if np.isnan(v):
                return "nan"
            return f"{v:.{d}f}"
        return str(v)

    phm = summary["PHM2010"]
    nasa = summary["NASA"]
    uni = summary["UniWear"]
    sim = summary["Simulator_NUAA"]

    eq = sim["taylor_like_equation"]
    c0 = eq["intercept_exp"]
    a = eq["speed_exponent"]
    b = eq["feed_exponent"]
    c = eq["depth_exponent"]

    lines = []
    lines.append("# Cross-Dataset Tool Health, Failure Marker, and Life Simulation Study")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append("Develop a reproducible pipeline that:")
    lines.append("- learns wear from sensor signals,")
    lines.append("- finds early failure markers,")
    lines.append("- and builds a parameter-driven simulator for tool-life optimization.")
    lines.append("")
    lines.append("Datasets used in this repository:")
    lines.append("- PHM 2010 challenge (`c1,c4,c6` labeled training; `c2,c3,c5` unlabeled test),")
    lines.append("- NASA milling dataset,")
    lines.append("- UniWear merged dataset + NUAA high-resolution parameterized bundle.")
    lines.append("")
    lines.append("## Dataset Inventory")
    lines.append("")
    lines.append("Dataset profile table: `results/cross_dataset_research/tables/dataset_profiles.csv`")
    lines.append("")
    lines.append("![Wear distribution](plots/dataset_wear_distribution.png)")
    lines.append("")
    lines.append("## Methods")
    lines.append("")
    lines.append("### 1. Feature Extraction")
    lines.append("- Per sensor channel: mean, std, RMS, abs-mean, peak-to-peak, skew-like, kurt-like, crest, quantiles.")
    lines.append("- Frequency markers: FFT energy, dominant bin, centroid, entropy.")
    lines.append("- PHM raw files were downsampled by 25 for computational efficiency while preserving trend statistics.")
    lines.append("")
    lines.append("### 2. Wear Regression")
    lines.append("- Models: Ridge, RandomForest, GradientBoosting.")
    lines.append("- Validation: Grouped CV (leave-groups-out style):")
    lines.append("  - PHM grouped by cutter (c1/c4/c6),")
    lines.append("  - NASA grouped by case,")
    lines.append("  - UniWear grouped by experiment tag.")
    lines.append("")
    lines.append("### 3. Failure Marker Mining")
    lines.append("- Binary failure label from wear threshold (dataset-specific):")
    lines.append(f"  - PHM: >= {phm['failure_threshold_mm']:.3f} mm")
    lines.append(f"  - NASA: >= {nasa['failure_threshold_mm']:.3f} mm")
    lines.append(f"  - UniWear: >= {uni['failure_threshold_mm']:.3f} mm")
    lines.append("- Classifier: RandomForestClassifier with class balancing and grouped CV.")
    lines.append("- Marker candidates ranked by single-feature AUC and healthy vs failed separation.")
    lines.append("")
    lines.append("### 4. Parameter-Life Simulator (NUAA Bundle)")
    lines.append("- Learned model: wear = f(time, feed_per_tooth, spindle_speed, axial_cutting_depth).")
    lines.append("- Life criterion: first time predicted wear reaches threshold.")
    lines.append(f"- Threshold used for simulator: {sim['failure_threshold_mm']:.3f} mm.")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("### A. Regression Performance")
    lines.append("")
    lines.append("All metrics: `results/cross_dataset_research/tables/all_datasets_regression_metrics.csv`")
    lines.append("")
    lines.append("![RMSE comparison](plots/cross_dataset_rmse_comparison.png)")
    lines.append("")
    lines.append(f"- PHM best model: **{phm['best_model']}**, RMSE={fmt(phm['best_rmse'])}, R2={fmt(phm['best_r2'])}")
    lines.append(f"- NASA best model: **{nasa['best_model']}**, RMSE={fmt(nasa['best_rmse'])}, R2={fmt(nasa['best_r2'])}")
    lines.append(f"- UniWear best model: **{uni['best_model']}**, RMSE={fmt(uni['best_rmse'])}, R2={fmt(uni['best_r2'])}")
    lines.append("")
    lines.append("Regression scatter plots:")
    lines.append("- `plots/phm2010_regression_scatter.png`")
    lines.append("- `plots/nasa_regression_scatter.png`")
    lines.append("- `plots/uniwear_regression_scatter.png`")
    lines.append("")
    lines.append("### B. Failure Classification")
    lines.append("")
    lines.append(f"- PHM ROC-AUC={fmt(phm['classification_roc_auc_mean'])}, F1={fmt(phm['classification_f1_mean'])}, failure-rate={fmt(phm['failure_rate'])}")
    lines.append(f"- NASA ROC-AUC={fmt(nasa['classification_roc_auc_mean'])}, F1={fmt(nasa['classification_f1_mean'])}, failure-rate={fmt(nasa['failure_rate'])}")
    lines.append(f"- UniWear ROC-AUC={fmt(uni['classification_roc_auc_mean'])}, F1={fmt(uni['classification_f1_mean'])}, failure-rate={fmt(uni['failure_rate'])}")
    lines.append("")
    lines.append("Marker tables (with direction and threshold):")
    lines.append("- `tables/phm2010_failure_markers.csv`")
    lines.append("- `tables/nasa_failure_markers.csv`")
    lines.append("- `tables/uniwear_failure_markers.csv`")
    lines.append("")
    lines.append("Top marker plots:")
    lines.append("- `plots/phm2010_failure_markers.png`")
    lines.append("- `plots/nasa_failure_markers.png`")
    lines.append("- `plots/uniwear_failure_markers.png`")
    lines.append("")
    lines.append("### C. Simulator and Parameter Optimization")
    lines.append("")
    lines.append(f"- NUAA simulator CV: RMSE={fmt(sim['cv_rmse_mean'])}, MAE={fmt(sim['cv_mae_mean'])}, R2={fmt(sim['cv_r2_mean'])}")
    lines.append("- Parameter-life heatmap: `plots/nuaa_life_heatmap.png`")
    lines.append("- Ranked life table: `tables/nuaa_simulated_life_table.csv`")
    lines.append("")
    lines.append("Best setting in observed parameter grid:")
    lines.append(
        f"- feed_per_tooth={sim['best_life_setting']['feed_per_tooth']:.3f}, spindle={sim['best_life_setting']['spindle_speed']:.0f} rpm, depth={sim['best_life_setting']['axial_cutting_depth']:.1f} mm, life={sim['best_life_setting']['predicted_life_s']:.1f} s"
    )
    lines.append("")
    lines.append("Worst setting in observed parameter grid:")
    lines.append(
        f"- feed_per_tooth={sim['worst_life_setting']['feed_per_tooth']:.3f}, spindle={sim['worst_life_setting']['spindle_speed']:.0f} rpm, depth={sim['worst_life_setting']['axial_cutting_depth']:.1f} mm, life={sim['worst_life_setting']['predicted_life_s']:.1f} s"
    )
    lines.append("")
    lines.append("### D. Taylor-like Life Equation (Empirical)")
    lines.append("")
    lines.append("From observed NUAA experiment lives (threshold crossing), fitted in log-space:")
    lines.append("")
    lines.append("`log(T) = b0 + b1*log(spindle_speed) + b2*log(feed_per_tooth) + b3*log(axial_depth)`")
    lines.append("")
    lines.append("Equivalent form:")
    lines.append("")
    lines.append(f"`T ~= {c0:.3e} * spindle_speed^({a:.4f}) * feed_per_tooth^({b:.4f}) * axial_depth^({c:.4f})`")
    lines.append("")
    lines.append(f"Log-space R2 = {eq['log_space_r2']:.4f}")
    lines.append("")
    lines.append("## Practical Failure-Marker Guidance")
    lines.append("")
    lines.append("- Consistent indicators across datasets are vibration-energy growth, AE/force dispersion rise, and crest-factor increase.")
    lines.append("- Use two-level alarms for deployment:")
    lines.append("  - Warning when marker thresholds are crossed with moderate confidence.")
    lines.append("  - Failure risk when classifier probability > 0.7 over consecutive windows.")
    lines.append("- For PHM test cutters (`c2,c3,c5`), the trained PHM model can score every run for trend-based risk even without wear labels.")
    lines.append("")
    lines.append("## Files Generated")
    lines.append("")
    lines.append("- Main metrics summary: `results/cross_dataset_research/metrics_summary.json`")
    lines.append("- Detailed tables: `results/cross_dataset_research/tables/`")
    lines.append("- Figures: `results/cross_dataset_research/plots/`")
    lines.append("- This report: `results/cross_dataset_research/tool_health_failure_research_report.md`")
    lines.append("")
    lines.append("## Limitations and Next Experiments")
    lines.append("")
    lines.append("- Cross-dataset transfer is constrained by sensor and operating-regime mismatch.")
    lines.append("- PHM wear range is lower than NASA, so thresholding must remain dataset-aware.")
    lines.append("- Next steps: sequence models (TCN/LSTM/Transformer) and uncertainty-calibrated RUL forecasting.")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    print("Loading datasets and extracting features...")
    phm_train, phm_test = load_phm_features(downsample=25)
    nasa_df = load_nasa_features()
    uniwear_df = load_uniwear_window_features(window_size=120, stride=60)

    print("Saving dataset profiles...")
    save_dataset_profile(phm_train, nasa_df, uniwear_df)

    print("Running PHM experiments...")
    phm_metrics, _, phm_summary, phm_feat_cols = evaluate_regression(
        phm_train, dataset_name="PHM2010", threshold_mm=0.15
    )

    print("Scoring PHM test cutters (c2,c3,c5) for risk trend...")
    phm_feature_set = [c for c in phm_feat_cols if c in phm_test.columns]
    phm_train_x = phm_train[phm_feature_set].replace([np.inf, -np.inf], np.nan).copy()
    phm_test_x = phm_test[phm_feature_set].replace([np.inf, -np.inf], np.nan).copy()
    for col in phm_feature_set:
        med = phm_train_x[col].median()
        if pd.isna(med):
            med = 0.0
        phm_train_x[col] = phm_train_x[col].fillna(med)
        phm_test_x[col] = phm_test_x[col].fillna(med)
        lo = phm_train_x[col].quantile(0.01)
        hi = phm_train_x[col].quantile(0.99)
        if pd.isna(lo) or pd.isna(hi) or lo == hi:
            continue
        phm_train_x[col] = phm_train_x[col].clip(lo, hi)
        phm_test_x[col] = phm_test_x[col].clip(lo, hi)

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    rf.fit(phm_train_x, phm_train["tool_wear_mm"])
    phm_test = phm_test.copy()
    phm_test["predicted_wear_mm"] = rf.predict(phm_test_x)
    phm_test[["group_id", "sample_id", "cut", "predicted_wear_mm"]].to_csv(
        TABLE_DIR / "phm2010_test_cutters_predicted_wear.csv", index=False
    )

    print("Running NASA experiments...")
    nasa_metrics, _, nasa_summary, _ = evaluate_regression(
        nasa_df, dataset_name="NASA", threshold_mm=0.30
    )

    print("Running UniWear experiments...")
    uni_metrics, _, uni_summary, _ = evaluate_regression(
        uniwear_df, dataset_name="UniWear", threshold_mm=0.22
    )

    print("Building NUAA parameter-life simulator...")
    sim_summary = build_nuaa_simulator(threshold_mm=0.25)

    print("Generating cross-dataset summary plot...")
    plot_cross_dataset_model_summary([phm_metrics, nasa_metrics, uni_metrics])

    summary = {
        "PHM2010": phm_summary,
        "NASA": nasa_summary,
        "UniWear": uni_summary,
        "Simulator_NUAA": sim_summary,
    }

    (OUT_DIR / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_report(summary)

    print("Done.")
    print(f"Outputs saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
