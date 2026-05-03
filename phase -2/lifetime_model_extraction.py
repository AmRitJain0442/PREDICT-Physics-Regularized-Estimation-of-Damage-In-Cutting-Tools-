from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import research_pipeline
from research_pipeline import (
    OUTPUTS,
    PHASE2,
    PLOTS,
    REFERENCE_DEPTH_MM,
    REFERENCE_FEED_MM_TOOTH,
    REFERENCE_SPEED_RPM,
    TRANSITION_WEAR_MM,
    load_nasa_case_level,
    load_nuaa_run_level,
    load_phm_cut_level,
)
from tool_life_simulator import MillingToolLifeModel


LIFE_OUTPUTS = OUTPUTS / "lifetime_modeling"
LIFE_PLOTS = PLOTS / "lifetime_modeling"
REPORT_PATH = LIFE_OUTPUTS / "phase2_lifetime_model_report.md"
RUN_LOG_PATH = LIFE_OUTPUTS / "lifetime_model_run_log.md"

NUAA_LIFE_THRESHOLD_MM = 0.25
NUAA_AUX_THRESHOLD_MM = 0.22
NASA_LIFE_THRESHOLD_MM = 0.30
PHM_LIFE_THRESHOLD_MM = 0.15
RANDOM_STATE = 42


@dataclass
class AFTFit:
    coefficients: pd.DataFrame
    sigma: float
    nll: float
    feature_cols: list[str]
    beta: np.ndarray


def ensure_dirs() -> None:
    LIFE_OUTPUTS.mkdir(parents=True, exist_ok=True)
    LIFE_PLOTS.mkdir(parents=True, exist_ok=True)


def interpolate_life(
    df: pd.DataFrame,
    time_col: str,
    wear_col: str,
    threshold_mm: float,
) -> tuple[float, int, float]:
    work = df.sort_values(time_col)
    t = work[time_col].to_numpy(dtype=float)
    w = work[wear_col].to_numpy(dtype=float)
    valid = np.isfinite(t) & np.isfinite(w)
    t = t[valid]
    w = w[valid]
    if len(t) == 0:
        return np.nan, 0, np.nan

    crossing = np.where(w >= threshold_mm)[0]
    if len(crossing) == 0:
        return float(t[-1]), 0, float(w[-1])

    idx = int(crossing[0])
    if idx == 0:
        return float(t[0]), 1, float(w[-1])

    t0, t1 = float(t[idx - 1]), float(t[idx])
    w0, w1 = float(w[idx - 1]), float(w[idx])
    frac = 0.0 if w1 == w0 else (threshold_mm - w0) / (w1 - w0)
    frac = float(np.clip(frac, 0.0, 1.0))
    return float(t0 + frac * (t1 - t0)), 1, float(w[-1])


def build_life_records() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nuaa = load_nuaa_run_level()
    nasa = load_nasa_case_level()
    phm = load_phm_cut_level()

    nuaa_rows = []
    for experiment_tag, group_df in nuaa.groupby("experiment_tag", sort=True):
        life_min, event, final_wear = interpolate_life(
            group_df,
            time_col="relative_time_min",
            wear_col="wear_mm",
            threshold_mm=NUAA_LIFE_THRESHOLD_MM,
        )
        aux_life_min, aux_event, _ = interpolate_life(
            group_df,
            time_col="relative_time_min",
            wear_col="wear_mm",
            threshold_mm=NUAA_AUX_THRESHOLD_MM,
        )
        first = group_df.iloc[0]
        nuaa_rows.append(
            {
                "dataset": "NUAA",
                "series_id": experiment_tag,
                "threshold_mm": NUAA_LIFE_THRESHOLD_MM,
                "life_min": life_min,
                "event": event,
                "censor_time_min": life_min if event == 0 else np.nan,
                "final_wear_mm": final_wear,
                "aux_threshold_mm": NUAA_AUX_THRESHOLD_MM,
                "aux_life_min": aux_life_min,
                "aux_event": aux_event,
                "initial_wear_mm": float(group_df["initial_wear_mm"].iloc[0]),
                "speed_rpm": float(first["speed_rpm"]),
                "feed_mm_tooth": float(first["feed_mm_tooth"]),
                "depth_mm": float(first["depth_mm"]),
                "log_speed_ref": np.log(float(first["speed_rpm"]) / REFERENCE_SPEED_RPM),
                "log_feed_ref": np.log(float(first["feed_mm_tooth"]) / REFERENCE_FEED_MM_TOOTH),
                "log_depth_ref": np.log(float(first["depth_mm"]) / REFERENCE_DEPTH_MM),
            }
        )

    nasa_rows = []
    for case, group_df in nasa.groupby("case", sort=True):
        life_min, event, final_wear = interpolate_life(
            group_df,
            time_col="relative_time_min",
            wear_col="VB",
            threshold_mm=NASA_LIFE_THRESHOLD_MM,
        )
        if not np.isfinite(life_min) or life_min <= 0.0:
            continue
        first = group_df.iloc[0]
        nasa_rows.append(
            {
                "dataset": "NASA",
                "series_id": f"case_{case}",
                "threshold_mm": NASA_LIFE_THRESHOLD_MM,
                "life_min": life_min,
                "event": event,
                "censor_time_min": life_min if event == 0 else np.nan,
                "final_wear_mm": final_wear,
                "speed_rpm": 826.0,
                "feed_mm_rev": float(first["feed"]),
                "depth_mm": float(first["DOC"]),
                "material_family": str(first["material_family"]),
                "material_steel": 1.0 if str(first["material_family"]) == "steel" else 0.0,
                "initial_wear_mm": float(group_df["initial_wear_mm"].iloc[0]),
                "log_feed_ref": np.log(float(first["feed"]) / 0.25),
                "log_depth_ref": np.log(float(first["DOC"]) / 0.75),
            }
        )

    phm_rows = []
    for cutter, group_df in phm.groupby("experiment_tag", sort=True):
        life_cut, event, final_wear = interpolate_life(
            group_df,
            time_col="relative_cut",
            wear_col="wear_mm",
            threshold_mm=PHM_LIFE_THRESHOLD_MM,
        )
        phm_rows.append(
            {
                "dataset": "PHM2010",
                "series_id": cutter,
                "threshold_mm": PHM_LIFE_THRESHOLD_MM,
                "life_cut": life_cut,
                "event": event,
                "final_wear_mm": final_wear,
                "initial_wear_mm": float(group_df["initial_wear_mm"].iloc[0]),
                "speed_rpm": 10400.0,
                "feed_rate_mm_min": 1555.0,
                "axial_depth_mm": 0.2,
                "radial_depth_mm": 0.125,
            }
        )

    return pd.DataFrame(nuaa_rows), pd.DataFrame(nasa_rows), pd.DataFrame(phm_rows)


def make_wear_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    t = np.maximum(df["relative_time_min"].to_numpy(dtype=float), 0.0)
    features["time_min"] = t
    features["sqrt_time"] = np.sqrt(t)
    features["log1p_time"] = np.log1p(t)
    features["time_pow_064"] = np.power(np.maximum(t, 1e-9), 0.64)
    features["log_speed_ref"] = np.log(df["speed_rpm"].to_numpy(dtype=float) / REFERENCE_SPEED_RPM)
    features["log_feed_ref"] = np.log(df["feed_mm_tooth"].to_numpy(dtype=float) / REFERENCE_FEED_MM_TOOTH)
    features["log_depth_ref"] = np.log(df["depth_mm"].to_numpy(dtype=float) / REFERENCE_DEPTH_MM)
    features["initial_wear_mm"] = df["initial_wear_mm"].to_numpy(dtype=float)
    features["feed_time"] = features["log_feed_ref"] * features["time_pow_064"]
    features["depth_time"] = features["log_depth_ref"] * features["time_pow_064"]
    features["speed_time"] = features["log_speed_ref"] * features["time_pow_064"]
    return features


def make_wear_models() -> dict[str, object]:
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e2))
        * RBF(length_scale=np.ones(11), length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
    )
    return {
        "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=0.25))]),
        "RandomForest": RandomForestRegressor(
            n_estimators=600,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.035,
            max_depth=3,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
        ),
        "GaussianProcess": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    GaussianProcessRegressor(
                        kernel=kernel,
                        normalize_y=True,
                        n_restarts_optimizer=2,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "MLP": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(32, 16),
                        activation="tanh",
                        alpha=0.01,
                        learning_rate_init=0.01,
                        max_iter=2500,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def predict_phase2_wear(df: pd.DataFrame, simulator: MillingToolLifeModel) -> np.ndarray:
    values = []
    for _, row in df.iterrows():
        values.append(
            simulator.wear_at_minutes(
                speed_rpm=float(row["speed_rpm"]),
                feed_mm_tooth=float(row["feed_mm_tooth"]),
                depth_mm=float(row["depth_mm"]),
                time_min=float(row["relative_time_min"]),
                initial_wear_mm=float(row["initial_wear_mm"]),
                calibration_factor=1.0,
                material_family="generic",
            )
        )
    return np.asarray(values, dtype=float)


def life_from_curve(times: np.ndarray, wear: np.ndarray, threshold_mm: float) -> tuple[float, int]:
    order = np.argsort(times)
    t = times[order].astype(float)
    w = np.maximum.accumulate(wear[order].astype(float))
    crossing = np.where(w >= threshold_mm)[0]
    if len(crossing) == 0:
        return float(t[-1]), 0
    idx = int(crossing[0])
    if idx == 0:
        return float(t[0]), 1
    t0, t1 = float(t[idx - 1]), float(t[idx])
    w0, w1 = float(w[idx - 1]), float(w[idx])
    frac = 0.0 if w1 == w0 else (threshold_mm - w0) / (w1 - w0)
    frac = float(np.clip(frac, 0.0, 1.0))
    return float(t0 + frac * (t1 - t0)), 1


def evaluate_nuaa_wear_models(
    nuaa_runs: pd.DataFrame,
    nuaa_life: pd.DataFrame,
    simulator: MillingToolLifeModel,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_df = make_wear_features(nuaa_runs)
    y = nuaa_runs["wear_mm"].to_numpy(dtype=float)
    groups = nuaa_runs["experiment_tag"].astype(str).to_numpy()
    logo = LeaveOneGroupOut()

    prediction_frames = []
    life_rows = []
    metric_rows = []

    phase2_pred = predict_phase2_wear(nuaa_runs, simulator)
    phase2_frame = nuaa_runs[["experiment_tag", "relative_time_min", "wear_mm"]].copy()
    phase2_frame["model"] = "Phase2Equation"
    phase2_frame["predicted_wear_mm"] = phase2_pred
    prediction_frames.append(phase2_frame)

    for experiment_tag, group_df in phase2_frame.groupby("experiment_tag", sort=True):
        observed = nuaa_life[nuaa_life["series_id"] == experiment_tag].iloc[0]
        pred_life, pred_event = life_from_curve(
            group_df["relative_time_min"].to_numpy(dtype=float),
            group_df["predicted_wear_mm"].to_numpy(dtype=float),
            NUAA_LIFE_THRESHOLD_MM,
        )
        life_rows.append(
            {
                "model": "Phase2Equation",
                "series_id": experiment_tag,
                "observed_life_min": float(observed["life_min"]),
                "observed_event": int(observed["event"]),
                "predicted_life_min": pred_life,
                "predicted_event": pred_event,
                "event_abs_error_min": abs(pred_life - float(observed["life_min"]))
                if int(observed["event"]) == 1
                else np.nan,
                "censor_bound_satisfied": bool(pred_life >= float(observed["life_min"]))
                if int(observed["event"]) == 0
                else np.nan,
            }
        )

    metric_rows.append(
        {
            "model": "Phase2Equation",
            "wear_rmse_mm": float(np.sqrt(mean_squared_error(y, phase2_pred))),
            "wear_mae_mm": float(mean_absolute_error(y, phase2_pred)),
            "wear_r2": float(r2_score(y, phase2_pred)),
        }
    )

    for model_name, model in make_wear_models().items():
        pred = np.zeros(len(nuaa_runs), dtype=float)
        for train_idx, test_idx in logo.split(feature_df, y, groups):
            model.fit(feature_df.iloc[train_idx], y[train_idx])
            pred[test_idx] = model.predict(feature_df.iloc[test_idx])

        frame = nuaa_runs[["experiment_tag", "relative_time_min", "wear_mm"]].copy()
        frame["model"] = model_name
        frame["predicted_wear_mm"] = pred
        prediction_frames.append(frame)

        metric_rows.append(
            {
                "model": model_name,
                "wear_rmse_mm": float(np.sqrt(mean_squared_error(y, pred))),
                "wear_mae_mm": float(mean_absolute_error(y, pred)),
                "wear_r2": float(r2_score(y, pred)),
            }
        )

        for experiment_tag, group_df in frame.groupby("experiment_tag", sort=True):
            observed = nuaa_life[nuaa_life["series_id"] == experiment_tag].iloc[0]
            pred_life, pred_event = life_from_curve(
                group_df["relative_time_min"].to_numpy(dtype=float),
                group_df["predicted_wear_mm"].to_numpy(dtype=float),
                NUAA_LIFE_THRESHOLD_MM,
            )
            life_rows.append(
                {
                    "model": model_name,
                    "series_id": experiment_tag,
                    "observed_life_min": float(observed["life_min"]),
                    "observed_event": int(observed["event"]),
                    "predicted_life_min": pred_life,
                    "predicted_event": pred_event,
                    "event_abs_error_min": abs(pred_life - float(observed["life_min"]))
                    if int(observed["event"]) == 1
                    else np.nan,
                    "censor_bound_satisfied": bool(pred_life >= float(observed["life_min"]))
                    if int(observed["event"]) == 0
                    else np.nan,
                }
            )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    life_predictions = pd.DataFrame(life_rows)
    metrics = pd.DataFrame(metric_rows)
    life_summary = (
        life_predictions.groupby("model", as_index=False)
        .agg(
            event_life_mae_min=("event_abs_error_min", "mean"),
            event_life_median_abs_error_min=("event_abs_error_min", "median"),
            censor_bound_satisfied_rate=("censor_bound_satisfied", lambda s: float(pd.Series(s).dropna().mean()) if len(pd.Series(s).dropna()) else np.nan),
        )
    )
    metrics = metrics.merge(life_summary, on="model", how="left").sort_values(
        ["event_life_mae_min", "wear_rmse_mm"],
        na_position="last",
    )
    return metrics, predictions, life_predictions


def fit_lognormal_aft(
    df: pd.DataFrame,
    feature_cols: list[str],
    time_col: str = "life_min",
    event_col: str = "event",
    l2: float = 0.2,
) -> AFTFit:
    work = df.dropna(subset=[time_col, event_col, *feature_cols]).copy()
    work = work[work[time_col] > 0.0].copy()
    x = np.column_stack([np.ones(len(work)), work[feature_cols].to_numpy(dtype=float)])
    log_t = np.log(work[time_col].to_numpy(dtype=float))
    event = work[event_col].to_numpy(dtype=int)
    p = x.shape[1]

    init_beta, *_ = np.linalg.lstsq(x[event == 1] if np.any(event == 1) else x, log_t[event == 1] if np.any(event == 1) else log_t, rcond=None)
    init = np.r_[init_beta, np.log(np.std(log_t) + 0.2)]

    def objective(params: np.ndarray) -> float:
        beta = params[:p]
        sigma = float(np.exp(params[p]))
        mu = x @ beta
        z = (log_t - mu) / sigma
        log_pdf = norm.logpdf(z) - np.log(sigma) - log_t
        log_surv = norm.logsf(z)
        nll = -float(np.sum(event * log_pdf + (1 - event) * log_surv))
        penalty = l2 * float(np.dot(beta[1:], beta[1:]))
        return nll + penalty

    result = minimize(objective, init, method="BFGS", options={"maxiter": 20_000})
    params = result.x
    beta = params[:p]
    sigma = float(np.exp(params[p]))
    names = ["intercept", *feature_cols]
    coef_df = pd.DataFrame(
        {
            "term": names,
            "coefficient": beta,
            "exp_coefficient": np.exp(beta),
        }
    )
    return AFTFit(
        coefficients=coef_df,
        sigma=sigma,
        nll=float(result.fun),
        feature_cols=feature_cols,
        beta=beta,
    )


def predict_aft_median(df: pd.DataFrame, fit: AFTFit) -> np.ndarray:
    x = np.column_stack([np.ones(len(df)), df[fit.feature_cols].to_numpy(dtype=float)])
    return np.exp(x @ fit.beta)


def bootstrap_aft(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_boot: int = 400,
    seed: int = 20260503,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    groups = df["series_id"].astype(str).tolist()
    attempts = 0
    max_attempts = n_boot * 40
    while len(rows) < n_boot and attempts < max_attempts:
        attempts += 1
        sampled = rng.choice(groups, size=len(groups), replace=True)
        boot = pd.concat([df[df["series_id"].astype(str) == group] for group in sampled], ignore_index=True)
        x = boot[feature_cols].to_numpy(dtype=float)
        if len(np.unique(sampled)) < len(feature_cols) + 1:
            continue
        if np.linalg.matrix_rank(np.column_stack([np.ones(len(x)), x])) < len(feature_cols) + 1:
            continue
        try:
            fit = fit_lognormal_aft(boot, feature_cols=feature_cols)
        except Exception:
            continue
        row = {"bootstrap_id": len(rows), "attempt_id": attempts, "sigma": fit.sigma}
        for _, coef_row in fit.coefficients.iterrows():
            row[str(coef_row["term"])] = float(coef_row["coefficient"])
        rows.append(row)

    if len(rows) < n_boot:
        raise RuntimeError(f"Only produced {len(rows)} AFT bootstrap draws.")
    return pd.DataFrame(rows)


def fit_life_equation_ols(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> tuple[pd.DataFrame, dict[str, float]]:
    work = df.dropna(subset=[target_col, *feature_cols]).copy()
    work = work[work[target_col] > 0.0]
    x = np.column_stack([np.ones(len(work)), work[feature_cols].to_numpy(dtype=float)])
    y = np.log(work[target_col].to_numpy(dtype=float))
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ beta
    names = ["intercept", *feature_cols]
    coef_df = pd.DataFrame({"term": names, "coefficient": beta, "exp_coefficient": np.exp(beta)})
    metrics = {
        "log_r2": float(r2_score(y, pred)) if len(work) > x.shape[1] else np.nan,
        "log_rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "records": int(len(work)),
    }
    return coef_df, metrics


def train_best_wear_model(nuaa_runs: pd.DataFrame, model_name: str) -> object:
    models = make_wear_models()
    if model_name == "Phase2Equation":
        return MillingToolLifeModel.from_json()
    model = models[model_name]
    feature_df = make_wear_features(nuaa_runs)
    y = nuaa_runs["wear_mm"].to_numpy(dtype=float)
    model.fit(feature_df, y)
    return model


def predict_wear_grid(model: object, input_df: pd.DataFrame, model_name: str) -> np.ndarray:
    if model_name == "Phase2Equation":
        return predict_phase2_wear(input_df, model)  # type: ignore[arg-type]
    return np.asarray(model.predict(make_wear_features(input_df)), dtype=float)  # type: ignore[attr-defined]


def extract_ml_life_surface(
    nuaa_runs: pd.DataFrame,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    model = train_best_wear_model(nuaa_runs, model_name)
    speeds = np.linspace(1750.0, 1850.0, 9)
    feeds = np.linspace(0.045, 0.055, 9)
    depths = np.linspace(2.5, 3.5, 5)
    times = np.linspace(0.0, 80.0, 320)

    rows = []
    for speed in speeds:
        for feed in feeds:
            for depth in depths:
                input_df = pd.DataFrame(
                    {
                        "relative_time_min": times,
                        "speed_rpm": speed,
                        "feed_mm_tooth": feed,
                        "depth_mm": depth,
                        "initial_wear_mm": 0.06,
                    }
                )
                wear = predict_wear_grid(model, input_df, model_name=model_name)
                wear = np.maximum.accumulate(np.maximum(wear, 0.0))
                life_min, event = life_from_curve(times, wear, NUAA_LIFE_THRESHOLD_MM)
                rows.append(
                    {
                        "model": model_name,
                        "speed_rpm": speed,
                        "feed_mm_tooth": feed,
                        "depth_mm": depth,
                        "life_min": life_min,
                        "event": event,
                    }
                )

    surface = pd.DataFrame(rows)
    event_surface = surface[surface["event"] == 1].copy()
    event_surface["log_speed_ref"] = np.log(event_surface["speed_rpm"] / REFERENCE_SPEED_RPM)
    event_surface["log_feed_ref"] = np.log(event_surface["feed_mm_tooth"] / REFERENCE_FEED_MM_TOOTH)
    event_surface["log_depth_ref"] = np.log(event_surface["depth_mm"] / REFERENCE_DEPTH_MM)
    coef_df, metrics = fit_life_equation_ols(
        event_surface,
        feature_cols=["log_speed_ref", "log_feed_ref", "log_depth_ref"],
        target_col="life_min",
    )
    metrics["surface_records"] = int(len(surface))
    metrics["surface_event_records"] = int(len(event_surface))
    return surface, coef_df, metrics


def equation_string(
    coef_df: pd.DataFrame,
    ref_terms: dict[str, tuple[str, float]],
    unit: str,
) -> str:
    intercept = float(coef_df[coef_df["term"] == "intercept"]["exp_coefficient"].iloc[0])
    parts = [f"T_{unit} = {intercept:.4g}"]
    for term, (symbol, ref) in ref_terms.items():
        if term not in set(coef_df["term"]):
            continue
        exponent = float(coef_df[coef_df["term"] == term]["coefficient"].iloc[0])
        parts.append(f"* ({symbol}/{ref:g})^{exponent:.4f}")
    return " ".join(parts)


def build_literature_matrix() -> pd.DataFrame:
    rows = [
        {
            "source": "Local blueprint PDF",
            "domain": "machining",
            "method": "tiered surrogate, physics-informed hybrid, optimization",
            "usable_transfer": "Keep analytical equation plus ML surrogate; expose uncertainty and optimize only within observed parameter ranges.",
            "url_or_path": "ML-Driven Cutting Tool Simulation and Life Optimization_ A Research Blueprint.pdf",
        },
        {
            "source": "NASA milling readme",
            "domain": "machining",
            "method": "case-level tool wear under feed/depth/material settings",
            "usable_transfer": "Use NASA as late-stage/material evidence; do not merge its feed units blindly with NUAA feed per tooth.",
            "url_or_path": "data/nasa_milling/Readme.pdf",
        },
        {
            "source": "PHM public dataset inventory",
            "domain": "PHM",
            "method": "run-to-failure dataset taxonomy",
            "usable_transfer": "Use time-to-event framing and keep censored records instead of dropping non-failures.",
            "url_or_path": "data list paper.pdf",
        },
        {
            "source": "QIT-CEMC Scientific Data 2025",
            "domain": "machining",
            "method": "full-life multimodal milling data",
            "usable_transfer": "Next external validation target for full lifecycle wear and force/torque features.",
            "url_or_path": "https://www.nature.com/articles/s41597-024-04345-2",
        },
        {
            "source": "Piecuch-Zabinski Scientific Data 2025",
            "domain": "machining",
            "method": "tool-wise grouped validation for tool-life estimation",
            "usable_transfer": "Validate by held-out tool, not random cycle split.",
            "url_or_path": "https://www.nature.com/articles/s41597-025-04923-y",
        },
        {
            "source": "PHM Society review 2018",
            "domain": "PHM",
            "method": "regression, RF, ANN, Bayesian linear regression, GPR for RUL",
            "usable_transfer": "Use model families as baselines and report whether the model predicts health or time-to-event.",
            "url_or_path": "https://papers.phmsociety.org/index.php/phmconf/article/download/462/phmc_18_462",
        },
        {
            "source": "Sankararaman and Goebel 2013",
            "domain": "PHM uncertainty",
            "method": "RUL uncertainty propagation",
            "usable_transfer": "Report intervals and uncertainty sources, not only point estimates.",
            "url_or_path": "https://papers.phmsociety.org/index.php/phmconf/article/view/2263",
        },
        {
            "source": "GPR RUL PHM 2022",
            "domain": "PHM",
            "method": "Gaussian process RUL with small data and uncertainty",
            "usable_transfer": "Include GPR as a small-data baseline; keep intervals as future work if calibrated externally.",
            "url_or_path": "https://papers.phmsociety.org/index.php/phmconf/article/view/3220",
        },
        {
            "source": "DeepSurv",
            "domain": "biomedicine",
            "method": "Cox neural network survival model",
            "usable_transfer": "Use survival/time-to-event framing for censored tool-life data; deep version is data-hungry here.",
            "url_or_path": "https://arxiv.org/abs/1606.00931",
        },
        {
            "source": "Physics-informed GP for tool wear",
            "domain": "machining",
            "method": "physical mean function plus GP residual",
            "usable_transfer": "For small labels, constrain probabilistic models with a wear law rather than pure black-box fitting.",
            "url_or_path": "https://pubmed.ncbi.nlm.nih.gov/37770369/",
        },
        {
            "source": "Drouillet et al. 2016",
            "domain": "machining",
            "method": "neural-network RUL using spindle-power RMS",
            "usable_transfer": "Use sensor-derived health indicators as additional life predictors once enough run-to-failure records exist.",
            "url_or_path": "https://impact.ornl.gov/en/publications/tool-life-predictions-in-milling-using-spindle-power-with-the-neu",
        },
        {
            "source": "PI-KAF 2025",
            "domain": "machining",
            "method": "physics-informed interpretable neural monitoring",
            "usable_transfer": "Prefer constrained, interpretable networks over unconstrained deep models for limited tool-wear labels.",
            "url_or_path": "https://www.sciencedirect.com/science/article/abs/pii/S0278612525002833",
        },
    ]
    return pd.DataFrame(rows)


def plot_wear_benchmark(metrics: pd.DataFrame, output_path: Path) -> None:
    ordered = metrics.sort_values("wear_rmse_mm")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].bar(ordered["model"], ordered["wear_rmse_mm"], color="#4E79A7")
    axes[0].set_title("Wear RMSE")
    axes[0].set_ylabel("mm")
    axes[1].bar(ordered["model"], ordered["event_life_mae_min"], color="#F28E2B")
    axes[1].set_title("Event-Life MAE")
    axes[1].set_ylabel("min")
    axes[2].bar(ordered["model"], ordered["censor_bound_satisfied_rate"], color="#59A14F")
    axes[2].set_title("Censored Lower Bounds")
    axes[2].set_ylim(0, 1.05)
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("NUAA Leave-One-Experiment-Out Lifetime Model Benchmark")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_life_equation_comparison(
    nuaa_life: pd.DataFrame,
    comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    event = nuaa_life[nuaa_life["event"] == 1]
    censored = nuaa_life[nuaa_life["event"] == 0]
    for model, part in comparison.groupby("model"):
        ax.scatter(part["observed_life_min"], part["predicted_life_min"], s=55, label=model, alpha=0.78)
    if len(censored):
        ax.scatter(
            censored["life_min"],
            censored["life_min"],
            marker=">",
            s=70,
            color="black",
            label="Observed lower bound",
        )
    lo = max(0.0, float(nuaa_life["life_min"].min()) - 2)
    hi = float(max(nuaa_life["life_min"].max(), comparison["predicted_life_min"].max())) + 4
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Observed life or censor time to 0.25 mm (min)")
    ax.set_ylabel("Predicted median life (min)")
    ax.set_title("NUAA Lifetime Equation Comparison")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_life_surface(surface: pd.DataFrame, output_path: Path) -> None:
    depths = sorted(surface["depth_mm"].unique())
    keep_depths = [depths[0], depths[len(depths) // 2], depths[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, depth in zip(axes, keep_depths):
        part = surface[np.isclose(surface["depth_mm"], depth)]
        pivot = part.pivot(index="feed_mm_tooth", columns="speed_rpm", values="life_min")
        im = ax.imshow(pivot.to_numpy(), origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)), [f"{x:.0f}" for x in pivot.columns], rotation=35)
        ax.set_yticks(range(len(pivot.index)), [f"{x:.3f}" for x in pivot.index])
        ax.set_title(f"Depth {depth:.2f} mm")
        ax.set_xlabel("Speed (rpm)")
        ax.set_ylabel("Feed per tooth")
        for i, feed in enumerate(pivot.index):
            for j, speed in enumerate(pivot.columns):
                ax.text(j, i, f"{pivot.loc[feed, speed]:.0f}", ha="center", va="center", color="white", fontsize=7)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label("Predicted life to 0.25 mm (min)")
    fig.suptitle("ML-Inverted Tool-Life Surface")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_aft_bootstrap(aft_bootstrap: pd.DataFrame, output_path: Path) -> None:
    terms = ["log_speed_ref", "log_feed_ref", "log_depth_ref"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, term in zip(axes, terms):
        values = aft_bootstrap[term].dropna()
        ax.hist(values, bins=28, color="#4E79A7", alpha=0.82)
        q025, median, q975 = np.quantile(values, [0.025, 0.5, 0.975])
        ax.axvspan(q025, q975, color="#F28E2B", alpha=0.18)
        ax.axvline(median, color="#E15759", linewidth=1.3)
        ax.set_title(term)
        ax.grid(alpha=0.22, axis="y")
    fig.suptitle("NUAA Censored AFT Coefficient Bootstrap")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_nasa_aft(nasa_life: pd.DataFrame, nasa_pred: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    merged = nasa_life.merge(nasa_pred, on="series_id")
    colors = {"cast_iron": "#4E79A7", "steel": "#E15759"}
    for material, part in merged.groupby("material_family"):
        ax.scatter(
            part["life_min"],
            part["predicted_life_min"],
            s=58,
            alpha=0.8,
            color=colors.get(material, "gray"),
            label=material,
        )
    lo = 0.0
    hi = float(max(merged["life_min"].max(), merged["predicted_life_min"].max())) + 5.0
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Observed life to 0.30 mm (min)")
    ax.set_ylabel("AFT predicted median life (min)")
    ax.set_title("NASA Material-Aware AFT Tool-Life Model")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.{digits}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    nuaa_life: pd.DataFrame,
    nasa_life: pd.DataFrame,
    phm_life: pd.DataFrame,
    wear_metrics: pd.DataFrame,
    best_model_name: str,
    aft_fit: AFTFit,
    aft_boot_summary: pd.DataFrame,
    ml_equation: pd.DataFrame,
    ml_equation_metrics: dict[str, float],
    observed_eq: pd.DataFrame,
    observed_eq_metrics: dict[str, float],
    nasa_aft: AFTFit,
    nasa_aft_metrics: pd.DataFrame,
    literature: pd.DataFrame,
) -> None:
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    overall_best_life = str(
        wear_metrics.sort_values(["event_life_mae_min", "wear_rmse_mm"]).iloc[0]["model"]
    )
    best_wear_rmse = str(wear_metrics.sort_values("wear_rmse_mm").iloc[0]["model"])
    aft_equation = equation_string(
        aft_fit.coefficients,
        {
            "log_speed_ref": ("n", REFERENCE_SPEED_RPM),
            "log_feed_ref": ("fz", REFERENCE_FEED_MM_TOOTH),
            "log_depth_ref": ("ap", REFERENCE_DEPTH_MM),
        },
        unit="min",
    )
    ml_equation_text = equation_string(
        ml_equation,
        {
            "log_speed_ref": ("n", REFERENCE_SPEED_RPM),
            "log_feed_ref": ("fz", REFERENCE_FEED_MM_TOOTH),
            "log_depth_ref": ("ap", REFERENCE_DEPTH_MM),
        },
        unit="min",
    )
    observed_equation_text = equation_string(
        observed_eq,
        {
            "log_speed_ref": ("n", REFERENCE_SPEED_RPM),
            "log_feed_ref": ("fz", REFERENCE_FEED_MM_TOOTH),
            "log_depth_ref": ("ap", REFERENCE_DEPTH_MM),
        },
        unit="min",
    )

    lines = [
        "# Phase 2 Lifetime Model Extraction Report",
        "",
        f"Generated: {generated_at}",
        "",
        "## What Changed Compared With the Previous Pass",
        "",
        "This run trains dedicated lifetime models. The previous extension quantified uncertainty around the analytical wear equation; this one builds threshold-crossing labels, trains wear-trajectory models, inverts the trained models into tool-life surfaces, and fits a survival-style accelerated-failure-time (AFT) equation that keeps censored runs instead of dropping them.",
        "",
        "## Lifetime Labels",
        "",
        markdown_table(
            pd.DataFrame(
                [
                    {
                        "dataset": "NUAA",
                        "records": len(nuaa_life),
                        "threshold": NUAA_LIFE_THRESHOLD_MM,
                        "events": int(nuaa_life["event"].sum()),
                        "censored": int((1 - nuaa_life["event"]).sum()),
                    },
                    {
                        "dataset": "NASA",
                        "records": len(nasa_life),
                        "threshold": NASA_LIFE_THRESHOLD_MM,
                        "events": int(nasa_life["event"].sum()),
                        "censored": int((1 - nasa_life["event"]).sum()),
                    },
                    {
                        "dataset": "PHM2010",
                        "records": len(phm_life),
                        "threshold": PHM_LIFE_THRESHOLD_MM,
                        "events": int(phm_life["event"].sum()),
                        "censored": int((1 - phm_life["event"]).sum()),
                    },
                ]
            ),
            ["dataset", "records", "threshold", "events", "censored"],
            digits=3,
        ),
        "",
        "NUAA and NASA are modeled in minutes. PHM2010 is recorded in cut index under fixed operating conditions, so it is kept as a reference life-label set rather than merged into the parameterized milling equation.",
        "",
        "## Trained Wear Models and Inverted Tool Life",
        "",
        markdown_table(
            wear_metrics,
            [
                "model",
                "wear_rmse_mm",
                "wear_mae_mm",
                "wear_r2",
                "event_life_mae_min",
                "censor_bound_satisfied_rate",
            ],
            digits=4,
        ),
        "",
        f"Best pointwise wear model by RMSE: `{best_wear_rmse}`.",
        "",
        f"Best overall event-life predictor in this run: `{overall_best_life}`.",
        "",
        f"Selected trained ML surface model for equation extraction: `{best_model_name}`. This is the best trained ML model by event-life MAE, but it should not be confused with the best overall lifetime predictor.",
        "",
        "Plot: `phase -2/plots/lifetime_modeling/nuaa_wear_life_model_benchmark.png`",
        "",
        "Interpretation: optimizing pointwise wear RMSE and optimizing threshold-crossing life are not the same objective. GradientBoosting has the lowest leave-one-experiment-out wear RMSE, but its threshold-crossing inversion is worse than the simpler Ridge model and the existing analytical equation.",
        "",
        "## Extracted Lifetime Equations",
        "",
        "### 1. Censored AFT Equation From NUAA Threshold-Crossing Labels",
        "",
        f"`{aft_equation}`",
        "",
        f"AFT sigma: `{aft_fit.sigma:.4f}`, negative log-likelihood: `{aft_fit.nll:.4f}`.",
        "",
        "Coefficient bootstrap summary:",
        "",
        markdown_table(aft_boot_summary, ["term", "count", "median", "q025", "q975"], digits=4),
        "",
        "Plot: `phase -2/plots/lifetime_modeling/nuaa_aft_coefficient_bootstrap.png`",
        "",
        "### 2. ML-Inverted Life Surface Equation",
        "",
        f"`{ml_equation_text}`",
        "",
        f"This log-linear surrogate explains the selected ML model's inverted life surface with log-space R2 `{ml_equation_metrics['log_r2']:.4f}` over `{int(ml_equation_metrics['surface_event_records'])}` crossing scenarios.",
        "",
        "The large speed exponent in this surrogate should be treated as a warning rather than a universal machining constant. The NUAA speed range is only 1750-1850 rpm, and the orthogonal grid creates collinearity between speed and the other factors.",
        "",
        "Plot: `phase -2/plots/lifetime_modeling/ml_life_surface_heatmap.png`",
        "",
        "### 3. Direct Observed-Life Equation",
        "",
        f"`{observed_equation_text}`",
        "",
        f"This equation is fit only to uncensored NUAA observed crossings and is therefore less robust. Log RMSE: `{observed_eq_metrics['log_rmse']:.4f}`.",
        "",
        "Plot: `phase -2/plots/lifetime_modeling/nuaa_life_equation_comparison.png`",
        "",
        "## NASA Material-Aware Lifetime Model",
        "",
        "NASA is kept separate because its feed variable is feed per revolution, its speed is fixed, and it uses cast iron/steel material families. The AFT model therefore uses feed, depth, material family, and initial wear.",
        "",
        markdown_table(nasa_aft.coefficients, ["term", "coefficient", "exp_coefficient"], digits=4),
        "",
        markdown_table(nasa_aft_metrics, ["metric", "value"], digits=4),
        "",
        "Plot: `phase -2/plots/lifetime_modeling/nasa_aft_life_model.png`",
        "",
        "## Literature and Method Transfer Notes",
        "",
        markdown_table(literature[["source", "domain", "method", "usable_transfer"]], ["source", "domain", "method", "usable_transfer"], digits=3),
        "",
        "## Research Conclusions",
        "",
        "1. Yes: this pass trains models and extracts tool-life equations from those trained models.",
        "2. The most defensible lifetime equation is the censored AFT equation because it uses threshold-crossing labels and retains censored NUAA runs.",
        "3. The ML-inverted equation is useful as a response-surface surrogate, but it is still only valid inside the NUAA parameter box: 1750-1850 rpm, 0.045-0.055 mm/tooth, 2.5-3.5 mm depth.",
        "4. The survival-analysis pivot from biology is directly useful: tool life is a time-to-event problem with censoring, so Cox/AFT/DeepSurv-style thinking is more appropriate than plain regression alone.",
        "5. Deep models should not be oversold on the current local lifetime labels. The right next move is external validation on QIT-CEMC or the 2025 Piecuch-Zabinski dataset, where many more full-life tools are available.",
        "",
        "## Generated Files",
        "",
        "- `phase -2/outputs/lifetime_modeling/nuaa_life_records.csv`",
        "- `phase -2/outputs/lifetime_modeling/nasa_life_records.csv`",
        "- `phase -2/outputs/lifetime_modeling/phm_life_records.csv`",
        "- `phase -2/outputs/lifetime_modeling/nuaa_wear_model_metrics.csv`",
        "- `phase -2/outputs/lifetime_modeling/nuaa_life_model_predictions.csv`",
        "- `phase -2/outputs/lifetime_modeling/nuaa_aft_coefficients.csv`",
        "- `phase -2/outputs/lifetime_modeling/nuaa_aft_bootstrap_summary.csv`",
        "- `phase -2/outputs/lifetime_modeling/ml_extracted_life_equation.csv`",
        "- `phase -2/outputs/lifetime_modeling/ml_life_surface.csv`",
        "- `phase -2/outputs/lifetime_modeling/nasa_aft_coefficients.csv`",
        "- `phase -2/outputs/lifetime_modeling/literature_method_transfer_matrix.csv`",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_log(best_model_name: str, wear_metrics: pd.DataFrame, aft_fit: AFTFit) -> None:
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    best_row = wear_metrics[wear_metrics["model"] == best_model_name].iloc[0]
    lines = [
        "# Lifetime Model Extraction Run Log",
        "",
        f"Run timestamp: {generated_at}",
        "",
        "## Command",
        "",
        "```powershell",
        '.\\.venv\\Scripts\\python.exe "phase -2\\lifetime_model_extraction.py"',
        "```",
        "",
        "## Actions",
        "",
        "- Rebuilt/loaded Phase 2 model context.",
        "- Built threshold-crossing life labels for NUAA, NASA, and PHM2010.",
        "- Trained NUAA wear trajectory models and inverted them into lifetime estimates.",
        "- Fit a censored log-normal AFT lifetime equation for NUAA.",
        "- Fit a NASA material-aware AFT lifetime model.",
        "- Extracted a log-linear Taylor-like equation from the selected trained ML life surface.",
        "- Wrote literature transfer notes covering machining, PHM, and survival-analysis methods.",
        "",
        "## Recorded Findings",
        "",
        f"- Selected ML model: {best_model_name}.",
        f"- Selected model wear RMSE: {best_row['wear_rmse_mm']:.5f} mm.",
        f"- Selected model event-life MAE: {best_row['event_life_mae_min']:.5f} min.",
        f"- NUAA AFT sigma: {aft_fit.sigma:.5f}.",
        "",
        "Git integrity note: this log uses the actual run timestamp.",
    ]
    RUN_LOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    if not (OUTPUTS / "phase2_model_summary.json").exists():
        research_pipeline.main()

    nuaa_runs = load_nuaa_run_level()
    simulator = MillingToolLifeModel.from_json()
    nuaa_life, nasa_life, phm_life = build_life_records()
    nuaa_life.to_csv(LIFE_OUTPUTS / "nuaa_life_records.csv", index=False)
    nasa_life.to_csv(LIFE_OUTPUTS / "nasa_life_records.csv", index=False)
    phm_life.to_csv(LIFE_OUTPUTS / "phm_life_records.csv", index=False)

    wear_metrics, wear_predictions, life_predictions = evaluate_nuaa_wear_models(
        nuaa_runs=nuaa_runs,
        nuaa_life=nuaa_life,
        simulator=simulator,
    )
    wear_metrics.to_csv(LIFE_OUTPUTS / "nuaa_wear_model_metrics.csv", index=False)
    wear_predictions.to_csv(LIFE_OUTPUTS / "nuaa_wear_model_predictions.csv", index=False)
    life_predictions.to_csv(LIFE_OUTPUTS / "nuaa_life_model_predictions.csv", index=False)

    ml_candidates = wear_metrics[wear_metrics["model"] != "Phase2Equation"].copy()
    best_model_name = str(
        ml_candidates.sort_values(["event_life_mae_min", "wear_rmse_mm"]).iloc[0]["model"]
    )

    aft_fit = fit_lognormal_aft(
        nuaa_life,
        feature_cols=["log_speed_ref", "log_feed_ref", "log_depth_ref"],
    )
    aft_fit.coefficients.to_csv(LIFE_OUTPUTS / "nuaa_aft_coefficients.csv", index=False)
    aft_boot = bootstrap_aft(
        nuaa_life,
        feature_cols=["log_speed_ref", "log_feed_ref", "log_depth_ref"],
    )
    aft_boot.to_csv(LIFE_OUTPUTS / "nuaa_aft_bootstrap.csv", index=False)
    aft_boot_summary = []
    for term in ["intercept", "log_speed_ref", "log_feed_ref", "log_depth_ref", "sigma"]:
        values = aft_boot[term].dropna().to_numpy(dtype=float)
        aft_boot_summary.append(
            {
                "term": term,
                "count": int(len(values)),
                "median": float(np.quantile(values, 0.5)),
                "q025": float(np.quantile(values, 0.025)),
                "q975": float(np.quantile(values, 0.975)),
            }
        )
    aft_boot_summary_df = pd.DataFrame(aft_boot_summary)
    aft_boot_summary_df.to_csv(LIFE_OUTPUTS / "nuaa_aft_bootstrap_summary.csv", index=False)

    surface, ml_equation, ml_equation_metrics = extract_ml_life_surface(
        nuaa_runs=nuaa_runs,
        model_name=best_model_name,
    )
    surface.to_csv(LIFE_OUTPUTS / "ml_life_surface.csv", index=False)
    ml_equation.to_csv(LIFE_OUTPUTS / "ml_extracted_life_equation.csv", index=False)
    (LIFE_OUTPUTS / "ml_extracted_life_equation_metrics.json").write_text(
        json.dumps(ml_equation_metrics, indent=2),
        encoding="utf-8",
    )

    observed_event_life = nuaa_life[nuaa_life["event"] == 1].copy()
    observed_eq, observed_eq_metrics = fit_life_equation_ols(
        observed_event_life,
        feature_cols=["log_speed_ref", "log_feed_ref", "log_depth_ref"],
        target_col="life_min",
    )
    observed_eq.to_csv(LIFE_OUTPUTS / "nuaa_observed_life_equation.csv", index=False)
    (LIFE_OUTPUTS / "nuaa_observed_life_equation_metrics.json").write_text(
        json.dumps(observed_eq_metrics, indent=2),
        encoding="utf-8",
    )

    comparison_rows = []
    aft_pred = predict_aft_median(nuaa_life, aft_fit)
    for model_name, predictions in [
        ("CensoredAFT", aft_pred),
    ]:
        for (_, row), pred in zip(nuaa_life.iterrows(), predictions):
            comparison_rows.append(
                {
                    "model": model_name,
                    "series_id": row["series_id"],
                    "observed_life_min": float(row["life_min"]),
                    "observed_event": int(row["event"]),
                    "predicted_life_min": float(pred),
                }
            )
    life_prediction_wide = life_predictions[life_predictions["model"].isin(["Phase2Equation", best_model_name])]
    for _, row in life_prediction_wide.iterrows():
        comparison_rows.append(
            {
                "model": str(row["model"]),
                "series_id": str(row["series_id"]),
                "observed_life_min": float(row["observed_life_min"]),
                "observed_event": int(row["observed_event"]),
                "predicted_life_min": float(row["predicted_life_min"]),
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(LIFE_OUTPUTS / "nuaa_life_equation_comparison.csv", index=False)

    nasa_aft = fit_lognormal_aft(
        nasa_life,
        feature_cols=["log_feed_ref", "log_depth_ref", "material_steel", "initial_wear_mm"],
        l2=0.25,
    )
    nasa_aft.coefficients.to_csv(LIFE_OUTPUTS / "nasa_aft_coefficients.csv", index=False)
    nasa_pred = nasa_life[["series_id"]].copy()
    nasa_pred["predicted_life_min"] = predict_aft_median(nasa_life, nasa_aft)
    nasa_pred.to_csv(LIFE_OUTPUTS / "nasa_aft_predictions.csv", index=False)
    nasa_merged = nasa_life.merge(nasa_pred, on="series_id")
    nasa_event = nasa_merged[nasa_merged["event"] == 1]
    nasa_aft_metrics = pd.DataFrame(
        [
            {"metric": "event_mae_min", "value": float(mean_absolute_error(nasa_event["life_min"], nasa_event["predicted_life_min"]))},
            {"metric": "event_rmse_min", "value": float(np.sqrt(mean_squared_error(nasa_event["life_min"], nasa_event["predicted_life_min"])))},
            {"metric": "event_r2", "value": float(r2_score(nasa_event["life_min"], nasa_event["predicted_life_min"]))},
            {"metric": "sigma", "value": float(nasa_aft.sigma)},
            {"metric": "negative_log_likelihood", "value": float(nasa_aft.nll)},
        ]
    )
    nasa_aft_metrics.to_csv(LIFE_OUTPUTS / "nasa_aft_metrics.csv", index=False)

    literature = build_literature_matrix()
    literature.to_csv(LIFE_OUTPUTS / "literature_method_transfer_matrix.csv", index=False)

    plot_wear_benchmark(wear_metrics, LIFE_PLOTS / "nuaa_wear_life_model_benchmark.png")
    plot_life_equation_comparison(
        nuaa_life=nuaa_life,
        comparison=comparison,
        output_path=LIFE_PLOTS / "nuaa_life_equation_comparison.png",
    )
    plot_life_surface(surface, LIFE_PLOTS / "ml_life_surface_heatmap.png")
    plot_aft_bootstrap(aft_boot, LIFE_PLOTS / "nuaa_aft_coefficient_bootstrap.png")
    plot_nasa_aft(nasa_life, nasa_pred, LIFE_PLOTS / "nasa_aft_life_model.png")

    write_report(
        nuaa_life=nuaa_life,
        nasa_life=nasa_life,
        phm_life=phm_life,
        wear_metrics=wear_metrics,
        best_model_name=best_model_name,
        aft_fit=aft_fit,
        aft_boot_summary=aft_boot_summary_df,
        ml_equation=ml_equation,
        ml_equation_metrics=ml_equation_metrics,
        observed_eq=observed_eq,
        observed_eq_metrics=observed_eq_metrics,
        nasa_aft=nasa_aft,
        nasa_aft_metrics=nasa_aft_metrics,
        literature=literature,
    )
    write_run_log(best_model_name=best_model_name, wear_metrics=wear_metrics, aft_fit=aft_fit)


if __name__ == "__main__":
    main()
