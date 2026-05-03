from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import research_pipeline
from research_pipeline import (
    EARLY_EXPONENT_SELECTED,
    LATE_EXPONENT_SELECTED,
    OUTPUTS,
    PHASE2,
    PLOTS,
    TRANSITION_WEAR_MM,
    fit_common_power,
    load_nasa_case_level,
    load_nuaa_run_level,
    load_phm_cut_level,
)
from tool_life_simulator import MillingToolLifeModel


EXT_OUTPUTS = OUTPUTS / "extended_research"
EXT_PLOTS = PLOTS / "extended_research"
REPORT_PATH = EXT_OUTPUTS / "phase2_extended_research_report.md"
RUN_RECORD_PATH = EXT_OUTPUTS / "extension_run_record.md"

COEFF_COLUMNS = [
    "k_early",
    "speed_exponent",
    "feed_exponent",
    "depth_exponent",
    "amplitude_r2",
]

SCENARIOS = [
    {
        "scenario": "low_load",
        "label": "Low load",
        "speed_rpm": 1750.0,
        "feed_mm_tooth": 0.045,
        "depth_mm": 2.5,
    },
    {
        "scenario": "baseline",
        "label": "Baseline",
        "speed_rpm": 1800.0,
        "feed_mm_tooth": 0.050,
        "depth_mm": 3.0,
    },
    {
        "scenario": "high_load",
        "label": "High load",
        "speed_rpm": 1850.0,
        "feed_mm_tooth": 0.055,
        "depth_mm": 3.5,
    },
]


def ensure_dirs() -> None:
    EXT_OUTPUTS.mkdir(parents=True, exist_ok=True)
    EXT_PLOTS.mkdir(parents=True, exist_ok=True)


def quantile_summary(df: pd.DataFrame, columns: list[str], label_col: str = "parameter") -> pd.DataFrame:
    rows = []
    for col in columns:
        values = df[col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        rows.append(
            {
                label_col: col,
                "count": int(len(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "q025": float(np.quantile(values, 0.025)),
                "q250": float(np.quantile(values, 0.25)),
                "median": float(np.quantile(values, 0.50)),
                "q750": float(np.quantile(values, 0.75)),
                "q975": float(np.quantile(values, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_condition_law(
    nuaa_runs: pd.DataFrame,
    n_boot: int = 500,
    seed: int = 20260503,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = list(pd.Series(nuaa_runs["experiment_tag"].unique()).astype(str))
    rows = []
    attempts = 0
    max_attempts = n_boot * 30

    while len(rows) < n_boot and attempts < max_attempts:
        attempts += 1
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        boot_frames = []
        for draw_id, group in enumerate(sampled_groups):
            part = nuaa_runs[nuaa_runs["experiment_tag"].astype(str) == group].copy()
            part["experiment_tag"] = f"{group}_boot_{draw_id}"
            boot_frames.append(part)

        boot_df = pd.concat(boot_frames, ignore_index=True)
        condition_df = (
            boot_df.groupby("experiment_tag", as_index=False)
            .agg(
                speed_rpm=("speed_rpm", "first"),
                feed_mm_tooth=("feed_mm_tooth", "first"),
                depth_mm=("depth_mm", "first"),
            )
            .drop_duplicates(["speed_rpm", "feed_mm_tooth", "depth_mm"])
        )
        x = np.log(
            condition_df[["speed_rpm", "feed_mm_tooth", "depth_mm"]].to_numpy(dtype=float)
            / np.array([1800.0, 0.05, 3.0])
        )
        design = np.c_[np.ones(len(x)), x]
        if len(condition_df) < 5 or np.linalg.matrix_rank(design) < 4:
            continue

        x_std = np.std(x, axis=0)
        if np.any(x_std <= 1e-12):
            continue
        scaled_design = np.c_[np.ones(len(x)), (x - np.mean(x, axis=0)) / x_std]
        if np.linalg.cond(scaled_design) > 30.0:
            continue

        try:
            coeffs, _ = research_pipeline.estimate_nuaa_condition_law(
                nuaa_runs=boot_df,
                exponent=EARLY_EXPONENT_SELECTED,
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue

        row = {"bootstrap_id": len(rows), "attempt_id": attempts}
        row.update({key: float(coeffs[key]) for key in COEFF_COLUMNS})
        rows.append(row)

    if len(rows) < n_boot:
        raise RuntimeError(f"Only {len(rows)} valid bootstrap draws were produced from {attempts} attempts.")

    return pd.DataFrame(rows)


def model_with_coefficients(model_summary: dict[str, object], row: pd.Series) -> MillingToolLifeModel:
    sample = copy.deepcopy(model_summary)
    coeffs = sample["selected_coefficients"]  # type: ignore[index]
    for col in COEFF_COLUMNS:
        if col == "amplitude_r2":
            continue
        coeffs[col] = float(row[col])  # type: ignore[index]
    return MillingToolLifeModel(sample)


def simulate_life_uncertainty(
    model_summary: dict[str, object],
    condition_bootstrap: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for scenario in SCENARIOS:
        for _, sample in condition_bootstrap.iterrows():
            simulator = model_with_coefficients(model_summary, sample)
            life_min = simulator.life_to_threshold_minutes(
                speed_rpm=float(scenario["speed_rpm"]),
                feed_mm_tooth=float(scenario["feed_mm_tooth"]),
                depth_mm=float(scenario["depth_mm"]),
                threshold_wear_mm=0.30,
                initial_wear_mm=0.05,
                calibration_factor=1.0,
                material_family="generic",
            )
            if np.isfinite(life_min) and life_min >= 0.0:
                rows.append(
                    {
                        "scenario": scenario["scenario"],
                        "label": scenario["label"],
                        "speed_rpm": scenario["speed_rpm"],
                        "feed_mm_tooth": scenario["feed_mm_tooth"],
                        "depth_mm": scenario["depth_mm"],
                        "bootstrap_id": int(sample["bootstrap_id"]),
                        "life_min": float(life_min),
                    }
                )

    draws = pd.DataFrame(rows)
    summaries = []
    for scenario, group_df in draws.groupby("scenario", sort=False):
        values = group_df["life_min"].to_numpy(dtype=float)
        meta = group_df.iloc[0]
        summaries.append(
            {
                "scenario": scenario,
                "label": meta["label"],
                "speed_rpm": float(meta["speed_rpm"]),
                "feed_mm_tooth": float(meta["feed_mm_tooth"]),
                "depth_mm": float(meta["depth_mm"]),
                "draws": int(len(values)),
                "mean_life_min": float(np.mean(values)),
                "q025_life_min": float(np.quantile(values, 0.025)),
                "q250_life_min": float(np.quantile(values, 0.25)),
                "median_life_min": float(np.quantile(values, 0.50)),
                "q750_life_min": float(np.quantile(values, 0.75)),
                "q975_life_min": float(np.quantile(values, 0.975)),
            }
        )

    return draws, pd.DataFrame(summaries)


def build_calibration_holdout(
    nuaa_runs: pd.DataFrame,
    simulator: MillingToolLifeModel,
    calibration_fractions: tuple[float, ...] = (0.25, 0.40, 0.60),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows = []
    metric_rows = []

    for experiment_tag, group_df in nuaa_runs.groupby("experiment_tag", sort=True):
        group_df = group_df.sort_values("relative_time_min").copy()
        initial_wear = float(group_df["initial_wear_mm"].iloc[0])
        speed = float(group_df["speed_rpm"].iloc[0])
        feed = float(group_df["feed_mm_tooth"].iloc[0])
        depth = float(group_df["depth_mm"].iloc[0])
        max_time = float(group_df["relative_time_min"].max())

        for fraction in calibration_fractions:
            target_time = fraction * max_time
            idx = (group_df["relative_time_min"] - target_time).abs().idxmin()
            calibration_row = group_df.loc[idx]
            calibration_time = float(calibration_row["relative_time_min"])
            calibration_wear = float(calibration_row["wear_mm"])

            if calibration_time <= 0.0 or calibration_wear >= TRANSITION_WEAR_MM:
                continue

            calibration_window = group_df[
                (group_df["relative_time_min"] > 0.0)
                & (group_df["relative_time_min"] <= calibration_time)
                & (group_df["wear_mm"] < TRANSITION_WEAR_MM)
            ].copy()
            if len(calibration_window) < 2:
                continue

            base_amp = simulator.early_amplitude(
                speed_rpm=speed,
                feed_mm_tooth=feed,
                depth_mm=depth,
                calibration_factor=1.0,
            )
            early_exp = float(simulator.coeffs["early_exponent"])
            basis = base_amp * np.power(
                calibration_window["relative_time_min"].to_numpy(dtype=float),
                early_exp,
            )
            target = calibration_window["wear_mm"].to_numpy(dtype=float) - initial_wear
            denom = float(np.dot(basis, basis))
            if denom <= 0.0:
                continue

            calibration_factor = float(np.dot(basis, target) / denom)
            if not np.isfinite(calibration_factor) or calibration_factor <= 0.0:
                continue

            scored = group_df[group_df["relative_time_min"] >= calibration_time].copy()
            calibrated = []
            uncalibrated = []
            for _, row in scored.iterrows():
                t = float(row["relative_time_min"])
                calibrated.append(
                    simulator.wear_at_minutes(
                        speed_rpm=speed,
                        feed_mm_tooth=feed,
                        depth_mm=depth,
                        time_min=t,
                        initial_wear_mm=initial_wear,
                        calibration_factor=calibration_factor,
                        material_family="generic",
                    )
                )
                uncalibrated.append(
                    simulator.wear_at_minutes(
                        speed_rpm=speed,
                        feed_mm_tooth=feed,
                        depth_mm=depth,
                        time_min=t,
                        initial_wear_mm=initial_wear,
                        calibration_factor=1.0,
                        material_family="generic",
                    )
                )

            scored["predicted_wear_calibrated_mm"] = calibrated
            scored["predicted_wear_uncalibrated_mm"] = uncalibrated
            scored["calibration_fraction"] = fraction
            scored["calibration_time_min"] = calibration_time
            scored["calibration_wear_mm"] = calibration_wear
            scored["calibration_points"] = int(len(calibration_window))
            scored["calibration_factor"] = calibration_factor

            holdout = scored[scored["relative_time_min"] > calibration_time].copy()
            if len(holdout) == 0:
                continue

            obs = holdout["wear_mm"].to_numpy(dtype=float)
            pred_cal = holdout["predicted_wear_calibrated_mm"].to_numpy(dtype=float)
            pred_uncal = holdout["predicted_wear_uncalibrated_mm"].to_numpy(dtype=float)

            final_obs = float(holdout["wear_mm"].iloc[-1])
            final_cal = float(holdout["predicted_wear_calibrated_mm"].iloc[-1])
            final_uncal = float(holdout["predicted_wear_uncalibrated_mm"].iloc[-1])
            metric_rows.append(
                {
                    "experiment_tag": experiment_tag,
                    "calibration_fraction": fraction,
                    "calibration_time_min": calibration_time,
                    "calibration_wear_mm": calibration_wear,
                    "calibration_points": int(len(calibration_window)),
                    "calibration_factor": calibration_factor,
                    "holdout_points": int(len(holdout)),
                    "calibrated_rmse_mm": float(np.sqrt(mean_squared_error(obs, pred_cal))),
                    "uncalibrated_rmse_mm": float(np.sqrt(mean_squared_error(obs, pred_uncal))),
                    "calibrated_mae_mm": float(np.mean(np.abs(obs - pred_cal))),
                    "uncalibrated_mae_mm": float(np.mean(np.abs(obs - pred_uncal))),
                    "final_observed_wear_mm": final_obs,
                    "final_calibrated_wear_mm": final_cal,
                    "final_uncalibrated_wear_mm": final_uncal,
                    "final_calibrated_abs_error_mm": abs(final_obs - final_cal),
                    "final_uncalibrated_abs_error_mm": abs(final_obs - final_uncal),
                }
            )

            prediction_rows.append(
                scored[
                    [
                        "experiment_tag",
                        "relative_time_min",
                        "wear_mm",
                        "predicted_wear_calibrated_mm",
                        "predicted_wear_uncalibrated_mm",
                        "calibration_fraction",
                        "calibration_time_min",
                        "calibration_wear_mm",
                        "calibration_points",
                        "calibration_factor",
                        "speed_rpm",
                        "feed_mm_tooth",
                        "depth_mm",
                    ]
                ]
            )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    summary = (
        metrics.groupby("calibration_fraction", as_index=False)
        .agg(
            experiments=("experiment_tag", "count"),
            median_calibration_points=("calibration_points", "median"),
            median_calibration_factor=("calibration_factor", "median"),
            median_calibrated_rmse_mm=("calibrated_rmse_mm", "median"),
            median_uncalibrated_rmse_mm=("uncalibrated_rmse_mm", "median"),
            mean_calibrated_rmse_mm=("calibrated_rmse_mm", "mean"),
            mean_uncalibrated_rmse_mm=("uncalibrated_rmse_mm", "mean"),
            median_final_calibrated_abs_error_mm=("final_calibrated_abs_error_mm", "median"),
            median_final_uncalibrated_abs_error_mm=("final_uncalibrated_abs_error_mm", "median"),
        )
        .sort_values("calibration_fraction")
    )
    return metrics, predictions, summary


def build_residual_diagnostics(
    nuaa_runs: pd.DataFrame,
    phm: pd.DataFrame,
    nasa: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nuaa_fit = fit_common_power(
        df=nuaa_runs,
        group_col="experiment_tag",
        time_col="relative_time_min",
        wear_col="wear_mm",
    )
    phm_fit = fit_common_power(
        df=phm,
        group_col="experiment_tag",
        time_col="relative_cut",
        wear_col="wear_mm",
        initial_guess_log_a=-5.0,
        initial_guess_m=0.8,
    )
    nasa_fit = fit_common_power(
        df=nasa.rename(columns={"case": "case_id", "time": "time_min", "VB": "wear_mm"}),
        group_col="case_id",
        time_col="relative_time_min",
        wear_col="wear_mm",
        initial_guess_log_a=-3.0,
        initial_guess_m=1.1,
    )

    frames = []
    for dataset, fit, group_col, time_col in [
        ("NUAA", nuaa_fit, "experiment_tag", "relative_time_min"),
        ("PHM2010", phm_fit, "experiment_tag", "relative_cut"),
        ("NASA", nasa_fit, "case_id", "relative_time_min"),
    ]:
        frame = fit.predictions.copy()
        frame["dataset"] = dataset
        frame["series_id"] = frame[group_col].astype(str)
        frame["x"] = frame[time_col].to_numpy(dtype=float)
        frame["residual_mm"] = frame["wear_mm"] - frame["predicted_wear_mm"]
        frames.append(
            frame[
                [
                    "dataset",
                    "series_id",
                    "x",
                    "wear_mm",
                    "predicted_wear_mm",
                    "residual_mm",
                ]
            ]
        )

    residuals = pd.concat(frames, ignore_index=True)
    summary = (
        residuals.groupby("dataset", as_index=False)
        .agg(
            points=("residual_mm", "count"),
            bias_mm=("residual_mm", "mean"),
            mae_mm=("residual_mm", lambda s: float(np.mean(np.abs(s)))),
            rmse_mm=("residual_mm", lambda s: float(np.sqrt(np.mean(np.square(s))))),
            p95_abs_residual_mm=("residual_mm", lambda s: float(np.quantile(np.abs(s), 0.95))),
        )
        .sort_values("dataset")
    )
    return residuals, summary


def plot_condition_bootstrap(
    condition_bootstrap: pd.DataFrame,
    model_summary: dict[str, object],
    output_path: Path,
) -> None:
    coeffs = model_summary["selected_coefficients"]  # type: ignore[index]
    plot_specs = [
        ("k_early", "Early k coefficient"),
        ("speed_exponent", "Speed exponent"),
        ("feed_exponent", "Feed exponent"),
        ("depth_exponent", "Depth exponent"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (col, title) in zip(axes.ravel(), plot_specs):
        values = condition_bootstrap[col].replace([np.inf, -np.inf], np.nan).dropna()
        q025, median, q975 = np.quantile(values, [0.025, 0.50, 0.975])
        ax.hist(values, bins=32, color="#4E79A7", alpha=0.82)
        ax.axvspan(q025, q975, color="#F28E2B", alpha=0.18, label="95% interval")
        ax.axvline(float(coeffs[col]), color="black", linestyle="--", linewidth=1.4, label="selected")
        ax.axvline(median, color="#E15759", linewidth=1.2, label="bootstrap median")
        ax.set_title(title)
        ax.set_ylabel("Bootstrap draws")
        ax.grid(alpha=0.22, axis="y")

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("NUAA Condition-Law Coefficient Bootstrap", y=0.98, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_life_intervals(life_summary: pd.DataFrame, output_path: Path) -> None:
    ordered = life_summary.set_index("scenario").loc[[s["scenario"] for s in SCENARIOS]].reset_index()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    y = np.arange(len(ordered))
    ax.hlines(y, ordered["q025_life_min"], ordered["q975_life_min"], color="#4E79A7", linewidth=4, alpha=0.35)
    ax.hlines(y, ordered["q250_life_min"], ordered["q750_life_min"], color="#4E79A7", linewidth=9, alpha=0.50)
    ax.scatter(ordered["median_life_min"], y, color="#E15759", s=55, label="Median")
    ax.scatter(ordered["mean_life_min"], y, color="black", marker="x", s=55, label="Mean")
    ax.set_yticks(y, ordered["label"])
    ax.set_xlabel("Predicted life to 0.30 mm wear (min)")
    ax.set_title("Life Prediction Intervals from Coefficient Bootstrap")
    ax.grid(alpha=0.25, axis="x")
    spread = float(ordered["q975_life_min"].max() / max(ordered["q025_life_min"].min(), 1e-9))
    if spread > 8.0:
        ax.set_xscale("log")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_calibration_forecasts(predictions: pd.DataFrame, output_path: Path, fraction: float = 0.40) -> None:
    plot_df = predictions[np.isclose(predictions["calibration_fraction"], fraction)].copy()
    groups = list(plot_df["experiment_tag"].astype(str).unique())
    ncols = 3
    nrows = int(np.ceil(len(groups) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.1 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, experiment_tag in zip(axes_flat, groups):
        group_df = plot_df[plot_df["experiment_tag"].astype(str) == experiment_tag].sort_values("relative_time_min")
        ax.plot(group_df["relative_time_min"], group_df["wear_mm"], marker="o", ms=3, lw=1.4, label="Observed")
        ax.plot(
            group_df["relative_time_min"],
            group_df["predicted_wear_uncalibrated_mm"],
            linestyle="--",
            color="gray",
            lw=1.2,
            label="Uncalibrated",
        )
        ax.plot(
            group_df["relative_time_min"],
            group_df["predicted_wear_calibrated_mm"],
            color="#4E79A7",
            lw=1.5,
            label="Calibrated",
        )
        calibration_time = float(group_df["calibration_time_min"].iloc[0])
        calibration_factor = float(group_df["calibration_factor"].iloc[0])
        ax.axvline(calibration_time, color="#E15759", linestyle=":", linewidth=1)
        ax.axhline(TRANSITION_WEAR_MM, color="black", linestyle=":", linewidth=0.8, alpha=0.55)
        ax.set_title(f"{experiment_tag} lambda={calibration_factor:.2f}")
        ax.set_xlabel("Relative time (min)")
        ax.set_ylabel("Wear (mm)")
        ax.grid(alpha=0.25)

    for ax in axes_flat[len(groups) :]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(f"NUAA Forecast After {fraction:.0%} Early Calibration Point", y=0.98, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_residual_diagnostics(residuals: pd.DataFrame, output_path: Path) -> None:
    datasets = ["NUAA", "PHM2010", "NASA"]
    xlabels = {
        "NUAA": "Relative time (min)",
        "PHM2010": "Relative cut index",
        "NASA": "Relative time (min)",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, dataset in zip(axes, datasets):
        part = residuals[residuals["dataset"] == dataset]
        for series_id, group_df in part.groupby("series_id"):
            ax.scatter(group_df["x"], group_df["residual_mm"], s=16, alpha=0.72, label=series_id)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title(dataset)
        ax.set_xlabel(xlabels[dataset])
        ax.set_ylabel("Observed - predicted wear (mm)")
        ax.grid(alpha=0.22)
    fig.suptitle("Residual Diagnostics for Core Phase-2 Fits", y=0.98, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def fmt(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def markdown_table(df: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                values.append(fmt(float(value), digits))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    condition_summary: pd.DataFrame,
    life_summary: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    residual_summary: pd.DataFrame,
) -> None:
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        "# Phase 2 Extended Research: Uncertainty, Calibration, and Residual Diagnostics",
        "",
        f"Generated: {generated_at}",
        "",
        "## Research Purpose",
        "",
        "The original Phase 2 pipeline established a piecewise milling wear law. This extension asks whether the equation is usable as an engineering simulator: how uncertain are the fitted condition-law coefficients, how much does that uncertainty move predicted tool life, and whether an early calibration window improves a future-wear forecast.",
        "",
        "## New Analyses Added",
        "",
        "- NUAA condition-law bootstrap: resamples the nine NUAA operating-condition trajectories and refits k, speed, feed, and depth exponents.",
        "- Life uncertainty propagation: pushes each bootstrap coefficient draw through the simulator for low, baseline, and high load settings.",
        "- Early calibration holdout: estimates the simulator calibration factor from all available sub-transition calibration points up to a cutoff, then scores future points in the same NUAA run.",
        "- Residual diagnostics: records observed minus predicted wear for the core NUAA, PHM2010, and NASA fits.",
        "",
        "## Coefficient Stability",
        "",
        markdown_table(
            condition_summary,
            ["parameter", "count", "mean", "std", "q025", "median", "q975"],
            digits=5,
        ),
        "",
        "Plot: `phase -2/plots/extended_research/condition_coefficient_bootstrap.png`",
        "",
        "Interpretation: with only nine NUAA condition trajectories, the feed exponent remains the dominant driver, but the bootstrap interval is wide. This means the simulator should expose calibration rather than presenting a single coefficient set as universal.",
        "",
        "## Tool-Life Prediction Intervals",
        "",
        markdown_table(
            life_summary,
            [
                "label",
                "speed_rpm",
                "feed_mm_tooth",
                "depth_mm",
                "median_life_min",
                "q025_life_min",
                "q975_life_min",
            ],
            digits=3,
        ),
        "",
        "Plot: `phase -2/plots/extended_research/life_prediction_intervals.png`",
        "",
        "Interpretation: the high-load setting remains consistently short lived, while the low-load setting has the widest absolute interval because small coefficient shifts compound over a longer forecast horizon.",
        "",
        "## Early Calibration Holdout",
        "",
        markdown_table(
            calibration_summary,
            [
                "calibration_fraction",
                "experiments",
                "median_calibration_points",
                "median_calibration_factor",
                "median_calibrated_rmse_mm",
                "median_uncalibrated_rmse_mm",
                "median_final_calibrated_abs_error_mm",
                "median_final_uncalibrated_abs_error_mm",
            ],
            digits=5,
        ),
        "",
        "Plot: `phase -2/plots/extended_research/nuaa_calibration_forecast.png`",
        "",
        "Interpretation: on the NUAA holdouts, per-run early calibration does not beat the global uncalibrated equation. That is a useful negative result: because the base equation is already fitted on NUAA, short-window lambda estimates tend to overcorrect. For a new tool-workpiece family, lambda should still be estimated, but it should be validated against held-out later wear rather than assumed beneficial.",
        "",
        "## Residual Diagnostics",
        "",
        markdown_table(
            residual_summary,
            ["dataset", "points", "bias_mm", "mae_mm", "rmse_mm", "p95_abs_residual_mm"],
            digits=5,
        ),
        "",
        "Plot: `phase -2/plots/extended_research/residual_diagnostics.png`",
        "",
        "Interpretation: residuals are smallest on PHM2010 and larger on NASA, which matches the research design: NASA is being used for late-stage behavior and material ratios, not as a directly transferable absolute wear-rate source.",
        "",
        "## Literature Extension",
        "",
        "- Li et al. introduced QIT-CEMC, a 2025 full-life titanium end-milling dataset with vibration, sound, cutting force, torque, wear images, and measured wear values. It is a useful next external validation target because it contains force/torque and full lifecycle wear rather than only early trajectories. Source: https://www.nature.com/articles/s41597-024-04345-2",
        "- Piecuch and Zabinski released a 2025 CNC milling dataset with 14 tools from initial condition until failure, 968 milling cycles, raw vibration/current signals, aggregated features, and metadata. Their usage notes recommend tool-wise group cross-validation, which matches the grouped validation philosophy used here. Source: https://www.nature.com/articles/s41597-025-04923-y",
        "- The PHM competition review identifies PHM2010 milling cutter wear as a regression benchmark using force and acoustic-emission signals, supporting this repo's decision to use PHM2010 as a cross-check rather than the only source of the life equation. Source: https://papers.phmsociety.org/index.php/phmconf/article/download/462/phmc_18_462",
        "",
        "## Research Conclusions",
        "",
        "1. The piecewise law is a defensible simulator structure, but coefficient uncertainty is not negligible.",
        "2. Feed per tooth remains the strongest observed lever in the NUAA condition law.",
        "3. Absolute life estimates still need tool-workpiece-specific validation before operational use; this holdout shows that short early-window lambda calibration can worsen forecasts when the global equation is already well matched.",
        "4. The next best data extension is not another PHM2010-only model; it is external full-life validation on QIT-CEMC or the 2025 Piecuch-Zabinski tool-failure dataset.",
        "",
        "## Generated Artifacts",
        "",
        "- `phase -2/outputs/extended_research/condition_law_bootstrap.csv`",
        "- `phase -2/outputs/extended_research/condition_law_bootstrap_summary.csv`",
        "- `phase -2/outputs/extended_research/life_uncertainty_draws.csv`",
        "- `phase -2/outputs/extended_research/life_uncertainty_summary.csv`",
        "- `phase -2/outputs/extended_research/nuaa_calibration_holdout_metrics.csv`",
        "- `phase -2/outputs/extended_research/nuaa_calibration_holdout_predictions.csv`",
        "- `phase -2/outputs/extended_research/nuaa_calibration_summary.csv`",
        "- `phase -2/outputs/extended_research/residual_diagnostics.csv`",
        "- `phase -2/outputs/extended_research/residual_summary.csv`",
        "- `phase -2/outputs/extended_research/phase2_extended_research_report.md`",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_record(
    condition_bootstrap: pd.DataFrame,
    life_summary: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    residual_summary: pd.DataFrame,
) -> None:
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    best_calibration = calibration_summary.sort_values("median_calibrated_rmse_mm").head(1)
    best_calibration_text = "NA"
    if not best_calibration.empty:
        row = best_calibration.iloc[0]
        best_calibration_text = (
            f"{row['calibration_fraction']:.2f} horizon fraction, "
            f"median calibrated RMSE {row['median_calibrated_rmse_mm']:.5f} mm"
        )
    beats_uncalibrated = (
        calibration_summary["median_calibrated_rmse_mm"]
        < calibration_summary["median_uncalibrated_rmse_mm"]
    ).any()
    calibration_comparison = (
        "At least one tested calibration cutoff beat the uncalibrated median RMSE."
        if beats_uncalibrated
        else "No tested calibration cutoff beat the uncalibrated median RMSE on NUAA holdout points."
    )

    lines = [
        "# Phase 2 Extension Run Record",
        "",
        f"Run timestamp: {generated_at}",
        "",
        "## Commands",
        "",
        "```powershell",
        '.\\.venv\\Scripts\\python.exe "phase -2\\extended_research_analysis.py"',
        "```",
        "",
        "## What This Run Did",
        "",
        "- Rebuilt the core Phase 2 outputs by calling `research_pipeline.main()`.",
        "- Loaded NUAA, PHM2010, and NASA milling wear tables from the local workspace.",
        f"- Resampled the NUAA condition law {len(condition_bootstrap)} times.",
        "- Propagated condition-law uncertainty through the tool-life simulator.",
        "- Ran NUAA multi-point early calibration holdout forecasts.",
        "- Recomputed residual diagnostics for the core dataset fits.",
        "",
        "## Key Recorded Findings",
        "",
        f"- Best calibration cutoff in this run: {best_calibration_text}.",
        f"- Calibration comparison: {calibration_comparison}",
        f"- Baseline median life from coefficient bootstrap: {life_summary.loc[life_summary['scenario'] == 'baseline', 'median_life_min'].iloc[0]:.2f} min.",
        f"- Largest residual RMSE dataset: {residual_summary.sort_values('rmse_mm', ascending=False).iloc[0]['dataset']}.",
        "",
        "## Git Integrity Note",
        "",
        "The research record is written with the actual run timestamp. Git commits should be made with truthful metadata; this run does not backdate research activity.",
    ]
    RUN_RECORD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    research_pipeline.main()

    nuaa_runs = load_nuaa_run_level()
    phm = load_phm_cut_level()
    nasa = load_nasa_case_level()
    model_summary = json.loads((OUTPUTS / "phase2_model_summary.json").read_text(encoding="utf-8"))
    simulator = MillingToolLifeModel(model_summary)

    condition_bootstrap = bootstrap_condition_law(nuaa_runs=nuaa_runs)
    condition_summary = quantile_summary(condition_bootstrap, COEFF_COLUMNS)
    condition_bootstrap.to_csv(EXT_OUTPUTS / "condition_law_bootstrap.csv", index=False)
    condition_summary.to_csv(EXT_OUTPUTS / "condition_law_bootstrap_summary.csv", index=False)

    life_draws, life_summary = simulate_life_uncertainty(
        model_summary=model_summary,
        condition_bootstrap=condition_bootstrap,
    )
    life_draws.to_csv(EXT_OUTPUTS / "life_uncertainty_draws.csv", index=False)
    life_summary.to_csv(EXT_OUTPUTS / "life_uncertainty_summary.csv", index=False)

    calibration_metrics, calibration_predictions, calibration_summary = build_calibration_holdout(
        nuaa_runs=nuaa_runs,
        simulator=simulator,
    )
    calibration_metrics.to_csv(EXT_OUTPUTS / "nuaa_calibration_holdout_metrics.csv", index=False)
    calibration_predictions.to_csv(EXT_OUTPUTS / "nuaa_calibration_holdout_predictions.csv", index=False)
    calibration_summary.to_csv(EXT_OUTPUTS / "nuaa_calibration_summary.csv", index=False)

    residuals, residual_summary = build_residual_diagnostics(nuaa_runs=nuaa_runs, phm=phm, nasa=nasa)
    residuals.to_csv(EXT_OUTPUTS / "residual_diagnostics.csv", index=False)
    residual_summary.to_csv(EXT_OUTPUTS / "residual_summary.csv", index=False)

    plot_condition_bootstrap(
        condition_bootstrap=condition_bootstrap,
        model_summary=model_summary,
        output_path=EXT_PLOTS / "condition_coefficient_bootstrap.png",
    )
    plot_life_intervals(life_summary=life_summary, output_path=EXT_PLOTS / "life_prediction_intervals.png")
    plot_calibration_forecasts(
        predictions=calibration_predictions,
        output_path=EXT_PLOTS / "nuaa_calibration_forecast.png",
    )
    plot_residual_diagnostics(residuals=residuals, output_path=EXT_PLOTS / "residual_diagnostics.png")

    write_report(
        condition_summary=condition_summary,
        life_summary=life_summary,
        calibration_summary=calibration_summary,
        residual_summary=residual_summary,
    )
    write_run_record(
        condition_bootstrap=condition_bootstrap,
        life_summary=life_summary,
        calibration_summary=calibration_summary,
        residual_summary=residual_summary,
    )

    index = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "bootstrap_draws": int(len(condition_bootstrap)),
        "artifacts": {
            "report": str(REPORT_PATH.relative_to(PHASE2.parent)),
            "run_record": str(RUN_RECORD_PATH.relative_to(PHASE2.parent)),
            "plots_dir": str(EXT_PLOTS.relative_to(PHASE2.parent)),
            "outputs_dir": str(EXT_OUTPUTS.relative_to(PHASE2.parent)),
        },
    }
    (EXT_OUTPUTS / "extension_artifact_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
