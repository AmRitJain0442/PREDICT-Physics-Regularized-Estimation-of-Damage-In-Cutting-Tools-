from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression


ROOT = Path(__file__).resolve().parent.parent
PHASE2 = Path(__file__).resolve().parent
OUTPUTS = PHASE2 / "outputs"
PLOTS = PHASE2 / "plots"
REFERENCES = PHASE2 / "references"

DATA_LIST_PDF = ROOT / "data list paper.pdf"
NUAA_PATH = ROOT / "data" / "uniwear" / "data" / "nuaa_orthogonal_bundle_high_resolution.csv"
PHM_ROOT = ROOT / "data" / "archive (1)"
NASA_META_PATH = ROOT / "data" / "nasa_milling" / "csv" / "metadata.csv"

REFERENCE_SPEED_RPM = 1800.0
REFERENCE_FEED_MM_TOOTH = 0.05
REFERENCE_DEPTH_MM = 3.0
TRANSITION_WEAR_MM = 0.25
EARLY_EXPONENT_SELECTED = 0.64
LATE_EXPONENT_SELECTED = 1.29


@dataclass
class CommonPowerFit:
    exponent: float
    rmse: float
    amplitude_by_group: dict[str, float]
    predictions: pd.DataFrame


def ensure_dirs() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    REFERENCES.mkdir(parents=True, exist_ok=True)


def maybe_extract_reference_pdf() -> None:
    if not DATA_LIST_PDF.exists():
        return

    pdftotext = shutil.which("pdftotext")
    if not pdftotext:
        return

    out_path = REFERENCES / "data_list_paper.txt"
    subprocess.run(
        [pdftotext, "-layout", str(DATA_LIST_PDF), str(out_path)],
        check=False,
        capture_output=True,
        text=True,
    )


def load_nuaa_run_level() -> pd.DataFrame:
    df = pd.read_csv(
        NUAA_PATH,
        usecols=[
            "experiment_tag",
            "experiment_csv_n",
            "timestamp",
            "tool_wear",
            "feed_per_tooth",
            "spindle_speed",
            "axial_cutting_depth",
        ],
    )
    run_level = (
        df.sort_values(["experiment_tag", "experiment_csv_n", "timestamp"])
        .groupby(["experiment_tag", "experiment_csv_n"], as_index=False)
        .agg(
            time_s=("timestamp", "max"),
            wear_mm=("tool_wear", "max"),
            feed_mm_tooth=("feed_per_tooth", "first"),
            speed_rpm=("spindle_speed", "first"),
            depth_mm=("axial_cutting_depth", "first"),
        )
    )
    run_level["time_min"] = run_level["time_s"] / 60.0
    run_level["relative_time_min"] = run_level.groupby("experiment_tag")["time_min"].transform(
        lambda s: s - s.min()
    )
    run_level["initial_wear_mm"] = run_level.groupby("experiment_tag")["wear_mm"].transform("first")
    run_level["delta_wear_mm"] = run_level["wear_mm"] - run_level["initial_wear_mm"]
    return run_level


def load_phm_cut_level() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for cutter in ["c1", "c4", "c6"]:
        wear_path = PHM_ROOT / cutter / f"{cutter}_wear.csv"
        wear_df = pd.read_csv(wear_path)
        wear_df["wear_mm"] = wear_df[["flute_1", "flute_2", "flute_3"]].mean(axis=1) / 1000.0
        wear_df["experiment_tag"] = cutter
        rows.append(wear_df[["experiment_tag", "cut", "wear_mm"]])

    df = pd.concat(rows, ignore_index=True).sort_values(["experiment_tag", "cut"])
    df["relative_cut"] = df.groupby("experiment_tag")["cut"].transform(lambda s: s - s.min())
    df["initial_wear_mm"] = df.groupby("experiment_tag")["wear_mm"].transform("first")
    df["delta_wear_mm"] = df["wear_mm"] - df["initial_wear_mm"]
    return df


def load_nasa_case_level() -> pd.DataFrame:
    df = pd.read_csv(NASA_META_PATH).dropna(subset=["VB"]).copy()
    df = df.sort_values(["case", "time"])
    df["relative_time_min"] = df.groupby("case")["time"].transform(lambda s: s - s.min())
    df["initial_wear_mm"] = df.groupby("case")["VB"].transform("first")
    df["delta_wear_mm"] = df["VB"] - df["initial_wear_mm"]
    df["material_family"] = df["material"].map({1: "cast_iron", 2: "steel"})
    return df


def fit_common_power(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    wear_col: str,
    initial_guess_log_a: float = -2.0,
    initial_guess_m: float = 0.7,
) -> CommonPowerFit:
    work = df.sort_values([group_col, time_col]).copy()
    groups = list(work[group_col].astype(str).unique())
    group_to_idx = {g: i for i, g in enumerate(groups)}
    idx = work[group_col].astype(str).map(group_to_idx).to_numpy()
    y = work["delta_wear_mm"].to_numpy(dtype=float)
    t = work[time_col].to_numpy(dtype=float)

    def residuals(params: np.ndarray) -> np.ndarray:
        log_a = params[:-1]
        m = float(np.exp(params[-1]))
        pred = np.exp(log_a[idx]) * np.power(np.maximum(t, 1e-9), m)
        return pred - y

    x0 = np.r_[np.full(len(groups), initial_guess_log_a), np.log(initial_guess_m)]
    result = least_squares(residuals, x0=x0, max_nfev=50_000)
    exponent = float(np.exp(result.x[-1]))
    amp = np.exp(result.x[:-1])
    pred = amp[idx] * np.power(np.maximum(t, 1e-9), exponent)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))

    predictions = work[[group_col, time_col, wear_col, "initial_wear_mm", "delta_wear_mm"]].copy()
    predictions["predicted_delta_wear_mm"] = pred
    predictions["predicted_wear_mm"] = predictions["initial_wear_mm"] + predictions["predicted_delta_wear_mm"]

    return CommonPowerFit(
        exponent=exponent,
        rmse=rmse,
        amplitude_by_group={group: float(amp[group_to_idx[group]]) for group in groups},
        predictions=predictions,
    )


def fit_fixed_power_amplitudes(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    wear_col: str,
    exponent: float,
) -> CommonPowerFit:
    work = df.sort_values([group_col, time_col]).copy()
    rows = []
    predictions = []
    for group, group_df in work.groupby(group_col, sort=True):
        x = np.power(np.maximum(group_df[time_col].to_numpy(dtype=float), 1e-9), exponent)
        y = group_df["delta_wear_mm"].to_numpy(dtype=float)
        denom = float(np.dot(x, x))
        amplitude = 0.0 if denom == 0.0 else float(np.dot(x, y) / denom)
        pred = amplitude * x
        rows.append({"group": str(group), "amplitude": amplitude})

        frame = group_df[[group_col, time_col, wear_col, "initial_wear_mm", "delta_wear_mm"]].copy()
        frame["predicted_delta_wear_mm"] = pred
        frame["predicted_wear_mm"] = frame["initial_wear_mm"] + frame["predicted_delta_wear_mm"]
        predictions.append(frame)

    pred_df = pd.concat(predictions, ignore_index=True)
    rmse = float(
        np.sqrt(np.mean((pred_df["predicted_delta_wear_mm"] - pred_df["delta_wear_mm"]) ** 2))
    )
    amp_df = pd.DataFrame(rows)

    return CommonPowerFit(
        exponent=exponent,
        rmse=rmse,
        amplitude_by_group=dict(zip(amp_df["group"], amp_df["amplitude"])),
        predictions=pred_df,
    )


def estimate_nuaa_condition_law(nuaa_runs: pd.DataFrame, exponent: float) -> tuple[dict[str, float], pd.DataFrame]:
    fit = fit_fixed_power_amplitudes(
        df=nuaa_runs,
        group_col="experiment_tag",
        time_col="relative_time_min",
        wear_col="wear_mm",
        exponent=exponent,
    )

    exp_df = (
        nuaa_runs.groupby("experiment_tag", as_index=False)
        .agg(
            speed_rpm=("speed_rpm", "first"),
            feed_mm_tooth=("feed_mm_tooth", "first"),
            depth_mm=("depth_mm", "first"),
        )
        .sort_values("experiment_tag")
    )
    exp_df["amplitude"] = exp_df["experiment_tag"].astype(str).map(fit.amplitude_by_group)

    x = np.log(
        exp_df[["speed_rpm", "feed_mm_tooth", "depth_mm"]].to_numpy(dtype=float)
        / np.array([REFERENCE_SPEED_RPM, REFERENCE_FEED_MM_TOOTH, REFERENCE_DEPTH_MM])
    )
    y = np.log(exp_df["amplitude"].to_numpy(dtype=float))
    reg = LinearRegression().fit(x, y)

    exp_df["predicted_amplitude"] = np.exp(reg.predict(x))
    coefficients = {
        "reference_speed_rpm": REFERENCE_SPEED_RPM,
        "reference_feed_mm_tooth": REFERENCE_FEED_MM_TOOTH,
        "reference_depth_mm": REFERENCE_DEPTH_MM,
        "k_early": float(np.exp(reg.intercept_)),
        "speed_exponent": float(reg.coef_[0]),
        "feed_exponent": float(reg.coef_[1]),
        "depth_exponent": float(reg.coef_[2]),
        "amplitude_r2": float(reg.score(x, y)),
    }
    return coefficients, exp_df


def estimate_nasa_late_stage_ratios(
    nasa: pd.DataFrame,
    early_exponent: float,
    late_exponent: float,
    transition_wear_mm: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for case, case_df in nasa.groupby("case", sort=True):
        case_df = case_df.sort_values("time").copy()
        if float(case_df["VB"].max()) < transition_wear_mm:
            continue

        initial_wear = float(case_df["VB"].iloc[0])
        prev = case_df[case_df["VB"] < transition_wear_mm].tail(1)
        nxt = case_df[case_df["VB"] >= transition_wear_mm].head(1)
        if prev.empty:
            transition_time = float(nxt["time"].iloc[0])
        else:
            t0 = float(prev["time"].iloc[0])
            w0 = float(prev["VB"].iloc[0])
            t1 = float(nxt["time"].iloc[0])
            w1 = float(nxt["VB"].iloc[0])
            frac = 0.0 if w1 == w0 else (transition_wear_mm - w0) / (w1 - w0)
            transition_time = t0 + frac * (t1 - t0)

        transition_time_rel = transition_time - float(case_df["time"].iloc[0])
        if transition_time_rel <= 0.0:
            continue

        early_amplitude = (transition_wear_mm - initial_wear) / (transition_time_rel**early_exponent)

        post = case_df[case_df["time"] >= transition_time].copy()
        dt = (post["time"] - transition_time).to_numpy(dtype=float)
        y = (post["VB"] - transition_wear_mm).to_numpy(dtype=float)
        valid = dt > 0.0
        dt = dt[valid]
        y = y[valid]
        if len(dt) < 2:
            continue

        basis = np.power(dt, late_exponent)
        late_amplitude = float(np.dot(basis, y) / np.dot(basis, basis))
        material_family = str(case_df["material_family"].iloc[0])
        rows.append(
            {
                "case": int(case),
                "material_family": material_family,
                "initial_wear_mm": initial_wear,
                "transition_time_min": transition_time,
                "transition_time_rel_min": transition_time_rel,
                "early_amplitude": early_amplitude,
                "late_amplitude": late_amplitude,
                "late_to_early_ratio": late_amplitude / early_amplitude,
                "feed_mm_rev": float(case_df["feed"].iloc[0]),
                "depth_mm": float(case_df["DOC"].iloc[0]),
            }
        )

    ratio_df = pd.DataFrame(rows).sort_values("case")
    summary = (
        ratio_df.groupby("material_family")["late_to_early_ratio"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    return ratio_df, summary


def build_dataset_profiles(
    nuaa_runs: pd.DataFrame,
    phm: pd.DataFrame,
    nasa: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for experiment_tag, group_df in nuaa_runs.groupby("experiment_tag", sort=True):
        rows.append(
            {
                "dataset": "NUAA",
                "series_id": experiment_tag,
                "points": int(len(group_df)),
                "start_time": float(group_df["time_min"].min()),
                "end_time": float(group_df["time_min"].max()),
                "start_wear_mm": float(group_df["wear_mm"].min()),
                "end_wear_mm": float(group_df["wear_mm"].max()),
                "speed_rpm": float(group_df["speed_rpm"].iloc[0]),
                "feed_mm_tooth": float(group_df["feed_mm_tooth"].iloc[0]),
                "depth_mm": float(group_df["depth_mm"].iloc[0]),
            }
        )

    for experiment_tag, group_df in phm.groupby("experiment_tag", sort=True):
        rows.append(
            {
                "dataset": "PHM2010",
                "series_id": experiment_tag,
                "points": int(len(group_df)),
                "start_time": float(group_df["cut"].min()),
                "end_time": float(group_df["cut"].max()),
                "start_wear_mm": float(group_df["wear_mm"].min()),
                "end_wear_mm": float(group_df["wear_mm"].max()),
                "speed_rpm": np.nan,
                "feed_mm_tooth": np.nan,
                "depth_mm": np.nan,
            }
        )

    for case, group_df in nasa.groupby("case", sort=True):
        rows.append(
            {
                "dataset": "NASA",
                "series_id": f"case_{case}",
                "points": int(len(group_df)),
                "start_time": float(group_df["time"].min()),
                "end_time": float(group_df["time"].max()),
                "start_wear_mm": float(group_df["VB"].min()),
                "end_wear_mm": float(group_df["VB"].max()),
                "speed_rpm": 826.0,
                "feed_mm_tooth": np.nan,
                "depth_mm": float(group_df["DOC"].iloc[0]),
            }
        )

    return pd.DataFrame(rows)


def plot_fit(
    observed: pd.DataFrame,
    group_col: str,
    time_col: str,
    wear_col: str,
    predicted_wear_col: str,
    title: str,
    xlabel: str,
    output_path: Path,
) -> None:
    unique_groups = list(observed[group_col].astype(str).unique())
    n_groups = len(unique_groups)
    ncols = 3
    nrows = int(np.ceil(n_groups / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.5 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, group in zip(axes_flat, unique_groups):
        group_df = observed[observed[group_col].astype(str) == group].sort_values(time_col)
        ax.plot(group_df[time_col], group_df[wear_col], marker="o", lw=1.5, label="Observed")
        ax.plot(group_df[time_col], group_df[predicted_wear_col], lw=1.5, label="Model")
        ax.set_title(str(group))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Wear (mm)")
        ax.grid(alpha=0.3)

    for ax in axes_flat[n_groups:]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(title, y=0.98, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_condition_fit(exp_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(exp_df["amplitude"], exp_df["predicted_amplitude"], s=55)
    min_val = min(float(exp_df["amplitude"].min()), float(exp_df["predicted_amplitude"].min()))
    max_val = max(float(exp_df["amplitude"].max()), float(exp_df["predicted_amplitude"].max()))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", linewidth=1)
    for _, row in exp_df.iterrows():
        ax.annotate(row["experiment_tag"], (row["amplitude"], row["predicted_amplitude"]), fontsize=8)
    ax.set_xlabel("Observed Early-Stage Amplitude")
    ax.set_ylabel("Predicted Early-Stage Amplitude")
    ax.set_title("NUAA Condition Law Fit")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_simulation_examples(model: dict[str, object], output_path: Path) -> None:
    from tool_life_simulator import MillingToolLifeModel

    simulator = MillingToolLifeModel(model)
    scenarios = [
        ("Low load", 1750.0, 0.045, 2.5),
        ("Mid load", 1800.0, 0.050, 3.0),
        ("High load", 1850.0, 0.055, 3.5),
    ]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for label, speed, feed, depth in scenarios:
        curve = simulator.curve_dataframe(
            speed_rpm=speed,
            feed_mm_tooth=feed,
            depth_mm=depth,
            initial_wear_mm=0.05,
            material_family="generic",
            minutes=np.linspace(0, 90, 240),
        )
        ax.plot(curve["time_min"], curve["wear_mm"], label=label)

    ax.axhline(TRANSITION_WEAR_MM, color="gray", linestyle="--", linewidth=1, label="Transition wear")
    ax.axhline(0.30, color="firebrick", linestyle=":", linewidth=1, label="0.30 mm threshold")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Predicted wear (mm)")
    ax.set_title("Phase-2 Simulator Example Curves")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_model_summary(
    nuaa_early_fit: CommonPowerFit,
    phm_early_fit: CommonPowerFit,
    nasa_late_fit: CommonPowerFit,
    condition_law: dict[str, float],
    late_ratio_summary: pd.DataFrame,
) -> dict[str, object]:
    late_factor_map = {
        "generic": 0.25,
        "cast_iron": 0.16,
        "steel": 0.40,
    }

    for _, row in late_ratio_summary.iterrows():
        family = str(row["material_family"])
        late_factor_map[family] = round(float(row["median"]), 4)

    return {
        "model_name": "phase2_bi_regime_milling_wear_law",
        "equation_family": "piecewise_power_law",
        "transition_wear_mm": TRANSITION_WEAR_MM,
        "units": {
            "wear": "mm",
            "time": "minutes",
            "speed": "rpm",
            "feed": "mm/tooth",
            "depth": "mm",
        },
        "selected_coefficients": {
            "early_exponent": EARLY_EXPONENT_SELECTED,
            "late_exponent": LATE_EXPONENT_SELECTED,
            **condition_law,
            "late_stage_factor_by_material_family": late_factor_map,
        },
        "evidence": {
            "nuaa_early_exponent_raw": nuaa_early_fit.exponent,
            "phm_early_exponent_raw": phm_early_fit.exponent,
            "nasa_late_exponent_raw": nasa_late_fit.exponent,
            "nuaa_early_rmse_mm": nuaa_early_fit.rmse,
            "phm_early_rmse_mm": phm_early_fit.rmse,
            "nasa_late_rmse_mm": nasa_late_fit.rmse,
        },
        "equations": {
            "condition_intensity_phi": (
                "(speed_rpm / 1800)^speed_exponent * "
                "(feed_mm_tooth / 0.05)^feed_exponent * "
                "(depth_mm / 3.0)^depth_exponent"
            ),
            "early_regime": (
                "wear_mm = initial_wear_mm + calibration_factor * k_early * phi * time_min^early_exponent"
            ),
            "late_regime": (
                "wear_mm = transition_wear_mm + calibration_factor * late_stage_factor * "
                "k_early * phi * (time_min - transition_time_min)^late_exponent"
            ),
        },
        "notes": [
            "The early-stage law is calibrated on the NUAA orthogonal milling bundle.",
            "PHM 2010 is used to verify that the early-stage exponent remains in the same range for another milling dataset.",
            "NASA run-to-failure milling data is used to estimate the late-stage acceleration exponent and material-family acceleration ratios.",
            "Use calibration_factor to adapt the absolute wear rate to a specific tool-workpiece pair before relying on absolute tool-life predictions.",
        ],
    }


def main() -> None:
    ensure_dirs()
    maybe_extract_reference_pdf()

    nuaa_runs = load_nuaa_run_level()
    phm = load_phm_cut_level()
    nasa = load_nasa_case_level()

    dataset_profiles = build_dataset_profiles(nuaa_runs, phm, nasa)
    dataset_profiles.to_csv(OUTPUTS / "dataset_profiles.csv", index=False)

    nuaa_linear = fit_fixed_power_amplitudes(
        df=nuaa_runs,
        group_col="experiment_tag",
        time_col="relative_time_min",
        wear_col="wear_mm",
        exponent=1.0,
    )
    nuaa_power = fit_common_power(
        df=nuaa_runs,
        group_col="experiment_tag",
        time_col="relative_time_min",
        wear_col="wear_mm",
    )
    pd.DataFrame(
        [
            {"dataset": "NUAA", "model": "linear_in_time", "exponent": 1.0, "rmse_mm": nuaa_linear.rmse},
            {
                "dataset": "NUAA",
                "model": "power_in_time",
                "exponent": nuaa_power.exponent,
                "rmse_mm": nuaa_power.rmse,
            },
        ]
    ).to_csv(OUTPUTS / "nuaa_candidate_models.csv", index=False)

    phm_power = fit_common_power(
        df=phm,
        group_col="experiment_tag",
        time_col="relative_cut",
        wear_col="wear_mm",
        initial_guess_log_a=-5.0,
        initial_guess_m=0.8,
    )
    nasa_power = fit_common_power(
        df=nasa.rename(columns={"case": "case_id", "time": "time_min", "VB": "wear_mm"}),
        group_col="case_id",
        time_col="relative_time_min",
        wear_col="wear_mm",
        initial_guess_log_a=-3.0,
        initial_guess_m=1.1,
    )

    condition_law, nuaa_amp_df = estimate_nuaa_condition_law(
        nuaa_runs=nuaa_runs,
        exponent=EARLY_EXPONENT_SELECTED,
    )
    nuaa_amp_df.to_csv(OUTPUTS / "nuaa_condition_amplitudes.csv", index=False)

    nasa_ratio_df, nasa_ratio_summary = estimate_nasa_late_stage_ratios(
        nasa=nasa,
        early_exponent=EARLY_EXPONENT_SELECTED,
        late_exponent=LATE_EXPONENT_SELECTED,
        transition_wear_mm=TRANSITION_WEAR_MM,
    )
    nasa_ratio_df.to_csv(OUTPUTS / "nasa_late_stage_case_ratios.csv", index=False)
    nasa_ratio_summary.to_csv(OUTPUTS / "nasa_late_stage_ratio_summary.csv", index=False)

    model_summary = build_model_summary(
        nuaa_early_fit=nuaa_power,
        phm_early_fit=phm_power,
        nasa_late_fit=nasa_power,
        condition_law=condition_law,
        late_ratio_summary=nasa_ratio_summary,
    )
    (OUTPUTS / "phase2_model_summary.json").write_text(json.dumps(model_summary, indent=2), encoding="utf-8")

    plot_fit(
        observed=nuaa_power.predictions.rename(columns={"experiment_tag": "group", "relative_time_min": "x"}),
        group_col="group",
        time_col="x",
        wear_col="wear_mm",
        predicted_wear_col="predicted_wear_mm",
        title="NUAA Early-Regime Power-Law Fit",
        xlabel="Relative time (min)",
        output_path=PLOTS / "nuaa_early_regime_fit.png",
    )
    plot_fit(
        observed=phm_power.predictions.rename(columns={"experiment_tag": "group", "relative_cut": "x"}),
        group_col="group",
        time_col="x",
        wear_col="wear_mm",
        predicted_wear_col="predicted_wear_mm",
        title="PHM 2010 Early-Regime Power-Law Fit",
        xlabel="Relative cut index",
        output_path=PLOTS / "phm_early_regime_fit.png",
    )
    plot_fit(
        observed=nasa_power.predictions.rename(columns={"case_id": "group", "relative_time_min": "x"}),
        group_col="group",
        time_col="x",
        wear_col="wear_mm",
        predicted_wear_col="predicted_wear_mm",
        title="NASA Late-Regime Power-Law Fit",
        xlabel="Relative time (min)",
        output_path=PLOTS / "nasa_late_regime_fit.png",
    )
    plot_condition_fit(nuaa_amp_df, PLOTS / "nuaa_condition_law_fit.png")
    plot_simulation_examples(model_summary, PLOTS / "simulation_example_curves.png")

    summary_table = pd.DataFrame(
        [
            {
                "item": "NUAA early exponent",
                "value": round(float(nuaa_power.exponent), 4),
                "notes": "Run-level fit on orthogonal milling experiments",
            },
            {
                "item": "PHM early exponent",
                "value": round(float(phm_power.exponent), 4),
                "notes": "Cut-level fit on PHM 2010 cutter wear traces",
            },
            {
                "item": "NASA late exponent",
                "value": round(float(nasa_power.exponent), 4),
                "notes": "Case-level fit on measured wear points",
            },
            {
                "item": "Selected early exponent",
                "value": EARLY_EXPONENT_SELECTED,
                "notes": "Rounded operating value for simulator",
            },
            {
                "item": "Selected late exponent",
                "value": LATE_EXPONENT_SELECTED,
                "notes": "Rounded operating value for simulator",
            },
            {
                "item": "Early k coefficient",
                "value": round(float(condition_law["k_early"]), 6),
                "notes": "Minutes-based NUAA condition law",
            },
            {
                "item": "Speed exponent",
                "value": round(float(condition_law["speed_exponent"]), 4),
                "notes": "Condition-intensity term around 1800 rpm",
            },
            {
                "item": "Feed exponent",
                "value": round(float(condition_law["feed_exponent"]), 4),
                "notes": "Condition-intensity term around 0.05 mm/tooth",
            },
            {
                "item": "Depth exponent",
                "value": round(float(condition_law["depth_exponent"]), 4),
                "notes": "Condition-intensity term around 3.0 mm axial depth",
            },
        ]
    )
    summary_table.to_csv(OUTPUTS / "key_results.csv", index=False)


if __name__ == "__main__":
    main()
