# Phase 2 Extended Research: Uncertainty, Calibration, and Residual Diagnostics

Generated: 2026-05-03T09:58:04+05:30

## Research Purpose

The original Phase 2 pipeline established a piecewise milling wear law. This extension asks whether the equation is usable as an engineering simulator: how uncertain are the fitted condition-law coefficients, how much does that uncertainty move predicted tool life, and whether an early calibration window improves a future-wear forecast.

## New Analyses Added

- NUAA condition-law bootstrap: resamples the nine NUAA operating-condition trajectories and refits k, speed, feed, and depth exponents.
- Life uncertainty propagation: pushes each bootstrap coefficient draw through the simulator for low, baseline, and high load settings.
- Early calibration holdout: estimates the simulator calibration factor from all available sub-transition calibration points up to a cutoff, then scores future points in the same NUAA run.
- Residual diagnostics: records observed minus predicted wear for the core NUAA, PHM2010, and NASA fits.

## Coefficient Stability

| parameter | count | mean | std | q025 | median | q975 |
| --- | --- | --- | --- | --- | --- | --- |
| k_early | 500 | 0.01824 | 0.00176 | 0.01558 | 0.01783 | 0.02183 |
| speed_exponent | 500 | 1.17526 | 4.62112 | -7.49808 | 1.16739 | 8.04424 |
| feed_exponent | 500 | 3.92111 | 1.22432 | 2.04563 | 3.99243 | 6.68021 |
| depth_exponent | 500 | 0.56199 | 0.71110 | -0.59012 | 0.51957 | 1.92473 |
| amplitude_r2 | 500 | 0.88162 | 0.09904 | 0.65545 | 0.89715 | 0.99805 |

Plot: `phase -2/plots/extended_research/condition_coefficient_bootstrap.png`

Interpretation: with only nine NUAA condition trajectories, the feed exponent remains the dominant driver, but the bootstrap interval is wide. This means the simulator should expose calibration rather than presenting a single coefficient set as universal.

## Tool-Life Prediction Intervals

| label | speed_rpm | feed_mm_tooth | depth_mm | median_life_min | q025_life_min | q975_life_min |
| --- | --- | --- | --- | --- | --- | --- |
| Low load | 1750.000 | 0.045 | 2.500 | 103.644 | 59.797 | 247.675 |
| Baseline | 1800.000 | 0.050 | 3.000 | 50.227 | 37.422 | 61.177 |
| High load | 1850.000 | 0.055 | 3.500 | 24.987 | 11.514 | 38.046 |

Plot: `phase -2/plots/extended_research/life_prediction_intervals.png`

Interpretation: the high-load setting remains consistently short lived, while the low-load setting has the widest absolute interval because small coefficient shifts compound over a longer forecast horizon.

## Early Calibration Holdout

| calibration_fraction | experiments | median_calibration_points | median_calibration_factor | median_calibrated_rmse_mm | median_uncalibrated_rmse_mm | median_final_calibrated_abs_error_mm | median_final_uncalibrated_abs_error_mm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.25000 | 7.00000 | 5.00000 | 0.73400 | 0.05045 | 0.04047 | 0.07498 | 0.06076 |
| 0.40000 | 8.00000 | 4.50000 | 0.77921 | 0.04750 | 0.02508 | 0.06926 | 0.03942 |
| 0.60000 | 8.00000 | 7.00000 | 0.91293 | 0.04230 | 0.02819 | 0.05253 | 0.03942 |

Plot: `phase -2/plots/extended_research/nuaa_calibration_forecast.png`

Interpretation: on the NUAA holdouts, per-run early calibration does not beat the global uncalibrated equation. That is a useful negative result: because the base equation is already fitted on NUAA, short-window lambda estimates tend to overcorrect. For a new tool-workpiece family, lambda should still be estimated, but it should be validated against held-out later wear rather than assumed beneficial.

## Residual Diagnostics

| dataset | points | bias_mm | mae_mm | rmse_mm | p95_abs_residual_mm |
| --- | --- | --- | --- | --- | --- |
| NASA | 146 | 0.00763 | 0.03149 | 0.04376 | 0.08825 |
| NUAA | 152 | -0.00162 | 0.01568 | 0.02113 | 0.04087 |
| PHM2010 | 945 | 0.00104 | 0.01070 | 0.01362 | 0.02437 |

Plot: `phase -2/plots/extended_research/residual_diagnostics.png`

Interpretation: residuals are smallest on PHM2010 and larger on NASA, which matches the research design: NASA is being used for late-stage behavior and material ratios, not as a directly transferable absolute wear-rate source.

## Literature Extension

- Li et al. introduced QIT-CEMC, a 2025 full-life titanium end-milling dataset with vibration, sound, cutting force, torque, wear images, and measured wear values. It is a useful next external validation target because it contains force/torque and full lifecycle wear rather than only early trajectories. Source: https://www.nature.com/articles/s41597-024-04345-2
- Piecuch and Zabinski released a 2025 CNC milling dataset with 14 tools from initial condition until failure, 968 milling cycles, raw vibration/current signals, aggregated features, and metadata. Their usage notes recommend tool-wise group cross-validation, which matches the grouped validation philosophy used here. Source: https://www.nature.com/articles/s41597-025-04923-y
- The PHM competition review identifies PHM2010 milling cutter wear as a regression benchmark using force and acoustic-emission signals, supporting this repo's decision to use PHM2010 as a cross-check rather than the only source of the life equation. Source: https://papers.phmsociety.org/index.php/phmconf/article/download/462/phmc_18_462

## Research Conclusions

1. The piecewise law is a defensible simulator structure, but coefficient uncertainty is not negligible.
2. Feed per tooth remains the strongest observed lever in the NUAA condition law.
3. Absolute life estimates still need tool-workpiece-specific validation before operational use; this holdout shows that short early-window lambda calibration can worsen forecasts when the global equation is already well matched.
4. The next best data extension is not another PHM2010-only model; it is external full-life validation on QIT-CEMC or the 2025 Piecuch-Zabinski tool-failure dataset.

## Generated Artifacts

- `phase -2/outputs/extended_research/condition_law_bootstrap.csv`
- `phase -2/outputs/extended_research/condition_law_bootstrap_summary.csv`
- `phase -2/outputs/extended_research/life_uncertainty_draws.csv`
- `phase -2/outputs/extended_research/life_uncertainty_summary.csv`
- `phase -2/outputs/extended_research/nuaa_calibration_holdout_metrics.csv`
- `phase -2/outputs/extended_research/nuaa_calibration_holdout_predictions.csv`
- `phase -2/outputs/extended_research/nuaa_calibration_summary.csv`
- `phase -2/outputs/extended_research/residual_diagnostics.csv`
- `phase -2/outputs/extended_research/residual_summary.csv`
- `phase -2/outputs/extended_research/phase2_extended_research_report.md`
