# Phase 2 Lifetime Model Extraction Report

Generated: 2026-05-03T11:37:43+05:30

## What Changed Compared With the Previous Pass

This run trains dedicated lifetime models. The previous extension quantified uncertainty around the analytical wear equation; this one builds threshold-crossing labels, trains wear-trajectory models, inverts the trained models into tool-life surfaces, and fits a survival-style accelerated-failure-time (AFT) equation that keeps censored runs instead of dropping them.

## Lifetime Labels

| dataset | records | threshold | events | censored |
| --- | --- | --- | --- | --- |
| NUAA | 9 | 0.250 | 5 | 4 |
| NASA | 15 | 0.300 | 15 | 0 |
| PHM2010 | 3 | 0.150 | 3 | 0 |

NUAA and NASA are modeled in minutes. PHM2010 is recorded in cut index under fixed operating conditions, so it is kept as a reference life-label set rather than merged into the parameterized milling equation.

## Trained Wear Models and Inverted Tool Life

| model | wear_rmse_mm | wear_mae_mm | wear_r2 | event_life_mae_min | censor_bound_satisfied_rate |
| --- | --- | --- | --- | --- | --- |
| Phase2Equation | 0.0969 | 0.0460 | -0.8811 | 5.3832 | 1.0000 |
| Ridge | 0.0505 | 0.0407 | 0.4883 | 6.7010 | 0.7500 |
| RandomForest | 0.0457 | 0.0337 | 0.5817 | 8.9847 | 1.0000 |
| GradientBoosting | 0.0394 | 0.0310 | 0.6896 | 11.6555 | 1.0000 |
| GaussianProcess | 0.0607 | 0.0459 | 0.2631 | 16.1923 | 1.0000 |
| MLP | 0.1159 | 0.0956 | -1.6890 | 19.0334 | 0.5000 |

Best pointwise wear model by RMSE: `GradientBoosting`.

Best overall event-life predictor in this run: `Phase2Equation`.

Selected trained ML surface model for equation extraction: `Ridge`. This is the best trained ML model by event-life MAE, but it should not be confused with the best overall lifetime predictor.

Plot: `phase -2/plots/lifetime_modeling/nuaa_wear_life_model_benchmark.png`

Interpretation: optimizing pointwise wear RMSE and optimizing threshold-crossing life are not the same objective. GradientBoosting has the lowest leave-one-experiment-out wear RMSE, but its threshold-crossing inversion is worse than the simpler Ridge model and the existing analytical equation.

## Extracted Lifetime Equations

### 1. Censored AFT Equation From NUAA Threshold-Crossing Labels

`T_min = 24.93 * (n/1800)^-0.0187 * (fz/0.05)^-1.7662 * (ap/3)^-0.3768`

AFT sigma: `0.5291`, negative log-likelihood: `21.2365`.

Coefficient bootstrap summary:

| term | count | median | q025 | q975 |
| --- | --- | --- | --- | --- |
| intercept | 400 | 3.2103 | 2.7199 | 3.8286 |
| log_speed_ref | 400 | -0.2186 | -6.2471 | 0.9129 |
| log_feed_ref | 400 | -2.1886 | -7.4945 | 0.1153 |
| log_depth_ref | 400 | -0.5467 | -2.4450 | 4.5558 |
| sigma | 400 | 0.1826 | 0.0000 | 0.6821 |

Plot: `phase -2/plots/lifetime_modeling/nuaa_aft_coefficient_bootstrap.png`

### 2. ML-Inverted Life Surface Equation

`T_min = 47.52 * (n/1800)^-10.4924 * (fz/0.05)^-4.6506 * (ap/3)^-0.1372`

This log-linear surrogate explains the selected ML model's inverted life surface with log-space R2 `0.9800` over `345` crossing scenarios.

The large speed exponent in this surrogate should be treated as a warning rather than a universal machining constant. The NUAA speed range is only 1750-1850 rpm, and the orthogonal grid creates collinearity between speed and the other factors.

Plot: `phase -2/plots/lifetime_modeling/ml_life_surface_heatmap.png`

### 3. Direct Observed-Life Equation

`T_min = 16.78 * (n/1800)^-12.6655 * (fz/0.05)^-4.1634 * (ap/3)^-2.4111`

This equation is fit only to uncensored NUAA observed crossings and is therefore less robust. Log RMSE: `0.3784`.

Plot: `phase -2/plots/lifetime_modeling/nuaa_life_equation_comparison.png`

## NASA Material-Aware Lifetime Model

NASA is kept separate because its feed variable is feed per revolution, its speed is fixed, and it uses cast iron/steel material families. The AFT model therefore uses feed, depth, material family, and initial wear.

| term | coefficient | exp_coefficient |
| --- | --- | --- |
| intercept | 4.0880 | 59.6220 |
| log_feed_ref | -0.3874 | 0.6788 |
| log_depth_ref | -1.1356 | 0.3212 |
| material_steel | -1.6069 | 0.2005 |
| initial_wear_mm | -3.1201 | 0.0442 |

| metric | value |
| --- | --- |
| event_mae_min | 2.5498 |
| event_rmse_min | 3.1544 |
| event_r2 | 0.9689 |
| sigma | 0.2459 |
| negative_log_likelihood | 43.6327 |

Plot: `phase -2/plots/lifetime_modeling/nasa_aft_life_model.png`

## Literature and Method Transfer Notes

| source | domain | method | usable_transfer |
| --- | --- | --- | --- |
| Local blueprint PDF | machining | tiered surrogate, physics-informed hybrid, optimization | Keep analytical equation plus ML surrogate; expose uncertainty and optimize only within observed parameter ranges. |
| NASA milling readme | machining | case-level tool wear under feed/depth/material settings | Use NASA as late-stage/material evidence; do not merge its feed units blindly with NUAA feed per tooth. |
| PHM public dataset inventory | PHM | run-to-failure dataset taxonomy | Use time-to-event framing and keep censored records instead of dropping non-failures. |
| QIT-CEMC Scientific Data 2025 | machining | full-life multimodal milling data | Next external validation target for full lifecycle wear and force/torque features. |
| Piecuch-Zabinski Scientific Data 2025 | machining | tool-wise grouped validation for tool-life estimation | Validate by held-out tool, not random cycle split. |
| PHM Society review 2018 | PHM | regression, RF, ANN, Bayesian linear regression, GPR for RUL | Use model families as baselines and report whether the model predicts health or time-to-event. |
| Sankararaman and Goebel 2013 | PHM uncertainty | RUL uncertainty propagation | Report intervals and uncertainty sources, not only point estimates. |
| GPR RUL PHM 2022 | PHM | Gaussian process RUL with small data and uncertainty | Include GPR as a small-data baseline; keep intervals as future work if calibrated externally. |
| DeepSurv | biomedicine | Cox neural network survival model | Use survival/time-to-event framing for censored tool-life data; deep version is data-hungry here. |
| Physics-informed GP for tool wear | machining | physical mean function plus GP residual | For small labels, constrain probabilistic models with a wear law rather than pure black-box fitting. |
| Drouillet et al. 2016 | machining | neural-network RUL using spindle-power RMS | Use sensor-derived health indicators as additional life predictors once enough run-to-failure records exist. |
| PI-KAF 2025 | machining | physics-informed interpretable neural monitoring | Prefer constrained, interpretable networks over unconstrained deep models for limited tool-wear labels. |

## Research Conclusions

1. Yes: this pass trains models and extracts tool-life equations from those trained models.
2. The most defensible lifetime equation is the censored AFT equation because it uses threshold-crossing labels and retains censored NUAA runs.
3. The ML-inverted equation is useful as a response-surface surrogate, but it is still only valid inside the NUAA parameter box: 1750-1850 rpm, 0.045-0.055 mm/tooth, 2.5-3.5 mm depth.
4. The survival-analysis pivot from biology is directly useful: tool life is a time-to-event problem with censoring, so Cox/AFT/DeepSurv-style thinking is more appropriate than plain regression alone.
5. Deep models should not be oversold on the current local lifetime labels. The right next move is external validation on QIT-CEMC or the 2025 Piecuch-Zabinski dataset, where many more full-life tools are available.

## Generated Files

- `phase -2/outputs/lifetime_modeling/nuaa_life_records.csv`
- `phase -2/outputs/lifetime_modeling/nasa_life_records.csv`
- `phase -2/outputs/lifetime_modeling/phm_life_records.csv`
- `phase -2/outputs/lifetime_modeling/nuaa_wear_model_metrics.csv`
- `phase -2/outputs/lifetime_modeling/nuaa_life_model_predictions.csv`
- `phase -2/outputs/lifetime_modeling/nuaa_aft_coefficients.csv`
- `phase -2/outputs/lifetime_modeling/nuaa_aft_bootstrap_summary.csv`
- `phase -2/outputs/lifetime_modeling/ml_extracted_life_equation.csv`
- `phase -2/outputs/lifetime_modeling/ml_life_surface.csv`
- `phase -2/outputs/lifetime_modeling/nasa_aft_coefficients.csv`
- `phase -2/outputs/lifetime_modeling/literature_method_transfer_matrix.csv`
