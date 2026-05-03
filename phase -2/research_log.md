# Phase 2 Research Log

## 2026-05-03

Scope: extend the existing milling tool-life research rather than replace it. The existing Phase 2 model already had a defensible piecewise early/late wear structure, so this pass focused on uncertainty, calibration behavior, residual diagnostics, and updated external dataset context.

Actions recorded:

- Read the top-level project README, Phase 2 README, source inventory, existing cross-dataset report, and Phase 2 pipeline/report/simulator code.
- Reviewed local data roles: NUAA for parameterized early wear, PHM2010 for early-regime cross-checking, and NASA for late-stage acceleration behavior.
- Added `extended_research_analysis.py` as a reproducible extension script.
- Added valid-design bootstrap filtering for the NUAA condition law so coefficient summaries are not dominated by rank-deficient resamples.
- Generated life prediction intervals from bootstrap coefficient draws.
- Tested multi-point early calibration holdout on NUAA and recorded the negative result: the global uncalibrated equation beats all tested early calibration cutoffs on median holdout RMSE.
- Generated residual diagnostics for NUAA, PHM2010, and NASA.
- Added literature update notes for the 2025 QIT-CEMC full-life milling dataset, the 2025 Piecuch-Zabinski tool-failure milling dataset, and the PHM competition review.
- Verified generated plot files are nonblank by checking image dimensions and pixel variance.
- Verified Python syntax with `py_compile`.
- Verified the simulator still runs for the baseline condition.

Primary generated record:

- `outputs/extended_research/extension_run_record.md`
- `outputs/extended_research/phase2_extended_research_report.md`

Git integrity note: this work should be committed and pushed with truthful Git metadata. The research log and generated run record use the actual run date.

## 2026-05-03, Lifetime Model Rework

Scope: answer the explicit question of whether a lifetime equation was produced by training models and extracting model results. The previous pass did not do that fully; this pass adds a dedicated lifetime model-extraction pipeline.

Actions recorded:

- Extracted/read local paper context from the ML-driven research blueprint, NASA milling documentation, and the PHM public dataset inventory.
- Reviewed external primary sources on QIT-CEMC, the 2025 tool-life milling dataset, PHM benchmark methodology, RUL uncertainty, Gaussian-process RUL, physics-informed Gaussian processes, spindle-power neural RUL, and DeepSurv-style survival modeling.
- Built threshold-crossing labels for NUAA, NASA, and PHM2010.
- Kept NUAA/NASA in minutes and PHM2010 in cut index to avoid mixing incompatible time bases.
- Trained Ridge, RandomForest, GradientBoosting, GaussianProcess, and MLP wear-trajectory models on NUAA with leave-one-experiment-out evaluation.
- Inverted trained wear models into threshold-crossing lifetime estimates.
- Fit a censored log-normal accelerated-failure-time equation to NUAA life labels.
- Bootstrapped the NUAA AFT equation coefficients.
- Extracted a Taylor-like lifetime equation from the selected trained ML response surface.
- Fit a separate material-aware NASA AFT model because NASA feed units and material families are not directly interchangeable with NUAA.
- Recorded a negative modeling result: the lowest wear-RMSE model is not the best lifetime model after threshold inversion.

Primary generated record:

- `outputs/lifetime_modeling/phase2_lifetime_model_report.md`
- `outputs/lifetime_modeling/lifetime_model_run_log.md`

## 2026-05-03, LaTeX Research Monograph

Scope: consolidate the repository research into a clean, standalone LaTeX manuscript with the requested author list, detailed tables, embedded plots, and editable LaTeX/TikZ explainer diagrams.

Actions recorded:

- Re-read the top-level README, Phase 2 report artifacts, extended research report, lifetime model report, cross-dataset results, VMD summary, modeling summary, feature-selection notes, and interpretation outputs.
- Created `phase -2/latex-research-book/predict_tool_life_research_book.tex` as a new 10pt report-style manuscript instead of overwriting the earlier Phase 2 LaTeX report.
- Used the title "From Sensor Signatures to Survival Curves: A Physics-Regularized Digital Twin for Milling Tool Wear and Tool-Life Forecasting".
- Added the requested authors: Amrit Lahari, Sukrit Agrawal, Pushkar Agrawal, and Amit Kumar Jain.
- Added LaTeX tables for dataset roles, failure thresholds, cross-dataset metrics, exponent evidence, bootstrap uncertainty, life intervals, calibration holdout results, residual diagnostics, trained-model inversion metrics, AFT coefficients, risk matrix, artifact index, and equation summary.
- Added TikZ diagrams for the research architecture, dataset evidence roles, piecewise wear schematic, lifetime extraction workflow, load-effect icons, decision logic, and next-step modeling roadmap.
- Embedded generated plots from Phase 2, extended research, lifetime modeling, cross-dataset analysis, VMD, feature selection, and SHAP interpretation.
- Compiled the manuscript with `pdflatex` and kept the source and final PDF as the committed deliverables.

Primary generated record:

- `latex-research-book/predict_tool_life_research_book.tex`
- `latex-research-book/predict_tool_life_research_book.pdf`
