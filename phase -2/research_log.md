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
