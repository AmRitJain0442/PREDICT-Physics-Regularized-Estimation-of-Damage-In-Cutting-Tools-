# Phase 2 Extension Run Record

Run timestamp: 2026-05-03T09:58:04+05:30

## Commands

```powershell
.\.venv\Scripts\python.exe "phase -2\extended_research_analysis.py"
```

## What This Run Did

- Rebuilt the core Phase 2 outputs by calling `research_pipeline.main()`.
- Loaded NUAA, PHM2010, and NASA milling wear tables from the local workspace.
- Resampled the NUAA condition law 500 times.
- Propagated condition-law uncertainty through the tool-life simulator.
- Ran NUAA multi-point early calibration holdout forecasts.
- Recomputed residual diagnostics for the core dataset fits.

## Key Recorded Findings

- Best calibration cutoff in this run: 0.60 horizon fraction, median calibrated RMSE 0.04230 mm.
- Calibration comparison: No tested calibration cutoff beat the uncalibrated median RMSE on NUAA holdout points.
- Baseline median life from coefficient bootstrap: 50.23 min.
- Largest residual RMSE dataset: NASA.

## Git Integrity Note

The research record is written with the actual run timestamp. Git commits should be made with truthful metadata; this run does not backdate research activity.
