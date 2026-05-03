# Lifetime Model Extraction Run Log

Run timestamp: 2026-05-03T11:37:44+05:30

## Command

```powershell
.\.venv\Scripts\python.exe "phase -2\lifetime_model_extraction.py"
```

## Actions

- Rebuilt/loaded Phase 2 model context.
- Built threshold-crossing life labels for NUAA, NASA, and PHM2010.
- Trained NUAA wear trajectory models and inverted them into lifetime estimates.
- Fit a censored log-normal AFT lifetime equation for NUAA.
- Fit a NASA material-aware AFT lifetime model.
- Extracted a log-linear Taylor-like equation from the selected trained ML life surface.
- Wrote literature transfer notes covering machining, PHM, and survival-analysis methods.

## Recorded Findings

- Selected ML model: Ridge.
- Selected model wear RMSE: 0.05055 mm.
- Selected model event-life MAE: 6.70096 min.
- NUAA AFT sigma: 0.52909.

Git integrity note: this log uses the actual run timestamp.
