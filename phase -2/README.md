# Phase 2: Milling Tool-Life Equation Research

This folder contains a data-backed phase-2 tool-life model for milling, built from the local NUAA, PHM 2010, and NASA milling datasets already present in the workspace. The work is structured as a reproducible research pipeline plus a small simulation engine.

## What Was Fitted

The analysis split the wear problem into two regimes instead of forcing a single Taylor-style law across the full trajectory:

- Early regime: NUAA and PHM 2010 both support a sub-linear wear-growth law with an exponent near `0.64`.
- Late regime: NASA run-to-failure data supports an accelerated wear-growth law with an exponent near `1.29`.

That leads to the selected phase-2 curve:

```text
phi = (n / 1800)^0.3600 * (fz / 0.05)^3.9560 * (ap / 3.0)^0.6881

For VB <= 0.25 mm:
VB(t) = VB0 + lambda * 0.01858 * phi * t^0.64

For VB > 0.25 mm:
VB(t) = 0.25 + lambda * rho * 0.01858 * phi * (t - t0.25)^1.29
```

Where:

- `VB(t)` is flank wear in `mm`
- `VB0` is the starting wear state
- `t` is time in `minutes`
- `n` is spindle speed in `rpm`
- `fz` is feed per tooth in `mm/tooth`
- `ap` is axial depth in `mm`
- `lambda` is a calibration factor for a specific tool-workpiece pair
- `rho` is the late-stage acceleration factor
- `t0.25` is the time at which the early equation reaches `0.25 mm`

The associated tool-life equation is analytic in each regime:

```text
If VBf <= 0.25:
Tf = ((VBf - VB0) / (lambda * 0.01858 * phi))^(1 / 0.64)

If VBf > 0.25:
T0.25 = ((0.25 - VB0) / (lambda * 0.01858 * phi))^(1 / 0.64)
Tf = T0.25 + ((VBf - 0.25) / (lambda * rho * 0.01858 * phi))^(1 / 1.29)
```

## Why This Structure

A single power law fit the NUAA early-stage data better than a linear-in-time model, but the NASA run-to-failure traces clearly accelerate later. PHM 2010 agreed with the NUAA early-stage exponent, which is why the final phase-2 model is piecewise instead of pretending that one exponent covers the entire curve.

The practical implication is important:

- The shape of the early wear region is supported directly by two milling datasets.
- The accelerated end-of-life region is supported by NASA case-level wear measurements.
- Absolute life still depends strongly on the tool-workpiece family, so `lambda` is exposed as a calibration knob instead of being hidden.

## Files

- `research_pipeline.py`: rebuilds the phase-2 tables, plots, and model JSON.
- `report_builder.py`: runs the deeper validation layer and generates the Word report.
- `tool_life_simulator.py`: CLI and reusable model wrapper.
- `outputs/phase2_model_summary.json`: selected coefficients and evidence.
- `outputs/key_results.csv`: compact coefficient table.
- `outputs/dataset_profiles.csv`: dataset inventory used in the fit.
- `plots/`: generated fit and simulation figures.
- `sources.md`: source inventory and research notes.

## Run

Rebuild the analysis:

```bash
.venv\Scripts\python.exe "phase -2\research_pipeline.py"
```

Generate the full Word report:

```bash
.venv\Scripts\python.exe "phase -2\report_builder.py"
```

Predict tool life:

```bash
.venv\Scripts\python.exe "phase -2\tool_life_simulator.py" --speed 1800 --feed 0.05 --depth 3.0 --threshold 0.30 --initial-wear 0.05 --material generic
```

Export a full wear curve:

```bash
.venv\Scripts\python.exe "phase -2\tool_life_simulator.py" --speed 1800 --feed 0.05 --depth 3.0 --threshold 0.30 --output-csv "phase -2\outputs\example_curve.csv"
```

## Limits

- The process-parameter exponents come primarily from the NUAA orthogonal bundle. They are best treated as a phase-2 empirical law, not a universal milling constant.
- Feed sensitivity is strong in the fitted data, while spindle-speed sensitivity is weaker and less identifiable because the NUAA speed range is narrow.
- PHM 2010 was used to confirm the early wear exponent, not to refit the condition exponents.
- NASA was used to estimate late-stage acceleration, not to transfer absolute wear-rate constants across materials without calibration.
