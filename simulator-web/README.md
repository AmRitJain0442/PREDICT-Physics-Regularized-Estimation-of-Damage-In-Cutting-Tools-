# Tool Life Simulator (Static)

This is a lightweight simulator page grounded in the generated research outputs:
- `results/cross_dataset_research/metrics_summary.json`
- `results/cross_dataset_research/tables/nuaa_simulated_life_table.csv`
- failure marker tables for PHM/NASA/UniWear

## Run

From project root:

```powershell
cd simulator-web
python build_simulator_data.py
python -m http.server 8000
```

Or manually:

```powershell
cd simulator-web
python -m http.server 8000
```

Then open:

`http://localhost:8000`

## Inputs

- Spindle speed
- Feed per tooth
- Axial depth
- Elapsed time
- Marker stress multipliers (vibration, AE, crest)
- Dataset threshold selector

## Outputs

- Estimated tool life
- Estimated wear
- Failure probability
- Status with remaining life
- Wear trajectory chart
- Top recommended parameter settings and marker references
