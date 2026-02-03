"""
Convert NASA Milling MATLAB file (mill.mat) to CSV format.

Produces:
  - metadata.csv : one row per run with scalar fields
  - signals/run_XXX.csv : per-run sensor signals (9000 samples × 6 channels)
"""

import os
import numpy as np
import pandas as pd
import scipy.io

MAT_PATH = "./data/nasa_milling/3. Milling/mill.mat"
OUT_DIR = "./data/nasa_milling/csv"
SIGNALS_DIR = os.path.join(OUT_DIR, "signals")

def main():
    os.makedirs(SIGNALS_DIR, exist_ok=True)

    print(f"Loading {MAT_PATH} ...")
    mat = scipy.io.loadmat(MAT_PATH)
    mill = mat["mill"]
    n_runs = mill.shape[1]
    print(f"Found {n_runs} runs.")

    # --- Metadata CSV ---
    metadata_fields = ["case", "run", "VB", "time", "DOC", "feed", "material"]
    signal_fields = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]

    meta_rows = []
    for i in range(n_runs):
        entry = mill[0, i]
        row = {}
        for f in metadata_fields:
            val = entry[f].flat[0]
            row[f] = val
        meta_rows.append(row)

    meta_df = pd.DataFrame(meta_rows)
    meta_path = os.path.join(OUT_DIR, "metadata.csv")
    meta_df.to_csv(meta_path, index=False)
    print(f"Saved metadata: {meta_path}  ({len(meta_df)} rows)")

    # --- Per-run signal CSVs ---
    for i in range(n_runs):
        entry = mill[0, i]
        sig_data = {}
        for f in signal_fields:
            sig_data[f] = entry[f].flatten()
        sig_df = pd.DataFrame(sig_data)

        run_id = int(entry["run"].flat[0])
        case_id = int(entry["case"].flat[0])
        fname = f"case{case_id}_run{run_id:03d}.csv"
        sig_path = os.path.join(SIGNALS_DIR, fname)
        sig_df.to_csv(sig_path, index=False)

        if (i + 1) % 20 == 0 or i == n_runs - 1:
            print(f"  Saved signal CSVs: {i + 1}/{n_runs}")

    print(f"\nDone. Output directory: {OUT_DIR}")
    print(f"  - metadata.csv ({len(meta_df)} runs)")
    print(f"  - signals/ ({n_runs} CSV files, 9000 samples each)")


if __name__ == "__main__":
    main()
