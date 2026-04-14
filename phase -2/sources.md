# Sources

## Local Sources Used Directly

- `data list paper.pdf`
  Purpose: source inventory and dataset reconnaissance.
- `data/uniwear/README.md`
  Purpose: documentation for the NUAA and PHM-derived Uniwear bundles and the orthogonal milling parameter ranges.
- `data/nasa_milling/Readme.pdf`
  Purpose: official experiment documentation for the NASA milling dataset, including fixed cutting speed and case-level feed/depth/material settings.
- `data/archive (1)/c1/c1_wear.csv`, `data/archive (1)/c4/c4_wear.csv`, `data/archive (1)/c6/c6_wear.csv`
  Purpose: PHM 2010 wear measurements used for early-regime exponent confirmation.
- `data/nasa_milling/csv/metadata.csv`
  Purpose: measured NASA flank-wear values for the late-regime fit.
- `data/uniwear/data/nuaa_orthogonal_bundle_high_resolution.csv`
  Purpose: primary source for the process-parameter wear law.

## External Sources Consulted

- PHM Society competition overview and PHM-related milling references:
  https://papers.phmsociety.org/index.php/phmconf/article/download/462/phmc_18_462
- Scientific Data 2025 open milling dataset paper:
  https://www.nature.com/articles/s41597-025-04923-y
- Sensors 2023 regression benchmark paper for milling tool-life estimation:
  https://doi.org/10.3390/s23239346

## Research Notes

- I did not fuse third-party web datasets directly into the phase-2 fit because the local workspace already includes three public milling datasets with different sensing and wear regimes.
- The most defensible phase-2 move was to use the local datasets first, then use web research mainly to identify expansion candidates and to confirm that newer open milling datasets now exist.
- The resulting model is intentionally structured around regime behavior:
  early wear from NUAA/PHM, accelerated wear from NASA.
