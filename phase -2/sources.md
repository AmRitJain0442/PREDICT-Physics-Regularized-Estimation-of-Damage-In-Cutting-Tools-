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
- Scientific Data 2025 QIT-CEMC full-life coated end-milling cutter wear dataset:
  https://www.nature.com/articles/s41597-024-04345-2
- PHM conference review paper covering PHM2010 milling cutter regression usage:
  https://papers.phmsociety.org/index.php/phmconf/article/download/462/phmc_18_462
- PHM Society 2013 paper on RUL uncertainty:
  https://papers.phmsociety.org/index.php/phmconf/article/view/2263
- PHM Society 2022 paper on Gaussian-process RUL:
  https://papers.phmsociety.org/index.php/phmconf/article/view/3220
- DeepSurv survival-analysis paper:
  https://arxiv.org/abs/1606.00931
- Physics-informed Gaussian-process tool-wear paper:
  https://pubmed.ncbi.nlm.nih.gov/37770369/
- Neural-network spindle-power tool-life paper:
  https://impact.ornl.gov/en/publications/tool-life-predictions-in-milling-using-spindle-power-with-the-neu
- PI-KAF physics-informed interpretable tool-wear monitoring paper:
  https://www.sciencedirect.com/science/article/abs/pii/S0278612525002833

## Research Notes

- I did not fuse third-party web datasets directly into the phase-2 fit because the local workspace already includes three public milling datasets with different sensing and wear regimes.
- The most defensible phase-2 move was to use the local datasets first, then use web research mainly to identify expansion candidates and to confirm that newer open milling datasets now exist.
- The resulting model is intentionally structured around regime behavior:
  early wear from NUAA/PHM, accelerated wear from NASA.
- The extended research pass adds uncertainty propagation and early-calibration diagnostics. It still does not directly ingest the 2025 external datasets because the current repository does not include those raw files and both are large enough to deserve a controlled data-ingestion plan.
- The lifetime model-extraction pass adds a survival-analysis perspective from biomedical time-to-event modeling, trains NUAA wear models, inverts trained models into life estimates, and fits censored accelerated-failure-time equations.
