# PREDICT: Physics-Regularized Estimation of Damage In Cutting Tools

A research framework for predicting cutting tool wear and optimizing machining parameters using machine learning and deep learning approaches.

## Abstract

This repository provides a comprehensive implementation for tool wear prediction in milling operations. The framework integrates traditional machine learning methods with physics-informed neural networks to achieve accurate wear estimation from multi-sensor signals. The system supports multiple benchmark datasets and provides baseline implementations achieving R² > 0.93 on standard evaluation metrics.

## Table of Contents

1. [Installation](#installation)
2. [Dataset Acquisition](#dataset-acquisition)
3. [Usage](#usage)
4. [Methodology](#methodology)
5. [Experimental Results](#experimental-results)
6. [Project Structure](#project-structure)
7. [Troubleshooting](#troubleshooting)
8. [References](#references)
9. [Citation](#citation)

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- scikit-learn 1.0+
- NumPy, Pandas, Matplotlib

### Environment Setup

**Option A: Using requirements.txt**
```bash
python3 -m venv tool_wear_env
source tool_wear_env/bin/activate  # On Windows: tool_wear_env\Scripts\activate
pip install -r requirements.txt
```

**Option B: Using setup script (Linux/macOS)**
```bash
chmod +x setup_environment.sh
./setup_environment.sh
source tool_wear_env/bin/activate
```

---

## Dataset Acquisition

### Supported Datasets

| Dataset | Description | Size | Complexity |
|---------|-------------|------|------------|
| **PHM 2010** | PHM Society Challenge dataset for milling tool wear | 500 MB | Moderate |
| **UniWear** | Preprocessed multi-condition wear dataset | 100 MB | Low |
| **NASA Milling** | NASA Prognostics Center milling dataset | 1 GB | Moderate |
| **NUAA** | Multi-condition cutting experiments | 200 MB | High |

### Automated Download

```bash
python download_data.py
# Select dataset option from menu
```

### Manual Download (PHM 2010)

1. Navigate to https://www.kaggle.com/datasets/vnigade/phm-2010-data-challenge-on-milling-machine
2. Download the dataset (requires Kaggle account)
3. Extract contents to `./data/phm2010/`

### Kaggle API Configuration

```bash
pip install kaggle
# Download kaggle.json from https://www.kaggle.com/settings
mkdir ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

python download_data.py
```

---

## Usage

### Running the Baseline Model

```bash
# Configure DATA_PATH in tool_wear_starter.py
python tool_wear_starter.py
```

### Expected Output

The baseline implementation performs the following operations:
- Loads and preprocesses PHM 2010 dataset
- Extracts 100+ statistical and frequency-domain features
- Trains Random Forest baseline model (R² > 0.93)
- Trains LSTM deep learning model
- Generates evaluation visualizations

Typical execution time: 5-10 minutes

Expected performance metrics:
- R²: 0.93-0.97
- RMSE: 0.02-0.04 mm

---

## Methodology

### Data Processing

The framework processes multi-sensor signals from milling operations:
- 6 cutting tools with progressive wear
- Force, vibration, and acoustic emission sensors
- Flank wear measurements per cutting pass

### Feature Engineering

The system extracts 100+ features per sample:

**Time-Domain Features:**
- Statistical moments (mean, standard deviation, RMS)
- Higher-order statistics (kurtosis, skewness)
- Signal characteristics (peak-to-peak, crest factor)

**Frequency-Domain Features:**
- FFT spectral energy
- Dominant frequency components
- Spectral distribution metrics

**Feature Dimensions:** 7 sensors × 15 features = 105 total features

### Model Architectures

**Random Forest Baseline:**
- Ensemble of decision trees
- Training time: ~30 seconds
- Provides feature importance analysis

**LSTM Network:**
- 2-layer bidirectional LSTM
- Dropout regularization
- Processes raw time-series directly

### Physics-Informed Extensions

Integration of Taylor's tool wear equation as a physics constraint:

```python
def taylor_constraint(speed, feed, depth, wear):
    n, a, b, C = 0.25, 0.5, 0.15, 400
    predicted_life = C / (speed**n * feed**a * depth**b)
    return (predicted_life - actual_life)**2
```

### Uncertainty Quantification

Monte Carlo Dropout for prediction uncertainty:

```python
model.train()  # Maintain dropout during inference
predictions = [model(x) for _ in range(100)]
mean_pred = np.mean(predictions)
std_pred = np.std(predictions)
```

---

## Experimental Results

### Performance Metrics

| Metric | Target | Baseline Performance |
|--------|--------|---------------------|
| RMSE | < 0.030 mm | 0.02-0.04 mm |
| R² | > 0.95 | 0.93-0.97 |
| MAE | < 0.020 mm | 0.015-0.025 mm |

### Computational Requirements

- Training time: ~5 minutes (Random Forest), ~30 minutes (LSTM)
- Inference time: < 100 ms per sample
- Memory: 4 GB RAM minimum

---

## Project Structure

```
PREDICT/
├── tool_wear_starter.py       # Main implementation
├── download_data.py           # Dataset acquisition utility
├── requirements.txt           # Python dependencies
├── setup_environment.sh       # Environment configuration
├── README.md                  # Documentation
│
├── data/                      # Dataset directory
│   ├── phm2010/
│   ├── uniwear/
│   └── nasa_milling/
│
└── results/                   # Output directory
    ├── models/                # Saved model weights
    ├── plots/                 # Visualization outputs
    └── logs/                  # Training logs
```

---

## Troubleshooting

### Module Import Errors

**"No module named 'torch'"**
```bash
pip install torch torchvision torchaudio
```

### Data Path Configuration

Update `DATA_PATH` in `tool_wear_starter.py`:
```python
DATA_PATH = "./data/phm2010"
```

### Memory Constraints

Reduce sequence length for LSTM training:
```python
if len(sensor_array) > 5000:
    indices = np.linspace(0, len(sensor_array)-1, 5000, dtype=int)
```

---

## References

### Primary Literature

1. PHM 2010 Data Challenge. Prognostics and Health Management Society.
2. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks. Journal of Computational Physics, 378, 686-707.
3. Li, X., et al. (2020). Tool wear prediction using XGBoost. Journal of Manufacturing Processes.
4. Wang, J., et al. (2023). BiLSTM with attention for tool wear monitoring. Mechanical Systems and Signal Processing.

### Resources

- PHM Society: https://www.phmsociety.org/
- NASA Prognostics Center: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/

---

## Citation

```bibtex
@misc{predict2025,
  title={PREDICT: Physics-Regularized Estimation of Damage In Cutting Tools},
  author={[Author Name]},
  year={2025},
  publisher={GitHub},
  url={https://github.com/[username]/PREDICT}
}
```

---

## License

This project is released for academic and research purposes.

---

## Quick Reference

```bash
# Environment setup
python3 -m venv tool_wear_env
source tool_wear_env/bin/activate
pip install -r requirements.txt

# Data acquisition
python download_data.py

# Execute baseline
python tool_wear_starter.py

# Jupyter notebook
jupyter notebook
```
