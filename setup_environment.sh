#!/bin/bash
# Tool Wear Prediction Project - Environment Setup

echo "Creating virtual environment..."
python3 -m venv tool_wear_env
source tool_wear_env/bin/activate

echo "Installing core dependencies..."
pip install --upgrade pip

# Core ML/DL libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas matplotlib seaborn scikit-learn scipy

# Signal processing
pip install pywavelets

# Progress tracking
pip install tqdm

# Jupyter for interactive development
pip install jupyter ipykernel
python -m ipykernel install --user --name=tool_wear_env

# Optional but recommended
pip install wandb optuna shap

echo "Setup complete! Activate with: source tool_wear_env/bin/activate"
