"""
Tool Wear Prediction - PHM 2010 Dataset
Starter Code for Data Loading and Baseline Model

This script will:
1. Load and explore the PHM 2010 milling dataset
2. Extract basic features from sensor signals
3. Build a baseline Random Forest model
4. Train a simple LSTM model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

print("Libraries imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")


# =============================================================================
# PART 1: DATA LOADING FOR PHM 2010 DATASET
# =============================================================================

class PHM2010DataLoader:
    """
    Loader for PHM 2010 Challenge Dataset
    
    Dataset Structure:
    - c1, c4, c6 (training): Each has multiple cut files with sensor data
    - c1_wear.csv, c4_wear.csv, c6_wear.csv: Tool wear measurements
    - Sensors: force_x, force_y, force_z, vib_table, vib_spindle, AE_table, AE_spindle
    """
    
    def __init__(self, data_path):
        """
        Args:
            data_path (str): Path to the PHM 2010 dataset folder
        """
        self.data_path = Path(data_path)
        self.cutters = ['c1', 'c4', 'c6']  # Training cutters
        
    def load_wear_data(self):
        """Load wear measurements for all cutters"""
        wear_data = {}
        
        for cutter in self.cutters:
            wear_file = self.data_path / f"{cutter}_wear.csv"
            if wear_file.exists():
                df = pd.read_csv(wear_file)
                wear_data[cutter] = df
                print(f"Loaded {cutter}: {len(df)} wear measurements")
            else:
                print(f"Warning: {wear_file} not found")
                
        return wear_data
    
    def load_sensor_data(self, cutter, cut_number):
        """
        Load sensor data for a specific cut
        
        Args:
            cutter (str): Cutter name (e.g., 'c1')
            cut_number (int): Cut number
            
        Returns:
            pd.DataFrame: Sensor readings
        """
        cut_file = self.data_path / cutter / f"{cutter}_cut_{cut_number}.csv"
        
        if cut_file.exists():
            return pd.read_csv(cut_file)
        else:
            return None
    
    def load_all_data(self):
        """Load all available sensor data with wear labels"""
        all_data = []
        wear_data = self.load_wear_data()
        
        for cutter in self.cutters:
            if cutter not in wear_data:
                continue
                
            wear_df = wear_data[cutter]
            
            print(f"\nProcessing {cutter}...")
            for idx, row in tqdm(wear_df.iterrows(), total=len(wear_df)):
                cut_num = row['cut']
                flank_wear = row['flank wear']
                
                # Load sensor data
                sensor_df = self.load_sensor_data(cutter, cut_num)
                
                if sensor_df is not None:
                    all_data.append({
                        'cutter': cutter,
                        'cut': cut_num,
                        'flank_wear': flank_wear,
                        'sensor_data': sensor_df
                    })
        
        return all_data


# =============================================================================
# PART 2: FEATURE ENGINEERING
# =============================================================================

class FeatureExtractor:
    """Extract statistical features from time-series sensor data"""
    
    @staticmethod
    def extract_statistical_features(signal):
        """
        Extract time-domain statistical features
        
        Args:
            signal (np.array): Time-series signal
            
        Returns:
            dict: Dictionary of features
        """
        features = {
            'mean': np.mean(signal),
            'std': np.std(signal),
            'max': np.max(signal),
            'min': np.min(signal),
            'rms': np.sqrt(np.mean(signal**2)),
            'kurtosis': pd.Series(signal).kurtosis(),
            'skewness': pd.Series(signal).skew(),
            'peak_to_peak': np.max(signal) - np.min(signal),
        }
        
        # Add percentiles
        for p in [25, 50, 75]:
            features[f'percentile_{p}'] = np.percentile(signal, p)
            
        return features
    
    @staticmethod
    def extract_frequency_features(signal, sampling_rate=50000):
        """
        Extract frequency-domain features using FFT
        
        Args:
            signal (np.array): Time-series signal
            sampling_rate (int): Sampling rate in Hz
            
        Returns:
            dict: Dictionary of frequency features
        """
        # Compute FFT
        fft_vals = np.fft.rfft(signal)
        fft_freq = np.fft.rfftfreq(len(signal), 1/sampling_rate)
        fft_power = np.abs(fft_vals)**2
        
        features = {
            'freq_mean': np.mean(fft_power),
            'freq_std': np.std(fft_power),
            'freq_max': np.max(fft_power),
            'dominant_freq': fft_freq[np.argmax(fft_power)],
            'freq_energy': np.sum(fft_power),
        }
        
        return features
    
    def extract_all_features(self, sensor_df):
        """
        Extract features from all sensor channels
        
        Args:
            sensor_df (pd.DataFrame): Sensor data with multiple channels
            
        Returns:
            dict: All extracted features
        """
        all_features = {}
        
        # Expected sensor columns
        sensor_cols = ['force_x', 'force_y', 'force_z', 
                      'vib_table', 'vib_spindle', 
                      'AE_table', 'AE_spindle']
        
        # Extract features for each available sensor
        for col in sensor_cols:
            if col in sensor_df.columns:
                signal = sensor_df[col].values
                
                # Time-domain features
                time_features = self.extract_statistical_features(signal)
                for feat_name, feat_val in time_features.items():
                    all_features[f'{col}_{feat_name}'] = feat_val
                
                # Frequency-domain features
                freq_features = self.extract_frequency_features(signal)
                for feat_name, feat_val in freq_features.items():
                    all_features[f'{col}_{feat_name}'] = feat_val
        
        return all_features


# =============================================================================
# PART 3: BASELINE MODEL - RANDOM FOREST
# =============================================================================

def train_baseline_model(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a Random Forest baseline model
    
    Args:
        X_train, X_test: Feature arrays
        y_train, y_test: Target (flank wear) arrays
        
    Returns:
        Trained model and metrics
    """
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST BASELINE MODEL")
    print("="*60)
    
    # Train model
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training...")
    rf_model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print(f"\nTraining RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': range(X_train.shape[1]),
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    return rf_model, {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'predictions': y_test_pred
    }


# =============================================================================
# PART 4: DEEP LEARNING MODEL - LSTM
# =============================================================================

class ToolWearDataset(Dataset):
    """PyTorch Dataset for tool wear prediction"""
    
    def __init__(self, sensor_data, wear_labels):
        """
        Args:
            sensor_data (list): List of sensor DataFrames
            wear_labels (np.array): Wear labels
        """
        self.sensor_data = sensor_data
        self.wear_labels = wear_labels
        
    def __len__(self):
        return len(self.wear_labels)
    
    def __getitem__(self, idx):
        # Get sensor data and convert to tensor
        sensor_df = self.sensor_data[idx]
        
        # Select sensor columns
        sensor_cols = ['force_x', 'force_y', 'force_z', 
                      'vib_table', 'vib_spindle', 
                      'AE_table', 'AE_spindle']
        available_cols = [col for col in sensor_cols if col in sensor_df.columns]
        
        # Extract sensor values
        sensor_array = sensor_df[available_cols].values
        
        # Downsample if too long (for memory efficiency)
        if len(sensor_array) > 10000:
            indices = np.linspace(0, len(sensor_array)-1, 10000, dtype=int)
            sensor_array = sensor_array[indices]
        
        # Convert to tensor
        x = torch.FloatTensor(sensor_array)
        y = torch.FloatTensor([self.wear_labels[idx]])
        
        return x, y


class LSTMToolWearModel(nn.Module):
    """LSTM-based tool wear prediction model"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMToolWearModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # x shape: (batch, sequence_length, input_size)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last hidden state
        last_hidden = hidden[-1]
        
        # Pass through fully connected layers
        output = self.fc(last_hidden)
        
        return output


def train_lstm_model(train_loader, val_loader, input_size, num_epochs=20, device='cpu'):
    """
    Train LSTM model
    
    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        input_size: Number of sensor channels
        num_epochs: Number of training epochs
        device: 'cpu' or 'cuda'
        
    Returns:
        Trained model and training history
    """
    print("\n" + "="*60)
    print("TRAINING LSTM MODEL")
    print("="*60)
    
    # Initialize model
    model = LSTMToolWearModel(input_size=input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    return model, history


# =============================================================================
# PART 5: VISUALIZATION
# =============================================================================

def plot_results(y_true, y_pred, title="Predictions vs Actual"):
    """Plot prediction results"""
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], 
             [y_true.min(), y_true.max()], 
             'r--', lw=2)
    plt.xlabel('Actual Wear (mm)')
    plt.ylabel('Predicted Wear (mm)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    residuals = y_true - y_pred
    plt.hist(residuals, bins=30, edgecolor='black')
    plt.xlabel('Prediction Error (mm)')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("TOOL WEAR PREDICTION - PHM 2010 DATASET")
    print("="*60)
    
    # ==================================================
    # STEP 1: Set your data path
    # ==================================================
    # CHANGE THIS to your actual data path
    DATA_PATH = "./phm2010_data"  # Update this path!
    
    print(f"\nData path: {DATA_PATH}")
    print("Make sure this folder contains c1/, c4/, c6/ subfolders and wear CSV files")
    
    # ==================================================
    # STEP 2: Load data
    # ==================================================
    loader = PHM2010DataLoader(DATA_PATH)
    
    # Load wear data first to check
    print("\nLoading wear measurements...")
    wear_data = loader.load_wear_data()
    
    if not wear_data:
        print("\nERROR: No wear data found!")
        print("Please download the PHM 2010 dataset and update DATA_PATH")
        exit(1)
    
    # Visualize wear progression for each cutter
    plt.figure(figsize=(12, 4))
    for i, (cutter, df) in enumerate(wear_data.items(), 1):
        plt.subplot(1, 3, i)
        plt.plot(df['cut'], df['flank wear'], marker='o')
        plt.xlabel('Cut Number')
        plt.ylabel('Flank Wear (mm)')
        plt.title(f'{cutter.upper()} Wear Progression')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('wear_progression.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Load all sensor data with labels
    print("\nLoading sensor data...")
    all_data = loader.load_all_data()
    
    if not all_data:
        print("\nERROR: No sensor data found!")
        exit(1)
    
    print(f"\nTotal samples loaded: {len(all_data)}")
    
    # ==================================================
    # STEP 3: Feature extraction
    # ==================================================
    print("\nExtracting features...")
    feature_extractor = FeatureExtractor()
    
    X_features = []
    y_labels = []
    
    for sample in tqdm(all_data):
        features = feature_extractor.extract_all_features(sample['sensor_data'])
        X_features.append(features)
        y_labels.append(sample['flank_wear'])
    
    # Convert to DataFrame and array
    X_df = pd.DataFrame(X_features)
    X = X_df.values
    y = np.array(y_labels)
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Wear range: {y.min():.4f} - {y.max():.4f} mm")
    
    # ==================================================
    # STEP 4: Train/test split and scaling
    # ==================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # ==================================================
    # STEP 5: Train baseline Random Forest model
    # ==================================================
    rf_model, rf_metrics = train_baseline_model(
        X_train_scaled, y_train, 
        X_test_scaled, y_test
    )
    
    # Plot results
    plot_results(y_test, rf_metrics['predictions'], 
                title="Random Forest: Predictions vs Actual")
    
    # ==================================================
    # STEP 6: Prepare data for LSTM (optional - more advanced)
    # ==================================================
    print("\n" + "="*60)
    print("LSTM MODEL (Optional - Comment out if you want to skip)")
    print("="*60)
    
    # Note: LSTM requires sequence data, so we keep sensor time series
    # Split indices for train/test
    train_idx = int(0.8 * len(all_data))
    train_sensor_data = [d['sensor_data'] for d in all_data[:train_idx]]
    test_sensor_data = [d['sensor_data'] for d in all_data[train_idx:]]
    train_wear = np.array([d['flank_wear'] for d in all_data[:train_idx]])
    test_wear = np.array([d['flank_wear'] for d in all_data[train_idx:]])
    
    # Create datasets
    train_dataset = ToolWearDataset(train_sensor_data, train_wear)
    test_dataset = ToolWearDataset(test_sensor_data, test_wear)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Determine input size (number of sensor channels)
    sample_sensors = train_sensor_data[0]
    sensor_cols = ['force_x', 'force_y', 'force_z', 
                  'vib_table', 'vib_spindle', 
                  'AE_table', 'AE_spindle']
    input_size = len([col for col in sensor_cols if col in sample_sensors.columns])
    
    print(f"\nInput size (sensor channels): {input_size}")
    
    # Train LSTM
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lstm_model, history = train_lstm_model(
        train_loader, test_loader, 
        input_size=input_size,
        num_epochs=20,
        device=device
    )
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('LSTM Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('lstm_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Try different features (wavelet, more frequency bands)")
    print("2. Implement physics-informed constraints")
    print("3. Add uncertainty quantification")
    print("4. Build the parameter optimization loop")
