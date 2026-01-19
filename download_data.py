"""
Quick Data Download Script
Downloads PHM 2010 and other datasets automatically
"""

import os
import urllib.request
import zipfile
from pathlib import Path

def download_file(url, destination):
    """Download a file with progress"""
    print(f"Downloading from {url}...")
    urllib.request.urlretrieve(url, destination)
    print(f"Downloaded to {destination}")

def download_phm2010_kaggle():
    """
    Download PHM 2010 from Kaggle
    
    NOTE: You need Kaggle API credentials for this to work!
    
    Setup:
    1. Go to https://www.kaggle.com/settings
    2. Click "Create New API Token"
    3. Save kaggle.json to ~/.kaggle/
    4. Run: chmod 600 ~/.kaggle/kaggle.json
    """
    print("\n" + "="*60)
    print("DOWNLOADING PHM 2010 FROM KAGGLE")
    print("="*60)
    
    try:
        import kaggle
        
        # Create data directory
        data_dir = Path("./data/phm2010")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Download dataset
        kaggle.api.dataset_download_files(
            'vnigade/phm-2010-data-challenge-on-milling-machine',
            path=str(data_dir),
            unzip=True
        )
        
        print(f"✓ PHM 2010 dataset downloaded to {data_dir}")
        return True
        
    except ImportError:
        print("ERROR: Kaggle package not installed")
        print("Install with: pip install kaggle")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        print("\nManual download instructions:")
        print("1. Go to: https://www.kaggle.com/datasets/vnigade/phm-2010-data-challenge-on-milling-machine")
        print("2. Click 'Download'")
        print("3. Extract to ./data/phm2010/")
        return False

def download_uniwear_github():
    """
    Download UniWear dataset from GitHub
    This is already preprocessed and ready to use!
    """
    print("\n" + "="*60)
    print("DOWNLOADING UNIWEAR DATASET")
    print("="*60)
    
    data_dir = Path("./data/uniwear")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("Cloning repository...")
    os.system(f"git clone https://github.com/katulu-io/uniwear-dataset.git {data_dir}")
    
    print(f"✓ UniWear dataset downloaded to {data_dir}")
    return True

def download_nasa_milling():
    """Download NASA milling dataset"""
    print("\n" + "="*60)
    print("DOWNLOADING NASA MILLING DATASET")
    print("="*60)
    
    print("Manual download required:")
    print("1. Go to: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/")
    print("2. Find 'Milling Data Set'")
    print("3. Download and extract to ./data/nasa_milling/")
    
    return False

if __name__ == "__main__":
    print("="*60)
    print("TOOL WEAR DATA DOWNLOADER")
    print("="*60)
    
    print("\nWhich dataset do you want to download?")
    print("1. PHM 2010 (Kaggle - requires API key)")
    print("2. UniWear (GitHub - easiest!)")
    print("3. NASA Milling (Manual download)")
    print("4. All datasets")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        download_phm2010_kaggle()
    elif choice == '2':
        download_uniwear_github()
    elif choice == '3':
        download_nasa_milling()
    elif choice == '4':
        download_phm2010_kaggle()
        download_uniwear_github()
        download_nasa_milling()
    else:
        print("Invalid choice!")
    
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)
    print("\nNext: Update DATA_PATH in tool_wear_starter.py and run the script!")
