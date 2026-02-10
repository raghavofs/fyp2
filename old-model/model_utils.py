import os
import json
import joblib
import numpy as np
from tensorflow.keras.models import load_model, save_model

def save_model_assets(model, scaler_X, scaler_y, baseline_errors, max_imfs, scaler_X_columns, base_path='model_assets'):
    """Saves all model assets, including the scaler's column structure."""
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    model.save(os.path.join(base_path, 'water_quality_model.keras'))
    joblib.dump(scaler_X, os.path.join(base_path, 'scaler_X.gz'))
    joblib.dump(scaler_y, os.path.join(base_path, 'scaler_y.gz'))
    
    # Create a single dictionary for all other JSON assets
    assets = {
        'baseline_errors': baseline_errors,
        'max_imfs': max_imfs,
        'scaler_X_columns': scaler_X_columns # Save the crucial column list
    }

    with open(os.path.join(base_path, 'assets.json'), 'w') as f:
        json.dump(assets, f, indent=4)
        
    print(f"\n[INFO] Model and all assets saved to '{base_path}' directory.")

def load_model_assets(base_path='model_assets'):
    """Loads all model assets, including the scaler's column structure."""
    if not os.path.exists(base_path):
        print(f"[ERROR] Assets directory '{base_path}' not found.")
        return None, None, None, None, None, None

    model_path = os.path.join(base_path, 'water_quality_model.keras')
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found in '{base_path}'.")
        return None, None, None, None, None, None
        
    scaler_X = joblib.load(os.path.join(base_path, 'scaler_X.gz'))
    scaler_y = joblib.load(os.path.join(base_path, 'scaler_y.gz'))
    
    with open(os.path.join(base_path, 'assets.json'), 'r') as f:
        assets = json.load(f)
    
    baseline_errors = assets['baseline_errors']
    max_imfs = assets['max_imfs']
    scaler_X_columns = assets['scaler_X_columns'] # Load the crucial column list
        
    print(f"\n[INFO] Model assets loaded from '{base_path}'.")
    
    # Return all 6 components
    return model_path, scaler_X, scaler_y, baseline_errors, max_imfs, scaler_X_columns