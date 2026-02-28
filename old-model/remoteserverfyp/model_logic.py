import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Layer
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from PyEMD import CEEMDAN

# Import the utility functions
from model_utils import save_model_assets, load_model_assets

# --- Constants ---
TIME_STEPS = 30
DATA_FILE = 'waterdata.csv'
ASSETS_PATH = 'model_assets'

# --- Self-Attention Layer Definition ---
class SelfAttention(Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_W', shape=(input_shape[-1], input_shape[-1]), initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='attention_b', shape=(input_shape[-1],), initializer='zeros', trainable=True)
        super(SelfAttention, self).build(input_shape)
    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

# --- Refactored Training Function ---
def train_and_baseline():
    """
    Loads the latest data, retrains the model, and saves the new assets.
    This is the core function for the quarterly retraining.
    """
    try:
        df = pd.read_csv(DATA_FILE, index_col='Date', parse_dates=True)
    except FileNotFoundError:
        print(f"[ERROR] Data file '{DATA_FILE}' not found.")
        return False
        
    feature_columns = df.columns
    
    print("[TRAINING] Applying CEEMDAN to find max number of IMFs...")
    ceemdan = CEEMDAN()
    all_imfs_raw = {col: ceemdan(df[col].values) for col in feature_columns}
    max_imfs = max(imfs.shape[0] for imfs in all_imfs_raw.values())
    print(f"[TRAINING] Determined max IMFs to be: {max_imfs}.")

    all_imfs_df = pd.DataFrame()
    for col in feature_columns:
        imfs = all_imfs_raw[col]
        num_imfs_found = imfs.shape[0]
        if num_imfs_found < max_imfs:
            padding = np.zeros((max_imfs - num_imfs_found, len(df)))
            imfs = np.vstack([imfs, padding])
        imfs_df = pd.DataFrame(imfs.T, index=df.index)
        imfs_df.columns = [f'{col}_imf_{i}' for i in range(max_imfs)]
        all_imfs_df = pd.concat([all_imfs_df, imfs_df], axis=1)
    
    scaler_X_columns = all_imfs_df.columns.tolist()
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_decomposed_scaled = scaler_X.fit_transform(all_imfs_df)
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_original_scaled = scaler_y.fit_transform(df[feature_columns])
    
    def create_sequences(input_data, output_data, time_steps):
        X, y = [], []
        for i in range(len(input_data) - time_steps):
            X.append(input_data[i:(i + time_steps)])
            y.append(output_data[i + time_steps])
        return np.array(X), np.array(y)
        
    X_seq, y_seq = create_sequences(X_decomposed_scaled, y_original_scaled, TIME_STEPS)
    
    print("[TRAINING] Building and compiling model...")
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_seq.shape[1], X_seq.shape[2])),
        MaxPooling1D(pool_size=2), LSTM(50, activation='relu', return_sequences=True), SelfAttention(), Dense(len(feature_columns))
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    print("\n[TRAINING] Training the new model...")
    # Note: No train/test split here, we train on ALL available data for deployment
    model.fit(X_seq, y_seq, epochs=50, batch_size=32, verbose=1)

    print("\n[TRAINING] Calculating new baseline errors...")
    y_pred_scaled = model.predict(X_seq)
    y_pred_actual = scaler_y.inverse_transform(y_pred_scaled)
    y_actual = scaler_y.inverse_transform(y_seq)
    errors = y_actual - y_pred_actual
    baseline_errors = {feature: np.std(errors[:, i]) for i, feature in enumerate(feature_columns)}
    
    save_model_assets(model, scaler_X, scaler_y, baseline_errors, max_imfs, scaler_X_columns, ASSETS_PATH)
    print("[TRAINING] New model and assets have been saved.")
    return True
