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
import matplotlib.pyplot as plt

# Import the final utility functions that handle all 6 assets
from model_utils import save_model_assets, load_model_assets

# --- Constants and Configuration ---
TIME_STEPS = 30
ANOMALY_THRESHOLD_MULTIPLIER = 3.0
ASSETS_PATH = 'model_assets'
PLOTS_PATH = 'plots'
STATE_FILE = os.path.join(ASSETS_PATH, 'simulation_state.json')

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

# --- Data Preparation with UNIFORM IMF PADDING & COLUMN SAVING ---
def prepare_data(filepath='waterdata.csv'):
    try:
        df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
        # Ensure all columns are numeric before processing!
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    except FileNotFoundError:
        print(f"[ERROR] Data file '{filepath}' not found.")
        return None, None, None, None, None, None, None, None, None, None
    # ... rest of your function ...

    feature_columns = df.columns
    
    print("[INFO] Applying CEEMDAN to find max number of IMFs...")
    ceemdan = CEEMDAN()
    all_imfs_raw = {col: ceemdan(df[col].values) for col in feature_columns}
    max_imfs = max(imfs.shape[0] for imfs in all_imfs_raw.values())
    print(f"[INFO] Determined max IMFs to be: {max_imfs}. All features will be padded.")

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
    
    # CRITICAL FIX: Save the exact column order the scaler is trained on
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
    train_size = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:train_size], X_seq[train_size:]
    y_train, y_test = y_seq[:train_size], y_seq[train_size:]
    
    return df, feature_columns, scaler_X, scaler_y, X_train, y_train, X_test, y_test, max_imfs, scaler_X_columns

# --- Model Training and Baselining Function ---
def train_and_baseline():
    res = prepare_data()
    if res[0] is None: return
    df, features, scaler_X, scaler_y, X_train, y_train, X_test, y_test, max_imfs, scaler_X_columns = res
    
    print("[INFO] Building and compiling model...")
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        MaxPooling1D(pool_size=2), LSTM(50, activation='relu', return_sequences=True), SelfAttention(), Dense(len(features))
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    print("\n[INFO] Training the model...")
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

    print("\n[INFO] Evaluating model and saving assets...")
    y_pred_scaled = model.predict(X_test)
    y_pred_actual = scaler_y.inverse_transform(y_pred_scaled)
    y_test_actual = scaler_y.inverse_transform(y_test)

    np.save(os.path.join(ASSETS_PATH, 'y_pred_actual.npy'), y_pred_actual)
    np.save(os.path.join(ASSETS_PATH, 'y_test_actual.npy'), y_test_actual)

    errors = y_test_actual - y_pred_actual
    baseline_errors = {feature: np.std(errors[:, i]) for i, feature in enumerate(features)}
    
    # CRITICAL FIX: Pass the column list to be saved along with other assets
    save_model_assets(model, scaler_X, scaler_y, baseline_errors, max_imfs, scaler_X_columns, ASSETS_PATH)

    initial_state = {'last_simulated_index': len(X_train) + TIME_STEPS}
    with open(STATE_FILE, 'w') as f: json.dump(initial_state, f)
    print(f"[INFO] Simulation state initialized.")

# --- Anomaly Detection Function ---
def check_for_anomalies(actual_row, predicted_row, baseline_errors, features):
    print("\n--- Anomaly Detection Report ---")
    alert_triggered = False
    for i, feature in enumerate(features):
        if feature in ['Turbidity_NTU', 'Ammonia_mgL', 'Nitrates_mgL', 'Phosphates_mgL', 'TDS_mgL', 'Conductivity_uS_cm']:
            predicted_val = predicted_row[0, i]
            actual_val = actual_row.iloc[0][feature]
            error = actual_val - predicted_val
            threshold = ANOMALY_THRESHOLD_MULTIPLIER * baseline_errors[feature]
            if error > threshold:
                alert_triggered = True
                print(f"  [ALERT!] {feature} is anomalously HIGH.")
                print(f"    - Actual: {actual_val:.2f}, Predicted: {predicted_val:.2f}")
                print(f"    - Exceeded error threshold of {threshold:.2f} by {(error - threshold):.2f}")
    if not alert_triggered:
        print("  [OK] No significant pollution increases detected.")
    print("---------------------------------")

# --- Simulation Function with ROBUST SCALING ---
def simulate_next_day():
    custom_objects = {'SelfAttention': SelfAttention}
    # CRITICAL FIX: Load all 6 assets, including the scaler_X_columns list
    loaded_assets = load_model_assets(ASSETS_PATH)
    if not loaded_assets: return
    model_path, scaler_X, scaler_y, baseline_errors, max_imfs, scaler_X_columns = loaded_assets
    model = load_model(model_path, custom_objects=custom_objects)
    df = pd.read_csv('waterdata.csv', index_col='Date', parse_dates=True)
    # --- FIX: Ensure index is datetime ---
    df.index = pd.to_datetime(df.index)
    features = df.columns
    with open(STATE_FILE, 'r') as f: state = json.load(f)
    current_index = state['last_simulated_index']
    
    if current_index >= len(df):
        print("\n[INFO] End of dataset reached. Resetting simulation.")
        current_index = int(len(df) * 0.8)

    print(f"\n--- Simulating for Date: {df.index[current_index].strftime('%Y-%m-%d')} ---")
    
    window_size = 200 
    start_index = max(0, current_index - window_size)
    data_window = df.iloc[start_index:current_index]

    ceemdan = CEEMDAN()
    all_imfs_window = pd.DataFrame()
    for col in features:
        imfs = ceemdan(data_window[col].values)
        num_imfs_found = imfs.shape[0]
        if num_imfs_found < max_imfs:
            padding = np.zeros((max_imfs - num_imfs_found, len(data_window)))
            imfs = np.vstack([imfs, padding])
        elif num_imfs_found > max_imfs:
            imfs = imfs[:max_imfs, :]
        imfs_df = pd.DataFrame(imfs.T, index=data_window.index)
        imfs_df.columns = [f'{col}_imf_{i}' for i in range(max_imfs)]
        all_imfs_window = pd.concat([all_imfs_window, imfs_df], axis=1)
    
    # --- THIS IS THE CRITICAL FIX ---
    # Reorder the columns to match the training data's structure EXACTLY before scaling
    try:
        all_imfs_window = all_imfs_window[scaler_X_columns]
    except KeyError as e:
        print(f"[ERROR] Column mismatch during simulation. The model may need retraining. Missing column: {e}")
        return
    # --- END OF FIX ---
    
    input_data_decomposed = all_imfs_window.tail(TIME_STEPS)
    input_data_scaled = scaler_X.transform(input_data_decomposed)
    input_data_reshaped = np.reshape(input_data_scaled, (1, TIME_STEPS, input_data_scaled.shape[1]))

    prediction_scaled = model.predict(input_data_reshaped)
    prediction_actual = scaler_y.inverse_transform(prediction_scaled)
    actual_data_row = df.iloc[[current_index]]
    
    print("\n--- Prediction vs. Actual ---")
    print("Prediction:", pd.Series(prediction_actual[0], index=features).round(2).to_dict())
    print("Actual:", actual_data_row[features].iloc[0].round(2).to_dict())
    
    check_for_anomalies(actual_data_row, prediction_actual, baseline_errors, features)
    
    state['last_simulated_index'] += 1
    with open(STATE_FILE, 'w') as f: json.dump(state, f)

# --- Performance Plotting Function ---
def generate_performance_plots():
    print("[INFO] Generating performance plots...")
    try:
        y_pred = np.load(os.path.join(ASSETS_PATH, 'y_pred_actual.npy'))
        y_test = np.load(os.path.join(ASSETS_PATH, 'y_test_actual.npy'))
        df = pd.read_csv('waterdata.csv')
        features = df.columns.drop('Date')
    except FileNotFoundError:
        print("[ERROR] Could not find prediction results. Please run training (Option 1) first.")
        return

    if not os.path.exists(PLOTS_PATH):
        os.makedirs(PLOTS_PATH)

    for i, feature in enumerate(features):
        plt.figure(figsize=(15, 6))
        plt.plot(y_test[:, i], color='blue', label='Actual Values')
        plt.plot(y_pred[:, i], color='red', linestyle='--', label='Predicted Values')
        plt.title(f'Model Performance for {feature}', fontsize=16)
        plt.xlabel('Time Step (in test set)', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.legend()
        plt.grid(True)
        
        safe_feature_name = feature.replace('/', '_').replace(' ', '_')
        plot_filename = os.path.join(PLOTS_PATH, f'performance_{safe_feature_name}.png')
        plt.savefig(plot_filename)
        plt.close()
        
        print(f"  - Saved plot for {feature} to {plot_filename}")
        
    print(f"\n[SUCCESS] All plots have been saved to the '{PLOTS_PATH}' directory.")

# --- Main Menu ---
def main():
    while True:
        print("\n--- Water Quality Monitoring Simulation ---")
        print("1. Train Model & Establish Error Baseline")
        print("2. Simulate Next Day's Reading & Check for Anomalies")
        print("3. Generate & Save Performance Plots")
        print("4. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            train_and_baseline()
        elif choice == '2':
            simulate_next_day()
        elif choice == '3':
            generate_performance_plots()
        elif choice == '4':
            break
        else:
            print("\n[ERROR] Invalid choice. Please try again.")

if __name__ == '__main__':
    main()