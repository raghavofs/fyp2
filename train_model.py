"""
Full two-stream water quality prediction pipeline.

Architecture (from diagram):
  Spatial Expert Stream:  Spectral bands -> GA-RF
  Temporal Expert Stream: CEEMDAN -> CNN -> LSTM -> Self-Attention
  Deep Fusion Engine:     Spatial + Temporal + Hydro-Met context -> Final prediction

Two model configurations:
  Model A: Hastings + Prescott, predicts Phosphorus + Nitrogen
  Model B: Hastings only, predicts all 5 WQ params (TP, TN, DO, Chl-a, Turbidity)
"""

import os
import sys
import json
import random
import warnings
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from PyEMD.EMD import EMD as _EMD_cls
from PyEMD.CEEMDAN import CEEMDAN as _CEEMDAN_cls
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed_clean")

# WQ_DIR and RUN_TAG can be overridden via command-line:
#   python train_model.py <wq_dir> <run_tag>
# e.g. python train_model.py data/augmented_real real_expansion
_WQ_DIR_DEFAULT = os.path.join(BASE_DIR, "data", "waterquality")
_RUN_TAG_DEFAULT = ""

SEED = 42
TIME_STEPS = 30
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
PATIENCE = 15  # early stopping

# Feature groups
SPECTRAL_BANDS = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2",
                  "RedEdge1", "RedEdge2", "RedEdge3"]

# Spectral indices derived from raw bands (computed in load_and_prepare_data)
SPECTRAL_INDICES = [
    "NDWI",          # (Green - NIR) / (Green + NIR) — water content
    "NDVI",          # (NIR - Red) / (NIR + Red) — vegetation/algae
    "NDTI",          # (Red - Green) / (Red + Green) — turbidity proxy
    "BR_ratio",      # Blue / Red — clarity indicator
    "GR_ratio",      # Green / Red — algae proxy
    "NIRG_ratio",    # NIR / Green — suspended sediment
    "RE1R_ratio",    # RedEdge1 / Red — chlorophyll index
    "SWIR_ratio",    # SWIR1 / SWIR2 — moisture content
    "Clay_Index",    # Red / Blue — clay/soil runoff marker (P adsorption proxy)
    "Particle_Size", # NIR - Red — fine silt vs coarse sand (P capacity proxy)
    "P_Load_Potential",  # Clay_Index × streamflow — runoff phosphorus loading
    "season_sin",    # sin(2π * DOY / 365) — seasonal cycle
    "season_cos",    # cos(2π * DOY / 365) — seasonal cycle
]

# Temporal summary stats computed over a 30-day window for each spectral feature
TEMPORAL_STATS = ["mean", "std", "slope", "min", "max", "last", "delta"]

# Base features for spatial stream (point-in-time indices + seasonal)
SPATIAL_BASE_FEATURES = SPECTRAL_BANDS + SPECTRAL_INDICES

# Full spatial feature list is built dynamically in load_and_prepare_data
# = base features (19) + temporal stats per band+index (19 × 7 = 133) = 152 total
SPATIAL_FEATURES = None  # set at runtime

HYDRO_MET_FEATURES = ["streamflow", "precipitation", "water_temp_c"]
TEMPORAL_FEATURES = HYDRO_MET_FEATURES + SPECTRAL_BANDS  # raw bands for temporal

# Model configs
MODEL_CONFIGS = {
    "model_a": {
        "name": "Model A - Phosphorus & Nitrogen",
        "stations": ["hastings", "prescott"],
        "targets": ["phosphorus", "nitrogen"],
    },
    "model_b": {
        "name": "Model B - All 5 WQ Parameters",
        "stations": ["hastings"],
        "targets": ["phosphorus", "nitrogen", "dissolved_oxygen_(do)",
                     "chlorophyll_a", "turbidity"],
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_spectral_indices(df):
    """Compute spectral indices and seasonal features from raw bands.
    Adds columns in-place and returns the dataframe."""
    eps = 1e-8  # avoid division by zero

    df["NDWI"] = (df["Green"] - df["NIR"]) / (df["Green"] + df["NIR"] + eps)
    df["NDVI"] = (df["NIR"] - df["Red"]) / (df["NIR"] + df["Red"] + eps)
    df["NDTI"] = (df["Red"] - df["Green"]) / (df["Red"] + df["Green"] + eps)
    df["BR_ratio"] = df["Blue"] / (df["Red"] + eps)
    df["GR_ratio"] = df["Green"] / (df["Red"] + eps)
    df["NIRG_ratio"] = df["NIR"] / (df["Green"] + eps)
    df["RE1R_ratio"] = df["RedEdge1"] / (df["Red"] + eps)
    df["SWIR_ratio"] = df["SWIR1"] / (df["SWIR2"] + eps)

    # Physics-based phosphorus features
    df["Clay_Index"] = df["Red"] / (df["Blue"] + eps)       # clay/soil runoff marker
    df["Particle_Size"] = df["NIR"] - df["Red"]             # fine silt (high P) vs coarse sand
    df["P_Load_Potential"] = (df["Red"] / (df["Blue"] + eps)) * df["streamflow"]  # runoff × sediment

    # Seasonal features from date
    doy = df["date"].dt.dayofyear
    df["season_sin"] = np.sin(2 * np.pi * doy / 365)
    df["season_cos"] = np.cos(2 * np.pi * doy / 365)

    return df


def compute_spatial_temporal_features(daily_df, idx, window=TIME_STEPS):
    """Compute temporal summary statistics of spectral features over a window.

    For each spectral band and index, computes:
    - mean, std, min, max over the window
    - slope (linear trend via polyfit)
    - last value (most recent day)
    - delta (last - first, i.e., net change over window)

    Returns a 1D array of all features for one sample.
    """
    features = SPECTRAL_BANDS + SPECTRAL_INDICES
    window_df = daily_df.iloc[idx - window:idx][features].values  # (window, n_features)

    result = []
    feature_names = []
    for j, feat in enumerate(features):
        col = window_df[:, j]
        result.append(np.nanmean(col))
        feature_names.append(f"{feat}_mean")
        result.append(np.nanstd(col))
        feature_names.append(f"{feat}_std")

        # Linear slope (trend)
        try:
            slope = np.polyfit(np.arange(len(col)), col, 1)[0]
        except (np.linalg.LinAlgError, ValueError):
            slope = 0.0
        result.append(slope)
        feature_names.append(f"{feat}_slope")

        result.append(np.nanmin(col))
        feature_names.append(f"{feat}_min")
        result.append(np.nanmax(col))
        feature_names.append(f"{feat}_max")
        result.append(col[-1])  # last value
        feature_names.append(f"{feat}_last")
        result.append(col[-1] - col[0])  # net change
        feature_names.append(f"{feat}_delta")

    return np.array(result, dtype=np.float64), feature_names


# ============================================================================
# 1. GA-RF Spatial Expert Stream
# ============================================================================
class GeneticAlgorithmFeatureSelector:
    """Simple GA for selecting optimal spectral band combinations for RF."""

    def __init__(self, n_features, pop_size=30, n_generations=20,
                 crossover_rate=0.7, mutation_rate=0.1, max_features=10):
        self.n_features = n_features
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_features = max_features  # hard cap on selected features

    def _init_population(self):
        pop = []
        for _ in range(self.pop_size):
            # Sparse initialization: select ~max_features features randomly
            chrom = np.zeros(self.n_features, dtype=int)
            n_sel = random.randint(2, min(self.max_features, self.n_features))
            indices = np.random.choice(self.n_features, n_sel, replace=False)
            chrom[indices] = 1
            pop.append(chrom)
        return np.array(pop)

    def _fitness(self, chrom, X_train, y_train, X_val, y_val):
        selected = np.where(chrom == 1)[0]
        if len(selected) == 0:
            return -1e6
        # Penalize selecting too many features (L0-style regularization)
        rf = RandomForestRegressor(n_estimators=50, random_state=SEED, n_jobs=-1)
        rf.fit(X_train[:, selected], y_train)
        preds = rf.predict(X_val[:, selected])
        mse = mean_squared_error(y_val, preds)
        # Penalty: 2% MSE increase per feature beyond max_features
        n_excess = max(0, len(selected) - self.max_features)
        penalty = mse * 0.02 * n_excess
        return -(mse + penalty)

    def _crossover(self, p1, p2):
        if random.random() < self.crossover_rate:
            pt = random.randint(1, self.n_features - 1)
            c1 = np.concatenate([p1[:pt], p2[pt:]])
            c2 = np.concatenate([p2[:pt], p1[pt:]])
            return c1, c2
        return p1.copy(), p2.copy()

    def _mutate(self, chrom):
        for i in range(self.n_features):
            if random.random() < self.mutation_rate:
                chrom[i] = 1 - chrom[i]
        if chrom.sum() == 0:
            chrom[random.randint(0, self.n_features - 1)] = 1
        # If over max_features, randomly drop excess
        while chrom.sum() > self.max_features:
            active = np.where(chrom == 1)[0]
            drop = np.random.choice(active)
            chrom[drop] = 0
        return chrom

    def run(self, X_train, y_train, X_val, y_val):
        pop = self._init_population()
        best_chrom = None
        best_fitness = -1e9

        for gen in range(self.n_generations):
            fitnesses = np.array([
                self._fitness(c, X_train, y_train, X_val, y_val) for c in pop
            ])

            gen_best_idx = np.argmax(fitnesses)
            if fitnesses[gen_best_idx] > best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best_chrom = pop[gen_best_idx].copy()

            # Tournament selection
            new_pop = [best_chrom.copy()]  # elitism
            while len(new_pop) < self.pop_size:
                i1, i2 = np.random.choice(self.pop_size, 2, replace=False)
                p1 = pop[i1] if fitnesses[i1] > fitnesses[i2] else pop[i2]
                i3, i4 = np.random.choice(self.pop_size, 2, replace=False)
                p2 = pop[i3] if fitnesses[i3] > fitnesses[i4] else pop[i4]
                c1, c2 = self._crossover(p1, p2)
                new_pop.append(self._mutate(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self._mutate(c2))
            pop = np.array(new_pop[:self.pop_size])

            if (gen + 1) % 5 == 0:
                n_sel = int(best_chrom.sum())
                print(f"    GA gen {gen+1}/{self.n_generations} | "
                      f"best MSE: {-best_fitness:.6f} | features: {n_sel}")

        selected_indices = np.where(best_chrom == 1)[0]
        return selected_indices, best_chrom


class SpatialExpertStream:
    """GA-RF spatial expert: uses genetic algorithm to select spectral bands,
    then trains a Random Forest on the selected features."""

    def __init__(self, band_names):
        self.band_names = band_names
        self.selected_indices = None
        self.selected_bands = None
        self.rf_model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def train(self, X_spectral, y_targets, train_mask, val_mask):
        X_train = self.scaler_X.fit_transform(X_spectral[train_mask])
        X_val = self.scaler_X.transform(X_spectral[val_mask])
        y_train = self.scaler_y.fit_transform(y_targets[train_mask])
        y_val = self.scaler_y.transform(y_targets[val_mask])

        # Use mean of targets for GA fitness (multi-output)
        y_train_mean = y_train.mean(axis=1)
        y_val_mean = y_val.mean(axis=1)

        print("  [GA] Running genetic algorithm for band selection...")
        ga = GeneticAlgorithmFeatureSelector(n_features=len(self.band_names))
        self.selected_indices, _ = ga.run(X_train, y_train_mean, X_val, y_val_mean)
        self.selected_bands = [self.band_names[i] for i in self.selected_indices]
        print(f"  [GA] Selected bands: {self.selected_bands}")

        print("  [RF] Training Random Forest on selected bands...")
        self.rf_model = RandomForestRegressor(
            n_estimators=200, max_depth=15, random_state=SEED, n_jobs=-1
        )
        self.rf_model.fit(X_train[:, self.selected_indices], y_train)

        # Evaluate
        y_pred_val = self.rf_model.predict(X_val[:, self.selected_indices])
        y_pred_actual = self.scaler_y.inverse_transform(y_pred_val)
        y_val_actual = self.scaler_y.inverse_transform(y_val)
        mse = mean_squared_error(y_val_actual, y_pred_actual)
        print(f"  [RF] Validation MSE: {mse:.6f}")

        return self

    def predict(self, X_spectral):
        """Returns spatial feature embeddings (RF leaf node predictions)."""
        X_scaled = self.scaler_X.transform(X_spectral)
        return self.rf_model.predict(X_scaled[:, self.selected_indices])

    def save(self, path):
        joblib.dump({
            "rf_model": self.rf_model,
            "selected_indices": self.selected_indices,
            "selected_bands": self.selected_bands,
            "scaler_X": self.scaler_X,
            "scaler_y": self.scaler_y,
            "band_names": self.band_names,
        }, path)

    def load(self, path):
        data = joblib.load(path)
        self.rf_model = data["rf_model"]
        self.selected_indices = data["selected_indices"]
        self.selected_bands = data["selected_bands"]
        self.scaler_X = data["scaler_X"]
        self.scaler_y = data["scaler_y"]
        self.band_names = data["band_names"]
        return self


# ============================================================================
# 2. CEEMDAN Decomposition
# ============================================================================
def apply_ceemdan(df, feature_cols):
    """Apply CEEMDAN decomposition to each feature column.
    Returns decomposed DataFrame and max_imfs count."""
    print("  [CEEMDAN] Decomposing features...")
    ceemdan = _CEEMDAN_cls(ext_EMD=_EMD_cls())
    ceemdan.noise_seed(SEED)

    all_imfs_raw = {}
    for col in feature_cols:
        try:
            imfs = ceemdan(df[col].values)
            all_imfs_raw[col] = imfs
        except Exception as e:
            print(f"    WARNING: CEEMDAN failed for {col}: {e}")
            # Fallback: use original signal as single IMF
            all_imfs_raw[col] = df[col].values.reshape(1, -1)

    max_imfs = max(imfs.shape[0] for imfs in all_imfs_raw.values())
    print(f"  [CEEMDAN] Max IMFs: {max_imfs}")

    # Pad all to max_imfs and build DataFrame
    all_imfs_df = pd.DataFrame(index=df.index)
    imf_columns = []
    for col in feature_cols:
        imfs = all_imfs_raw[col]
        n_imfs = imfs.shape[0]
        if n_imfs < max_imfs:
            padding = np.zeros((max_imfs - n_imfs, len(df)))
            imfs = np.vstack([imfs, padding])
        for i in range(max_imfs):
            col_name = f"{col}_imf_{i}"
            all_imfs_df[col_name] = imfs[i]
            imf_columns.append(col_name)

    return all_imfs_df, max_imfs, imf_columns


# ============================================================================
# 3. PyTorch Dataset and Temporal Model
# ============================================================================
class TimeSeriesDataset(Dataset):
    def __init__(self, X_temporal, X_spatial, X_context, y, mask=None):
        self.X_temporal = torch.FloatTensor(X_temporal)
        self.X_spatial = torch.FloatTensor(X_spatial)
        self.X_context = torch.FloatTensor(X_context)
        self.y = torch.FloatTensor(y)
        self.mask = torch.FloatTensor(mask) if mask is not None else torch.ones_like(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.X_temporal[idx], self.X_spatial[idx],
                self.X_context[idx], self.y[idx], self.mask[idx])


class SelfAttention(nn.Module):
    """Self-attention layer for temporal sequence weighting."""
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x):
        # x: (batch, seq_len, hidden)
        energy = torch.tanh(self.W(x))
        attention = torch.softmax(self.v(energy), dim=1)
        context = (attention * x).sum(dim=1)
        return context, attention


class TemporalExpertStream(nn.Module):
    """CEEMDAN-CNN-LSTM-SA temporal processing model."""
    def __init__(self, n_input_features, hidden_size=64):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input_features, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.lstm = nn.LSTM(64, hidden_size, num_layers=2,
                            batch_first=True, dropout=0.2)
        self.attention = SelfAttention(hidden_size)
        self.output_size = hidden_size

    def forward(self, x):
        # x: (batch, time_steps, features)
        x = x.permute(0, 2, 1)  # (batch, features, time_steps) for Conv1D
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # back to (batch, seq, features) for LSTM
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        return context


class TemporalHead(nn.Module):
    """Prediction head for the temporal stream."""
    def __init__(self, hidden_size, n_targets):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, n_targets),
        )

    def forward(self, x):
        return self.head(x)


class SpatialHead(nn.Module):
    """Prediction head for the spatial (GA-RF) stream."""
    def __init__(self, input_dim, n_targets):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_targets),
        )

    def forward(self, x):
        return self.head(x)


class AttentionFusionGate(nn.Module):
    """Attention-based late fusion: learns dynamic per-sample weights for
    each expert stream based on the hydro-met context and stream hidden states.

    Outputs interpretable weights (w_temporal, w_spatial) that sum to 1,
    and a fused prediction = w_temporal * temporal_pred + w_spatial * spatial_pred.
    """
    def __init__(self, temporal_hidden, spatial_dim, context_dim, n_targets):
        super().__init__()
        gate_input = temporal_hidden + spatial_dim + context_dim
        self.gate = nn.Sequential(
            nn.Linear(gate_input, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # 2 experts
        )
        self.n_targets = n_targets

    def forward(self, temporal_hidden, spatial_feat, context_feat,
                temporal_pred, spatial_pred):
        # Compute attention weights from hidden representations + context
        gate_input = torch.cat([temporal_hidden, spatial_feat, context_feat], dim=-1)
        logits = self.gate(gate_input)  # (batch, 2)
        weights = torch.softmax(logits, dim=-1)  # (batch, 2)

        w_temporal = weights[:, 0:1]  # (batch, 1)
        w_spatial = weights[:, 1:2]   # (batch, 1)

        # Weighted combination
        fused = w_temporal * temporal_pred + w_spatial * spatial_pred

        return fused, weights


class FullModel(nn.Module):
    """Two-stream model with attention-based late fusion.

    Each stream independently predicts targets. An attention gate learns
    dynamic weights based on context, producing interpretable fusion weights.
    """
    def __init__(self, n_temporal_features, n_spatial_features,
                 n_context_features, n_targets):
        super().__init__()
        self.temporal_stream = TemporalExpertStream(n_temporal_features)

        # Independent prediction heads
        self.temporal_head = TemporalHead(self.temporal_stream.output_size, n_targets)
        self.spatial_head = SpatialHead(n_spatial_features, n_targets)

        # Attention-based fusion gate
        self.fusion_gate = AttentionFusionGate(
            temporal_hidden=self.temporal_stream.output_size,
            spatial_dim=n_spatial_features,
            context_dim=n_context_features,
            n_targets=n_targets,
        )

    def forward(self, x_temporal, x_spatial, x_context):
        # Independent stream processing
        temporal_hidden = self.temporal_stream(x_temporal)
        temporal_pred = self.temporal_head(temporal_hidden)
        spatial_pred = self.spatial_head(x_spatial)

        # Attention-based fusion
        fused_pred, fusion_weights = self.fusion_gate(
            temporal_hidden, x_spatial, x_context,
            temporal_pred, spatial_pred,
        )

        return fused_pred, temporal_pred, spatial_pred, fusion_weights


# ============================================================================
# 4. Data Pipeline
# ============================================================================
WQ_DIR = _WQ_DIR_DEFAULT  # overridden in __main__ if CLI args provided


def _load_raw_wq_samples(station, targets):
    """Load actual WQ measurement dates and values (not interpolated)."""
    path = os.path.join(WQ_DIR, f"{station}_wq.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["ActivityStartDate"])
    df["ResultMeasureValue"] = pd.to_numeric(df["ResultMeasureValue"], errors="coerce")

    # Map characteristic names to our target column names
    char_map = {
        "Phosphorus": "phosphorus",
        "Nitrogen": "nitrogen",
        "Dissolved oxygen (DO)": "dissolved_oxygen_(do)",
        "Chlorophyll a": "chlorophyll_a",
        "Turbidity": "turbidity",
    }
    df["target_name"] = df["CharacteristicName"].map(char_map)
    df = df.dropna(subset=["target_name", "ResultMeasureValue"])

    # Pivot to wide format: one row per date with target columns
    pivot = df.pivot_table(
        index="date", columns="target_name",
        values="ResultMeasureValue", aggfunc="mean"
    ).reset_index()

    # Only keep dates that have at least one target we care about
    available = [t for t in targets if t in pivot.columns]
    if not available:
        return pd.DataFrame()

    return pivot


def load_and_prepare_data(config_key):
    """Load data and prepare train/val/test splits.

    Key design: input features are daily-interpolated for temporal windows,
    but targets come ONLY from actual WQ measurement dates.
    """
    config = MODEL_CONFIGS[config_key]
    print(f"\n{'='*60}")
    print(f"Preparing data for: {config['name']}")
    print(f"Stations: {config['stations']}")
    print(f"Targets: {config['targets']}")
    print(f"{'='*60}")

    temporal_feature_cols = HYDRO_MET_FEATURES + SPECTRAL_BANDS

    all_temporal_sequences = []
    all_spatial_at_t = []
    all_context_at_t = []
    all_y_at_t = []
    all_masks = []  # for masked loss (1=real sample, 0=missing)
    all_dates = []

    for station in config["stations"]:
        # Load daily interpolated data (for input features)
        daily_path = os.path.join(DATA_DIR, f"{station}_daily.csv")
        daily_df = pd.read_csv(daily_path, parse_dates=["date"])
        daily_df = daily_df.sort_values("date").reset_index(drop=True)

        # Compute spectral indices for spatial stream
        daily_df = compute_spectral_indices(daily_df)

        # Load actual WQ measurement dates
        wq_samples = _load_raw_wq_samples(station, config["targets"])
        if wq_samples.empty:
            print(f"  WARNING: No WQ samples for {station}, skipping")
            continue

        # Build date -> index mapping in daily data
        date_to_idx = {d: i for i, d in enumerate(daily_df["date"])}

        # Find WQ sample dates that fall within our daily data (after TIME_STEPS days)
        sample_dates = sorted(wq_samples["date"].values)
        valid_sample_indices = []
        for sd in sample_dates:
            sd_ts = pd.Timestamp(sd)
            if sd_ts in date_to_idx:
                idx = date_to_idx[sd_ts]
                if idx >= TIME_STEPS:
                    valid_sample_indices.append((sd_ts, idx))

        print(f"  {station}: {len(valid_sample_indices)} valid WQ sample dates "
              f"(of {len(sample_dates)} total)")

        if len(valid_sample_indices) == 0:
            continue

        # CEEMDAN on full daily input features (interpolated - this is fine for inputs)
        print(f"    Running CEEMDAN on {station}...")
        ceemdan_df, max_imfs, imf_cols = apply_ceemdan(daily_df, temporal_feature_cols)
        scaler_ceemdan = MinMaxScaler()
        ceemdan_scaled = scaler_ceemdan.fit_transform(ceemdan_df.values)

        # Build sequences only for actual WQ sample dates
        spatial_feature_names = None
        for sample_date, daily_idx in valid_sample_indices:
            # Temporal: 30-day window of CEEMDAN-decomposed input features
            temporal_seq = ceemdan_scaled[daily_idx - TIME_STEPS:daily_idx]
            all_temporal_sequences.append(temporal_seq)

            # Spatial: temporal summary stats of spectral features over 30-day window
            spat_feats, spatial_feature_names = compute_spatial_temporal_features(
                daily_df, daily_idx, window=TIME_STEPS)
            all_spatial_at_t.append(spat_feats)

            # Context: hydro-met on the sample date
            all_context_at_t.append(daily_df.iloc[daily_idx][HYDRO_MET_FEATURES].values.astype(float))

            # Target: actual WQ measurement (with mask for missing targets)
            wq_row = wq_samples[wq_samples["date"] == sample_date]
            target_vals = []
            mask_vals = []
            for t in config["targets"]:
                if t in wq_row.columns and not wq_row[t].isna().all():
                    target_vals.append(float(wq_row[t].values[0]))
                    mask_vals.append(1.0)
                else:
                    target_vals.append(0.0)  # placeholder
                    mask_vals.append(0.0)  # masked out in loss

            all_y_at_t.append(target_vals)
            all_masks.append(mask_vals)
            all_dates.append(sample_date)

    X_temporal = np.array(all_temporal_sequences)
    X_spatial_seq = np.nan_to_num(np.array(all_spatial_at_t), nan=0.0)  # handle NaN from stats
    X_context_seq = np.array(all_context_at_t)
    y_seq = np.array(all_y_at_t)
    mask_seq = np.array(all_masks)
    dates = np.array(all_dates)

    # Set global SPATIAL_FEATURES for GA-RF band naming
    global SPATIAL_FEATURES
    SPATIAL_FEATURES = spatial_feature_names

    n_imf_features = X_temporal.shape[2]
    n_samples = len(y_seq)

    print(f"\n  Final dataset shapes:")
    print(f"    X_temporal: {X_temporal.shape}  (samples, {TIME_STEPS}, {n_imf_features} IMF features)")
    print(f"    X_spatial:  {X_spatial_seq.shape}  ({len(SPATIAL_FEATURES)} spatio-temporal features)")
    print(f"    X_context:  {X_context_seq.shape}")
    print(f"    y:          {y_seq.shape}")
    print(f"    mask:       {mask_seq.shape}")
    for i, t in enumerate(config["targets"]):
        n_real = int(mask_seq[:, i].sum())
        print(f"      {t}: {n_real}/{n_samples} real samples")

    # --- Train/Val/Test split (70/15/15 chronological) ---
    n = n_samples
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    splits = {
        "train": slice(0, train_end),
        "val": slice(train_end, val_end),
        "test": slice(val_end, n),
    }

    print(f"\n  Splits: train={train_end}, val={val_end - train_end}, test={n - val_end}")

    return {
        "X_temporal": X_temporal,
        "X_spatial": X_spatial_seq,
        "X_context": X_context_seq,
        "y": y_seq,
        "mask": mask_seq,
        "dates": dates,
        "splits": splits,
        "n_imf_features": n_imf_features,
        "config": config,
    }


# ============================================================================
# 5. Training Loop
# ============================================================================
def masked_mse_loss(pred, target, mask):
    """MSE loss that only considers targets with real measurements."""
    diff = (pred - target) ** 2
    masked_diff = diff * mask
    # Average over non-masked elements
    n_valid = mask.sum()
    if n_valid == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return masked_diff.sum() / n_valid


def train_full_model(config_key):
    set_seed()
    data = load_and_prepare_data(config_key)
    config = data["config"]

    splits = data["splits"]
    n_targets = len(config["targets"])
    train_sl = splits["train"]
    val_sl = splits["val"]
    test_sl = splits["test"]

    # --- Scale targets (fit on TRAIN only, using masked real values) ---
    scaler_y = MinMaxScaler()
    y_train = data["y"][train_sl]
    mask_train = data["mask"][train_sl]
    y_fit = y_train.copy()
    for col in range(n_targets):
        valid_mask = mask_train[:, col] == 1
        if valid_mask.sum() > 0:
            col_mean = y_train[valid_mask, col].mean()
            y_fit[~valid_mask, col] = col_mean
    scaler_y.fit(y_fit)
    y_scaled = scaler_y.transform(data["y"])

    # --- Scale spatial and context features (fit on TRAIN only) ---
    scaler_spatial = MinMaxScaler()
    scaler_spatial.fit(data["X_spatial"][train_sl])
    X_spatial_scaled = scaler_spatial.transform(data["X_spatial"])

    scaler_context = MinMaxScaler()
    scaler_context.fit(data["X_context"][train_sl])
    X_context_scaled = scaler_context.transform(data["X_context"])

    # ===========================================
    # Train GA-RF Spatial Stream
    # ===========================================
    print(f"\n{'='*60}")
    print("Training GA-RF Spatial Expert Stream")
    print(f"{'='*60}")

    rf_train_mask = data["mask"][train_sl][:, 0] == 1
    rf_val_mask = data["mask"][val_sl][:, 0] == 1

    spatial_stream = SpatialExpertStream(SPATIAL_FEATURES)
    X_spat_train = X_spatial_scaled[train_sl][rf_train_mask]
    y_rf_train = y_scaled[train_sl][rf_train_mask]
    X_spat_val = X_spatial_scaled[val_sl][rf_val_mask]
    y_rf_val = y_scaled[val_sl][rf_val_mask]

    if len(X_spat_train) > 5 and len(X_spat_val) > 2:
        full_train = np.zeros(len(X_spat_train) + len(X_spat_val), dtype=bool)
        full_val = np.zeros(len(X_spat_train) + len(X_spat_val), dtype=bool)
        full_train[:len(X_spat_train)] = True
        full_val[len(X_spat_train):] = True
        X_spat_combined = np.vstack([X_spat_train, X_spat_val])
        y_rf_combined = np.vstack([y_rf_train, y_rf_val])
        spatial_stream.train(X_spat_combined, y_rf_combined, full_train, full_val)
    else:
        print("  WARNING: Not enough RF training data, using simple RF")
        spatial_stream.selected_indices = list(range(len(SPATIAL_FEATURES)))
        spatial_stream.selected_bands = SPATIAL_FEATURES
        spatial_stream.rf_model = RandomForestRegressor(n_estimators=100, random_state=SEED)
        spatial_stream.scaler_X.fit(X_spatial_scaled[train_sl])
        spatial_stream.scaler_y.fit(y_scaled[train_sl])
        X_t = spatial_stream.scaler_X.transform(X_spatial_scaled[train_sl])
        spatial_stream.rf_model.fit(X_t, spatial_stream.scaler_y.transform(y_scaled[train_sl]))

    # Get spatial predictions as features for the neural net
    spatial_preds = spatial_stream.predict(data["X_spatial"])
    scaler_spatial_preds = MinMaxScaler()
    scaler_spatial_preds.fit(spatial_preds[train_sl])
    spatial_preds_scaled = scaler_spatial_preds.transform(spatial_preds)

    # ===========================================
    # Build PyTorch Datasets
    # ===========================================
    train_ds = TimeSeriesDataset(
        data["X_temporal"][train_sl], spatial_preds_scaled[train_sl],
        X_context_scaled[train_sl], y_scaled[train_sl], data["mask"][train_sl])
    val_ds = TimeSeriesDataset(
        data["X_temporal"][val_sl], spatial_preds_scaled[val_sl],
        X_context_scaled[val_sl], y_scaled[val_sl], data["mask"][val_sl])
    test_ds = TimeSeriesDataset(
        data["X_temporal"][test_sl], spatial_preds_scaled[test_sl],
        X_context_scaled[test_sl], y_scaled[test_sl], data["mask"][test_sl])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # ===========================================
    # Build and Train Full Model
    # ===========================================
    print(f"\n{'='*60}")
    print("Training Attention-Based Two-Stream Fusion Model")
    print(f"{'='*60}")

    model = FullModel(
        n_temporal_features=data["n_imf_features"],
        n_spatial_features=n_targets,
        n_context_features=len(HYDRO_MET_FEATURES),
        n_targets=n_targets,
    ).to(DEVICE)

    print(f"  Device: {DEVICE}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7)

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    # Auxiliary loss weight for individual stream heads
    AUX_WEIGHT = 0.3

    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        epoch_loss = 0
        n_samples = 0
        for x_temp, x_spat, x_ctx, y_batch, m_batch in train_loader:
            x_temp, x_spat, x_ctx = x_temp.to(DEVICE), x_spat.to(DEVICE), x_ctx.to(DEVICE)
            y_batch, m_batch = y_batch.to(DEVICE), m_batch.to(DEVICE)

            optimizer.zero_grad()
            fused, temp_pred, spat_pred, weights = model(x_temp, x_spat, x_ctx)

            # Combined loss: fused + auxiliary per-stream losses
            loss_fused = masked_mse_loss(fused, y_batch, m_batch)
            loss_temp = masked_mse_loss(temp_pred, y_batch, m_batch)
            loss_spat = masked_mse_loss(spat_pred, y_batch, m_batch)
            loss = loss_fused + AUX_WEIGHT * (loss_temp + loss_spat)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss_fused.item() * m_batch.sum().item()
            n_samples += m_batch.sum().item()
        epoch_loss = epoch_loss / max(n_samples, 1)
        train_losses.append(epoch_loss)

        # --- Validate ---
        model.eval()
        val_loss, val_n = 0, 0
        with torch.no_grad():
            for x_temp, x_spat, x_ctx, y_batch, m_batch in val_loader:
                x_temp, x_spat, x_ctx = x_temp.to(DEVICE), x_spat.to(DEVICE), x_ctx.to(DEVICE)
                y_batch, m_batch = y_batch.to(DEVICE), m_batch.to(DEVICE)
                fused, _, _, _ = model(x_temp, x_spat, x_ctx)
                loss = masked_mse_loss(fused, y_batch, m_batch)
                val_loss += loss.item() * m_batch.sum().item()
                val_n += m_batch.sum().item()
        val_loss = val_loss / max(val_n, 1)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Train: {epoch_loss:.6f} | Val: {val_loss:.6f} | LR: {lr:.1e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    model.eval()

    # ===========================================
    # Evaluate: Per-stream + Fused + Weights
    # ===========================================
    print(f"\n{'='*60}")
    print("Test Set Evaluation (Per-Stream + Fused)")
    print(f"{'='*60}")

    all_fused, all_temp, all_spat, all_weights, all_true, all_masks_out = [], [], [], [], [], []
    with torch.no_grad():
        for x_temp, x_spat, x_ctx, y_batch, m_batch in test_loader:
            x_temp, x_spat, x_ctx = x_temp.to(DEVICE), x_spat.to(DEVICE), x_ctx.to(DEVICE)
            fused, temp_pred, spat_pred, weights = model(x_temp, x_spat, x_ctx)
            all_fused.append(fused.cpu().numpy())
            all_temp.append(temp_pred.cpu().numpy())
            all_spat.append(spat_pred.cpu().numpy())
            all_weights.append(weights.cpu().numpy())
            all_true.append(y_batch.numpy())
            all_masks_out.append(m_batch.numpy())

    fused_sc = np.concatenate(all_fused)
    temp_sc = np.concatenate(all_temp)
    spat_sc = np.concatenate(all_spat)
    test_weights = np.concatenate(all_weights)   # (n_test, 2)
    y_true_sc = np.concatenate(all_true)
    test_masks = np.concatenate(all_masks_out)

    # Inverse transform
    fused_pred = scaler_y.inverse_transform(fused_sc)
    temp_pred_inv = scaler_y.inverse_transform(temp_sc)
    spat_pred_inv = scaler_y.inverse_transform(spat_sc)
    y_true = scaler_y.inverse_transform(y_true_sc)

    # Compute per-stream and fused metrics
    def _compute_metrics(y_true, y_pred, mask, targets):
        m = {}
        for i, t in enumerate(targets):
            valid = mask[:, i] == 1
            n = int(valid.sum())
            if n < 2:
                m[t] = {"MSE": None, "MAE": None, "R2": None, "n_samples": n}
            else:
                m[t] = {
                    "MSE": mean_squared_error(y_true[valid, i], y_pred[valid, i]),
                    "MAE": mean_absolute_error(y_true[valid, i], y_pred[valid, i]),
                    "R2": r2_score(y_true[valid, i], y_pred[valid, i]),
                    "n_samples": n,
                }
        return m

    metrics_fused = _compute_metrics(y_true, fused_pred, test_masks, config["targets"])
    metrics_temporal = _compute_metrics(y_true, temp_pred_inv, test_masks, config["targets"])
    metrics_spatial = _compute_metrics(y_true, spat_pred_inv, test_masks, config["targets"])

    print(f"\n  {'Target':>25s} | {'Stream':>10s} | {'MSE':>10s} | {'MAE':>10s} | {'R²':>8s} | n")
    print("  " + "-" * 80)
    for t in config["targets"]:
        for label, m in [("Temporal", metrics_temporal), ("Spatial", metrics_spatial), ("Fused", metrics_fused)]:
            v = m[t]
            if v["R2"] is not None:
                print(f"  {t:>25s} | {label:>10s} | {v['MSE']:10.6f} | {v['MAE']:10.6f} | {v['R2']:8.4f} | {v['n_samples']}")
            else:
                print(f"  {t:>25s} | {label:>10s} | {'SKIP':>10s} | {'SKIP':>10s} | {'SKIP':>8s} | {v['n_samples']}")
        print("  " + "-" * 80)

    # Attention weight analysis
    w_temp_mean = test_weights[:, 0].mean()
    w_spat_mean = test_weights[:, 1].mean()
    print(f"\n  Avg attention weights: Temporal={w_temp_mean:.4f}, Spatial={w_spat_mean:.4f}")

    # ===========================================
    # Save Everything
    # ===========================================
    run_tag = globals().get("_ACTIVE_RUN_TAG", "")
    tag_suffix = f"_{run_tag}" if run_tag else ""
    save_dir = os.path.join(BASE_DIR, "trained_models", config_key + tag_suffix)
    os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(save_dir, "fusion_model.pt"))
    spatial_stream.save(os.path.join(save_dir, "spatial_stream.joblib"))

    joblib.dump({
        "scaler_y": scaler_y,
        "scaler_spatial": scaler_spatial,
        "scaler_context": scaler_context,
        "scaler_spatial_preds": scaler_spatial_preds,
    }, os.path.join(save_dir, "scalers.joblib"))

    all_metrics = {
        "fused": metrics_fused,
        "temporal_stream": metrics_temporal,
        "spatial_stream": metrics_spatial,
        "attention_weights": {
            "mean_temporal": float(w_temp_mean),
            "mean_spatial": float(w_spat_mean),
            "per_sample_temporal": test_weights[:, 0].tolist(),
            "per_sample_spatial": test_weights[:, 1].tolist(),
        },
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump({
            "config": config,
            "metrics": all_metrics,
            "n_imf_features": data["n_imf_features"],
            "best_val_loss": best_val_loss,
            "epochs_trained": len(train_losses),
        }, f, indent=2, default=str)

    # ===========================================
    # Generate Comprehensive Plots
    # ===========================================
    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    test_dates = data["dates"][test_sl]

    # 1. Loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"{config['name']} - Training Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "loss_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Per-target: 3-way comparison (temporal vs spatial vs fused)
    for i, target in enumerate(config["targets"]):
        valid = test_masks[:, i] == 1
        if valid.sum() < 2:
            continue
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={"height_ratios": [3, 1]})

        # Top: predictions
        ax = axes[0]
        ax.plot(test_dates[valid], y_true[valid, i], "ko-", label="Actual", ms=6)
        ax.plot(test_dates[valid], temp_pred_inv[valid, i], "b^--", label=f"Temporal (R²={metrics_temporal[target]['R2']:.3f})", ms=5, alpha=0.8)
        ax.plot(test_dates[valid], spat_pred_inv[valid, i], "rs--", label=f"Spatial (R²={metrics_spatial[target]['R2']:.3f})", ms=5, alpha=0.8)
        ax.plot(test_dates[valid], fused_pred[valid, i], "gD-", label=f"Fused (R²={metrics_fused[target]['R2']:.3f})", ms=5, alpha=0.9)
        ax.set_ylabel(target)
        ax.set_title(f"{config['name']} - {target}: Stream Comparison")
        ax.legend()
        ax.grid(True)

        # Bottom: attention weights over time
        ax2 = axes[1]
        ax2.bar(test_dates[valid], test_weights[valid, 0], width=5, label="Temporal weight", color="steelblue", alpha=0.7)
        ax2.bar(test_dates[valid], test_weights[valid, 1], width=5, bottom=test_weights[valid, 0], label="Spatial weight", color="coral", alpha=0.7)
        ax2.set_ylabel("Attention Weight")
        ax2.set_xlabel("Date")
        ax2.set_ylim(0, 1)
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"stream_comparison_{target}.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # 3. Attention weights distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(test_weights[:, 0], bins=20, color="steelblue", alpha=0.8, edgecolor="black")
    axes[0].set_xlabel("Temporal Weight")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Temporal Stream Weight Distribution")
    axes[0].axvline(w_temp_mean, color="red", linestyle="--", label=f"Mean={w_temp_mean:.3f}")
    axes[0].legend()

    axes[1].hist(test_weights[:, 1], bins=20, color="coral", alpha=0.8, edgecolor="black")
    axes[1].set_xlabel("Spatial Weight")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Spatial Stream Weight Distribution")
    axes[1].axvline(w_spat_mean, color="red", linestyle="--", label=f"Mean={w_spat_mean:.3f}")
    axes[1].legend()

    plt.suptitle(f"{config['name']} - Attention Weight Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "attention_weights.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Attention weights vs hydro-met context
    context_test = X_context_scaled[test_sl]
    context_names = HYDRO_MET_FEATURES
    fig, axes = plt.subplots(1, len(context_names), figsize=(5 * len(context_names), 4))
    if len(context_names) == 1:
        axes = [axes]
    for j, cname in enumerate(context_names):
        axes[j].scatter(context_test[:, j], test_weights[:, 0], alpha=0.6, c="steelblue", s=30, label="Temporal w")
        axes[j].scatter(context_test[:, j], test_weights[:, 1], alpha=0.6, c="coral", s=30, label="Spatial w")
        axes[j].set_xlabel(cname)
        axes[j].set_ylabel("Attention Weight")
        axes[j].legend()
        axes[j].grid(True, alpha=0.3)
    plt.suptitle(f"{config['name']} - Weights vs Context Features", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "weights_vs_context.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 5. Per-stream accuracy bar chart
    targets_with_data = [t for t in config["targets"] if metrics_fused[t]["R2"] is not None]
    if targets_with_data:
        x_pos = np.arange(len(targets_with_data))
        width = 0.25
        fig, ax = plt.subplots(figsize=(10, 6))
        r2_temp = [metrics_temporal[t]["R2"] for t in targets_with_data]
        r2_spat = [metrics_spatial[t]["R2"] for t in targets_with_data]
        r2_fused = [metrics_fused[t]["R2"] for t in targets_with_data]
        ax.bar(x_pos - width, r2_temp, width, label="Temporal (CEEMDAN-CNN-LSTM-SA)", color="steelblue")
        ax.bar(x_pos, r2_spat, width, label="Spatial (GA-RF)", color="coral")
        ax.bar(x_pos + width, r2_fused, width, label="Attention Fused", color="green")
        ax.set_xlabel("Target")
        ax.set_ylabel("R² Score")
        ax.set_title(f"{config['name']} - Per-Stream R² Comparison")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(targets_with_data, rotation=30, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "r2_comparison.png"), dpi=150, bbox_inches="tight")
        plt.close()

    print(f"\n  All assets saved to: {save_dir}")
    print(f"  Plots saved to: {plots_dir}")

    return model, all_metrics


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    # CLI: python train_model.py [wq_dir] [run_tag]
    if len(sys.argv) >= 2:
        WQ_DIR = os.path.join(BASE_DIR, sys.argv[1]) if not os.path.isabs(sys.argv[1]) else sys.argv[1]
    if len(sys.argv) >= 3:
        _ACTIVE_RUN_TAG = sys.argv[2]
    else:
        _ACTIVE_RUN_TAG = ""

    print("=" * 60)
    print("WATER QUALITY PREDICTION - FULL PIPELINE")
    print(f"Device: {DEVICE}")
    print(f"WQ data: {WQ_DIR}")
    if _ACTIVE_RUN_TAG:
        print(f"Run tag: {_ACTIVE_RUN_TAG}")
    print("=" * 60)

    print("\n\n" + "#" * 60)
    print("# MODEL A: Phosphorus + Nitrogen (Hastings + Prescott)")
    print("#" * 60)
    model_a, metrics_a = train_full_model("model_a")

    print("\n\n" + "#" * 60)
    print("# MODEL B: All 5 WQ Parameters (Hastings only)")
    print("#" * 60)
    model_b, metrics_b = train_full_model("model_b")

    print("\n\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    for label, metrics in [("Model A", metrics_a), ("Model B", metrics_b)]:
        print(f"\n{label} - Fused metrics:")
        for t, m in metrics["fused"].items():
            r2 = m['R2']
            print(f"  {t}: R²={r2:.4f}" if r2 is not None else f"  {t}: SKIPPED")
        wt = metrics["attention_weights"]["mean_temporal"]
        ws = metrics["attention_weights"]["mean_spatial"]
        print(f"  Avg weights: Temporal={wt:.3f}, Spatial={ws:.3f}")
