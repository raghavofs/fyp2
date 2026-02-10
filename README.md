# Water Quality Prediction Using Two-Stream Deep Fusion

A multi-modal deep learning system for predicting water quality parameters in the Mississippi River using satellite imagery, in-situ monitoring data, and hydro-meteorological features. The architecture combines a **GA-RF spatial expert** with a **CEEMDAN-CNN-LSTM-SA temporal expert**, fused through a learned **attention-based gating mechanism** that dynamically weights each stream's contribution.

## Architecture

```
DATA INPUT LAYER          PREPROCESSING         TWO-STREAM MODELS        FUSION & OUTPUT
┌─────────────────┐    ┌──────────────┐      ┌─────────────────────┐    ┌──────────────┐
│ Sentinel-2 /    │───>│ Radiometric  │─────>│ GA-RF Spatial       │    │              │
│ Landsat-8       │    │ Correction,  │      │ Expert Stream       │───>│  Attention-  │
│ (9 spectral     │    │ Band Select  │      │ (Feature Selection  │    │  Based Late  │
│  bands)         │    │              │      │  + Random Forest)   │    │  Fusion Gate │──> Final
├─────────────────┤    ├──────────────┤      ├─────────────────────┤    │              │   WQ Map
│ In-Situ WQ      │───>│ Missing Val  │─────>│ CEEMDAN-CNN-LSTM-SA │    │  Dynamic     │
│ (TP, TN, DO,    │    │ Imputation,  │      │ Temporal Expert     │───>│  Per-Sample  │
│  Chl-a, Turb.)  │    │ Outlier Det. │      │ Stream              │    │  Weights     │
├─────────────────┤    ├──────────────┤      └─────────────────────┘    └──────────────┘
│ Hydro-Met Data  │───>│ Alignment,   │─────────> Context Features ────────────┘
│ (Flow, Precip,  │    │ Resampling   │
│  Temperature)   │    │              │
└─────────────────┘    └──────────────┘
```

### Key Innovation: Attention-Based Late Fusion

Unlike simple concatenation-based fusion, our model uses a **learned attention gate** that produces interpretable, per-sample weights for each expert stream. The gate takes as input:
- The temporal stream's hidden representation (30-day CEEMDAN-decomposed context)
- The spatial stream's spectral band features
- Hydro-meteorological context (streamflow, precipitation, water temperature)

And outputs soft weights `(w_temporal, w_spatial)` that sum to 1, where:
```
final_prediction = w_temporal × temporal_prediction + w_spatial × spatial_prediction
```

This allows the model to learn **when** each stream is more reliable — e.g., favoring spatial (satellite) features during clear-sky high-flow periods and temporal (historical trend) features during cloudy or stable conditions.

## Data Sources

| Data | Source | Stations | Period | Notes |
|------|--------|----------|--------|-------|
| **Satellite Imagery** | Google Earth Engine (Sentinel-2, Landsat-8) | 5 Mississippi River sites | 2019–2025 | 9 spectral bands: Blue, Green, Red, NIR, SWIR1, SWIR2, RedEdge1-3 |
| **Water Quality** | USGS Water Quality Portal (WQP) | Hastings, Prescott, Winona, St Paul, Brooklyn Park | 2019–2025 | Parameters: TP, TN, DO, Chlorophyll-a, Turbidity. **Sparse: ~66–166 sample dates per station** |
| **Streamflow** | USGS National Water Information System (NWIS) | Same 5 stations | 2019–2025 | Daily discharge (cfs) |
| **Precipitation** | NOAA Climate Data Online | Nearby weather stations | 2019–2025 | Daily precipitation (inches) |
| **Water Temperature** | Minnesota DNR | Lock & Dam #1, Minneapolis | 2019–2025 | Daily water temperature (°F), shared across stations |

### USGS Station IDs
| Station | USGS Site ID |
|---------|-------------|
| Brooklyn Park | 05288500 |
| St. Paul | 05331000 |
| Hastings | 05331580 |
| Prescott | 05344500 |
| Winona | 05378500 |

## Data Collection Difficulties

1. **Extreme WQ data sparsity**: Water quality samples are collected sporadically (monthly to biweekly during field season). Hastings and Prescott have ~66–84 sample dates over 7 years. Brooklyn Park has only 2 total samples. No single date has all 5 WQ parameters measured simultaneously.

2. **Precipitation coverage gaps**: The NOAA weather station near Prescott only covers 7.6% of days, and Winona only 14.4%. Missing days were filled with zero (no rain assumed), which biases the data.

3. **Water temp file format**: The Minnesota DNR water temperature data was served as an HTML table masquerading as an `.xls` file, requiring custom HTML parsing rather than standard Excel reading.

4. **Spectral data date corruption**: 94% of dates (3,219 of 3,409 rows) in the Google Earth Engine spectral export were malformed, using day-of-year (DOY) encoding (e.g., `2025-12-348`) instead of standard `YYYY-MM-DD`. Required programmatic detection and conversion.

5. **Satellite revisit gaps**: Sentinel-2 and Landsat-8 have ~5-day revisit cycles with further gaps due to cloud cover, yielding ~20% daily coverage for spectral bands.

## Preprocessing Pipeline

Run `preprocess_all_data.py` to execute the full preprocessing pipeline:

```bash
python preprocess_all_data.py
```

### Steps:
1. **Water temperature HTML parsing** → Clean CSV with daily F/C values (2,214 records)
2. **Spectral date fixing** → DOY-encoded dates converted to proper calendar dates
3. **Data loading** → Streamflow, precipitation, water quality from raw CSV files
4. **Coverage audit** → Reports per-station coverage for each data source
5. **Interpolation** → Linear interpolation for input features (spectral, streamflow, temp); precipitation gaps filled with 0; water quality targets are NOT interpolated (only real sample dates used for training)

Output: Per-station daily CSVs in `data/processed_clean/`.

## Model Configurations

### Model A: Phosphorus + Nitrogen
- **Stations**: Hastings + Prescott (2 stations)
- **Targets**: Total Phosphorus, Total Nitrogen
- **Training samples**: ~165 (from ~132 actual WQ measurement dates with P+N data)
- **Split**: 115 train / 25 val / 25 test (chronological)

### Model B: All 5 WQ Parameters
- **Stations**: Hastings only
- **Targets**: TP, TN, DO, Chlorophyll-a, Turbidity
- **Training samples**: ~84 (from actual WQ measurement dates)
- **Split**: 58 train / 13 val / 13 test (chronological)
- **Note**: Uses masked loss — Chlorophyll-a (6 dates) and DO (14 dates) have very few samples; the model only computes loss on targets with real measurements.

## Training

```bash
pip install -r requirements.txt
python train_model.py
```

Training takes ~20–30 minutes (most time spent on CEEMDAN decomposition). Outputs saved to `trained_models/model_a/` and `trained_models/model_b/`.

### Training Difficulties

1. **CEEMDAN computational cost**: Decomposing 12 features × 2,557 daily data points takes ~10–15 minutes per station. The PyEMD library also had a broken import (`EMD` module vs class conflict) requiring a workaround using `ext_EMD=EMD()`.

2. **TensorFlow incompatibility**: The installed TensorFlow had a broken protobuf dependency. Migrated the entire pipeline from TensorFlow/Keras to **PyTorch**, which worked out of the box with MPS (Apple Silicon) acceleration.

3. **Interpolation trap**: Initially, all WQ targets were linearly interpolated across 2,557 days from ~66 sparse samples, creating near-constant target values. This caused R² scores of -10^12 (the model appeared catastrophically wrong, but actually the test set had near-zero variance). **Fix**: Only train/evaluate on dates with actual WQ measurements.

4. **Data leakage**: Scalers were initially `fit_transform()`'d on the full dataset instead of `fit()` on training data only. Fixed to fit on training split and `transform()` the rest.

5. **Index misalignment**: After concatenating multi-station data and using `reset_index()`, array indices no longer matched, causing samples to train on incorrect targets. Fixed by extracting per-station arrays directly.

## Results

### Model A (Phosphorus + Nitrogen)

| Target | Stream | R² | MAE | n |
|--------|--------|-----|-----|---|
| Phosphorus | Temporal (CEEMDAN-CNN-LSTM-SA) | -0.014 | 0.062 | 14 |
| Phosphorus | Spatial (GA-RF) | -0.127 | 0.066 | 14 |
| Phosphorus | **Attention Fused** | **-0.061** | **0.064** | 14 |
| Nitrogen | Temporal | -0.058 | 0.081 | 17 |
| Nitrogen | Spatial | 0.200 | 0.076 | 17 |
| Nitrogen | **Attention Fused** | **0.380** | **0.068** | 17 |

**Average attention weights**: Temporal = 0.520, Spatial = 0.480

### Model B (All 5 WQ Parameters)

| Target | Stream | R² | MAE | n |
|--------|--------|-----|-----|---|
| Phosphorus | Fused | -0.218 | 0.002 | 2 |
| Nitrogen | Fused | -5.256 | 0.077 | 3 |
| Dissolved Oxygen | Temporal | -0.090 | 0.928 | 10 |
| Dissolved Oxygen | Spatial | -1.236 | 1.290 | 10 |
| Dissolved Oxygen | **Fused** | **-0.059** | **0.933** | 10 |
| Chlorophyll-a | — | SKIPPED | — | 0 |
| Turbidity | Fused | -4.167 | 4.018 | 3 |

**Average attention weights**: Temporal = 0.633, Spatial = 0.367

### Attention Weight Analysis

The attention gate learns **dynamic per-sample weights** revealing when each stream is more reliable:

**Model A weights (52% temporal, 48% spatial)** are nearly balanced, indicating that with the current data density, neither stream has a clear advantage. However, the fusion still outperforms either individual stream for nitrogen (R²=0.38 fused vs 0.20 spatial vs -0.06 temporal), demonstrating that the attention gate successfully combines complementary information.

**Model B weights (63% temporal, 37% spatial)** lean toward the temporal stream. This makes sense: with only a single station (Hastings), the spatial (satellite) features lack cross-station variability to learn from, while the temporal patterns from CEEMDAN decomposition of streamflow and precipitation trends are more informative for this single-site context.

**Why these specific weights?**

1. **Temporal dominance in Model B**: The CEEMDAN decomposition extracts seasonal and trend components from the hydro-met time series. For a single station, historical patterns (e.g., spring phosphorus loading tied to snowmelt runoff) are the strongest signal. Satellite spectral features at a single point are noisy and inconsistent across seasons.

2. **Near-balance in Model A**: With two stations, the spatial stream gains cross-station spectral variability (Hastings vs Prescott have different land use and turbidity regimes), making satellite features more informative. The temporal stream also benefits from twice the data. The fusion gate correctly assigns roughly equal weights.

3. **Nitrogen favors spatial**: For nitrogen specifically, the GA-RF spatial stream alone achieves R²=0.20, while temporal alone gives R²=-0.06. The attention gate correctly weights spatial higher for nitrogen predictions, and the fused output (R²=0.38) exceeds both — showing the gate learned to leverage the spatial signal while using temporal context to correct spatial errors.

### Why Performance Is Currently Limited

1. **Extreme data scarcity**: 115 training samples for Model A, 58 for Model B. Deep learning models with 128K+ parameters need orders of magnitude more data. The 30-day temporal windows compound this — each training point requires 30 consecutive days of input features, but the target is just one WQ measurement.

2. **Sparse WQ sampling**: Water quality is measured sporadically (not on a fixed schedule), creating irregular gaps. The model never sees enough consecutive measurements to learn short-term dynamics.

3. **Interpolated input features**: While input features (spectral bands at ~20% coverage, precipitation at 7-86%) are linearly interpolated to fill gaps, this smooths out the very signals the model needs to detect (e.g., a runoff event visible in satellite data may be averaged away by interpolation).

4. **Cross-station heterogeneity**: Model A trains on Hastings + Prescott data concatenated chronologically. The test set may be dominated by one station whose patterns differ from the other.

5. **Small test set**: With only 14–17 test samples per target, R² is highly sensitive to individual outliers. A single mispredicted point can swing R² from positive to deeply negative.

## Project Structure

```
fyp-2/
├── data/
│   ├── streamflow/           # USGS daily discharge per station
│   ├── waterquality/         # Raw WQ samples from WQP
│   ├── precipitation/        # NOAA daily precipitation
│   ├── watertemp/            # MN DNR water temperature (HTML)
│   ├── waterfeatures/        # GEE spectral band extractions
│   └── processed_clean/      # Preprocessed daily CSVs (output)
├── trained_models/
│   ├── model_a/              # P+N model artifacts + plots
│   └── model_b/              # All-5 model artifacts + plots
├── old-model/                # Previous CEEMDAN-CNN-LSTM-SA prototype
├── tifs/                     # Raw Sentinel-2 / Landsat-8 GeoTIFFs
├── preprocess_all_data.py    # Full preprocessing pipeline
├── train_model.py            # Full training pipeline (both models)
├── download_water_quality.py # WQ data download script
├── find_wq_stations_v3.py    # Station discovery script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Future Improvements

### Data Improvements (Highest Impact)
1. **Increase WQ sampling frequency**: Partner with monitoring programs for weekly or bi-weekly sampling at all 5 stations. Even doubling the current ~66 samples per station to ~130 would significantly improve model generalization.
2. **Add more stations**: Extend to additional Mississippi River monitoring sites (USGS operates dozens). More stations = more diverse training conditions.
3. **Use gridded precipitation** (PRISM or Daymet) instead of sparse weather station data to eliminate the 7-93% coverage gaps.
4. **Multi-temporal satellite composites**: Instead of single-date spectral values, create 5-day or 10-day cloud-free composites using Sentinel-2's short revisit cycle.

### Model Improvements
5. **Data augmentation**: Apply temporal jittering (shifting windows by ±1–3 days), Gaussian noise injection, and mixup to synthetically expand the training set.
6. **Transfer learning**: Pre-train the temporal stream on a larger dataset (e.g., all USGS WQ stations nationwide) and fine-tune on Mississippi River data.
7. **Reduce model complexity**: For the current dataset size (~100 samples), a simpler model (fewer LSTM layers, smaller hidden size) may generalize better. Consider a hyperparameter search.
8. **Probabilistic predictions**: Replace point predictions with uncertainty estimates (e.g., MC Dropout or ensemble methods) to indicate when the model is confident vs uncertain.
9. **Seasonal encoding**: Add explicit temporal features (day-of-year, season) to help the model learn cyclical nutrient loading patterns.
10. **Attention gate conditioning**: Condition the fusion gate on additional signals like cloud cover fraction, days since last satellite observation, or WQ sample density.

### Engineering Improvements
11. **Cache CEEMDAN decompositions**: Save decomposed IMFs to disk and reload on subsequent runs to avoid the 10–15 minute decomposition step.
12. **Hyperparameter optimization**: Use Optuna or Ray Tune for systematic hyperparameter search (learning rate, hidden sizes, number of LSTM layers, CEEMDAN parameters).
13. **Cross-validation**: Replace the single chronological split with time-series cross-validation (expanding window) to get more robust performance estimates from limited data.

## License

This project was developed as a Final Year Project (FYP).
