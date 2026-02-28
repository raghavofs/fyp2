# Water Quality Prediction Using Two-Stream Deep Fusion

A multi-modal deep learning system for predicting river water quality from satellite imagery and hydro-meteorological time series. Validated on two separate datasets:
- **Mississippi River, Minnesota** — 5 USGS stations, 2019–2025, sparse WQ data (~80 samples/station)
- **Danube River, Serbia** — 7 ICPDR/GFQA_v3 stations, 2013–2023, richer WQ data (~150 samples/parameter)

The architecture combines a **per-target GA-RF spatial expert** with a **CEEMDAN-CNN-LSTM-SA temporal expert**, fused through a learned **attention-based gating mechanism** that dynamically weights each stream's contribution using a **ContextEncoder LSTM** trained on recent hydro-meteorological history.

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

### Key Innovations

**1. Per-target GA-RF spatial stream**: A separate Genetic Algorithm and Random Forest runs for each WQ parameter, allowing each target to find its own optimal spectral feature subset (max 15 features). Previously a single GA averaged all targets, forcing a counterproductive compromise between DO (temperature-driven), EC (dilution-driven), and Chl-a (optical-driven) feature needs.

**2. ContextEncoder LSTM for dynamic fusion**: The fusion gate receives a 16-dimensional LSTM encoding of the 30-day hydro-meteorological history rather than a single-day scalar snapshot. This allows the gate to recognise hydrological regimes (flood pulse, summer low-flow, spring melt) and weight the expert streams appropriately.

**3. Physics-informed spatial features**:
- `DO_sat(T)` — Garcia-Gordon oxygen saturation at measured temperature
- `EC_dilution = median_Q / Q` — conservative tracer dilution physics
- `log_Q_anomaly` — log deviation from 30-day mean flow (nutrient loading events)

**4. Z-score target normalisation**: `StandardScaler` applied to targets before computing loss, preventing high-magnitude targets (EC: hundreds µS/cm) from dominating the gradient over low-magnitude targets (TP: ~0.1 mg/L).

**5. Attention-Based Late Fusion**: Produces interpretable per-sample weights `(w_temporal, w_spatial)` summing to 1:
```
final_prediction = w_temporal × temporal_prediction + w_spatial × spatial_prediction
```

## Data Sources

### Mississippi River (Models A & B)

| Data | Source | Stations | Period | Notes |
|------|--------|----------|--------|-------|
| **Satellite Imagery** | Google Earth Engine (Sentinel-2, Landsat-8) | 5 sites | 2019–2025 | 9 bands; DOY date encoding bug fixed |
| **Water Quality** | USGS Water Quality Portal (WQP) | Hastings, Prescott, Winona, St Paul, Brooklyn Park | 2019–2025 | TP, TN, DO, Chl-a, Turbidity — **sparse: ~66–166 sample dates/station** |
| **Streamflow** | USGS NWIS | Same 5 stations | 2019–2025 | Daily discharge (cfs) |
| **Precipitation** | NOAA Climate Data Online | Nearby weather stations | 2019–2025 | Daily (inches); gaps filled with 0 |
| **Water Temperature** | Minnesota DNR | Lock & Dam #1, Minneapolis | 2019–2025 | Daily (°F→°C), shared across stations |

| Station | USGS Site ID |
|---------|-------------|
| Brooklyn Park | 05288500 |
| St. Paul | 05331000 |
| Hastings | 05331580 |
| Prescott | 05344500 |
| Winona | 05378500 |

### Danube River — Serbia (Model C)

| Data | Source | Stations | Period | Notes |
|------|--------|----------|--------|-------|
| **Satellite Imagery** | Google Earth Engine (Sentinel-2, Landsat-8) | 7 ICPDR stations | 2012–2023 | DOY date bug fixed; L8 RedEdge −9999 → NaN |
| **Water Quality** | GFQA_v3 (per-parameter CSVs) | All 7 stations | 2013–2023 | DO, TP, NO3N, EC, Chl-a — **~125–180 samples/parameter/station** |
| **Streamflow** | ICPDR discharge.csv | All 7 stations | 2012–2023 | Daily (m³/s), renamed to `streamflow` |
| **Weather** | ICPDR weather.csv | All 7 stations | 2012–2023 | Daily precipitation (mm) + air temp (°C) |

| Station ID | Location | River km |
|------------|----------|---------|
| SRB00001 | Bezdan | 1425 |
| SRB00040 | Bogojevo | 1367 |
| SRB00002 | Novi Sad | 1255 |
| SRB00003 | Zemun (Belgrade) | 1173 |
| SRB00041 | Smederevo | 1116 |
| SRB00005 | Banatska Palanka | 1077 |
| SRB00006 | Tekija (Iron Gate) | 931 |

## Data Collection Difficulties

1. **Extreme WQ data sparsity**: Water quality samples are collected sporadically (monthly to biweekly during field season). Hastings and Prescott have ~66–84 sample dates over 7 years. Brooklyn Park has only 2 total samples. No single date has all 5 WQ parameters measured simultaneously.

2. **Precipitation coverage gaps**: The NOAA weather station near Prescott only covers 7.6% of days, and Winona only 14.4%. Missing days were filled with zero (no rain assumed), which biases the data.

3. **Water temp file format**: The Minnesota DNR water temperature data was served as an HTML table masquerading as an `.xls` file, requiring custom HTML parsing rather than standard Excel reading.

4. **Spectral data date corruption**: 94% of dates (3,219 of 3,409 rows) in the Google Earth Engine spectral export were malformed, using day-of-year (DOY) encoding (e.g., `2025-12-348`) instead of standard `YYYY-MM-DD`. Required programmatic detection and conversion.

5. **Satellite revisit gaps**: Sentinel-2 and Landsat-8 have ~5-day revisit cycles with further gaps due to cloud cover, yielding ~20% daily coverage for spectral bands.

## Preprocessing Pipeline

### Mississippi River
```bash
python preprocess_all_data.py
```
Steps: water temperature HTML parsing → DOY date fix → streamflow/precip loading → coverage audit → linear interpolation to daily. Output: `data/processed_clean/`.

### Danube River
```bash
python preprocess_danube.py
```
Steps: DOY date fix + Landsat-8 RedEdge sentinel replacement → discharge/weather loading → GFQA_v3 WQ extraction → coverage audit → daily interpolation. Output: `data/danube_processed/processed_clean/`.

### Data Augmentation (Strategy C — Jitter)
```bash
python augment_data.py                # Mississippi
python augment_data.py --danube       # Danube
```
Creates 3 augmented copies per real WQ sample with 1–3 day temporal jitter and 5% Gaussian noise on target values. Originals tagged `source="original"`. Output: `data/augmented_jitter/` (Mississippi) or `data/danube_augmented_jitter/` (Danube).

**Key design**: input features are linearly interpolated daily (satellites, streamflow, temperature) but **WQ targets are never interpolated** — only actual measurement dates are used as training samples.

## Model Configurations

### Model A: Mississippi — Phosphorus + Nitrogen
- **Stations**: Hastings + Prescott (2 stations)
- **Targets**: Total Phosphorus, Total Nitrogen
- **Training samples**: ~165 (from ~132 actual WQ measurement dates; jitter augmentation ×4 for training)
- **Split**: 70% train (real+augmented) / 15% val (real only) / 15% test (real only)

### Model B: Mississippi — All 5 WQ Parameters
- **Stations**: Hastings only
- **Targets**: TP, TN, DO, Chlorophyll-a, Turbidity
- **Training samples**: ~84 (actual WQ measurement dates)
- **Note**: Uses masked loss — Chlorophyll-a (6 dates) and DO (14 dates) have very few samples

### Model C: Danube — 5 WQ Parameters
- **Stations**: All 7 Serbian Danube stations (Bezdan → Tekija)
- **Targets**: Dissolved Oxygen, Total Phosphorus, Nitrate-N, Electrical Conductance, Chlorophyll-a
- **Training samples**: ~125–180 per parameter per station (GFQA_v3); jitter augmentation ×4 for training
- **Note**: BOD (oxygen demand) excluded — no satellite spectral signal; including it degraded all other targets
- **Split**: 70% train (real+augmented) / 15% val (real only) / 15% test (real only)

## Training

```bash
pip install -r requirements.txt

# Mississippi
python train_model.py data/waterquality baseline --model a
python train_model.py data/augmented_jitter jitter_v4 --model a
python train_model.py data/augmented_jitter cv5 --mode kfold --folds 5 --model a
python train_model.py data/augmented_jitter tune --mode tune --trials 12 --folds 3 --model a

# Danube
python train_model.py data/danube_processed danube_v1 --model c
python train_model.py data/danube_augmented_jitter danube_aug --model c
python train_model.py data/danube_augmented_jitter danube_tune --mode tune --trials 12 --folds 3 --model c
```

Training time: ~20–30 min per run (dominated by CEEMDAN decomposition). With per-target GA (5 targets for Model C), add ~5–10 min for the GA-RF phase. Outputs saved to `trained_models/{model}_{tag}/`.

### Training Difficulties

1. **CEEMDAN computational cost**: Decomposing 12 features × 2,557 daily data points takes ~10–15 minutes per station. The PyEMD library also had a broken import (`EMD` module vs class conflict) requiring a workaround using `ext_EMD=EMD()`.

2. **TensorFlow incompatibility**: The installed TensorFlow had a broken protobuf dependency. Migrated the entire pipeline from TensorFlow/Keras to **PyTorch**, which worked out of the box with MPS (Apple Silicon) acceleration.

3. **Interpolation trap**: Initially, all WQ targets were linearly interpolated across 2,557 days from ~66 sparse samples, creating near-constant target values. This caused R² scores of -10^12 (the model appeared catastrophically wrong, but actually the test set had near-zero variance). **Fix**: Only train/evaluate on dates with actual WQ measurements.

4. **Data leakage**: Scalers were initially `fit_transform()`'d on the full dataset instead of `fit()` on training data only. Fixed to fit on training split and `transform()` the rest.

5. **Index misalignment**: After concatenating multi-station data and using `reset_index()`, array indices no longer matched, causing samples to train on incorrect targets. Fixed by extracting per-station arrays directly.

## Results

### Model A Best Results (Mississippi — Phosphorus + Nitrogen, v4 + Jitter Augmentation)

Best single-split results after GA cap at 10→15 features, physics features, and jitter augmentation:

| Target | Temporal R² | Spatial R² | Fused R² |
|--------|-------------|------------|----------|
| **Phosphorus** | **0.379** | **0.332** | **0.419** |
| **Nitrogen** | **0.541** | **0.247** | **0.605** |

**Average attention weights**: Temporal ≈ 0.787, Spatial ≈ 0.213

**K-Fold cross-validation (5-fold, TimeSeriesSplit)**:

| Target | Mean Fused R² | Std |
|--------|--------------|-----|
| Phosphorus | 0.196 | ±0.132 |
| Nitrogen | 0.597 | ±0.080 |

Nitrogen is robust across splits (~0.60); Phosphorus is noisier due to smaller sample count (~68 measurements).

### Model C Baseline Results (Danube — pre-tuning, architectural overhaul)

7 Serbian Danube stations, 5 WQ targets. Results from initial run before per-target GA and ContextEncoder tuning:

| Target | Temporal R² | Fused R² | Notes |
|--------|-------------|----------|-------|
| Dissolved Oxygen | 0.41 | 0.41 | Temperature-driven signal |
| Nitrate-N | 0.52 | 0.52 | Seasonal agriculture signal |
| Total Phosphorus | −0.09 | −0.09 | Needs physics features |
| Electrical Conductance | 0.32 | 0.32 | Dilution signal present |
| Chlorophyll-a | −0.12 | −0.12 | Optically complex |

Per-target GA, ContextEncoder, StandardScaler, and physics features (DO_sat, EC_dilution, log_Q_anomaly) are expected to improve TP and Chl-a substantially.

### Attention Weight Analysis

The fusion gate learns **dynamic per-sample weights** using a ContextEncoder LSTM over 30-day hydro-met history:

- **Temporal dominance** (~0.79 weight in best runs): The CEEMDAN decomposition extracts seasonal and trend components. For sparse WQ data, historical patterns (e.g., spring phosphorus loading tied to snowmelt) are the strongest signal.
- **Spatial contributes when spectral is informative**: Nitrogen specifically shows the spatial stream catching runoff events that the temporal stream misses (GA-RF picks up Clay_Index, P_Load_Potential features).
- **Per-target GA enables target-specific fusion**: DO features are temperature-linked optical properties; EC features are dilution-linked streamflow metrics. Shared-GA was a counterproductive compromise.

### Why Performance Varies

1. **Extreme data scarcity (Mississippi)**: ~80 WQ samples per station. The 30-day temporal windows compound this — each training point requires 30 consecutive days of input features with only one sparse WQ target.
2. **Sparse WQ sampling**: Water quality is measured sporadically creating irregular gaps. Jitter augmentation (Strategy C: 3 copies × ±1–3 day shift + 5% Gaussian noise) partially addresses this.
3. **Small test sets**: With only 14–17 Mississippi test samples per target, R² is sensitive to individual outliers. K-fold CV provides more robust estimates.
4. **BOD excluded**: Biochemical oxygen demand has no satellite spectral signal — including it degraded all other targets through shared loss gradient.

## Project Structure

```
fyp-2/
├── data/
│   ├── streamflow/                    # USGS daily discharge per station (Mississippi)
│   ├── waterquality/                  # Raw WQ samples from WQP (Mississippi)
│   ├── precipitation/                 # NOAA daily precipitation (Mississippi)
│   ├── watertemp/                     # MN DNR water temperature HTML files
│   ├── waterfeatures/                 # GEE spectral band extractions (Mississippi)
│   ├── processed_clean/               # Preprocessed daily CSVs (Mississippi, 5 stations)
│   ├── augmented_jitter/              # Jitter-augmented WQ samples (Mississippi)
│   └── danube_processed/
│       ├── Danube_Satellite_Data_2012_2023 (1).csv  # Raw GEE satellite export
│       ├── discharge.csv              # Danube daily discharge (7 stations)
│       ├── weather.csv                # Danube daily weather (7 stations)
│       ├── dissolved_oxygen.csv       # GFQA_v3 WQ (O2-Dis parameter)
│       ├── phosphorus.csv             # GFQA_v3 WQ (TP parameter)
│       ├── nitrogen_oxidized.csv      # GFQA_v3 WQ (NO3N parameter)
│       ├── electrical_conductance.csv # GFQA_v3 WQ (EC parameter)
│       ├── oxygen_demand.csv          # GFQA_v3 WQ (BOD — excluded from training)
│       ├── chlorophyll.csv            # GFQA_v3 WQ (Chl-a parameter)
│       └── processed_clean/           # Preprocessed daily CSVs (Danube, 7 stations)
├── data/danube_augmented_jitter/      # Jitter-augmented WQ samples (Danube)
├── trained_models/                    # Experiment output directories
│   ├── model_a_{tag}/                 # P+N Mississippi artifacts + plots
│   ├── model_b_{tag}/                 # All-5 Mississippi artifacts + plots
│   └── model_c_{tag}/                 # Danube artifacts + plots
├── old-model/                         # Previous CEEMDAN-CNN-LSTM-SA prototype
├── tifs/                              # Raw Sentinel-2 / Landsat-8 GeoTIFFs
├── preprocess_all_data.py             # Mississippi preprocessing pipeline
├── preprocess_danube.py               # Danube preprocessing pipeline
├── augment_data.py                    # Strategy C jitter augmentation (both rivers)
├── train_model.py                     # Full training pipeline (Models A, B, C)
├── download_water_quality.py          # WQ data download script (USGS WQP)
├── find_wq_stations_v3.py             # Station discovery script
├── RESULTS_SUMMARY.md                 # Iterative version history + result tables
├── PROJECT_REPORT.md                  # Full academic project report
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Future Improvements

### Data Improvements (Highest Impact)
1. **Increase WQ sampling frequency**: Partner with monitoring programs for weekly or bi-weekly sampling at all 5 Mississippi stations. Even doubling the current ~66 samples per station would significantly improve generalization.
2. **Add more Mississippi stations**: Extend to additional USGS monitoring sites along the Upper Mississippi. More stations = more diverse training conditions for the spatial stream.
3. **Use gridded precipitation** (PRISM or Daymet) instead of sparse weather station data to eliminate the 7–93% Mississippi coverage gaps.
4. **Multi-temporal satellite composites**: Instead of single-date spectral values, create 5-day or 10-day cloud-free composites using Sentinel-2's short revisit cycle.

### Model Improvements
5. **Transfer learning**: Pre-train the temporal stream on a larger dataset (e.g., all USGS WQ stations nationwide) and fine-tune on Mississippi River data.
6. **Probabilistic predictions**: Replace point predictions with uncertainty estimates (MC Dropout or deep ensembles) to indicate when the model is confident vs uncertain.
7. **Attention gate conditioning on satellite quality**: Add cloud cover fraction and days-since-last-observation as context signals — the spatial stream should be downweighted when satellite data is stale or cloud-contaminated.
8. **Extend Danube cross-validation**: Run full 5-fold TimeSeriesSplit CV on Model C to confirm per-target GA improvements are robust across splits.
9. **Graph-based spatial stream**: Replace per-station independent RF with a graph neural network that models upstream-downstream nutrient transport along the Danube river network.

### Engineering Improvements
10. **Optuna integration**: Replace the random search hyperparameter tuner with Optuna's Tree-structured Parzen Estimator (TPE) for more efficient trial selection.
11. **Streaming data ingestion**: Build an automated pipeline to pull new USGS and Danube WQ samples daily and retrain incrementally.

## License

This project was developed as a Final Year Project (FYP).
