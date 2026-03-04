# Water Quality Prediction Using Two-Stream Deep Fusion

A multi-modal deep learning system for predicting river water quality from satellite imagery and hydro-meteorological time series. Validated on two separate datasets:
- **Mississippi River, Minnesota** — 5 USGS stations, 2019–2025
- **Danube River, Serbia** — 7 ICPDR/GFQA_v3 stations, 2013–2023

The architecture combines a **per-target GA-RF spatial expert** with a **CEEMDAN-CNN-LSTM-SA temporal expert**, fused through a learned **attention-based gating mechanism**.

---

## Quick Start

```bash
pip install -r requirements.txt

# Preprocess data (run once — Danube only; Mississippi data already processed)
python preprocess_danube.py          # Danube (7 stations → data/danube_processed/processed_clean/)

# Train
python train_model.py data/danube_processed danube_v1 --model c
python train_model.py data/waterquality baseline --model a
```

---

## All Commands Reference

### 1. Standard Training (70/15/15 split)

```bash
# Model A — Mississippi: Phosphorus + Nitrogen (hastings + prescott)
python train_model.py data/waterquality baseline --model a

# Model B — Mississippi: all 5 WQ parameters (hastings only)
python train_model.py data/waterquality baseline --model b

# Model C — Danube: all 6 WQ parameters (7 stations)
python train_model.py data/danube_processed danube_v1 --model c

# Both Mississippi models in one run
python train_model.py data/waterquality baseline --model both

# With augmented data (jitter strategy)
python train_model.py data/augmented_jitter jitter_v4 --model a
python train_model.py data/danube_augmented_jitter danube_aug --model c

# With GAN-augmented data
python train_model.py data/danube_processed danube_gan_v1 --model c \
  --danube-wq-dir data/danube_augmented_gan
```

---

### 2. K-Fold Cross-Validation

```bash
# 5-fold CV — Model A
python train_model.py data/waterquality cv5 --mode kfold --folds 5 --model a

# 5-fold CV — Model C (Danube)
python train_model.py data/danube_processed danube_cv5 --mode kfold --folds 5 --model c

# 3-fold (faster, useful during iteration)
python train_model.py data/danube_processed danube_cv3 --mode kfold --folds 3 --model c
```

---

### 3. Hyperparameter Tuning

#### Default: Random Search
```bash
# 12 trials, 3-fold CV per trial
python train_model.py data/waterquality tune --mode tune --trials 12 --folds 3 --model a
python train_model.py data/danube_processed danube_tune --mode tune --trials 12 --folds 3 --model c

# Fewer trials for a quick sweep
python train_model.py data/danube_processed danube_tune --mode tune --trials 6 --folds 2 --model c
```

#### Bayesian Optimization — Optuna TPE
Uses Tree-structured Parzen Estimators; learns which hyperparameter regions are promising and concentrates trials there. Outperforms random search after ~6 trials.
```bash
python train_model.py data/danube_processed danube_tpe --mode tune --trials 12 --folds 3 \
  --model c --hpo-method tpe

python train_model.py data/waterquality tune_tpe --mode tune --trials 12 --folds 3 \
  --model a --hpo-method tpe
```

#### BOHB — Bayesian Optimization + Hyperband (Successive Halving)
Runs cheap evaluations (few GA gens) first, prunes bad configs early, runs full GA only on survivors. Best when evaluations are expensive (they are).
```bash
python train_model.py data/danube_processed danube_bohb --mode tune --trials 12 --folds 3 \
  --model c --hpo-method bohb

python train_model.py data/waterquality tune_bohb --mode tune --trials 12 --folds 3 \
  --model a --hpo-method bohb
```

---

### 4. Skipping CEEMDAN (Fastest Iteration)

CEEMDAN decomposition takes ~10–15 min per station on first run. Two ways to skip it:

#### 4a. Use the cache (automatic after first run)
After the first training run, IMFs are written to `.ceemdan_cache/` inside the data directory.
Subsequent runs load from cache automatically — no flag needed.

```bash
# First run: CEEMDAN computed and cached (~10–15 min per station)
python train_model.py data/danube_processed danube_v1 --model c

# Second run: cache loaded automatically (seconds, not minutes)
python train_model.py data/danube_processed danube_v2 --model c

# Force recompute (e.g. if temporal features changed)
python train_model.py data/danube_processed danube_v3 --model c --no-cache-imfs
```

#### 4b. Freeze the entire temporal stream (`--load-temporal`)
Loads temporal stream (CNN-LSTM-SA) weights from a previous run and freezes them.
Only the spatial stream (GA-RF) and fusion gate are retrained.
Reduces runtime by **~80%** — from 4–8 hr to 30–55 min.

```bash
# Step 1: Full training run to get a temporal stream baseline
python train_model.py data/danube_processed danube_v1 --model c

# Step 2: Iterate on spatial/fusion only — freeze temporal from step 1
python train_model.py data/danube_processed danube_spatial_v2 --model c \
  --load-temporal trained_models/model_c_danube_v1

# Combine with HPO
python train_model.py data/danube_processed danube_tpe_fast --mode tune \
  --trials 12 --folds 3 --model c \
  --load-temporal trained_models/model_c_danube_v1 \
  --hpo-method tpe

# Combine with K-fold CV
python train_model.py data/danube_processed danube_cv_fast --mode kfold --folds 5 \
  --model c --load-temporal trained_models/model_c_danube_v1
```

**Expected runtimes with `--load-temporal` + cached CEEMDAN:**

| Model | Full pipeline | Cached + frozen temporal |
|-------|--------------|--------------------------|
| Model A (2 targets × 2 stations) | 3–5 hr | **~12–20 min** |
| Model B (5 targets × 1 station) | 4–6 hr | **~15–25 min** |
| Model C (6 targets × 7 stations) | 8–14 hr | **~35–55 min** |

---

### 5. Alternative Feature Selector (CMA-ES instead of GA)

CMA-ES adapts a full covariance matrix over generations, converging ~40% faster than the binary-chromosome GA to comparable feature subsets.

```bash
# Use CMA-ES for spatial feature selection (any model)
python train_model.py data/danube_processed danube_cmaes --model c \
  --feature-selector cmaes

# CMA-ES + load-temporal (fastest possible iteration on spatial stream)
python train_model.py data/danube_processed danube_cmaes_fast --model c \
  --feature-selector cmaes \
  --load-temporal trained_models/model_c_danube_v1

# Requires: pip install cma
# Falls back to scipy differential_evolution if cma is not installed
```

---

### 6. Data Augmentation

#### CTGAN augmentation (GAN-based, learns joint distributions) — **recommended**
Fits a Conditional Tabular GAN per WQ parameter, generates statistically realistic synthetic samples.
```bash
# Quick smoke test (5 epochs, DO only, 10 rows — runs in ~30 sec)
python augment_danube_gan.py --smoke

# Full run: all 6 parameters, 3× augmentation, 200 epochs (~15–30 min total)
python augment_danube_gan.py --multiplier 3 --epochs 200

# Single parameter
python augment_danube_gan.py --param dissolved_oxygen.csv --epochs 200

# Custom multiplier
python augment_danube_gan.py --multiplier 5 --epochs 300

# Output: data/danube_augmented_gan/
# Requires: pip install ctgan
```

---

### 7. Anomaly Detection + Pollution Attribution

Detects statistically anomalous WQ measurements using rolling Z-scores, identifies the most likely upstream pollution source, and matches to known industrial facilities.

```bash
# Full run — all parameters, all years
python anomaly_detector.py

# Filter by parameter
python anomaly_detector.py --param electrical_conductance
python anomaly_detector.py --param dissolved_oxygen total_phosphorus

# Filter by date range
python anomaly_detector.py --from-date 2017-01-01 --to-date 2019-12-31

# Adjust sensitivity (lower z = more sensitive; default 2.5)
python anomaly_detector.py --z-thresh 2.0
python anomaly_detector.py --z-thresh 3.0   # stricter — only clear anomalies

# Custom output directory
python anomaly_detector.py --output-dir anomaly_reports/my_run

# Custom industry database
python anomaly_detector.py --industry-db data/industry_lookup/overpass_industries.json

# Combine options
python anomaly_detector.py --param nitrate_n --from-date 2015-01-01 \
  --z-thresh 2.0 --output-dir anomaly_reports/nitrate_analysis

# Output: anomaly_report.csv + anomaly_report.json in the output directory
```

#### LLM-enriched attribution (routes high-severity anomalies to Claude)
Anomalies with Z ≥ 3.5 get a structured LLM analysis: primary source, confidence, suggested action.
```bash
export ANTHROPIC_API_KEY=sk-ant-...   # or use --api-key

# Enrich the latest anomaly report
python llm_attributor.py

# Smoke test: process only the highest-severity event
python llm_attributor.py --smoke

# Run detector first, then enrich automatically
python llm_attributor.py --run-detector

# Custom report + threshold
python llm_attributor.py \
  --report anomaly_reports/my_run/anomaly_report.json \
  --z-high 4.0

# Use OpenAI / other provider: swap call_claude() in llm_attributor.py
# (see comments in the function for drop-in replacements)
```

---

### 8. Industry Database

#### Curated database (already present)
`data/industry_lookup/danube_industries.json` — 12 real Serbian Danube facilities with
effluent parameters, river km, and stretch assignments. Used by default.

#### Live OpenStreetMap lookup
Queries the Overpass API for industrial facilities along the Serbian Danube.
```bash
# Fetch and print (no file written)
python fetch_industries_overpass.py --smoke

# Fetch and save
python fetch_industries_overpass.py

# Fetch and merge with existing curated database (deduplicates by proximity)
python fetch_industries_overpass.py --merge

# Custom output path
python fetch_industries_overpass.py --out data/industry_lookup/combined.json

# Use merged database with anomaly detector
python anomaly_detector.py --industry-db data/industry_lookup/overpass_industries.json
```

---

### 9. Live Satellite Imagery Fetch (Google Earth Engine)

Fetches updated Sentinel-2 + Landsat-8 imagery for all 7 Danube stations.

```bash
# First-time authentication (one-time setup)
python fetch_gee_imagery.py --auth

# Smoke test: Bezdan only, last 30 days, prints result (no file written)
python fetch_gee_imagery.py --smoke

# Full fetch: 2012-12-01 to today
python fetch_gee_imagery.py

# Update: fetch only recent data
python fetch_gee_imagery.py --start 2024-01-01

# Specific stations only
python fetch_gee_imagery.py --station SRB00001 SRB00002

# Merge with existing CSV instead of overwriting
python fetch_gee_imagery.py --start 2024-01-01 --merge

# After fetching, rerun preprocessing to incorporate new imagery
python preprocess_danube.py

# Requires: pip install earthengine-api  + gcloud auth or ee.Authenticate()
```

---

### 10. Preprocessing

```bash
# Danube — run once after data is in place
python preprocess_danube.py
# Output: data/danube_processed/processed_clean/{SRB*}_daily.csv

# Mississippi data is already processed; preprocess_all_data.py is in support_scripts/ for reference

# These must be run before training (Danube only for active experiments)
```

---

### 11. Standalone HPO Strategy Tests

```bash
# CMA-ES feature selector smoke test (no model training, ~10 sec)
python hpo_strategies.py --smoke-cmaes

# Run Optuna TPE on model_a (requires full training — slow)
python hpo_strategies.py --tpe

# Run BOHB on model_a (requires full training — slow)
python hpo_strategies.py --bohb
```

---

## Full Pipeline Examples

### Danube — from scratch
```bash
python preprocess_danube.py
python augment_danube_gan.py --multiplier 3 --epochs 200      # ~30 min
python train_model.py data/danube_processed danube_v1 --model c    # first run (caches CEEMDAN)
python anomaly_detector.py --output-dir anomaly_reports/danube_full
```

### Danube — fast iteration (after first run)
```bash
# Change spatial features / GA settings, retrain only spatial stream
python train_model.py data/danube_processed danube_v2 --model c \
  --load-temporal trained_models/model_c_danube_v1   # ~40 min vs 8+ hr
```

### Danube — HPO with frozen temporal
```bash
python train_model.py data/danube_processed danube_tpe --mode tune \
  --trials 12 --folds 3 --model c \
  --load-temporal trained_models/model_c_danube_v1 \
  --hpo-method tpe    # ~2–3 hr vs 12+ hr full HPO
```

### Danube — CMA-ES feature selector + TPE HPO
```bash
python train_model.py data/danube_processed danube_cmaes_tpe --mode tune \
  --trials 12 --folds 3 --model c \
  --feature-selector cmaes \
  --hpo-method tpe \
  --load-temporal trained_models/model_c_danube_v1
```

---

## Architecture

```
DATA INPUT LAYER          PREPROCESSING         TWO-STREAM MODELS        FUSION & OUTPUT
┌─────────────────┐    ┌──────────────┐      ┌─────────────────────┐    ┌──────────────┐
│ Sentinel-2 /    │───>│ Radiometric  │─────>│ GA-RF Spatial       │    │              │
│ Landsat-8       │    │ Correction,  │      │ Expert Stream       │───>│  Attention-  │
│ (9 spectral     │    │ Band Select  │      │ (Feature Selection  │    │  Based Late  │
│  bands)         │    │              │      │  + Random Forest)   │    │  Fusion Gate │──> WQ Pred.
├─────────────────┤    ├──────────────┤      ├─────────────────────┤    │              │
│ In-Situ WQ      │───>│ Missing Val  │─────>│ CEEMDAN-CNN-LSTM-SA │    │  Dynamic     │
│ (TP, NO3N, DO,  │    │ Imputation,  │      │ Temporal Expert     │───>│  Per-Sample  │
│  EC, Chl-a)     │    │ Outlier Det. │      │ Stream              │    │  Weights     │
├─────────────────┤    ├──────────────┤      └─────────────────────┘    └──────────────┘
│ Hydro-Met Data  │───>│ Alignment,   │─────────> Context Features ────────────┘
│ (Flow, Precip,  │    │ Resampling   │
│  Temperature)   │    │              │
└─────────────────┘    └──────────────┘
```

---

## Project Structure

```
fyp-2/
│
│── ─── CORE PIPELINE ──────────────────────────────────────────────────────────
├── train_model.py               # Main training pipeline — Models A, B, C
├── preprocess_danube.py         # Danube preprocessing (DOY fix, daily CSVs)
├── augment_danube_gan.py        # CTGAN-based WQ augmentation (Danube)
├── hpo_strategies.py            # Optuna TPE / BOHB / CMA-ES feature selector
│
│── ─── ANOMALY & ATTRIBUTION ──────────────────────────────────────────────────
├── anomaly_detector.py          # WQ anomaly detection + rule-based attribution
├── llm_attributor.py            # LLM-enriched attribution (Claude / OpenAI)
├── fetch_industries_overpass.py # Live industry lookup via Overpass / OpenStreetMap
├── fetch_gee_imagery.py         # Live satellite fetch via GEE Python API
│
│── ─── CONFIG & DOCS ──────────────────────────────────────────────────────────
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
├── RESULTS_SUMMARY.md
│
│── ─── LEGACY (not needed to run) ─────────────────────────────────────────────
├── support_scripts/             # Legacy scripts — not needed to run the pipeline
│   ├── find_wq_stations*.py     # USGS station discovery (superseded)
│   ├── download_water_quality.py# WQ download from USGS WQP (superseded)
│   ├── download_wq_final.py     # Final WQ download (superseded)
│   ├── preprocess_data.py       # Original single-file preprocessing (superseded)
│   ├── preprocess_all_data.py   # Mississippi preprocessing (data already processed)
│   ├── augment_data.py          # Jitter augmentation (superseded by CTGAN)
│   └── tifverif.py              # GeoTIFF verification tool (one-off)
│
│── ─── DATA ───────────────────────────────────────────────────────────────────
├── data/
│   ├── danube_processed/
│   │   ├── Danube_Satellite_Data_2012_2023 (1).csv  # Raw GEE export
│   │   ├── discharge.csv                            # Daily discharge (m³/s)
│   │   ├── weather.csv                              # Daily precip + air temp
│   │   ├── dissolved_oxygen.csv                     # GFQA_v3: O2-Dis
│   │   ├── phosphorus.csv                           # GFQA_v3: TP
│   │   ├── nitrogen_oxidized.csv                    # GFQA_v3: NO3N
│   │   ├── electrical_conductance.csv               # GFQA_v3: EC
│   │   ├── oxygen_demand.csv                        # GFQA_v3: BOD (excluded)
│   │   ├── chlorophyll.csv                          # GFQA_v3: Chl-a
│   │   └── processed_clean/                         # Preprocessed daily CSVs
│   │       └── .ceemdan_cache/                      # Auto-generated IMF cache
│   ├── danube_augmented_jitter/                     # Jitter-augmented WQ
│   ├── danube_augmented_gan/                        # CTGAN-augmented WQ
│   └── industry_lookup/
│       ├── danube_industries.json                   # 12 curated Serbian facilities
│       └── overpass_industries.json                 # Live OSM fetch (generated)
│
│── ─── OUTPUTS ────────────────────────────────────────────────────────────────
├── trained_models/
│   └── model_{a,b,c}_{tag}/     # Plots + results.json (binaries gitignored)
├── anomaly_reports/
│   └── {timestamp}/
│       ├── anomaly_report.csv
│       ├── anomaly_report.json
│       └── llm_attributed_report.json
│
└── old-model/                   # Previous CEEMDAN-CNN-LSTM prototype
```

---

## Data Sources

### Danube River — Serbia (Model C)

| Data | Source | Stations | Period |
|------|--------|----------|--------|
| Satellite | Google Earth Engine (S2 + L8) | 7 ICPDR | 2012–2023 |
| Water Quality | GFQA_v3 | All 7 | 2013–2023 |
| Streamflow | ICPDR discharge.csv | All 7 | 2012–2023 |
| Weather | ICPDR weather.csv | All 7 | 2012–2023 |

| Station ID | Location | River km |
|------------|----------|---------|
| SRB00001 | Bezdan | 1425 |
| SRB00040 | Bogojevo | 1367 |
| SRB00002 | Novi Sad | 1255 |
| SRB00003 | Zemun (Belgrade) | 1173 |
| SRB00041 | Smederevo | 1116 |
| SRB00005 | Banatska Palanka | 1077 |
| SRB00006 | Tekija (Iron Gate) | 931 |

### Mississippi River (Models A & B)

| Data | Source | Stations | Period |
|------|--------|----------|--------|
| Satellite | Google Earth Engine (S2 + L8) | 5 USGS | 2019–2025 |
| Water Quality | USGS WQP | 5 stations | 2019–2025 |
| Streamflow | USGS NWIS | 5 stations | 2019–2025 |
| Precipitation | NOAA CDO | Nearby stations | 2019–2025 |
| Water Temp | MN DNR | Minneapolis | 2019–2025 |

---

## Results

### Model A — Mississippi (Best: v4 + jitter augmentation)

| Target | Temporal R² | Spatial R² | Fused R² |
|--------|-------------|------------|----------|
| Phosphorus | 0.379 | 0.332 | **0.419** |
| Nitrogen | 0.541 | 0.247 | **0.605** |

**K-fold CV (5-fold)**: Nitrogen: 0.597 ± 0.080 | Phosphorus: 0.196 ± 0.132

**Average attention weights**: Temporal ≈ 0.787, Spatial ≈ 0.213

### Model C — Danube (Baseline, pre-tuning)

| Target | Temporal R² | Fused R² |
|--------|-------------|----------|
| Dissolved Oxygen | 0.41 | 0.41 |
| Nitrate-N | 0.52 | 0.52 |
| Electrical Conductance | 0.32 | 0.32 |
| Total Phosphorus | −0.09 | −0.09 |
| Chlorophyll-a | −0.12 | −0.12 |

---

## Dependencies

```bash
pip install -r requirements.txt

# Core ML
numpy pandas scikit-learn torch matplotlib joblib

# Signal decomposition
EMD-signal==1.6.4          # imports as PyEMD

# Augmentation
ctgan>=0.7.0               # CTGAN-based WQ augmentation

# HPO strategies
optuna>=3.6.0              # Bayesian HPO (TPE + BOHB)
cma>=3.3.0                 # CMA-ES feature selector (scipy fallback if absent)

# Satellite fetch
earthengine-api>=0.1.380   # GEE Python API

# LLM attribution
anthropic>=0.30.0          # Claude API (swap call_claude() for OpenAI/Mistral/Ollama)
```

---

## Key Design Notes

**Why WQ targets are never interpolated**: With ~130 sparse measurements per parameter, interpolation fills 4,000 days from ~130 points. Training on interpolated targets gives R² of −10^12 (near-zero test variance). Only actual measurement dates are used as training samples; satellite/streamflow features are interpolated.

**CEEMDAN caching**: IMFs are expensive to compute (~15 min/station). Cached to `.ceemdan_cache/{station}.npz` after first run. Use `--no-cache-imfs` to force recompute when temporal features change.

**Temporal stream freezing**: `--load-temporal` loads CNN-LSTM-SA weights and sets `requires_grad=False`. Only GA-RF, ContextEncoder, and FusionGate are trained. Reduces runtime ~80%.

**Per-target feature selection**: A separate GA (or CMA-ES) and RF runs per WQ parameter. DO needs temperature-optical features; EC needs dilution features; Chl-a needs optical indices. Shared-GA was a counterproductive compromise.
