"""
Data augmentation strategies for water quality prediction.

Three independent approaches, each writing to its own directory:
  B) Real data expansion: download WQ from additional Mississippi River stations
     + lag-based upstream→downstream transfer
  C) Legitimate augmentation: Gaussian noise + temporal jitter on real samples
  A) Synthetic physics-based generation (for pipeline validation only)

Each approach produces augmented *_wq.csv files in the same format as the
originals, so train_model.py can consume them without changes.
"""

import os
import math
import numpy as np
import pandas as pd
from dataretrieval import wqp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WQ_DIR = os.path.join(BASE_DIR, "data", "waterquality")
DANUBE_WQ_DIR = os.path.join(BASE_DIR, "data", "danube_processed")

DANUBE_STATIONS = {
    "SRB00001": "Bezdan",
    "SRB00040": "Bogojevo",
    "SRB00002": "Novi Sad",
    "SRB00003": "Zemun",
    "SRB00041": "Smederevo",
    "SRB00005": "Banatska Palanka",
    "SRB00006": "Tekija",
}

# (filename, parameter_code, human-readable label)
DANUBE_WQ_FILES = [
    ("dissolved_oxygen.csv",       "O2-Dis",  "dissolved_oxygen"),
    ("phosphorus.csv",             "TP",       "total_phosphorus"),
    ("nitrogen_oxidized.csv",      "NO3N",     "nitrate_n"),
    ("electrical_conductance.csv", "EC",       "electrical_conductance"),
    ("oxygen_demand.csv",          "BOD",      "oxygen_demand"),
    ("chlorophyll.csv",            "Chl-a",    "chlorophyll_a"),
]

# Our five target stations and their coordinates (lat, lon)
STATION_COORDS = {
    "brooklynpark": (45.094, -93.348),
    "stpaul":       (44.945, -93.090),
    "hastings":     (44.737, -92.852),
    "prescott":     (44.748, -92.802),
    "winona":       (44.055, -91.639),
}

CHARACTERISTICS = [
    "Phosphorus", "Nitrogen", "Dissolved oxygen (DO)",
    "Chlorophyll a", "Turbidity",
]

# Mississippi River station order (upstream → downstream) with approximate
# river-mile distances between consecutive stations and lat/lon.
# Used for lag-based transfer.
RIVER_ORDER = [
    # (station_name, river_mile_from_source, lat, lon)
    ("brooklynpark", 0,   45.094, -93.348),
    ("stpaul",       25,  44.945, -93.090),
    ("hastings",     55,  44.737, -92.852),
    ("prescott",     60,  44.748, -92.802),   # at St. Croix confluence
    ("winona",       135, 44.055, -91.639),
]

# Expanded bounding box to capture more upstream/downstream stations
BBOX_EXPANDED = "-94.0,43.5,-91.0,46.0"


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def estimate_travel_days(upstream_mile, downstream_mile):
    """Estimate river travel time in days between two river-mile points.

    Typical Mississippi flow in MN: 2-4 mph average.
    Conservative estimate: ~2 mph = ~48 miles/day.
    """
    distance_miles = abs(downstream_mile - upstream_mile)
    avg_speed_miles_per_day = 48  # conservative
    days = distance_miles / avg_speed_miles_per_day
    return max(1, round(days))


# ============================================================================
# Strategy B: Download real WQ from additional stations + lag transfer
# ============================================================================

def strategy_b_real_expansion():
    """Download WQ data from additional Mississippi River stations in a wider
    bounding box, then use lag-based transfer to create pseudo-samples for
    our target stations."""
    out_dir = os.path.join(BASE_DIR, "data", "augmented_real")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("STRATEGY B: Real data expansion + lag-based transfer")
    print("=" * 70)

    # --- Step 1: Download all WQ in expanded bbox ---
    print("\n[B.1] Downloading WQ data from expanded bounding box...")
    # Use original bbox (expanded one returns too much data and causes parse errors)
    bbox = "-93.5,43.9,-91.4,45.2"
    print(f"  BBox: {bbox}")

    # Get site metadata
    sites_df, _ = wqp.what_sites(bBox=bbox)
    site_info = {}
    for _, row in sites_df.iterrows():
        try:
            sid = row["MonitoringLocationIdentifier"]
            lat = float(row["LatitudeMeasure"])
            lon = float(row["LongitudeMeasure"])
            name = str(row.get("MonitoringLocationName", ""))
            stype = str(row.get("MonitoringLocationTypeName", ""))
            site_info[sid] = {"lat": lat, "lon": lon, "name": name, "type": stype}
        except (ValueError, TypeError, KeyError):
            pass
    print(f"  Found {len(site_info)} total monitoring stations")

    # Download WQ results — go back to 2015 for more data
    all_dfs = []
    for char in CHARACTERISTICS:
        print(f"    Downloading {char}...")
        try:
            chunk, _ = wqp.get_results(
                bBox=bbox,
                characteristicName=[char],
                startDateLo="01-01-2015",
                startDateHi="12-31-2025",
            )
            all_dfs.append(chunk)
            print(f"      → {len(chunk)} records")
        except Exception as e:
            print(f"      → Failed: {e}")
    df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    print(f"  Downloaded {len(df)} total WQ records")

    # Identify columns
    id_col = [c for c in df.columns if "MonitoringLocationIdentifier" in c][0]
    char_col = [c for c in df.columns if "CharacteristicName" in c][0]
    result_col = [c for c in df.columns if c == "ResultMeasureValue"
                  or "ResultMeasure/MeasureValue" in c]
    result_col = result_col[0] if result_col else None
    date_col = [c for c in df.columns if "ActivityStartDate" in c][0]

    # Add coordinates
    df["_lat"] = df[id_col].map(lambda x: site_info.get(x, {}).get("lat"))
    df["_lon"] = df[id_col].map(lambda x: site_info.get(x, {}).get("lon"))
    df["_name"] = df[id_col].map(lambda x: site_info.get(x, {}).get("name", ""))
    df["_type"] = df[id_col].map(lambda x: site_info.get(x, {}).get("type", ""))
    df = df.dropna(subset=["_lat", "_lon"])

    # Filter to river/stream stations on or near the Mississippi
    river_mask = df["_type"].str.contains("River|Stream", case=False, na=False)
    miss_mask = df["_name"].str.contains("Mississippi|Miss R|MISS R", case=False, na=False)
    river_df = df[river_mask & miss_mask].copy()
    print(f"  Mississippi River/Stream records: {len(river_df)}")

    if len(river_df) < 50:
        print("  Falling back to all river/stream records")
        river_df = df[river_mask].copy()
        print(f"  All River/Stream records: {len(river_df)}")

    # Clean up values
    river_df["value"] = pd.to_numeric(river_df[result_col], errors="coerce")
    river_df["date"] = pd.to_datetime(river_df[date_col], errors="coerce")
    river_df = river_df.dropna(subset=["value", "date"])

    # --- Step 2: For each source station, estimate which target station
    # it's closest to on the river and compute travel lag ---
    print("\n[B.2] Computing lag-based transfers...")

    # Build river-mile lookup
    river_mile_lookup = {name: mile for name, mile, _, _ in RIVER_ORDER}
    river_coord_lookup = {name: (lat, lon) for name, mile, lat, lon in RIVER_ORDER}

    # For each unique source monitoring station, find nearest target station
    source_stations = river_df[id_col].unique()
    print(f"  {len(source_stations)} unique source monitoring stations")

    transfer_records = []  # (target_station, date, characteristic, value)

    for src_sid in source_stations:
        src_data = river_df[river_df[id_col] == src_sid]
        src_lat = src_data["_lat"].iloc[0]
        src_lon = src_data["_lon"].iloc[0]
        src_name = src_data["_name"].iloc[0]

        # Find closest target station
        min_dist = float("inf")
        closest_target = None
        for tgt_name, (tgt_lat, tgt_lon) in river_coord_lookup.items():
            d = haversine_km(src_lat, src_lon, tgt_lat, tgt_lon)
            if d < min_dist:
                min_dist = d
                closest_target = tgt_name

        if min_dist < 5:
            # Within 5km - this IS one of our stations essentially, use directly
            for _, row in src_data.iterrows():
                transfer_records.append({
                    "target_station": closest_target,
                    "date": row["date"],
                    "CharacteristicName": row[char_col],
                    "ResultMeasureValue": row["value"],
                    "source": f"direct_{src_sid}",
                    "lag_days": 0,
                })
        elif min_dist < 80:
            # Within 80km on the river - use lag transfer
            # Estimate river mile of source station (approximate from lat)
            # The river generally flows N→S in MN, so we use latitude as proxy
            src_river_mile = _estimate_river_mile(src_lat, src_lon)
            tgt_river_mile = river_mile_lookup.get(closest_target, 0)

            lag_days = estimate_travel_days(src_river_mile, tgt_river_mile)
            is_upstream = src_river_mile < tgt_river_mile

            if is_upstream:
                # Source is upstream → water flows to target
                lag_sign = +lag_days  # shift forward
            else:
                # Source is downstream → we "look back" in time
                lag_sign = -lag_days  # shift backward

            for _, row in src_data.iterrows():
                # Add noise to account for dilution, tributaries, etc.
                noise_factor = np.random.normal(1.0, 0.05 * (min_dist / 10))
                noisy_value = row["value"] * max(0.5, min(1.5, noise_factor))

                transfer_records.append({
                    "target_station": closest_target,
                    "date": row["date"] + pd.Timedelta(days=lag_sign),
                    "CharacteristicName": row[char_col],
                    "ResultMeasureValue": round(noisy_value, 4),
                    "source": f"lag_{src_sid}_d{lag_sign}",
                    "lag_days": lag_sign,
                })

    transfer_df = pd.DataFrame(transfer_records)
    print(f"  Generated {len(transfer_df)} transfer records")

    # --- Step 3: Merge with existing data and save ---
    print("\n[B.3] Merging with existing data...")

    for station in STATION_COORDS:
        # Load existing
        orig_path = os.path.join(WQ_DIR, f"{station}_wq.csv")
        if os.path.exists(orig_path):
            orig = pd.read_csv(orig_path)
        else:
            orig = pd.DataFrame()

        # Get transfer data for this station
        xfer = transfer_df[transfer_df["target_station"] == station].copy()

        if xfer.empty:
            if not orig.empty:
                orig.to_csv(os.path.join(out_dir, f"{station}_wq.csv"), index=False)
                print(f"  {station}: {len(orig)} original records (no transfers)")
            continue

        # Format transfer data to match original schema
        xfer_formatted = pd.DataFrame({
            "ActivityStartDate": xfer["date"].dt.strftime("%Y-%m-%d"),
            "CharacteristicName": xfer["CharacteristicName"],
            "ResultMeasureValue": xfer["ResultMeasureValue"],
            "source": xfer["source"],
        })

        # Keep original columns, add source column
        if not orig.empty:
            orig["source"] = "original"
            # Combine
            combined = pd.concat([orig, xfer_formatted], ignore_index=True)
        else:
            combined = xfer_formatted

        # Deduplicate: keep original if same date+characteristic exists
        combined["_date_char"] = combined["ActivityStartDate"].astype(str) + "_" + combined["CharacteristicName"]
        combined = combined.drop_duplicates(subset=["_date_char"], keep="first")
        combined = combined.drop(columns=["_date_char"])

        out_path = os.path.join(out_dir, f"{station}_wq.csv")
        combined.to_csv(out_path, index=False)

        n_orig = len(orig) if not orig.empty else 0
        n_new = len(combined) - n_orig
        unique_dates = pd.to_datetime(combined["ActivityStartDate"]).nunique()
        print(f"  {station}: {n_orig} original + {n_new} new = {len(combined)} total "
              f"({unique_dates} unique dates)")

    print(f"\n  Saved to: {out_dir}")
    return out_dir


def _estimate_river_mile(lat, lon):
    """Rough river-mile estimate based on latitude.
    In MN, the Mississippi flows roughly N to S.
    Brooklyn Park (45.09) ≈ mile 0, Winona (44.05) ≈ mile 135.
    """
    lat_range = 45.094 - 44.055  # ~1.04 degrees
    mile_range = 135
    miles = (45.094 - lat) / lat_range * mile_range
    return max(0, min(150, miles))


# ============================================================================
# Strategy C: Legitimate data augmentation (noise + temporal jitter)
# ============================================================================

def strategy_c_augmentation():
    """Apply Gaussian noise and temporal jitter to real WQ samples.

    - Gaussian noise: add N(0, σ) where σ = 5% of the parameter's std dev
    - Temporal jitter: shift sample dates by ±1-3 days (the water quality
      doesn't change drastically in 1-3 days)
    - Creates 3 augmented copies per real sample
    """
    out_dir = os.path.join(BASE_DIR, "data", "augmented_jitter")
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("STRATEGY C: Gaussian noise + temporal jitter augmentation")
    print("=" * 70)

    N_COPIES = 3  # augmented copies per real sample
    NOISE_SCALE = 0.05  # 5% of std dev
    MAX_JITTER_DAYS = 3

    for station in STATION_COORDS:
        orig_path = os.path.join(WQ_DIR, f"{station}_wq.csv")
        if not os.path.exists(orig_path):
            continue

        orig = pd.read_csv(orig_path)
        if len(orig) == 0:
            continue

        orig["source"] = "original"

        # Compute per-characteristic std devs for noise scaling
        char_stds = {}
        for char_name in CHARACTERISTICS:
            vals = pd.to_numeric(
                orig.loc[orig["CharacteristicName"] == char_name, "ResultMeasureValue"],
                errors="coerce"
            ).dropna()
            if len(vals) > 1:
                char_stds[char_name] = vals.std()
            else:
                char_stds[char_name] = 0.0

        augmented_rows = []
        for _, row in orig.iterrows():
            for copy_i in range(N_COPIES):
                new_row = row.copy()

                # Temporal jitter: shift date by ±1-3 days
                jitter_days = np.random.choice([-3, -2, -1, 1, 2, 3])
                try:
                    orig_date = pd.to_datetime(row["ActivityStartDate"])
                    new_date = orig_date + pd.Timedelta(days=jitter_days)
                    new_row["ActivityStartDate"] = new_date.strftime("%Y-%m-%d")
                except Exception:
                    pass

                # Gaussian noise on measurement value
                val = pd.to_numeric(row["ResultMeasureValue"], errors="coerce")
                if not np.isnan(val):
                    char = row["CharacteristicName"]
                    sigma = char_stds.get(char, 0.0) * NOISE_SCALE
                    noise = np.random.normal(0, sigma) if sigma > 0 else 0
                    new_val = max(0, val + noise)  # can't go negative
                    new_row["ResultMeasureValue"] = round(new_val, 4)

                new_row["source"] = f"augmented_copy{copy_i + 1}_jitter{jitter_days}d"
                augmented_rows.append(new_row)

        augmented_df = pd.DataFrame(augmented_rows)
        combined = pd.concat([orig, augmented_df], ignore_index=True)

        out_path = os.path.join(out_dir, f"{station}_wq.csv")
        combined.to_csv(out_path, index=False)

        unique_dates = pd.to_datetime(combined["ActivityStartDate"]).nunique()
        print(f"  {station}: {len(orig)} original × {N_COPIES + 1} = {len(combined)} total "
              f"({unique_dates} unique dates)")

    print(f"\n  Saved to: {out_dir}")
    return out_dir


# ============================================================================
# Strategy A: Synthetic physics-based generation (VALIDATION ONLY)
# ============================================================================

def strategy_a_synthetic():
    """Generate synthetic WQ targets from spectral/hydro-met features using
    physics-inspired rules. THIS IS FOR PIPELINE VALIDATION ONLY — the model
    will be learning the formula we define, not real water physics.

    Based on known spectral-WQ correlations:
    - Phosphorus correlates with Red/NIR (sediment proxy)
    - Nitrogen correlates with Green/Blue (dissolved organics)
    - Chlorophyll-a correlates with RedEdge (vegetation/algae signal)
    - DO inversely correlates with temperature
    - Turbidity correlates with Red/NIR (suspended sediment)
    """
    out_dir = os.path.join(BASE_DIR, "data", "augmented_synthetic")
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("STRATEGY A: Synthetic physics-based generation (VALIDATION ONLY)")
    print("=" * 70)

    DATA_DIR = os.path.join(BASE_DIR, "data", "processed_clean")

    for station in ["hastings", "prescott", "stpaul", "winona"]:
        daily_path = os.path.join(DATA_DIR, f"{station}_daily.csv")
        if not os.path.exists(daily_path):
            continue

        daily = pd.read_csv(daily_path, parse_dates=["date"])
        daily = daily.sort_values("date")

        # Sample every 7th day to get realistic measurement frequency
        sampled = daily.iloc[::7].copy()
        np.random.seed(42)

        # Seasonality signal
        doy = sampled["date"].dt.dayofyear
        season = np.sin(2 * np.pi * (doy - 150) / 365)

        rows = []
        for i, (_, row) in enumerate(sampled.iterrows()):
            s = season.iloc[i]
            red = row.get("Red", 0.1)
            nir = row.get("NIR", 0.1)
            green = row.get("Green", 0.1)
            blue = row.get("Blue", 0.1)
            re1 = row.get("RedEdge1", green)  # fallback
            temp = row.get("water_temp_c", 15)

            # Phosphorus: sediment proxy (Red + NIR)
            tp = 0.1 + (red * 0.8) + (nir * 0.4) + (s * 0.05)
            tp += np.random.normal(0, 0.02)
            tp = np.clip(tp, 0.05, 1.0)

            # Nitrogen: dissolved organics (Green + Blue)
            tn = 2.0 + (green * 5.0) + (blue * 2.0) + (s * 1.0)
            tn += np.random.normal(0, 0.5)
            tn = np.clip(tn, 1.0, 15.0)

            # Dissolved oxygen: inversely correlated with temperature
            do_val = 14.0 - (temp * 0.2) + (s * -1.5)
            do_val += np.random.normal(0, 0.3)
            do_val = np.clip(do_val, 4.0, 14.0)

            # Chlorophyll-a: algae (RedEdge)
            chl = 10.0 + (re1 * 100.0) + (s * 20.0)
            chl += np.random.normal(0, 5.0)
            chl = np.clip(chl, 1.0, 200.0)

            # Turbidity: suspended sediment (Red + NIR)
            turb = 5.0 + (red * 50.0) + (nir * 30.0) + (s * 3.0)
            turb += np.random.normal(0, 2.0)
            turb = np.clip(turb, 1.0, 500.0)

            date_str = row["date"].strftime("%Y-%m-%d")
            for char_name, val in [
                ("Phosphorus", tp), ("Nitrogen", tn),
                ("Dissolved oxygen (DO)", do_val),
                ("Chlorophyll a", chl), ("Turbidity", turb),
            ]:
                rows.append({
                    "ActivityStartDate": date_str,
                    "CharacteristicName": char_name,
                    "ResultMeasureValue": round(val, 4),
                    "source": "synthetic_physics",
                })

        synth_df = pd.DataFrame(rows)
        out_path = os.path.join(out_dir, f"{station}_wq.csv")
        synth_df.to_csv(out_path, index=False)

        unique_dates = synth_df["ActivityStartDate"].nunique()
        print(f"  {station}: {len(synth_df)} synthetic records ({unique_dates} unique dates)")

    print(f"\n  Saved to: {out_dir}")
    print("  ⚠ WARNING: This data is SYNTHETIC — for pipeline validation only!")
    return out_dir


# ============================================================================
# Strategy C (Danube): noise + temporal jitter on Danube WQ samples
# ============================================================================

def strategy_c_augmentation_danube(src_dir=None, out_dir=None):
    """Apply Gaussian noise and temporal jitter to Danube WQ samples.

    Same logic as strategy_c_augmentation but reads/writes the per-parameter
    CSV format used by the Danube dataset instead of USGS {station}_wq.csv files.

    Args:
        src_dir: directory containing the raw Danube WQ CSVs (defaults to
                 data/danube_processed)
        out_dir: output directory (defaults to data/danube_augmented_jitter)

    Output files mirror the input filenames so that _load_danube_wq_samples
    in train_model.py can read them unchanged when pointed at the output dir.
    """
    if src_dir is None:
        src_dir = DANUBE_WQ_DIR
    if out_dir is None:
        out_dir = os.path.join(BASE_DIR, "data", "danube_augmented_jitter")
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("STRATEGY C (DANUBE): Gaussian noise + temporal jitter augmentation")
    print("=" * 70)

    N_COPIES = 3
    NOISE_SCALE = 0.05
    MAX_JITTER_DAYS = 3

    for filename, param_code, label in DANUBE_WQ_FILES:
        src_path = os.path.join(src_dir, filename)
        if not os.path.exists(src_path):
            print(f"  SKIP {filename}: not found in {src_dir}")
            continue

        df = pd.read_csv(src_path)
        # Normalise column names (strip whitespace)
        df.columns = df.columns.str.strip()

        # Filter to the one parameter code we care about per file
        df = df[df["Parameter_Code"] == param_code].copy()
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        df = df.dropna(subset=["Value"])
        df["Sample_Date"] = pd.to_datetime(df["Sample_Date"])
        df["source"] = "original"

        if df.empty:
            print(f"  SKIP {label}: no usable rows after filtering to {param_code}")
            continue

        # Per-station std dev for noise scaling
        station_stds = (
            df.groupby("Station_ID")["Value"].std().fillna(0).to_dict()
        )

        augmented_rows = []
        for _, row in df.iterrows():
            sigma = station_stds.get(row["Station_ID"], 0.0) * NOISE_SCALE
            for copy_i in range(N_COPIES):
                new_row = row.copy()

                # Temporal jitter
                jitter_days = np.random.choice([-3, -2, -1, 1, 2, 3])
                new_row["Sample_Date"] = (
                    row["Sample_Date"] + pd.Timedelta(days=jitter_days)
                ).strftime("%Y-%m-%d")

                # Gaussian noise
                noise = np.random.normal(0, sigma) if sigma > 0 else 0.0
                new_row["Value"] = round(max(0.0, row["Value"] + noise), 4)
                new_row["source"] = f"augmented_copy{copy_i+1}_jitter{jitter_days:+d}d"
                augmented_rows.append(new_row)

        augmented_df = pd.DataFrame(augmented_rows)
        # Restore Sample_Date to string in original rows too
        df["Sample_Date"] = df["Sample_Date"].dt.strftime("%Y-%m-%d")

        combined = pd.concat([df, augmented_df], ignore_index=True)
        combined = combined.sort_values(["Station_ID", "Sample_Date"]).reset_index(drop=True)

        out_path = os.path.join(out_dir, filename)
        combined.to_csv(out_path, index=False)

        n_orig = len(df)
        n_total = len(combined)
        n_stations = combined["Station_ID"].nunique()
        print(f"  {label:<28s}: {n_orig} original × {N_COPIES+1} = {n_total} total "
              f"({n_stations} stations)")

    print(f"\n  Saved to: {out_dir}")
    print(f"  To train: python train_model.py {os.path.relpath(out_dir, BASE_DIR)} "
          f"danube_jitter --model c")
    return out_dir


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Water Quality Data Augmentation")
    parser.add_argument("--danube", action="store_true",
                        help="Run Strategy C for Danube dataset instead of Mississippi")
    parser.add_argument("--strategy", choices=["a", "b", "c", "all"], default="all",
                        help="Which strategy to run for Mississippi (default: all)")
    args = parser.parse_args()

    print("Water Quality Data Augmentation Pipeline")
    print("=" * 70)

    if args.danube:
        dir_c_danube = strategy_c_augmentation_danube()
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"  C/Danube (noise + jitter): {dir_c_danube}")
    else:
        dir_b = dir_c = dir_a = None

        if args.strategy in ("b", "all"):
            try:
                dir_b = strategy_b_real_expansion()
            except Exception as e:
                print(f"\n  Strategy B failed (likely no internet): {e}")

        if args.strategy in ("c", "all"):
            dir_c = strategy_c_augmentation()

        if args.strategy in ("a", "all"):
            dir_a = strategy_a_synthetic()

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        if dir_b:
            print(f"  B (real expansion):  {dir_b}")
        if dir_c:
            print(f"  C (noise + jitter):  {dir_c}")
        if dir_a:
            print(f"  A (synthetic):       {dir_a}")
        print("\nTo train with augmented data, pass the output dir as the first argument:")
        print("  python train_model.py data/augmented_jitter jitter_v1 --model a")
