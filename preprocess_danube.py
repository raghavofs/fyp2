"""
Preprocess Danube River data for the water quality model pipeline.
7 Serbian Danube monitoring stations (SRB00001–SRB00006, SRB00040, SRB00041).

Steps:
1. Fix DOY-encoded dates in satellite CSV and replace -9999 (Landsat 8 sentinel)
2. Load discharge, weather, and water quality CSVs
3. Build per-station daily interpolated CSVs in data/danube_processed/processed_clean/

Output column schema (matches Mississippi processed_clean format so train_model.py
can reuse the same load_and_prepare_data function):
  date, streamflow, precipitation, water_temp_c,
  Blue, Green, Red, NIR, SWIR1, SWIR2, RedEdge1, RedEdge2, RedEdge3,
  dissolved_oxygen, total_phosphorus, nitrate_n,
  electrical_conductance, oxygen_demand, chlorophyll_a,
  station_id
"""

import os
import datetime
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DANUBE_DIR = os.path.join(BASE_DIR, "data", "danube_processed")
OUTPUT_DIR = os.path.join(DANUBE_DIR, "processed_clean")
os.makedirs(OUTPUT_DIR, exist_ok=True)

STATIONS = {
    "SRB00001": "Bezdan",
    "SRB00040": "Bogojevo",
    "SRB00002": "Novi Sad",
    "SRB00003": "Zemun",
    "SRB00041": "Smederevo",
    "SRB00005": "Banatska Palanka",
    "SRB00006": "Tekija",
}

# 30-day lookback buffer before first 2013 WQ measurement
DATE_RANGE = ("2012-12-01", "2023-12-31")

BAND_COLS = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2",
             "RedEdge1", "RedEdge2", "RedEdge3"]
REDEDGE_COLS = ["RedEdge1", "RedEdge2", "RedEdge3"]

# WQ file -> (filename, parameter_code_to_keep, output_column_name)
WQ_EXTRACTIONS = [
    ("dissolved_oxygen.csv",       "O2-Dis",  "dissolved_oxygen"),
    ("phosphorus.csv",             "TP",       "total_phosphorus"),
    ("nitrogen_oxidized.csv",      "NO3N",     "nitrate_n"),
    ("electrical_conductance.csv", "EC",       "electrical_conductance"),
    ("oxygen_demand.csv",          "BOD",      "oxygen_demand"),
    ("chlorophyll.csv",            "Chl-a",    "chlorophyll_a"),
]

WQ_COLS = [col for _, _, col in WQ_EXTRACTIONS]


# ---------------------------------------------------------------------------
# Step 1: Fix satellite dates and clean -9999
# ---------------------------------------------------------------------------
def fix_satellite_data():
    print("=" * 60)
    print("1. Fixing Danube satellite dates and sentinel values")
    print("=" * 60)

    sat_path = os.path.join(DANUBE_DIR, "Danube_Satellite_Data_2012_2023 (1).csv")
    df = pd.read_csv(sat_path)
    print(f"  Loaded {len(df)} rows")

    # Fix DOY-encoded dates: "2013-04-118" → year=2013, DOY=118
    fixed_dates = []
    bad_count = 0
    for raw_date in df["image_date"]:
        parts = str(raw_date).split("-")
        year = int(parts[0])
        day_part = int(parts[2])  # the third segment is always the DOY
        if day_part > 31:
            actual = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day_part - 1)
            fixed_dates.append(actual.strftime("%Y-%m-%d"))
            bad_count += 1
        else:
            # Already a valid day number — reconstruct with correct month
            month = int(parts[1])
            fixed_dates.append(f"{year:04d}-{month:02d}-{day_part:02d}")

    df["image_date"] = pd.to_datetime(fixed_dates)
    print(f"  Fixed {bad_count} DOY-encoded dates")

    # Replace -9999 sentinel with NaN for RedEdge bands (L8 has none)
    for col in REDEDGE_COLS:
        n_bad = (df[col] == -9999).sum()
        df[col] = df[col].replace(-9999, np.nan)
        if n_bad:
            print(f"  Replaced {n_bad} sentinel -9999 values in {col}")

    # Also replace any remaining -9999 in other band columns
    for col in ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]:
        df[col] = df[col].replace(-9999, np.nan)

    out_path = os.path.join(OUTPUT_DIR, "satellite_fixed.csv")
    df.to_csv(out_path, index=False)
    print(f"  Date range: {df['image_date'].min().date()} → {df['image_date'].max().date()}")
    print(f"  Saved to {out_path}")
    return df


# ---------------------------------------------------------------------------
# Step 2: Load discharge, weather, and WQ data
# ---------------------------------------------------------------------------
def load_discharge():
    path = os.path.join(DANUBE_DIR, "discharge.csv")
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.rename(columns={"Station_ID": "station_id", "Date": "date",
                             "river_discharge_m3s": "streamflow"})
    print(f"  Discharge: {len(df)} rows, "
          f"{df['date'].min().date()} → {df['date'].max().date()}")
    return df[["station_id", "date", "streamflow"]]


def load_weather():
    path = os.path.join(DANUBE_DIR, "weather.csv")
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.rename(columns={"Station_ID": "station_id", "Date": "date",
                             "precipitation_mm": "precipitation",
                             "temperature_mean_c": "water_temp_c"})
    print(f"  Weather:   {len(df)} rows, "
          f"{df['date'].min().date()} → {df['date'].max().date()}")
    return df[["station_id", "date", "precipitation", "water_temp_c"]]


def load_wq_data():
    """Load all 6 WQ parameter files and merge into a wide-format DataFrame."""
    frames = []
    for filename, param_code, col_name in WQ_EXTRACTIONS:
        path = os.path.join(DANUBE_DIR, filename)
        if not os.path.exists(path):
            print(f"  WARNING: {filename} not found, skipping {col_name}")
            continue
        df = pd.read_csv(path, parse_dates=["Sample_Date"])
        df = df.rename(columns={"Station_ID": "station_id",
                                 "Sample_Date": "date",
                                 "Parameter_Code": "param_code",
                                 "Value": "value"})
        # Filter to desired parameter code
        df = df[df["param_code"] == param_code].copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        # Average multiple readings on same date/station
        df = (df.groupby(["station_id", "date"])["value"]
                .mean()
                .reset_index()
                .rename(columns={"value": col_name}))
        n = len(df)
        print(f"  WQ {col_name:<25s}: {n:>5} station-date records")
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    # Outer-merge all parameters on (station_id, date)
    wq = frames[0]
    for f in frames[1:]:
        wq = wq.merge(f, on=["station_id", "date"], how="outer")
    return wq


# ---------------------------------------------------------------------------
# Step 3: Build per-station daily interpolated CSVs
# ---------------------------------------------------------------------------
def build_daily_datasets(satellite_df, discharge_df, weather_df, wq_df):
    print("\n" + "=" * 60)
    print("3. Building per-station daily datasets")
    print("=" * 60)

    full_range = pd.date_range(DATE_RANGE[0], DATE_RANGE[1], freq="D")
    total_days = len(full_range)

    all_frames = []

    for station_id, station_name in STATIONS.items():
        print(f"\n--- {station_id} ({station_name}) ---")

        daily = pd.DataFrame({"date": full_range})

        # -- Streamflow --
        sf = discharge_df[discharge_df["station_id"] == station_id][["date", "streamflow"]].copy()
        sf = sf.drop_duplicates("date").sort_values("date")
        daily = daily.merge(sf, on="date", how="left")
        missing = daily["streamflow"].isna().sum()
        daily["streamflow"] = daily["streamflow"].interpolate(method="linear", limit_direction="both")
        print(f"  Streamflow:  {total_days - missing}/{total_days} days ({100*(total_days-missing)/total_days:.1f}%), interpolated {missing}")

        # -- Precipitation (fill missing with 0) --
        wx = weather_df[weather_df["station_id"] == station_id][["date", "precipitation", "water_temp_c"]].copy()
        wx = wx.drop_duplicates("date").sort_values("date")
        daily = daily.merge(wx, on="date", how="left")
        miss_pr = daily["precipitation"].isna().sum()
        miss_wt = daily["water_temp_c"].isna().sum()
        daily["precipitation"] = daily["precipitation"].fillna(0.0)
        daily["water_temp_c"] = daily["water_temp_c"].interpolate(method="linear", limit_direction="both")
        print(f"  Precip:      filled {miss_pr} missing days with 0")
        print(f"  Water temp:  interpolated {miss_wt} missing days")

        # -- Satellite spectral bands --
        sp = satellite_df[satellite_df["station_id"] == station_id].copy()
        if not sp.empty:
            sp = sp.rename(columns={"image_date": "date"})
            sp = (sp.groupby("date")[BAND_COLS].mean().reset_index())
            n_sat = len(sp)
            daily = daily.merge(sp, on="date", how="left")
            missing_sat = daily["Blue"].isna().sum()
            for col in BAND_COLS:
                daily[col] = daily[col].interpolate(method="linear", limit_direction="both")
            print(f"  Satellite:   {n_sat} obs → interpolated {missing_sat} missing days")
        else:
            print(f"  Satellite:   NO DATA for {station_id}")
            for col in BAND_COLS:
                daily[col] = np.nan

        # -- Water quality targets (keep sparse, do NOT interpolate) --
        if not wq_df.empty:
            wq_st = wq_df[wq_df["station_id"] == station_id][["date"] + WQ_COLS].copy()
            wq_st = wq_st.drop_duplicates("date")
            daily = daily.merge(wq_st, on="date", how="left")
            for col in WQ_COLS:
                n_real = daily[col].notna().sum()
                print(f"  {col:<28s}: {n_real} actual measurements")
        else:
            for col in WQ_COLS:
                daily[col] = np.nan

        daily["station_id"] = station_id

        # Reorder columns
        ordered_cols = (
            ["date", "streamflow", "precipitation", "water_temp_c"]
            + BAND_COLS
            + WQ_COLS
            + ["station_id"]
        )
        daily = daily[ordered_cols]

        out_path = os.path.join(OUTPUT_DIR, f"{station_id}_daily.csv")
        daily.to_csv(out_path, index=False)
        print(f"  → Saved {len(daily)} rows to {out_path}")

        all_frames.append(daily)

    # Combined file
    combined = pd.concat(all_frames, ignore_index=True)
    combined_path = os.path.join(OUTPUT_DIR, "all_stations_daily.csv")
    combined.to_csv(combined_path, index=False)
    print(f"\n  → Combined: {len(combined)} rows saved to {combined_path}")


# ---------------------------------------------------------------------------
# Coverage audit
# ---------------------------------------------------------------------------
def audit_coverage(satellite_df, discharge_df, weather_df, wq_df):
    print("\n" + "=" * 60)
    print("2. Coverage audit per station")
    print("=" * 60)

    full_range = pd.date_range(DATE_RANGE[0], DATE_RANGE[1], freq="D")
    total_days = len(full_range)

    for station_id, station_name in STATIONS.items():
        print(f"\n  {station_id} ({station_name})")

        sf_days = len(discharge_df[discharge_df["station_id"] == station_id]["date"].unique())
        wx_days = len(weather_df[weather_df["station_id"] == station_id]["date"].unique())
        sat_days = len(satellite_df[satellite_df["station_id"] == station_id]["image_date"].unique())

        print(f"    Discharge:  {sf_days:>5}/{total_days} ({100*sf_days/total_days:.1f}%)")
        print(f"    Weather:    {wx_days:>5}/{total_days} ({100*wx_days/total_days:.1f}%)")
        print(f"    Satellite:  {sat_days:>5}/{total_days} ({100*sat_days/total_days:.1f}%)")

        if not wq_df.empty:
            wq_st = wq_df[wq_df["station_id"] == station_id]
            for col in WQ_COLS:
                n = wq_st[col].notna().sum() if col in wq_st.columns else 0
                print(f"    {col:<28s}: {n:>4} samples")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Step 1
    satellite_df = fix_satellite_data()

    # Step 2
    print("\n" + "=" * 60)
    print("2. Loading supporting data sources")
    print("=" * 60)
    discharge_df = load_discharge()
    weather_df = load_weather()
    wq_df = load_wq_data()

    # Audit
    audit_coverage(satellite_df, discharge_df, weather_df, wq_df)

    # Step 3
    build_daily_datasets(satellite_df, discharge_df, weather_df, wq_df)

    print("\n" + "=" * 60)
    print("DONE. Processed files saved to:", OUTPUT_DIR)
    print("=" * 60)
