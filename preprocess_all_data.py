"""
Preprocess all data sources for the water quality model pipeline.
1. Parse water temp HTML -> clean CSV
2. Fix malformed DOY dates in spectral data
3. Audit date coverage across all sources per station
4. Interpolate gaps to produce consistent daily time series
"""

import os
import re
import datetime
import pandas as pd
import numpy as np
from io import StringIO

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(DATA_DIR, "processed_clean")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Station mapping: name -> USGS site ID
STATIONS = {
    "brooklynpark": "05288500",
    "stpaul": "05331000",
    "hastings": "05331580",
    "prescott": "05344500",
    "winona": "05378500",
}

# Reverse mapping
SITE_TO_NAME = {v: k for k, v in STATIONS.items()}

# Precipitation station name mapping (from NOAA names to our station names)
PRECIP_NAME_MAP = {
    "BROOKLYN PARK": "brooklynpark",
    "ST PAUL PARK": "stpaul",
    "HASTINGS": "hastings",
    "PRESCOTT": "prescott",
    "WINONA": "winona",
}

DATE_RANGE = ("2019-01-01", "2025-12-31")


# ---------------------------------------------------------------------------
# 1. Parse water temperature HTML -> CSV
# ---------------------------------------------------------------------------
def parse_water_temp():
    print("=" * 60)
    print("1. Parsing water temperature HTML file")
    print("=" * 60)

    filepath = os.path.join(DATA_DIR, "watertemp", "watertemp.xls")

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    # Extract date and temperature values from HTML table rows
    # Pattern: date in one <td>, temperature in next <td>
    pattern = r'<td><div align="center">\s*([\d/]+\s+[\d:]+)\s*</div></td>\s*<td><div align="center">\s*([\d.]+)\s*</div></td>'
    matches = re.findall(pattern, html)

    if not matches:
        print("  ERROR: Could not parse any data from HTML file")
        return None

    rows = []
    for date_str, temp_str in matches:
        try:
            dt = pd.to_datetime(date_str, format="%m/%d/%Y %H:%M")
            temp = float(temp_str)
            rows.append({"date": dt.date(), "water_temp_f": temp})
        except (ValueError, TypeError):
            continue

    df = pd.DataFrame(rows)
    # Average if multiple readings per day
    df = df.groupby("date").agg({"water_temp_f": "mean"}).reset_index()
    df["date"] = pd.to_datetime(df["date"])

    # Convert F to C
    df["water_temp_c"] = (df["water_temp_f"] - 32) * 5 / 9

    outpath = os.path.join(OUTPUT_DIR, "water_temp.csv")
    df.to_csv(outpath, index=False)
    print(f"  Parsed {len(df)} daily records")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Saved to {outpath}")
    return df


# ---------------------------------------------------------------------------
# 2. Fix spectral data dates (DOY format issue)
# ---------------------------------------------------------------------------
def fix_spectral_dates():
    print("\n" + "=" * 60)
    print("2. Fixing spectral data dates")
    print("=" * 60)

    filepath = os.path.join(DATA_DIR, "waterfeatures", "Mississippi_River_Spectral_Data_2019_2025.csv")
    df = pd.read_csv(filepath)

    fixed_dates = []
    bad_count = 0
    for raw_date in df["Date"]:
        parts = str(raw_date).split("-")
        year = int(parts[0])
        day_part = int(parts[2])

        if day_part > 31:
            # Day-of-year format: use DOY to compute actual date
            actual_date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day_part - 1)
            fixed_dates.append(actual_date.strftime("%Y-%m-%d"))
            bad_count += 1
        else:
            fixed_dates.append(raw_date)

    df["Date"] = pd.to_datetime(fixed_dates)

    # Map station IDs to names
    df["station_name"] = df["station_id"].astype(str).str.zfill(8).map(SITE_TO_NAME)

    outpath = os.path.join(OUTPUT_DIR, "spectral_data_fixed.csv")
    df.to_csv(outpath, index=False)
    print(f"  Fixed {bad_count} malformed DOY dates")
    print(f"  Total records: {len(df)}")
    print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"  Saved to {outpath}")
    return df


# ---------------------------------------------------------------------------
# 3. Load and standardize all data sources
# ---------------------------------------------------------------------------
def load_streamflow():
    """Load streamflow data for all stations."""
    frames = []
    for station_name, site_id in STATIONS.items():
        filepath = os.path.join(DATA_DIR, "streamflow", f"{station_name}.csv")
        if not os.path.exists(filepath):
            print(f"  WARNING: No streamflow data for {station_name}")
            continue
        df = pd.read_csv(filepath, skiprows=[1])  # skip the "5s,15s,..." format row
        df = df.rename(columns={"datetime": "date", "flow": "streamflow"})
        df["date"] = pd.to_datetime(df["date"])
        df["station_name"] = station_name
        df["streamflow"] = pd.to_numeric(df["streamflow"], errors="coerce")
        frames.append(df[["date", "station_name", "streamflow"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_precipitation():
    """Load precipitation data, mapped to nearest station."""
    filepath = os.path.join(DATA_DIR, "precipitation", "4224543.csv")
    df = pd.read_csv(filepath)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["PRCP"] = pd.to_numeric(df["PRCP"], errors="coerce")

    # Map NOAA station names to our station names
    def map_station(name):
        for key, val in PRECIP_NAME_MAP.items():
            if key in str(name).upper():
                return val
        return None

    df["station_name"] = df["NAME"].apply(map_station)
    df = df.dropna(subset=["station_name"])
    df = df.rename(columns={"DATE": "date", "PRCP": "precipitation"})
    return df[["date", "station_name", "precipitation"]]


def load_water_quality():
    """Load water quality data for all stations."""
    frames = []
    for station_name in STATIONS:
        filepath = os.path.join(DATA_DIR, "waterquality", f"{station_name}_wq.csv")
        if not os.path.exists(filepath):
            continue
        df = pd.read_csv(filepath)
        if "ActivityStartDate" in df.columns:
            df = df.rename(columns={"ActivityStartDate": "date"})
        df["date"] = pd.to_datetime(df["date"])
        df["station_name"] = station_name

        # Pivot characteristics into columns
        if "CharacteristicName" in df.columns and "ResultMeasureValue" in df.columns:
            df["ResultMeasureValue"] = pd.to_numeric(df["ResultMeasureValue"], errors="coerce")
            pivot = df.pivot_table(
                index=["date", "station_name"],
                columns="CharacteristicName",
                values="ResultMeasureValue",
                aggfunc="mean",
            ).reset_index()
            # Clean column names
            pivot.columns = [c.lower().replace(" ", "_") if c not in ["date", "station_name"] else c for c in pivot.columns]
            frames.append(pivot)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# 4. Audit and report coverage
# ---------------------------------------------------------------------------
def audit_coverage(streamflow_df, precip_df, wq_df, spectral_df, watertemp_df):
    print("\n" + "=" * 60)
    print("3. Data coverage audit")
    print("=" * 60)

    full_range = pd.date_range(DATE_RANGE[0], DATE_RANGE[1], freq="D")
    total_days = len(full_range)

    for station in STATIONS:
        print(f"\n--- {station} ---")

        # Streamflow
        sf = streamflow_df[streamflow_df["station_name"] == station]
        sf_dates = set(sf["date"].dt.date)
        sf_coverage = len(sf_dates)
        print(f"  Streamflow:     {sf_coverage:>5}/{total_days} days ({100*sf_coverage/total_days:.1f}%)")

        # Precipitation
        pr = precip_df[precip_df["station_name"] == station]
        pr_dates = set(pr["date"].dt.date)
        pr_coverage = len(pr_dates)
        print(f"  Precipitation:  {pr_coverage:>5}/{total_days} days ({100*pr_coverage/total_days:.1f}%)")

        # Water quality (sparse by nature - sample-based)
        wq = wq_df[wq_df["station_name"] == station] if not wq_df.empty else pd.DataFrame()
        wq_dates = set(wq["date"].dt.date) if not wq.empty else set()
        wq_coverage = len(wq_dates)
        print(f"  Water quality:  {wq_coverage:>5}/{total_days} days ({100*wq_coverage/total_days:.1f}%) [sample-based]")

        # Spectral data (satellite revisit ~5 days)
        sp = spectral_df[spectral_df["station_name"] == station]
        sp_dates = set(sp["Date"].dt.date)
        sp_coverage = len(sp_dates)
        print(f"  Spectral:       {sp_coverage:>5}/{total_days} days ({100*sp_coverage/total_days:.1f}%) [satellite revisit]")

        # Water temp (single station, applies to all)
        wt_dates = set(watertemp_df["date"].dt.date) if watertemp_df is not None else set()
        wt_coverage = len(wt_dates)
        print(f"  Water temp:     {wt_coverage:>5}/{total_days} days ({100*wt_coverage/total_days:.1f}%) [shared]")


# ---------------------------------------------------------------------------
# 5. Build interpolated per-station daily datasets
# ---------------------------------------------------------------------------
def build_daily_dataset(streamflow_df, precip_df, wq_df, spectral_df, watertemp_df):
    print("\n" + "=" * 60)
    print("4. Building interpolated daily datasets per station")
    print("=" * 60)

    full_range = pd.date_range(DATE_RANGE[0], DATE_RANGE[1], freq="D")

    for station in STATIONS:
        print(f"\n--- {station} ---")

        # Start with full date range
        daily = pd.DataFrame({"date": full_range})

        # -- Streamflow (daily, interpolate gaps) --
        sf = streamflow_df[streamflow_df["station_name"] == station][["date", "streamflow"]].copy()
        sf = sf.drop_duplicates(subset="date").sort_values("date")
        daily = daily.merge(sf, on="date", how="left")
        missing_sf = daily["streamflow"].isna().sum()
        daily["streamflow"] = daily["streamflow"].interpolate(method="linear", limit_direction="both")
        print(f"  Streamflow: interpolated {missing_sf} missing days")

        # -- Precipitation (interpolate gaps) --
        pr = precip_df[precip_df["station_name"] == station][["date", "precipitation"]].copy()
        pr = pr.drop_duplicates(subset="date").sort_values("date")
        daily = daily.merge(pr, on="date", how="left")
        missing_pr = daily["precipitation"].isna().sum()
        # For precipitation, fill missing with 0 (no rain assumed) rather than interpolating
        daily["precipitation"] = daily["precipitation"].fillna(0.0)
        print(f"  Precipitation: filled {missing_pr} missing days with 0")

        # -- Water temperature (shared, interpolate) --
        if watertemp_df is not None and not watertemp_df.empty:
            wt = watertemp_df[["date", "water_temp_c"]].copy()
            wt = wt.drop_duplicates(subset="date").sort_values("date")
            daily = daily.merge(wt, on="date", how="left")
            missing_wt = daily["water_temp_c"].isna().sum()
            daily["water_temp_c"] = daily["water_temp_c"].interpolate(method="linear", limit_direction="both")
            print(f"  Water temp: interpolated {missing_wt} missing days")

        # -- Spectral data (satellite revisit ~5 days, interpolate) --
        sp = spectral_df[spectral_df["station_name"] == station].copy()
        band_cols = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "RedEdge1", "RedEdge2", "RedEdge3"]
        if not sp.empty:
            # Average duplicate dates (multiple satellite passes)
            sp = sp.groupby("Date")[band_cols].mean().reset_index()
            sp = sp.rename(columns={"Date": "date"})
            daily = daily.merge(sp, on="date", how="left")
            missing_sp = daily["Blue"].isna().sum()
            for col in band_cols:
                daily[col] = daily[col].interpolate(method="linear", limit_direction="both")
            print(f"  Spectral bands: interpolated {missing_sp} missing days (9 bands)")
        else:
            print(f"  Spectral bands: NO DATA for this station")

        # -- Water quality targets (sparse - interpolate for training) --
        wq = wq_df[wq_df["station_name"] == station].copy() if not wq_df.empty else pd.DataFrame()
        if not wq.empty:
            wq_cols = [c for c in wq.columns if c not in ["date", "station_name"]]
            wq = wq.drop(columns=["station_name"])
            wq = wq.groupby("date")[wq_cols].mean().reset_index()
            daily = daily.merge(wq, on="date", how="left")
            for col in wq_cols:
                present = daily[col].notna().sum()
                daily[col] = daily[col].interpolate(method="linear", limit_direction="both")
                print(f"  {col}: {present} samples -> interpolated to daily")
        else:
            print(f"  Water quality: NO DATA for this station")

        daily["station_name"] = station

        # Save
        outpath = os.path.join(OUTPUT_DIR, f"{station}_daily.csv")
        daily.to_csv(outpath, index=False)
        print(f"  -> Saved {len(daily)} rows to {outpath}")

    # Also save a combined version
    all_frames = []
    for station in STATIONS:
        path = os.path.join(OUTPUT_DIR, f"{station}_daily.csv")
        if os.path.exists(path):
            all_frames.append(pd.read_csv(path))
    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined_path = os.path.join(OUTPUT_DIR, "all_stations_daily.csv")
        combined.to_csv(combined_path, index=False)
        print(f"\n  -> Combined dataset: {len(combined)} rows saved to {combined_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Step 1: Parse water temp
    watertemp_df = parse_water_temp()

    # Step 2: Fix spectral dates
    spectral_df = fix_spectral_dates()

    # Step 3: Load other sources
    print("\n" + "=" * 60)
    print("Loading other data sources...")
    print("=" * 60)
    streamflow_df = load_streamflow()
    print(f"  Streamflow: {len(streamflow_df)} records")
    precip_df = load_precipitation()
    print(f"  Precipitation: {len(precip_df)} records")
    wq_df = load_water_quality()
    print(f"  Water quality: {len(wq_df)} records")

    # Step 4: Audit
    audit_coverage(streamflow_df, precip_df, wq_df, spectral_df, watertemp_df)

    # Step 5: Build daily interpolated datasets
    build_daily_dataset(streamflow_df, precip_df, wq_df, spectral_df, watertemp_df)

    print("\n" + "=" * 60)
    print("DONE! All processed data saved to:", OUTPUT_DIR)
    print("=" * 60)
