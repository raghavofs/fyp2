"""
Preprocess and merge all data sources into a unified dataset
aligned to a 5-day cadence (matching satellite imagery).

Data sources:
  - Streamflow (daily) -> 5-day mean
  - Water quality (irregular, ~21-day) -> interpolated to 5-day
  - (Satellite bands to be added later)

Outputs one CSV per station into data/processed/
"""

import os
import pandas as pd
import numpy as np

# Stations and their available data
STATIONS = {
    "hastings": {
        "streamflow": "data/streamflow/hastings.csv",
        "wq": "data/waterquality/hastings_wq.csv",
    },
    "prescott": {
        "streamflow": "data/streamflow/prescott.csv",
        "wq": "data/waterquality/prescott_wq.csv",
    },
    "winona": {
        "streamflow": "data/streamflow/winona.csv",
        "wq": "data/waterquality/winona_wq.csv",
    },
    "stpaul": {
        "streamflow": "data/streamflow/stpaul.csv",
        "wq": "data/waterquality/stpaul_wq.csv",
    },
    "brooklynpark": {
        "streamflow": "data/streamflow/brooklynpark.csv",
        "wq": "data/waterquality/brooklynpark_wq.csv",
    },
}

# WQ parameters to keep (dropping Chlorophyll a and DO due to sparsity)
WQ_PARAMS = ["Phosphorus", "Nitrogen", "Turbidity"]

# 5-day cadence matching satellite revisit
RESAMPLE_DAYS = 5

OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_streamflow(filepath):
    """Load streamflow CSV, return daily series."""
    df = pd.read_csv(filepath, skiprows=[1])
    df["date"] = pd.to_datetime(df["datetime"])
    df["flow"] = pd.to_numeric(df["flow"], errors="coerce")
    df = df[["date", "flow"]].dropna().set_index("date").sort_index()
    return df


def load_water_quality(filepath):
    """Load WQ CSV, pivot to wide format with one value per date per parameter.
    Duplicate readings on the same date are averaged."""
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["ActivityStartDate"])
    df["value"] = pd.to_numeric(df["ResultMeasureValue"], errors="coerce")

    # Keep only the parameters we want
    df = df[df["CharacteristicName"].isin(WQ_PARAMS)]

    # Average duplicate readings per (date, parameter)
    grouped = df.groupby(["date", "CharacteristicName"])["value"].mean().reset_index()

    # Pivot to wide format: columns = parameter names
    wide = grouped.pivot(index="date", columns="CharacteristicName", values="value")
    wide.columns = [c.lower().replace(" ", "_") for c in wide.columns]
    wide = wide.sort_index()

    return wide


def interpolate_to_5day(streamflow_df, wq_df, start_date=None, end_date=None):
    """Align all data to a 5-day grid using interpolation.

    Streamflow: 5-day rolling mean (dense daily data -> 5-day samples).
    Water quality: linear interpolation between sparse irregular readings.
    """
    # Determine date range from overlapping data
    if start_date is None:
        start_date = max(streamflow_df.index.min(), wq_df.index.min())
    if end_date is None:
        end_date = min(streamflow_df.index.max(), wq_df.index.max())

    # Create the 5-day date grid
    date_grid = pd.date_range(start=start_date, end=end_date, freq=f"{RESAMPLE_DAYS}D")

    # --- Streamflow: 5-day rolling mean, then sample at grid points ---
    flow_5d = streamflow_df["flow"].rolling(window=RESAMPLE_DAYS, center=True).mean()
    flow_resampled = flow_5d.reindex(date_grid, method="nearest", tolerance="2D")
    flow_out = flow_resampled.to_frame("streamflow")

    # --- Water quality: reindex to grid and interpolate ---
    # First, reindex WQ to include both original dates AND the 5-day grid
    all_dates = wq_df.index.union(date_grid).sort_values()
    wq_expanded = wq_df.reindex(all_dates)

    # Linear interpolation (only between known points, no extrapolation)
    wq_interpolated = wq_expanded.interpolate(method="time", limit_area="inside")

    # Sample at the 5-day grid points
    wq_out = wq_interpolated.reindex(date_grid)

    # --- Merge ---
    merged = flow_out.join(wq_out, how="outer")
    merged.index.name = "date"

    return merged


def process_station(name, paths):
    """Full pipeline for one station."""
    print(f"\n{'=' * 60}")
    print(f"  {name.upper()}")
    print(f"{'=' * 60}")

    # Load data
    streamflow = load_streamflow(paths["streamflow"])
    print(f"  Streamflow: {len(streamflow)} daily records "
          f"({streamflow.index.min().date()} to {streamflow.index.max().date()})")

    wq = load_water_quality(paths["wq"])
    print(f"  Water quality: {len(wq)} observation dates")
    print(f"    Parameters available: {wq.columns.tolist()}")
    for col in wq.columns:
        non_null = wq[col].notna().sum()
        print(f"    {col}: {non_null} readings")

    if wq.empty or len(wq) < 5:
        print(f"  SKIPPING — not enough WQ data (need >= 5 readings)")
        return None

    # Interpolate and merge
    merged = interpolate_to_5day(streamflow, wq)

    # Report coverage
    print(f"\n  Output: {len(merged)} rows at {RESAMPLE_DAYS}-day intervals")
    print(f"  Date range: {merged.index.min().date()} to {merged.index.max().date()}")
    print(f"  Coverage (non-null):")
    for col in merged.columns:
        n = merged[col].notna().sum()
        pct = 100 * n / len(merged)
        print(f"    {col}: {n}/{len(merged)} ({pct:.0f}%)")

    # Show gap stats for interpolated WQ
    print(f"\n  Interpolation quality:")
    for col in [c for c in merged.columns if c != "streamflow"]:
        if col in wq.columns:
            n_original = wq[col].notna().sum()
            n_interpolated = merged[col].notna().sum()
            n_synthetic = n_interpolated - n_original
            if n_interpolated > 0:
                pct_real = 100 * n_original / n_interpolated
                print(f"    {col}: {n_original} real + {n_synthetic} interpolated "
                      f"= {n_interpolated} total ({pct_real:.0f}% real)")

    # Save
    outpath = os.path.join(OUTPUT_DIR, f"{name}_5day.csv")
    merged.to_csv(outpath)
    print(f"\n  Saved to: {outpath}")

    return merged


def main():
    print("Preprocessing all stations to 5-day cadence")
    print(f"Parameters: {WQ_PARAMS}")
    print(f"Output: {OUTPUT_DIR}/")

    results = {}
    for name, paths in STATIONS.items():
        if not os.path.exists(paths["wq"]):
            print(f"\n  {name}: no WQ file found, skipping")
            continue
        results[name] = process_station(name, paths)

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, df in results.items():
        if df is not None:
            complete = df.dropna()
            print(f"  {name:15s}: {len(df):4d} total rows, "
                  f"{len(complete):4d} complete rows ({100*len(complete)/len(df):.0f}%)")
        else:
            print(f"  {name:15s}: skipped (insufficient data)")


if __name__ == "__main__":
    main()
