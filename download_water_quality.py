"""
Download water quality data from USGS for Mississippi River stations.
Parameters: Phosphorus, Nitrogen, Dissolved Oxygen, Chlorophyll a, Turbidity
Date range: 2019-2025
"""

import os
import pandas as pd
from dataretrieval import nwis, wqp

# Station mapping (matching streamflow stations)
STATIONS = {
    "brooklynpark": "05288500",
    "stpaul": "05331000",
    "hastings": "05331580",
    "prescott": "05344500",
    "winona": "05378500",
}

# USGS parameter codes for water quality
# See: https://help.waterdata.usgs.gov/parameter_listing
PARAM_CODES = {
    "phosphorus": "00665",       # Phosphorus, total (mg/L)
    "nitrogen": "00631",         # Nitrogen, nitrate + nitrite (mg/L)
    "dissolved_oxygen": "00300", # Dissolved oxygen (mg/L)
    "chlorophyll_a": "70953",    # Chlorophyll a (ug/L)
    "turbidity": "63680",        # Turbidity (FNU)
}

START_DATE = "2019-01-01"
END_DATE = "2025-12-31"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "waterquality")


def download_usgs_qw():
    """Try downloading via NWIS qwdata (quantitative water quality)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_param_codes = list(PARAM_CODES.values())
    site_list = list(STATIONS.values())

    print(f"Downloading water quality data for sites: {site_list}")
    print(f"Parameters: {list(PARAM_CODES.keys())}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print()

    # --- Method 1: Try NWIS instantaneous/daily values for each parameter ---
    for station_name, site_no in STATIONS.items():
        print(f"--- {station_name} (site {site_no}) ---")
        station_frames = []

        for param_name, param_code in PARAM_CODES.items():
            try:
                # Try daily values first
                df, metadata = nwis.get_dv(
                    sites=site_no,
                    parameterCd=param_code,
                    start=START_DATE,
                    end=END_DATE,
                )
                if not df.empty:
                    df = df.reset_index()
                    df["parameter"] = param_name
                    df["site_name"] = station_name
                    station_frames.append(df)
                    print(f"  {param_name}: {len(df)} daily records")
                    continue
            except Exception:
                pass

            try:
                # Fall back to instantaneous values
                df, metadata = nwis.get_iv(
                    sites=site_no,
                    parameterCd=param_code,
                    start=START_DATE,
                    end=END_DATE,
                )
                if not df.empty:
                    df = df.reset_index()
                    df["parameter"] = param_name
                    df["site_name"] = station_name
                    station_frames.append(df)
                    print(f"  {param_name}: {len(df)} instantaneous records")
                    continue
            except Exception:
                pass

            print(f"  {param_name}: no NWIS data found")

        if station_frames:
            combined = pd.concat(station_frames, ignore_index=True)
            outpath = os.path.join(OUTPUT_DIR, f"{station_name}.csv")
            combined.to_csv(outpath, index=False)
            print(f"  -> Saved to {outpath}")
        print()

    # --- Method 2: Also try Water Quality Portal for sample data ---
    print("=" * 60)
    print("Trying Water Quality Portal (WQP) for sample-based data...")
    print("=" * 60)

    wqp_characteristics = [
        "Phosphorus",
        "Nitrogen",
        "Dissolved oxygen (DO)",
        "Chlorophyll a",
        "Turbidity",
    ]

    for station_name, site_no in STATIONS.items():
        print(f"\n--- {station_name} (site {site_no}) ---")
        usgs_site = f"USGS-{site_no}"

        try:
            df, metadata = wqp.get_results(
                siteid=usgs_site,
                characteristicName=wqp_characteristics,
                startDateLo="01-01-2019",
                startDateHi="12-31-2025",
            )

            if df is not None and not df.empty:
                # Keep useful columns
                keep_cols = [
                    c for c in df.columns
                    if any(k in c.lower() for k in [
                        "date", "time", "character", "result", "measure",
                        "unit", "site", "activity", "sample"
                    ])
                ]
                if keep_cols:
                    df_slim = df[keep_cols]
                else:
                    df_slim = df

                outpath = os.path.join(OUTPUT_DIR, f"{station_name}_wqp.csv")
                df_slim.to_csv(outpath, index=False)
                print(f"  WQP: {len(df)} records -> {outpath}")

                # Show summary
                if "CharacteristicName" in df.columns:
                    print("  Breakdown:")
                    for char, count in df["CharacteristicName"].value_counts().items():
                        print(f"    {char}: {count}")
            else:
                print("  WQP: no data found")

        except Exception as e:
            print(f"  WQP error: {e}")

    print("\nDone! Check output in:", OUTPUT_DIR)


if __name__ == "__main__":
    download_usgs_qw()
