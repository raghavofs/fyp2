"""
Download all water quality data in the Mississippi River corridor (MN),
then match to nearest streamflow stations and export per-station CSVs.
"""

import os
import math
import pandas as pd
from dataretrieval import wqp

STATION_COORDS = {
    "brooklynpark": (45.094, -93.348),
    "stpaul":       (44.945, -93.090),
    "hastings":     (44.737, -92.852),
    "prescott":     (44.748, -92.802),
    "winona":       (44.055, -91.639),
}

CHARACTERISTICS = [
    "Phosphorus",
    "Nitrogen",
    "Dissolved oxygen (DO)",
    "Chlorophyll a",
    "Turbidity",
]

BBOX = "-93.5,43.9,-91.4,45.2"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "waterquality")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))

# Step 1: Get station metadata with coordinates
print("Step 1: Fetching station metadata...")
sites_df, _ = wqp.what_sites(bBox=BBOX)
site_coords = {}
for _, row in sites_df.iterrows():
    try:
        sid = row["MonitoringLocationIdentifier"]
        lat = float(row["LatitudeMeasure"])
        lon = float(row["LongitudeMeasure"])
        name = str(row["MonitoringLocationName"])
        site_coords[sid] = (lat, lon, name)
    except (ValueError, TypeError, KeyError):
        pass
print(f"  {len(site_coords)} stations with coordinates")

# Step 2: Download ALL water quality data in the bounding box
print("\nStep 2: Downloading all WQ data in bounding box...")
print(f"  Characteristics: {CHARACTERISTICS}")
df, _ = wqp.get_results(
    bBox=BBOX,
    characteristicName=CHARACTERISTICS,
    startDateLo="01-01-2019",
    startDateHi="12-31-2025",
)
print(f"  Total records: {len(df)}")

# Identify columns
id_col = [c for c in df.columns if "MonitoringLocationIdentifier" in c][0]
char_col = [c for c in df.columns if "CharacteristicName" in c][0]
result_col = [c for c in df.columns if c == "ResultMeasureValue" or "ResultMeasure/MeasureValue" in c]
result_col = result_col[0] if result_col else None
date_col = [c for c in df.columns if "ActivityStartDate" in c][0]

print(f"  Columns used: {id_col}, {char_col}, {date_col}, {result_col}")

# Step 3: Add coordinates to the data
print("\nStep 3: Matching stations to coordinates...")
df["_lat"] = df[id_col].map(lambda x: site_coords.get(x, (None, None, None))[0])
df["_lon"] = df[id_col].map(lambda x: site_coords.get(x, (None, None, None))[1])
df["_name"] = df[id_col].map(lambda x: site_coords.get(x, (None, None, None))[2])

has_coords = df.dropna(subset=["_lat", "_lon"])
print(f"  Records with coordinates: {len(has_coords)} / {len(df)}")

# Filter to Mississippi River stations
miss_mask = has_coords["_name"].str.contains("Mississippi|Miss R|MISS R|MISSISSIPPI", case=False, na=False)
miss_df = has_coords[miss_mask]
print(f"  Mississippi River records: {len(miss_df)}")

# If not enough Mississippi data, also include all nearby river data
if len(miss_df) < 100:
    print("  (Using all stations due to limited Mississippi-specific data)")
    miss_df = has_coords

# Step 4: For each streamflow site, find the best nearby WQ station(s)
print("\nStep 4: Finding best WQ stations for each streamflow site...")
print("=" * 90)

SEARCH_RADIUS_KM = 20  # Search within 20 km

for stn_name, (stn_lat, stn_lon) in STATION_COORDS.items():
    print(f"\n{'─' * 90}")
    print(f"  {stn_name.upper()}")
    print(f"{'─' * 90}")

    # Calculate distance for each record
    nearby = miss_df.copy()
    nearby["_dist_km"] = nearby.apply(
        lambda r: haversine_km(stn_lat, stn_lon, r["_lat"], r["_lon"]), axis=1
    )
    nearby = nearby[nearby["_dist_km"] <= SEARCH_RADIUS_KM]

    if nearby.empty:
        print(f"  No WQ data within {SEARCH_RADIUS_KM} km")
        continue

    # Show available stations ranked by (distance, record count)
    station_summary = nearby.groupby(id_col).agg(
        records=(char_col, "size"),
        dist_km=("_dist_km", "first"),
        name=("_name", "first"),
        params=(char_col, lambda x: list(x.unique())),
    ).sort_values(["dist_km"])

    print(f"  Stations within {SEARCH_RADIUS_KM} km:")
    for sid, row in station_summary.head(10).iterrows():
        print(f"    {row['dist_km']:5.1f} km | {row['records']:4d} records | {sid:40s} | {row['name'][:45]}")
        print(f"           params: {row['params']}")

    # Pick stations: prefer closest Mississippi River station with most parameters
    # Combine data from all stations within 5 km to maximize coverage
    close_data = nearby[nearby["_dist_km"] <= 5].copy()
    if close_data.empty:
        # Fall back to closest station
        closest_sid = station_summary.index[0]
        close_data = nearby[nearby[id_col] == closest_sid].copy()
        print(f"\n  Using closest station: {closest_sid} ({station_summary.loc[closest_sid, 'dist_km']:.1f} km)")
    else:
        print(f"\n  Combining data from {close_data[id_col].nunique()} stations within 5 km")

    # Select useful columns for output
    keep_cols = [date_col, char_col]
    if result_col:
        keep_cols.append(result_col)
    # Add unit columns
    unit_cols = [c for c in df.columns if "Unit" in c and "Result" in c]
    keep_cols.extend(unit_cols)
    # Add station info
    keep_cols.extend([id_col, "_name", "_dist_km"])
    # Add any other result-related columns
    for c in df.columns:
        if c not in keep_cols and any(k in c for k in ["ResultDetection", "ResultStatus", "ResultComment"]):
            keep_cols.append(c)

    keep_cols = [c for c in keep_cols if c in close_data.columns]
    output = close_data[keep_cols].sort_values(date_col)

    # Summary
    print(f"\n  Data summary:")
    for char_name, count in output[char_col].value_counts().items():
        print(f"    {char_name}: {count}")
    print(f"    Date range: {output[date_col].min()} to {output[date_col].max()}")
    print(f"    Total records: {len(output)}")

    # Save
    outpath = os.path.join(OUTPUT_DIR, f"{stn_name}_wq.csv")
    output.to_csv(outpath, index=False)
    print(f"\n  Saved to: {outpath}")

# Also save the full combined dataset
print("\n" + "=" * 90)
full_out = os.path.join(OUTPUT_DIR, "all_mississippi_wq.csv")
keep = [date_col, char_col, id_col, "_name", "_lat", "_lon"]
if result_col:
    keep.append(result_col)
keep.extend(unit_cols)
keep = [c for c in keep if c in miss_df.columns]
miss_df[keep].sort_values(date_col).to_csv(full_out, index=False)
print(f"Saved full Mississippi River dataset ({len(miss_df)} records) to: {full_out}")
print("\nDone!")
