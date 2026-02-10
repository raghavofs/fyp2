"""
Find water quality monitoring stations near our 5 streamflow sites
along the Mississippi River in Minnesota.
"""

from dataretrieval import wqp
import pandas as pd

# Approximate lat/lon for each streamflow station
STATION_COORDS = {
    "brooklynpark": (45.094, -93.348),
    "stpaul":       (44.945, -93.090),
    "hastings":     (44.737, -92.852),
    "prescott":     (44.748, -92.802),
    "winona":       (44.055, -91.639),
}

# Bounding box covering the full river stretch (with padding)
# South to North: ~44.0 to 45.2, West to East: ~-93.5 to -91.5
BBOX = (-93.5, 43.9, -91.4, 45.2)

CHARACTERISTICS = [
    "Phosphorus",
    "Nitrogen",
    "Dissolved oxygen (DO)",
    "Chlorophyll a",
    "Turbidity",
]

print("Searching Water Quality Portal for stations in bounding box...")
print(f"  bbox: {BBOX}")
print(f"  characteristics: {CHARACTERISTICS}")
print(f"  date range: 2019-2025")
print()

try:
    df, meta = wqp.get_results(
        bBox=",".join(str(x) for x in BBOX),
        characteristicName=CHARACTERISTICS,
        startDateLo="01-01-2019",
        startDateHi="12-31-2025",
    )
except Exception as e:
    print(f"Error: {e}")
    exit(1)

if df is None or df.empty:
    print("No data found.")
    exit(0)

print(f"Total records found: {len(df)}")
print()

# Identify key columns
site_col = [c for c in df.columns if "MonitoringLocationIdentifier" in c]
name_col = [c for c in df.columns if "MonitoringLocationName" in c]
char_col = [c for c in df.columns if "CharacteristicName" in c]
lat_col = [c for c in df.columns if "Latitude" in c]
lon_col = [c for c in df.columns if "Longitude" in c]

if not site_col:
    print("Columns available:", df.columns.tolist())
    exit(1)

site_col = site_col[0]
char_col = char_col[0] if char_col else None
name_col = name_col[0] if name_col else None
lat_col = lat_col[0] if lat_col else None
lon_col = lon_col[0] if lon_col else None

# Summary by station
print("=" * 80)
print("STATIONS WITH WATER QUALITY DATA (sorted by record count)")
print("=" * 80)

station_counts = df.groupby(site_col).size().sort_values(ascending=False)

for site_id, count in station_counts.head(30).items():
    site_data = df[df[site_col] == site_id]

    name = site_data[name_col].iloc[0] if name_col else "N/A"
    lat = site_data[lat_col].iloc[0] if lat_col else "N/A"
    lon = site_data[lon_col].iloc[0] if lon_col else "N/A"

    chars = site_data[char_col].value_counts().to_dict() if char_col else {}

    print(f"\n{site_id} — {name}")
    print(f"  Location: ({lat}, {lon})  |  Total records: {count}")
    print(f"  Parameters: {chars}")

# Also show which stations are closest to our streamflow sites
print()
print("=" * 80)
print("CLOSEST WQ STATIONS TO EACH STREAMFLOW SITE")
print("=" * 80)

if lat_col and lon_col:
    unique_sites = df.drop_duplicates(subset=[site_col])

    for stn_name, (stn_lat, stn_lon) in STATION_COORDS.items():
        print(f"\n--- {stn_name} ({stn_lat}, {stn_lon}) ---")

        # Calculate rough distance
        unique_sites = unique_sites.copy()
        unique_sites["_dist"] = (
            (unique_sites[lat_col].astype(float) - stn_lat) ** 2 +
            (unique_sites[lon_col].astype(float) - stn_lon) ** 2
        ) ** 0.5

        nearest = unique_sites.nsmallest(5, "_dist")

        for _, row in nearest.iterrows():
            sid = row[site_col]
            rec_count = station_counts.get(sid, 0)
            site_chars = df[df[site_col] == sid][char_col].value_counts().to_dict() if char_col else {}
            dist_deg = row["_dist"]
            dist_km = dist_deg * 111  # rough conversion
            print(f"  {sid} — {row.get(name_col, 'N/A')}")
            print(f"    ~{dist_km:.1f} km away | {rec_count} records | {site_chars}")
