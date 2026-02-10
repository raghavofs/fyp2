"""
Find water quality stations on the Mississippi River near our streamflow sites.
"""

from dataretrieval import wqp
import pandas as pd
import math

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

bbox = "-93.5,43.9,-91.4,45.2"

print("Fetching station metadata...")
sites_df, _ = wqp.what_sites(bBox=bbox)
print(f"Found {len(sites_df)} total sites")

# Use explicit column names
id_col = "MonitoringLocationIdentifier"
name_col = "MonitoringLocationName"
lat_col = "LatitudeMeasure"
lon_col = "LongitudeMeasure"
type_col = "MonitoringLocationTypeName"

# Filter to river/stream sites
river_types = sites_df[type_col].unique()
print(f"\nSite types: {river_types}")

river_df = sites_df[sites_df[type_col].isin(["River/Stream", "Stream", "River/Stream Perennial"])]
print(f"River/Stream sites: {len(river_df)}")

# Further filter to sites mentioning Mississippi
miss_df = river_df[river_df[name_col].str.contains("Mississippi|Miss R|MISSISSIPPI", case=False, na=False)]
print(f"Mississippi River sites: {len(miss_df)}")

if len(miss_df) > 0:
    print("\nAll Mississippi River monitoring sites:")
    for _, row in miss_df.iterrows():
        print(f"  {row[id_col]:45s} | {row[name_col][:60]:60s} | ({row[lat_col]}, {row[lon_col]})")

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

# For each streamflow site, find closest Mississippi River WQ station
print("\n" + "=" * 90)
print("CLOSEST MISSISSIPPI RIVER WQ STATIONS TO EACH STREAMFLOW SITE")
print("=" * 90)

best_stations = {}

for stn_name, (stn_lat, stn_lon) in STATION_COORDS.items():
    print(f"\n--- {stn_name.upper()} ({stn_lat}, {stn_lon}) ---")

    distances = []
    for _, row in miss_df.iterrows():
        try:
            slat = float(row[lat_col])
            slon = float(row[lon_col])
            d = haversine_km(stn_lat, stn_lon, slat, slon)
            distances.append((d, row[id_col], row[name_col], slat, slon))
        except (ValueError, TypeError):
            pass

    distances.sort()

    for d, site_id, sname, slat, slon in distances[:5]:
        print(f"  {d:5.1f} km | {site_id:45s} | {sname[:55]}")

    if distances:
        best_stations[stn_name] = distances[0]

# Also check all river/stream sites (not just Mississippi) within 5km
print("\n" + "=" * 90)
print("ALL RIVER/STREAM WQ STATIONS WITHIN 5 KM OF EACH STREAMFLOW SITE")
print("=" * 90)

for stn_name, (stn_lat, stn_lon) in STATION_COORDS.items():
    print(f"\n--- {stn_name.upper()} ({stn_lat}, {stn_lon}) ---")

    distances = []
    for _, row in river_df.iterrows():
        try:
            slat = float(row[lat_col])
            slon = float(row[lon_col])
            d = haversine_km(stn_lat, stn_lon, slat, slon)
            if d < 5:
                distances.append((d, row[id_col], row[name_col], slat, slon))
        except (ValueError, TypeError):
            pass

    distances.sort()
    if not distances:
        print("  (none within 5 km)")

    for d, site_id, sname, slat, slon in distances[:10]:
        print(f"  {d:5.1f} km | {site_id:45s} | {sname[:55]}")

# Check data at best stations
print("\n" + "=" * 90)
print("DATA AVAILABILITY AT CLOSEST MISSISSIPPI RIVER STATIONS")
print("=" * 90)

for stn_name, (dist, site_id, site_name, slat, slon) in best_stations.items():
    print(f"\n--- {stn_name} -> {site_id} ({dist:.1f} km) ---")
    print(f"    {site_name}")

    try:
        df, _ = wqp.get_results(
            siteid=site_id,
            characteristicName=CHARACTERISTICS,
            startDateLo="01-01-2019",
            startDateHi="12-31-2025",
        )
        if df is not None and not df.empty:
            char_col = [c for c in df.columns if "CharacteristicName" in c][0]
            print(f"  Total records: {len(df)}")
            for char_name, count in df[char_col].value_counts().items():
                print(f"    {char_name}: {count}")
        else:
            print("  No data for these parameters in 2019-2025")
    except Exception as e:
        print(f"  Error: {e}")
