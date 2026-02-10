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

# Step 1: Get station metadata in the bounding box
print("Step 1: Fetching station metadata...")
bbox = "-93.5,43.9,-91.4,45.2"

sites_df, _ = wqp.what_sites(bBox=bbox)
print(f"  Found {len(sites_df)} total sites in bounding box")
print(f"  Columns: {sites_df.columns.tolist()}")

# Find lat/lon columns
lat_col = [c for c in sites_df.columns if 'lat' in c.lower()][0]
lon_col = [c for c in sites_df.columns if 'lon' in c.lower()][0]
id_col = [c for c in sites_df.columns if 'identifier' in c.lower()][0]
name_col = [c for c in sites_df.columns if 'name' in c.lower()][0]

print(f"  Using columns: {id_col}, {name_col}, {lat_col}, {lon_col}")

# Step 2: Get results for Mississippi River search
print("\nStep 2: Fetching WQ data filtered to Mississippi River...")

# Search for USGS sites on Mississippi River
for search_term in ["Mississippi"]:
    try:
        river_sites, _ = wqp.what_sites(
            bBox=bbox,
            # Filter to sites with these characteristics available
        )
        break
    except Exception as e:
        print(f"  Error: {e}")

# Build a lookup of site coords
site_coords = {}
for _, row in sites_df.iterrows():
    try:
        site_coords[row[id_col]] = (float(row[lat_col]), float(row[lon_col]), str(row[name_col]))
    except (ValueError, TypeError):
        pass

# Step 3: Find closest WQ stations to each streamflow site
print(f"\nStep 3: Finding closest stations to each streamflow site...")
print(f"  Total sites with coordinates: {len(site_coords)}")

# Filter to sites that mention Mississippi or are USGS sites
mississippi_sites = {
    k: v for k, v in site_coords.items()
    if "mississippi" in v[2].lower() or "miss" in v[2].lower() or k.startswith("USGS-")
}
print(f"  Mississippi River / USGS sites: {len(mississippi_sites)}")

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

# For each streamflow site, find the closest WQ stations
print("\n" + "=" * 90)
print("CLOSEST WATER QUALITY STATIONS TO EACH STREAMFLOW SITE")
print("=" * 90)

best_stations = {}

for stn_name, (stn_lat, stn_lon) in STATION_COORDS.items():
    print(f"\n{'─' * 90}")
    print(f"  {stn_name.upper()} (streamflow site at {stn_lat}, {stn_lon})")
    print(f"{'─' * 90}")

    # Calculate distances - first check Mississippi sites, then all
    distances = []
    for site_id, (slat, slon, sname) in site_coords.items():
        d = haversine_km(stn_lat, stn_lon, slat, slon)
        if d < 15:  # Within 15 km
            distances.append((d, site_id, sname, slat, slon))

    distances.sort()

    # Show top 10 nearest
    for d, site_id, sname, slat, slon in distances[:10]:
        is_miss = "mississippi" in sname.lower() or "miss" in sname.lower()
        marker = " ** MISSISSIPPI **" if is_miss else ""
        print(f"  {d:5.1f} km | {site_id:40s} | {sname[:50]}{marker}")

    # Pick the best Mississippi River station if available
    miss_nearby = [(d, sid, sn) for d, sid, sn, _, _ in distances if "mississippi" in sn.lower() or "miss" in sn.lower()]
    if miss_nearby:
        best_stations[stn_name] = miss_nearby[0]
        print(f"\n  >>> Best match: {miss_nearby[0][1]} ({miss_nearby[0][2][:50]}) at {miss_nearby[0][0]:.1f} km")
    elif distances:
        best_stations[stn_name] = (distances[0][0], distances[0][1], distances[0][2])
        print(f"\n  >>> Closest: {distances[0][1]} ({distances[0][2][:50]}) at {distances[0][0]:.1f} km")

# Step 4: Check what data is available at the best stations
print("\n" + "=" * 90)
print("CHECKING DATA AVAILABILITY AT BEST STATIONS")
print("=" * 90)

for stn_name, (dist, site_id, site_name) in best_stations.items():
    print(f"\n--- {stn_name} -> {site_id} ({site_name[:50]}, {dist:.1f} km away) ---")

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
            print("  No data for these parameters")
    except Exception as e:
        print(f"  Error: {e}")

print("\n\nRECOMMENDED STATION MAPPING:")
print("=" * 90)
for stn_name, (dist, site_id, site_name) in best_stations.items():
    print(f"  {stn_name:15s} -> {site_id} ({site_name[:50]}) [{dist:.1f} km]")
