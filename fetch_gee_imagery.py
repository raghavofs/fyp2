"""
fetch_gee_imagery.py — Fetch Sentinel-2 + Landsat 8 satellite imagery for
Danube monitoring stations using the Google Earth Engine Python API.

Output matches the format of Danube_Satellite_Data_2012_2023 (1).csv so the
existing preprocess_danube.py pipeline can consume it directly.

Authentication:
  First run:  python fetch_gee_imagery.py --auth
  Subsequent: runs automatically via stored credentials (~/.config/earthengine/)

Usage:
  python fetch_gee_imagery.py                        # full 2012-12-01 → today
  python fetch_gee_imagery.py --start 2023-01-01     # recent update
  python fetch_gee_imagery.py --smoke                # 1 station, last 30 days, print only
  python fetch_gee_imagery.py --auth                 # re-authenticate

Dependencies:
  pip install earthengine-api
  gcloud auth (or ee.Authenticate() on first run)
"""

import os
import sys
import argparse
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
GEE_OUT    = os.path.join(BASE_DIR, "data", "danube_processed")
DEFAULT_OUT = os.path.join(GEE_OUT, "Danube_Satellite_Data_GEE_Live.csv")

# ── Station definitions ───────────────────────────────────────────────────────
STATIONS = [
    {"id": "SRB00001", "name": "Bezdan",          "lat": 45.8542, "lon": 18.8586},
    {"id": "SRB00040", "name": "Bogojevo",         "lat": 45.5291, "lon": 19.0780},
    {"id": "SRB00002", "name": "Novi Sad",         "lat": 45.2244, "lon": 19.8419},
    {"id": "SRB00003", "name": "Zemun (Belgrade)", "lat": 44.8489, "lon": 20.4172},
    {"id": "SRB00041", "name": "Smederevo",        "lat": 44.6960, "lon": 20.9592},
    {"id": "SRB00005", "name": "Banatska Palanka", "lat": 44.8244, "lon": 21.3450},
    {"id": "SRB00006", "name": "Tekija",           "lat": 44.6961, "lon": 22.4113},
]

BUFFER_M = 500   # 500-metre buffer around each station point

# ── Band definitions ──────────────────────────────────────────────────────────
S2_BANDS  = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12"]
S2_NAMES  = ["Blue", "Green", "Red", "RedEdge1", "RedEdge2", "RedEdge3",
             "NIR", "SWIR1", "SWIR2"]

L8_BANDS  = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
L8_NAMES  = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
# L8 has no Red Edge; those columns will be NaN in the output


# ── GEE helpers ───────────────────────────────────────────────────────────────

def init_ee():
    """Initialise the Earth Engine API, guiding the user through auth if needed."""
    try:
        import ee
    except ImportError:
        print("ERROR: earthengine-api not installed.")
        print("  pip install earthengine-api")
        sys.exit(1)

    try:
        ee.Initialize(opt_url="https://earthengine.googleapis.com")
        return ee
    except Exception:
        print("Earth Engine credentials not found or expired.")
        print("Run:  python fetch_gee_imagery.py --auth")
        print("Or:   earthengine authenticate")
        sys.exit(1)


def mask_s2_clouds(image):
    """Apply SCL-based cloud mask to Sentinel-2 SR image."""
    import ee
    scl = image.select("SCL")
    # SCL classes to mask: 3=cloud shadow, 7=unclassified, 8=cloud medium, 9=cloud high, 10=cirrus
    cloud_mask = scl.neq(3).And(scl.neq(7)).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return image.updateMask(cloud_mask).divide(10000)   # scale reflectance to 0–1


def mask_l8_clouds(image):
    """Apply QA_PIXEL bitmask cloud mask to Landsat 8 C2 SR image."""
    import ee
    qa = image.select("QA_PIXEL")
    cloud_bit  = 1 << 3   # bit 3: cloud
    shadow_bit = 1 << 4   # bit 4: cloud shadow
    dilated    = 1 << 1   # bit 1: dilated cloud
    clear_mask = (qa.bitwiseAnd(cloud_bit).eq(0)
                   .And(qa.bitwiseAnd(shadow_bit).eq(0))
                   .And(qa.bitwiseAnd(dilated).eq(0)))
    return image.updateMask(clear_mask).multiply(0.0000275).add(-0.2)   # scale to SR


def add_indices(image, satellite: str):
    """Compute and add NDVI, NDWI, MNDWI, NDTI, EVI, FAI to the image."""
    import ee
    if satellite == "S2":
        blue   = image.select("Blue")
        green  = image.select("Green")
        red    = image.select("Red")
        nir    = image.select("NIR")
        swir1  = image.select("SWIR1")
        swir2  = image.select("SWIR2")
    else:   # L8
        blue   = image.select("Blue")
        green  = image.select("Green")
        red    = image.select("Red")
        nir    = image.select("NIR")
        swir1  = image.select("SWIR1")
        swir2  = image.select("SWIR2")

    ndvi  = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    ndwi  = green.subtract(nir).divide(green.add(nir)).rename("NDWI")
    mndwi = green.subtract(swir1).divide(green.add(swir1)).rename("MNDWI")
    ndti  = red.subtract(green).divide(red.add(green)).rename("NDTI")
    evi   = (nir.subtract(red)
               .multiply(2.5)
               .divide(nir.add(red.multiply(6))
                          .subtract(blue.multiply(7.5))
                          .add(1))
               .rename("EVI"))
    # FAI = NIR − Red − (SWIR1 − Red) × 0.4
    fai   = (nir.subtract(red)
               .subtract(swir1.subtract(red).multiply(0.4))
               .rename("FAI"))

    return image.addBands([ndvi, ndwi, mndwi, ndti, evi, fai])


def extract_station_timeseries(
    ee,
    station: dict,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch S2 + L8 time series for one station and return a DataFrame."""
    point   = ee.Geometry.Point([station["lon"], station["lat"]])
    aoi     = point.buffer(BUFFER_M)

    all_rows = []

    # ── Sentinel-2 ────────────────────────────────────────────────────────────
    s2_col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(aoi)
          .filterDate(start_date, end_date)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
          .map(mask_s2_clouds)
          .select(S2_BANDS, S2_NAMES)
          .map(lambda img: add_indices(img, "S2"))
    )

    index_bands  = ["NDVI", "NDWI", "MNDWI", "NDTI", "EVI", "FAI"]
    all_s2_bands = S2_NAMES + index_bands

    def extract_s2(image):
        date_str  = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd")
        n_pixels  = image.select("Blue").unmask(0).gt(0).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi, scale=10).get("Blue")
        means = image.select(all_s2_bands).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi, scale=10)
        return ee.Feature(None, means
                          .set("image_date", date_str)
                          .set("satellite", "S2")
                          .set("valid_pixels", n_pixels))

    s2_fc = s2_col.map(extract_s2)
    try:
        s2_info = s2_fc.getInfo()
        for feat in s2_info["features"]:
            p = feat["properties"]
            row = {"image_date": p.get("image_date"),
                   "satellite":  "S2",
                   "valid_pixels": p.get("valid_pixels", 0)}
            for b in all_s2_bands:
                row[b] = p.get(b)
            all_rows.append(row)
    except Exception as e:
        print(f"    [WARN] S2 extraction failed for {station['id']}: {e}")

    # ── Landsat 8 ─────────────────────────────────────────────────────────────
    l8_col = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
          .filterBounds(aoi)
          .filterDate(start_date, end_date)
          .filter(ee.Filter.lt("CLOUD_COVER", 60))
          .map(mask_l8_clouds)
          .select(L8_BANDS, L8_NAMES)
          .map(lambda img: add_indices(img, "L8"))
    )

    l8_bands = L8_NAMES + index_bands

    def extract_l8(image):
        date_str = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd")
        n_pixels = image.select("Blue").unmask(0).gt(0).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi, scale=30).get("Blue")
        means = image.select(l8_bands).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi, scale=30)
        return ee.Feature(None, means
                          .set("image_date", date_str)
                          .set("satellite", "L8")
                          .set("valid_pixels", n_pixels))

    l8_fc = l8_col.map(extract_l8)
    try:
        l8_info = l8_fc.getInfo()
        for feat in l8_info["features"]:
            p = feat["properties"]
            row = {"image_date": p.get("image_date"),
                   "satellite":  "L8",
                   "valid_pixels": p.get("valid_pixels", 0)}
            for b in l8_bands:
                row[b] = p.get(b)
            # L8 has no Red Edge — leave as NaN
            for re in ["RedEdge1", "RedEdge2", "RedEdge3"]:
                row[re] = None
            all_rows.append(row)
    except Exception as e:
        print(f"    [WARN] L8 extraction failed for {station['id']}: {e}")

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df.insert(0, "station_id",   station["id"])
    df.insert(1, "station_name", station["name"])

    # Sort by date, prefer S2 over L8 on same day
    df["image_date"] = pd.to_datetime(df["image_date"])
    df = df.sort_values(["image_date", "satellite"],
                        ascending=[True, True]).reset_index(drop=True)

    return df


def merge_with_existing(new_df: pd.DataFrame, existing_path: str) -> pd.DataFrame:
    """Append new observations to existing CSV, deduplicating on (station_id, image_date, satellite)."""
    if not os.path.exists(existing_path):
        return new_df

    existing = pd.read_csv(existing_path)
    # Normalise date column name
    date_col = "image_date" if "image_date" in existing.columns else "date"
    existing = existing.rename(columns={date_col: "image_date"})
    existing["image_date"] = pd.to_datetime(existing["image_date"], errors="coerce")

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["station_id", "image_date", "satellite"], keep="last"
    ).sort_values(["station_id", "image_date"]).reset_index(drop=True)

    n_new = len(combined) - len(existing)
    print(f"  Merged with existing ({len(existing)} rows) → {len(combined)} total ({n_new} new)")
    return combined


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch Danube satellite imagery from Google Earth Engine"
    )
    parser.add_argument(
        "--start", default="2012-12-01",
        help="Start date YYYY-MM-DD (default: 2012-12-01)",
    )
    parser.add_argument(
        "--end", default=None,
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--out", default=DEFAULT_OUT,
        help=f"Output CSV path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--station", nargs="+", default=None,
        help="Restrict to specific station IDs (e.g. SRB00001 SRB00002)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Quick test: 1 station (Bezdan), last 30 days, print only — no file write",
    )
    parser.add_argument(
        "--auth", action="store_true",
        help="Force GEE re-authentication and exit",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge new observations with existing CSV instead of overwriting",
    )
    args = parser.parse_args()

    # ── Handle auth request ───────────────────────────────────────────────────
    if args.auth:
        try:
            import ee
        except ImportError:
            print("pip install earthengine-api")
            sys.exit(1)
        ee.Authenticate()
        print("Authentication complete. Run without --auth to fetch data.")
        return

    ee = init_ee()

    end_date   = args.end or datetime.today().strftime("%Y-%m-%d")
    start_date = args.start

    if args.smoke:
        smoke_start = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")
        start_date  = smoke_start
        stations_to_run = [STATIONS[0]]   # Bezdan only
        print(f"[SMOKE TEST] Station: {STATIONS[0]['id']}, "
              f"date range: {start_date} → {end_date}")
    else:
        stations_to_run = (
            [s for s in STATIONS if s["id"] in args.station]
            if args.station else STATIONS
        )

    print(f"\nGEE Satellite Fetch — Danube Serbia")
    print(f"  Date range : {start_date} → {end_date}")
    print(f"  Stations   : {[s['id'] for s in stations_to_run]}")
    print(f"  Buffer     : {BUFFER_M} m")
    print()

    all_frames = []
    for station in stations_to_run:
        print(f"  [{station['id']}] {station['name']} ...")
        df = extract_station_timeseries(ee, station, start_date, end_date)
        if df.empty:
            print(f"    No data returned (check date range / cloud cover filter)")
            continue
        all_frames.append(df)
        # Report band stats
        band_cols = [c for c in df.columns if c not in
                     ("station_id", "station_name", "image_date", "satellite", "valid_pixels")]
        n_s2 = (df["satellite"] == "S2").sum()
        n_l8 = (df["satellite"] == "L8").sum()
        print(f"    {len(df)} images (S2={n_s2}, L8={n_l8}) | "
              f"bands: {len(band_cols)} | "
              f"valid_pixels mean: {df['valid_pixels'].mean():.0f}")

    if not all_frames:
        print("\n  No data retrieved. Check GEE authentication and date range.")
        return

    result = pd.concat(all_frames, ignore_index=True)

    if args.smoke:
        print("\n[SMOKE] Sample output:")
        print(result[["station_id", "image_date", "satellite", "Blue", "Green",
                       "Red", "NIR", "NDVI", "NDWI"]].head(10).to_string(index=False))
        print(f"\n[SMOKE] Total: {len(result)} rows — file not written.")
        return

    if args.merge:
        result = merge_with_existing(result, args.out)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"\n  Saved {len(result)} rows → {args.out}")
    print("\n  To reprocess with new imagery:")
    print(f"    python preprocess_danube.py  (reads {args.out})")


if __name__ == "__main__":
    main()
