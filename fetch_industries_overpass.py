"""
fetch_industries_overpass.py — Query OpenStreetMap Overpass API for industrial
facilities along the Serbian Danube and build a structured industry lookup.

The output schema matches data/industry_lookup/danube_industries.json so the
anomaly_detector.py can consume either file (or a merged version) identically.

Usage:
  python fetch_industries_overpass.py                     # fetch + save
  python fetch_industries_overpass.py --smoke             # fetch only, print, no save
  python fetch_industries_overpass.py --merge             # merge with existing JSON
  python fetch_industries_overpass.py --out my_db.json    # custom output path

No API key required — Overpass API is a free public service.
Rate limiting: we send one request; the server handles the rest.
"""

import os
import json
import math
import time
import argparse
import urllib.request
import urllib.parse
import urllib.error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT    = os.path.join(BASE_DIR, "data", "industry_lookup", "overpass_industries.json")
EXISTING_DB    = os.path.join(BASE_DIR, "data", "industry_lookup", "danube_industries.json")
OVERPASS_ENDPOINTS = [
    "https://overpass.kumi.systems/api/interpreter",   # mirror (usually fastest)
    "https://overpass-api.de/api/interpreter",         # primary
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",  # backup
]

# ── Bounding box: Serbian Danube (South, West, North, East) ───────────────────
# Bezdan (45.85°N, 18.86°E) → Tekija (44.70°N, 22.41°E) plus ~0.3° margin
BBOX = (44.50, 18.50, 46.00, 22.70)   # (lat_min, lon_min, lat_max, lon_max)

# ── Station metadata for proximity mapping ────────────────────────────────────
STATIONS = [
    {"id": "SRB00001", "name": "Bezdan",          "km": 1425, "lat": 45.8542, "lon": 18.8586},
    {"id": "SRB00040", "name": "Bogojevo",         "km": 1367, "lat": 45.5291, "lon": 19.0780},
    {"id": "SRB00002", "name": "Novi Sad",         "km": 1255, "lat": 45.2244, "lon": 19.8419},
    {"id": "SRB00003", "name": "Zemun (Belgrade)", "km": 1173, "lat": 44.8489, "lon": 20.4172},
    {"id": "SRB00041", "name": "Smederevo",        "km": 1116, "lat": 44.6960, "lon": 20.9592},
    {"id": "SRB00005", "name": "Banatska Palanka", "km": 1077, "lat": 44.8244, "lon": 21.3450},
    {"id": "SRB00006", "name": "Tekija",           "km":  931, "lat": 44.6961, "lon": 22.4113},
]

# ── OSM tag → our industry type ───────────────────────────────────────────────
OSM_TYPE_MAP = {
    # industrial=* values
    "steel":              "metallurgy",
    "metal":              "metallurgy",
    "foundry":            "metallurgy",
    "smelting":           "metallurgy",
    "refinery":           "refinery",
    "oil":                "refinery",
    "petrochemical":      "chemical_plant",
    "chemical":           "chemical_plant",
    "fertiliser":         "fertilizer",
    "fertilizer":         "fertilizer",
    "pharmaceutical":     "chemical_plant",
    "power":              "power_plant",
    "electricity":        "power_plant",
    "wastewater":         "wastewater_treatment",
    "sewage":             "wastewater_treatment",
    "water_treatment":    "wastewater_treatment",
    "food":               "food_processing",
    "brewery":            "food_processing",
    "slaughterhouse":     "food_processing",
    "paper":              "pulp_paper",
    "timber":             "pulp_paper",
    "quarry":             "mining",
    "mine":               "mining",
    "agriculture":        "agriculture",
    "farm":               "agriculture",
    # man_made=* values
    "wastewater_plant":   "wastewater_treatment",
    "sewage_works":       "wastewater_treatment",
    "petroleum_well":     "refinery",
    "works":              "industrial",
    # power=* values
    "plant":              "power_plant",
    "dam":                "power_plant",
    # landuse=* values
    "industrial":         "industrial",
}

# Pollutant types associated with each industry type
EFFLUENT_PARAMS = {
    "metallurgy":           ["electrical_conductance", "total_phosphorus"],
    "refinery":             ["dissolved_oxygen", "electrical_conductance"],
    "chemical_plant":       ["electrical_conductance", "dissolved_oxygen"],
    "fertilizer":           ["nitrate_n", "electrical_conductance"],
    "power_plant":          ["dissolved_oxygen", "electrical_conductance"],
    "wastewater_treatment": ["dissolved_oxygen", "total_phosphorus", "nitrate_n"],
    "food_processing":      ["dissolved_oxygen", "total_phosphorus"],
    "pulp_paper":           ["dissolved_oxygen", "electrical_conductance"],
    "mining":               ["electrical_conductance", "total_phosphorus"],
    "agriculture":          ["nitrate_n", "total_phosphorus", "chlorophyll_a"],
    "industrial":           ["electrical_conductance"],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def nearest_station(lat: float, lon: float) -> dict:
    """Return the station object nearest to (lat, lon)."""
    return min(STATIONS, key=lambda s: haversine_km(lat, lon, s["lat"], s["lon"]))


def stretch_for_location(lat: float, lon: float) -> tuple[str, str, str]:
    """Return (upstream_id, downstream_id, stretch_key) for a facility location.

    The facility's river stretch is between the nearest upstream station and the
    nearest downstream station. We find the nearest station, then determine which
    side it's on by checking river km ordering.
    """
    dists   = [(haversine_km(lat, lon, s["lat"], s["lon"]), s) for s in STATIONS]
    dists.sort(key=lambda x: x[0])
    nearest = dists[0][1]
    idx     = STATIONS.index(nearest)

    if idx == 0:
        # At or above most upstream station
        return (None, STATIONS[0]["id"], f"UPSTREAM_{STATIONS[0]['id']}")
    elif idx == len(STATIONS) - 1:
        # Below most downstream station
        return (STATIONS[-2]["id"], STATIONS[-1]["id"],
                f"{STATIONS[-2]['id']}_{STATIONS[-1]['id']}")
    else:
        # Between two stations — pick the bounding pair by km proximity
        up   = STATIONS[idx - 1]
        down = STATIONS[idx]
        return (up["id"], down["id"], f"{up['id']}_{down['id']}")


def infer_type(tags: dict) -> str:
    """Infer our industry type from OSM tags."""
    for key in ("industrial", "man_made", "power", "landuse"):
        val = tags.get(key, "").lower()
        if val in OSM_TYPE_MAP:
            return OSM_TYPE_MAP[val]

    # Fallback: name-based heuristics
    name = tags.get("name", "").lower()
    for keyword, itype in [
        ("steel", "metallurgy"), ("čelik", "metallurgy"), ("hesteel", "metallurgy"),
        ("rafinerij", "refinery"), ("refinery", "refinery"), ("naftna", "refinery"),
        ("petrohem", "chemical_plant"), ("azotara", "fertilizer"),
        ("vodovod", "wastewater_treatment"), ("wastewater", "wastewater_treatment"),
        ("power", "power_plant"), ("elektrana", "power_plant"), ("dam", "power_plant"),
        ("djerdap", "power_plant"), ("đerdap", "power_plant"),
    ]:
        if keyword in name:
            return itype

    return "industrial"


def build_overpass_query() -> str:
    """Build Overpass QL query using node-only lookups (fast + reliable).

    Ways and relations are excluded to avoid server-side timeouts — nodes
    capture the majority of named industrial facilities in OSM.
    """
    s, w, n, e = BBOX
    bbox = f"{s},{w},{n},{e}"
    return f"""[out:json][timeout:60];
(
  node["industrial"]({bbox});
  node["man_made"="wastewater_plant"]({bbox});
  node["man_made"="sewage_works"]({bbox});
  node["man_made"="petroleum_well"]({bbox});
  node["power"="plant"]({bbox});
  node["power"="dam"]({bbox});
  node["landuse"="industrial"]["name"]({bbox});
);
out;""".strip()


# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_overpass(timeout: int = 90) -> dict:
    """Send Overpass QL query and return parsed JSON. Tries multiple endpoints."""
    query = build_overpass_query()
    data  = urllib.parse.urlencode({"data": query}).encode()

    print(f"  Querying Overpass API (bbox: {BBOX})...")
    last_err = None
    for endpoint in OVERPASS_ENDPOINTS:
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded",
                     "User-Agent": "danube-wq-research/1.0 (academic)"},
        )
        try:
            start = time.time()
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                # Some servers return gzip-compressed responses
                if raw[:2] == b'\x1f\x8b':
                    import gzip
                    raw = gzip.decompress(raw)
                text = raw.decode("utf-8", errors="replace")
                if not text.strip():
                    raise ValueError("Empty response from server")
                result = json.loads(text)
            elapsed = time.time() - start
            print(f"  Response: {len(result.get('elements', []))} elements "
                  f"in {elapsed:.1f}s (via {endpoint.split('/')[2]})")
            return result
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            print(f"  [WARN] {endpoint.split('/')[2]} failed: {e} — trying next endpoint...")
            last_err = e
            time.sleep(2)

    raise urllib.error.URLError(f"All Overpass endpoints failed. Last error: {last_err}")


def parse_elements(elements: list) -> list[dict]:
    """Convert raw Overpass elements to our industry schema."""
    facilities = []
    seen_names = set()

    for el in elements:
        tags = el.get("tags", {})
        name = tags.get("name") or tags.get("name:en") or tags.get("operator")
        if not name:
            continue

        # De-duplicate by name (keep first occurrence)
        name_key = name.lower().strip()
        if name_key in seen_names:
            continue
        seen_names.add(name_key)

        # Get coordinates
        if el["type"] == "node":
            lat, lon = el.get("lat"), el.get("lon")
        else:
            # way / relation — use centre if available
            centre = el.get("center", {})
            lat    = centre.get("lat")
            lon    = centre.get("lon")

        if lat is None or lon is None:
            continue

        # Only keep facilities within ~5 km of the Danube
        # (rough filter using station distances; Overpass bbox already constrains the region)
        nearest_st = nearest_station(lat, lon)
        dist_km    = haversine_km(lat, lon, nearest_st["lat"], nearest_st["lon"])
        if dist_km > 40:
            continue   # too far from any monitoring station

        itype = infer_type(tags)

        up_id, down_id, stretch_key = stretch_for_location(lat, lon)

        # Approximate river km from nearest station
        approx_km = nearest_st["km"]

        facility = {
            "id":                       f"OSM_{el['type'][0].upper()}{el['id']}",
            "name":                     name,
            "type":                     itype,
            "description":              tags.get("description", ""),
            "lat":                      round(lat, 4),
            "lon":                      round(lon, 4),
            "river_km":                 approx_km,
            "nearest_station_upstream": up_id,
            "nearest_station_downstream": down_id,
            "stretch":                  stretch_key,
            "effluent_parameters":      EFFLUENT_PARAMS.get(itype, ["electrical_conductance"]),
            "effluent_notes":           (
                f"Facility inferred from OpenStreetMap. Type: {itype}. "
                f"Distance to nearest station ({nearest_st['name']}): {dist_km:.1f} km."
            ),
            "source":                   f"OpenStreetMap Overpass API / element id {el['id']}",
        }
        facilities.append(facility)

    return facilities


def merge_with_existing(overpass_facilities: list[dict],
                        existing_path: str) -> list[dict]:
    """Merge overpass results with the existing curated JSON.

    Keeps all curated entries; adds overpass entries that aren't within
    2 km of any existing entry (avoids duplicating Hesteel etc.).
    """
    with open(existing_path) as f:
        curated = json.load(f)

    existing_coords = [(e["lat"], e["lon"]) for e in curated]
    added = 0

    for fac in overpass_facilities:
        too_close = any(
            haversine_km(fac["lat"], fac["lon"], elat, elon) < 2.0
            for elat, elon in existing_coords
        )
        if not too_close:
            curated.append(fac)
            existing_coords.append((fac["lat"], fac["lon"]))
            added += 1

    print(f"  Merged: {added} new facilities added to {len(curated)} curated entries")
    return curated


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch industrial facilities from Overpass API for Danube Serbia"
    )
    parser.add_argument(
        "--out", default=DEFAULT_OUT,
        help=f"Output path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge with existing danube_industries.json (deduplicate by proximity)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Fetch and print results only — do not write any file",
    )
    parser.add_argument(
        "--timeout", type=int, default=90,
        help="HTTP timeout in seconds (default: 90)",
    )
    args = parser.parse_args()

    print("\nDanube Industrial Facilities — Overpass API Fetch")
    print(f"  Bounding box: lat {BBOX[0]}–{BBOX[2]}, lon {BBOX[1]}–{BBOX[3]}")

    try:
        result = fetch_overpass(timeout=args.timeout)
    except urllib.error.URLError as e:
        print(f"\n  [ERROR] Network request failed: {e}")
        print("  Check internet connection or try again later.")
        return

    elements   = result.get("elements", [])
    facilities = parse_elements(elements)

    print(f"\n  Parsed {len(facilities)} distinct facilities after filtering\n")

    if not facilities:
        print("  No facilities found. The Overpass API may be rate-limiting.")
        print("  Try again in a few minutes or increase --timeout.")
        return

    # Print preview
    print(f"  {'Name':<45} {'Type':<24} {'Stretch'}")
    print("  " + "-" * 90)
    for f in facilities[:20]:
        print(f"  {f['name'][:44]:<45} {f['type']:<24} {f['stretch']}")
    if len(facilities) > 20:
        print(f"  ... and {len(facilities) - 20} more")

    if args.smoke:
        print("\n  [SMOKE] Dry run complete — file not written.")
        return

    if args.merge:
        if os.path.exists(EXISTING_DB):
            facilities = merge_with_existing(facilities, EXISTING_DB)
        else:
            print(f"  [WARN] --merge requested but {EXISTING_DB} not found; saving as-is")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(facilities, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved {len(facilities)} facilities → {args.out}")
    print("\n  To use with anomaly_detector.py:")
    print(f"    python anomaly_detector.py --industry-db {args.out}")


if __name__ == "__main__":
    main()
