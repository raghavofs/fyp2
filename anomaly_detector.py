"""
anomaly_detector.py — Danube WQ anomaly detection + pollution source attribution

Pipeline:
  1. Load actual WQ measurements from GFQA_v3 CSVs (all stations, all dates)
  2. For each station × parameter, flag anomalies using two independent checks:
       a) Rolling Z-score: value vs 90-day rolling median/MAD
       b) Seasonal baseline: value vs same-month mean across all years
     Anomaly requires BOTH checks to fire (reduces false positives).
     Trend vs. acute classification based on whether the rolling mean itself
     is elevated (sustained trend) or flat (sudden spike).
  3. Cross-station comparison: find the most upstream station that shows the
     anomaly — the pollution source lies in the stretch upstream of it.
  4. Attribute likely industries from data/industry_lookup/danube_industries.json
     filtered by stretch and pollutant type.
  5. Write full report to anomaly_reports/{timestamp}/

Usage:
  python anomaly_detector.py
  python anomaly_detector.py --z-thresh 2.0 --window 60 --output-dir my_reports
  python anomaly_detector.py --param dissolved_oxygen --from-date 2018-01-01
"""

import os
import json
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Station metadata — ordered upstream → downstream
# ============================================================
STATIONS = [
    {"id": "SRB00001", "name": "Bezdan",           "km": 1425},
    {"id": "SRB00040", "name": "Bogojevo",          "km": 1367},
    {"id": "SRB00002", "name": "Novi Sad",          "km": 1255},
    {"id": "SRB00003", "name": "Zemun (Belgrade)",  "km": 1173},
    {"id": "SRB00041", "name": "Smederevo",         "km": 1116},
    {"id": "SRB00005", "name": "Banatska Palanka",  "km": 1077},
    {"id": "SRB00006", "name": "Tekija",            "km":  931},
]
STATION_IDS   = [s["id"]   for s in STATIONS]
STATION_NAMES = {s["id"]: s["name"] for s in STATIONS}
STATION_KM    = {s["id"]: s["km"]   for s in STATIONS}

# ============================================================
# WQ parameter configuration
# Maps our target names → GFQA parameter codes + anomaly direction
# ============================================================
PARAM_CONFIG = {
    "dissolved_oxygen": {
        "file": "dissolved_oxygen.csv",
        "code": "O2-Dis",
        "anomaly_direction": "below",   # low DO is the hazard
        "unit": "mg/L",
        "normal_range": (6, 12),
    },
    "total_phosphorus": {
        "file": "phosphorus.csv",
        "code": "TP",
        "anomaly_direction": "above",
        "unit": "mg/L",
        "normal_range": (0.01, 0.4),
    },
    "nitrate_n": {
        "file": "nitrogen_oxidized.csv",
        "code": "NO3N",
        "anomaly_direction": "above",
        "unit": "mg/L",
        "normal_range": (0.5, 6.0),
    },
    "electrical_conductance": {
        "file": "electrical_conductance.csv",
        "code": "EC",
        "anomaly_direction": "above",
        "unit": "µS/cm",
        "normal_range": (200, 600),
    },
    "chlorophyll_a": {
        "file": "chlorophyll.csv",
        "code": "Chl-a",
        "anomaly_direction": "above",
        "unit": "µg/L",
        "normal_range": (1, 30),
    },
}

# What kinds of industries to look for based on which parameter spikes / drops
POLLUTANT_INDUSTRY_TYPES = {
    "dissolved_oxygen": {
        "direction_label": "LOW",
        "industry_types": ["wastewater_treatment", "chemical_plant", "refinery",
                           "food_processing", "pulp_paper", "agriculture"],
        "mechanism": (
            "Low dissolved oxygen indicates high organic or chemical oxygen demand. "
            "Likely sources: municipal sewage overflows, organic industrial effluent "
            "(food processing, refinery), or nutrient-driven algal decomposition."
        ),
    },
    "total_phosphorus": {
        "direction_label": "HIGH",
        "industry_types": ["wastewater_treatment", "agriculture", "fertilizer",
                           "chemical_plant", "metallurgy"],
        "mechanism": (
            "Elevated phosphorus from agricultural runoff (fertilisers), "
            "municipal wastewater (detergents, human waste), or industrial effluent "
            "(metallurgical slag, phosphate processing)."
        ),
    },
    "nitrate_n": {
        "direction_label": "HIGH",
        "industry_types": ["agriculture", "fertilizer", "wastewater_treatment",
                           "food_processing", "chemical_plant"],
        "mechanism": (
            "Elevated nitrate from agricultural fertiliser application, "
            "animal husbandry drainage, nitrogen fertiliser plant effluent, "
            "or secondary treatment of municipal wastewater."
        ),
    },
    "electrical_conductance": {
        "direction_label": "HIGH",
        "industry_types": ["metallurgy", "mining", "chemical_plant", "refinery",
                           "power_plant", "fertilizer"],
        "mechanism": (
            "High electrical conductance indicates elevated dissolved ion load — "
            "salts, metals, or acids from industrial discharge. Common sources: "
            "steel plant cooling water, mining drainage, chemical plant process water."
        ),
    },
    "chlorophyll_a": {
        "direction_label": "HIGH",
        "industry_types": ["agriculture", "wastewater_treatment", "fertilizer"],
        "mechanism": (
            "Algal bloom driven by nutrient enrichment (eutrophication). "
            "Chlorophyll spikes are secondary indicators — trace upstream phosphorus "
            "or nitrate anomalies as primary cause."
        ),
    },
}


# ============================================================
# 1. Data loading
# ============================================================

def load_wq_measurements(wq_dir: str) -> dict[str, pd.DataFrame]:
    """Load actual WQ measurements from GFQA CSVs for all parameters.

    Returns dict: {param_name -> DataFrame(date, station_id, value)}
    """
    wq_data = {}
    for param, cfg in PARAM_CONFIG.items():
        path = os.path.join(wq_dir, cfg["file"])
        if not os.path.exists(path):
            print(f"  [WARN] Missing: {path}")
            continue

        df = pd.read_csv(path, parse_dates=["Sample_Date"])
        df = df[df["Parameter_Code"] == cfg["code"]].copy()
        df = df.rename(columns={"Station_ID": "station_id",
                                 "Sample_Date": "date",
                                 "Value": "value"})[["date", "station_id", "value"]]
        df = df.dropna(subset=["value"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        df = df.sort_values("date").reset_index(drop=True)

        wq_data[param] = df
        print(f"  Loaded {param}: {len(df)} measurements across "
              f"{df['station_id'].nunique()} stations "
              f"({df['date'].min().date()} → {df['date'].max().date()})")

    return wq_data


def load_industry_db(db_path: str) -> list[dict]:
    with open(db_path) as f:
        return json.load(f)


# ============================================================
# 2. Anomaly detection
# ============================================================

def detect_anomalies(
    df: pd.DataFrame,
    param: str,
    z_thresh: float = 2.5,
    window: int = 90,
    min_obs: int = 3,
) -> list[dict]:
    """Detect anomalies for a single station × parameter time series.

    Two independent checks must both fire to flag an anomaly:
      1. Rolling Z-score: (value - rolling_median) / (1.4826 * rolling_MAD) > z_thresh
         Uses median absolute deviation (robust to outliers). Looks back `window` days.
      2. Seasonal baseline: value deviates > z_thresh σ from the same-month mean
         across all years. This distinguishes genuine anomalies from seasonal trends.

    Anomaly type:
      - "acute":  current value spikes but the rolling mean itself is flat
                  (sudden event — industrial accident, spill)
      - "trend":  both rolling mean and current value are elevated
                  (sustained pressure — ongoing discharge, season-long fertiliser runoff)
    """
    cfg = PARAM_CONFIG[param]
    direction = cfg["anomaly_direction"]
    anomalies = []

    df = df.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    # ── Rolling statistics (window = last N calendar days) ──────────────────
    # Use .rolling on time-indexed data so irregular sampling is handled correctly
    roll = df["value"].rolling(f"{window}D", min_periods=min_obs)
    df["roll_median"] = roll.median()
    df["roll_mad"]    = roll.apply(lambda x: np.median(np.abs(x - np.median(x))),
                                    raw=True)
    df["roll_std_est"] = df["roll_mad"] * 1.4826   # consistent std estimate

    # ── Seasonal statistics (same calendar month, all years) ────────────────
    df["month"] = df.index.month
    month_stats = df.groupby("month")["value"].agg(["mean", "std"]).rename(
        columns={"mean": "seasonal_mean", "std": "seasonal_std"}
    )
    df = df.join(month_stats, on="month")

    # ── Per-observation anomaly scoring ─────────────────────────────────────
    for ts, row in df.iterrows():
        v = row["value"]

        # Skip if not enough history
        if pd.isna(row["roll_median"]) or row["roll_std_est"] < 1e-6:
            continue

        # Rolling Z-score — PRIMARY detection gate
        # (signed: positive = above median; for DO, spike means drop so we negate)
        roll_z = (v - row["roll_median"]) / row["roll_std_est"]
        if direction == "below":
            roll_z = -roll_z

        if roll_z < z_thresh:
            continue   # value is within normal range of recent history

        # Seasonal Z-score — used for CLASSIFICATION only, not as a gate
        seas_z = 0.0
        if pd.notna(row["seasonal_std"]) and row["seasonal_std"] > 1e-6:
            seas_z = (v - row["seasonal_mean"]) / row["seasonal_std"]
            if direction == "below":
                seas_z = -seas_z

        # Classify anomaly type:
        #   "acute"  — rolling baseline is flat; current value spikes unexpectedly
        #              → likely a sudden industrial event / spill
        #   "trend"  — rolling median itself is elevated vs seasonal norm
        #              → sustained discharge pressure (ongoing pollution / seasonal crop runoff)
        roll_median_z = 0.0
        if pd.notna(row["seasonal_std"]) and row["seasonal_std"] > 1e-6:
            roll_median_z = (row["roll_median"] - row["seasonal_mean"]) / row["seasonal_std"]
            if direction == "below":
                roll_median_z = -roll_median_z

        anomaly_type = "trend" if roll_median_z > 1.0 else "acute"

        anomalies.append({
            "date":           ts.date(),
            "param":          param,
            "value":          round(v, 4),
            "roll_median":    round(row["roll_median"], 4),
            "seasonal_mean":  round(row["seasonal_mean"], 4),
            "roll_z":         round(roll_z, 2),
            "seas_z":         round(seas_z, 2),
            "anomaly_type":   anomaly_type,
            "unit":           cfg["unit"],
        })

    return anomalies


# ============================================================
# 3. Cross-station stretch identification
# ============================================================

def find_pollution_stretch(
    anomaly_station_ids: set,
    param: str,
    anomaly_type: str,
) -> dict:
    """Identify the river stretch most likely containing the pollution source.

    Logic: find the most upstream station showing the anomaly. The source lies
    in the stretch between that station and the next station upstream (which is
    presumably clean). If the most upstream station (Bezdan) is flagged, the
    source may be in Hungary or Croatia upstream of Serbia.

    Returns a dict with stretch metadata.
    """
    if not anomaly_station_ids:
        return {}

    # Find first (most upstream) anomaly station in the ordered list
    for i, st in enumerate(STATIONS):
        if st["id"] not in anomaly_station_ids:
            continue

        # This is the most upstream anomaly station
        if i == 0:
            return {
                "stretch_label":     f"Upstream of {st['name']} (possible transboundary source)",
                "upstream_station":  None,
                "downstream_station": st["id"],
                "km_from":           st["km"],
                "km_to":             st["km"] + 150,
                "stretch_key":       f"UPSTREAM_{st['id']}",
                "note":              "Anomaly at most upstream Serbian station — source may be in Hungary/Croatia.",
            }
        else:
            prev = STATIONS[i - 1]
            return {
                "stretch_label":      f"{prev['name']} (km {prev['km']}) → {st['name']} (km {st['km']})",
                "upstream_station":   prev["id"],
                "downstream_station": st["id"],
                "km_from":            st["km"],
                "km_to":              prev["km"],
                "stretch_key":        f"{prev['id']}_{st['id']}",
                "note":               (
                    f"{len(anomaly_station_ids)} station(s) show anomaly; "
                    f"source most likely in the {prev['km'] - st['km']} km stretch."
                ),
            }

    return {}


# ============================================================
# 4. Industry attribution
# ============================================================

def attribute_industries(
    stretch_key: str,
    param: str,
    industry_db: list[dict],
) -> list[dict]:
    """Return industries in the given river stretch that match the pollutant type."""
    if not stretch_key:
        return []

    relevant_types = POLLUTANT_INDUSTRY_TYPES.get(param, {}).get("industry_types", [])
    matches = []

    for facility in industry_db:
        # Check stretch match
        if facility.get("stretch") != stretch_key:
            ds = facility.get("nearest_station_downstream") or ""
            us = facility.get("nearest_station_upstream") or ""
            if not (stretch_key.startswith(us) or (ds and stretch_key.endswith(ds))):
                continue

        # Check pollutant match
        if param not in facility.get("effluent_parameters", []):
            continue

        matches.append({
            "facility_id":   facility["id"],
            "name":          facility["name"],
            "type":          facility["type"],
            "river_km":      facility["river_km"],
            "effluent_notes": facility["effluent_notes"],
        })

    return matches


# ============================================================
# 5. Report generation
# ============================================================

def generate_report(events: list[dict], output_dir: str) -> None:
    """Write anomaly report as CSV (human-readable) and JSON (machine-readable)."""
    os.makedirs(output_dir, exist_ok=True)

    if not events:
        print("\n  No anomalies detected with current thresholds.")
        return

    # ── Flatten for CSV ─────────────────────────────────────────────────────
    rows = []
    for ev in events:
        industries = ev.get("attributed_industries", [])
        industry_names  = "; ".join(f["name"] for f in industries) if industries else "None identified"
        industry_types  = "; ".join(f["type"] for f in industries) if industries else ""
        industry_km     = "; ".join(str(f["river_km"]) for f in industries) if industries else ""
        mechanism       = POLLUTANT_INDUSTRY_TYPES.get(ev["param"], {}).get("mechanism", "")

        rows.append({
            "date":              ev["date"],
            "station_id":        ev["station_id"],
            "station_name":      STATION_NAMES.get(ev["station_id"], ev["station_id"]),
            "parameter":         ev["param"],
            "measured_value":    ev["value"],
            "unit":              ev["unit"],
            "rolling_median":    ev["roll_median"],
            "seasonal_mean":     ev["seasonal_mean"],
            "rolling_zscore":    ev["roll_z"],
            "seasonal_zscore":   ev["seas_z"],
            "anomaly_type":      ev["anomaly_type"],
            "direction":         POLLUTANT_INDUSTRY_TYPES.get(ev["param"], {}).get("direction_label", ""),
            "river_stretch":     ev.get("stretch_label", ""),
            "km_from":           ev.get("km_from", ""),
            "km_to":             ev.get("km_to", ""),
            "attributed_facilities": industry_names,
            "facility_types":    industry_types,
            "facility_km":       industry_km,
            "mechanism":         mechanism,
        })

    df = pd.DataFrame(rows).sort_values(["date", "station_id", "parameter"])

    csv_path  = os.path.join(output_dir, "anomaly_report.csv")
    json_path = os.path.join(output_dir, "anomaly_report.json")

    df.to_csv(csv_path, index=False)

    with open(json_path, "w") as f:
        json.dump(events, f, indent=2, default=str)

    # ── Terminal summary ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  ANOMALY REPORT — {len(events)} events detected")
    print(f"{'='*70}")
    print(f"\n  {'Date':<12} {'Station':<22} {'Parameter':<24} {'Value':>8}  "
          f"{'Type':<8}  {'Likely Source'}")
    print("  " + "-" * 100)

    for _, row in df.iterrows():
        val_str = f"{row['measured_value']:.3f} {row['unit']}"
        source  = row["attributed_facilities"][:45] + "…" \
                  if len(row["attributed_facilities"]) > 45 \
                  else row["attributed_facilities"]
        print(f"  {str(row['date']):<12} {row['station_name']:<22} "
              f"{row['parameter']:<24} {val_str:>14}  "
              f"{row['anomaly_type']:<8}  {source}")

    print(f"\n  Files written:")
    print(f"    {csv_path}")
    print(f"    {json_path}")

    # ── Per-parameter summary ────────────────────────────────────────────────
    print(f"\n  Summary by parameter:")
    for param, grp in df.groupby("parameter"):
        n_acute = (grp["anomaly_type"] == "acute").sum()
        n_trend = (grp["anomaly_type"] == "trend").sum()
        print(f"    {param:<24} {len(grp):3d} events  "
              f"({n_acute} acute, {n_trend} trend)  "
              f"stations: {', '.join(grp['station_name'].unique())}")


# ============================================================
# 6. Main orchestration
# ============================================================

def run(
    wq_dir: str,
    industry_db_path: str,
    output_dir: str,
    z_thresh: float = 2.5,
    window: int = 90,
    params: list[str] | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
) -> list[dict]:

    print(f"\n{'='*70}")
    print("  DANUBE WQ ANOMALY DETECTOR")
    print(f"  Z-threshold: {z_thresh}  |  Rolling window: {window} days")
    if from_date or to_date:
        print(f"  Date range: {from_date or 'start'} → {to_date or 'end'}")
    print(f"{'='*70}\n")

    # Load data
    print("Loading WQ measurements...")
    wq_data = load_wq_measurements(wq_dir)
    industry_db = load_industry_db(industry_db_path)
    print(f"  Industry database: {len(industry_db)} facilities loaded\n")

    target_params = params if params else list(PARAM_CONFIG.keys())

    all_events = []

    for param in target_params:
        if param not in wq_data:
            continue

        df_param = wq_data[param].copy()

        # Date filtering
        if from_date:
            df_param = df_param[df_param["date"] >= pd.Timestamp(from_date)]
        if to_date:
            df_param = df_param[df_param["date"] <= pd.Timestamp(to_date)]

        print(f"  [{param}]  {len(df_param)} measurements")
        param_anomalies = {}   # station_id → list of anomaly dicts

        for station_id in STATION_IDS:
            st_df = df_param[df_param["station_id"] == station_id][["date", "value"]].copy()
            if len(st_df) < 6:
                continue

            events = detect_anomalies(st_df, param, z_thresh=z_thresh, window=window)
            if events:
                param_anomalies[station_id] = events
                print(f"    {STATION_NAMES[station_id]:<22}: {len(events)} anomalies")

        if not param_anomalies:
            continue

        # Group anomalies by date to find cross-station patterns
        # Build lookup: date → set of anomalous station IDs
        date_to_stations: dict = {}
        for sid, evs in param_anomalies.items():
            for ev in evs:
                d = ev["date"]
                date_to_stations.setdefault(d, set()).add(sid)

        # For each anomaly event, assign stretch and industries
        for sid, evs in param_anomalies.items():
            for ev in evs:
                d = ev["date"]
                anomaly_stations_today = date_to_stations.get(d, {sid})
                dominant_type = max(
                    (e["anomaly_type"] for e in (param_anomalies.get(s, []) for s in anomaly_stations_today)
                     for e in e),
                    key=lambda t: 0 if t == "acute" else 1,
                    default="acute",
                )

                stretch = find_pollution_stretch(anomaly_stations_today, param, dominant_type)
                industries = attribute_industries(
                    stretch.get("stretch_key", ""), param, industry_db
                )

                all_events.append({
                    **ev,
                    "station_id":            sid,
                    "stations_on_same_date": sorted(anomaly_stations_today),
                    **stretch,
                    "attributed_industries": industries,
                })

    generate_report(all_events, output_dir)
    return all_events


def main():
    parser = argparse.ArgumentParser(
        description="Danube WQ anomaly detection + pollution source attribution"
    )
    parser.add_argument(
        "--wq-dir",
        default=os.path.join(BASE_DIR, "data", "danube_processed"),
        help="Directory containing GFQA WQ CSVs (default: data/danube_processed/)",
    )
    parser.add_argument(
        "--industry-db",
        default=os.path.join(BASE_DIR, "data", "industry_lookup", "danube_industries.json"),
        help="Path to industry lookup JSON",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(BASE_DIR, "anomaly_reports",
                             datetime.now().strftime("%Y%m%d_%H%M%S")),
        help="Directory to write report files",
    )
    parser.add_argument(
        "--z-thresh", type=float, default=2.5,
        help="Z-score threshold for anomaly flagging (default: 2.5). "
             "Lower = more sensitive. Both rolling and seasonal Z must exceed this.",
    )
    parser.add_argument(
        "--window", type=int, default=90,
        help="Rolling baseline window in days (default: 90)",
    )
    parser.add_argument(
        "--param", nargs="+", default=None,
        choices=list(PARAM_CONFIG.keys()),
        help="Restrict analysis to specific parameters",
    )
    parser.add_argument(
        "--from-date", default=None,
        help="Start date filter, e.g. 2018-01-01",
    )
    parser.add_argument(
        "--to-date", default=None,
        help="End date filter, e.g. 2022-12-31",
    )
    args = parser.parse_args()

    run(
        wq_dir=args.wq_dir,
        industry_db_path=args.industry_db,
        output_dir=args.output_dir,
        z_thresh=args.z_thresh,
        window=args.window,
        params=args.param,
        from_date=args.from_date,
        to_date=args.to_date,
    )


if __name__ == "__main__":
    main()
