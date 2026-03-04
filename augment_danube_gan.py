"""
augment_danube_gan.py — CTGAN-based augmentation for sparse Danube WQ measurements.

Strategy:
  For each WQ parameter, fit a Conditional Tabular GAN (CTGAN) on existing
  measurements (station_id × month × season × value), generate N synthetic
  samples, assign plausible dates (sampled from real dates ± jitter), and
  write output in the same GFQA CSV format as the source data.

  The output directory (`data/danube_augmented_gan/`) is a drop-in replacement
  for `data/danube_processed/` — pass it via --danube-wq-dir in train_model.py.

Usage:
  python augment_danube_gan.py                      # all params, 3× augmentation
  python augment_danube_gan.py --multiplier 5       # 5× the original sample count
  python augment_danube_gan.py --param dissolved_oxygen
  python augment_danube_gan.py --smoke              # DO only, 5 epochs, 10 rows

Dependencies:
  pip install ctgan
"""

import os
import argparse
import warnings
import json

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WQ_SRC   = os.path.join(BASE_DIR, "data", "danube_processed")
WQ_OUT   = os.path.join(BASE_DIR, "data", "danube_augmented_gan")

STATION_NAMES = {
    "SRB00001": "Bezdan",
    "SRB00040": "Bogojevo",
    "SRB00002": "Novi Sad",
    "SRB00003": "Zemun",
    "SRB00041": "Smederevo",
    "SRB00005": "Banatska Palanka",
    "SRB00006": "Tekija",
}

WQ_FILES = [
    ("dissolved_oxygen.csv",       "O2-Dis",  "mg/l"),
    ("phosphorus.csv",             "TP",       "mg/l"),
    ("nitrogen_oxidized.csv",      "NO3N",     "mg/l"),
    ("electrical_conductance.csv", "EC",       "µS/cm"),
    ("oxygen_demand.csv",          "BOD",      "mg/l"),
    ("chlorophyll.csv",            "Chl-a",    "µg/l"),
]

SEASON_MAP = {12: "winter", 1: "winter", 2: "winter",
              3: "spring",  4: "spring", 5: "spring",
              6: "summer",  7: "summer", 8: "summer",
              9: "autumn", 10: "autumn", 11: "autumn"}

DATE_MIN = pd.Timestamp("2013-01-01")
DATE_MAX = pd.Timestamp("2023-12-31")


# ── Feature engineering ────────────────────────────────────────────────────────

def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Convert GFQA rows to a tabular feature table for CTGAN.

    Output columns: station_id (cat), month (int), season (cat), value (float)
    """
    df = df.copy()
    df["date"]   = pd.to_datetime(df["Sample_Date"])
    df["month"]  = df["date"].dt.month.astype(int)
    df["season"] = df["month"].map(SEASON_MAP)
    df["value"]  = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df[df["value"] > 0]

    return df[["station_id", "month", "season", "value"]].rename(
        columns={"station_id": "station_id"}
    )


def assign_dates(synthetic: pd.DataFrame, real_dates: pd.DataFrame,
                 rng: np.random.Generator) -> pd.Series:
    """Assign plausible synthetic dates.

    For each synthetic (station_id, month) pair:
      1. Find real measurement dates for that station+month.
      2. Sample one date at random.
      3. Add ±14-day jitter.
      4. Clip to [DATE_MIN, DATE_MAX].
    Falls back to a random date in the correct month if no real dates exist.
    """
    assigned = []
    for _, row in synthetic.iterrows():
        sid   = row["station_id"]
        month = int(row["month"])

        candidates = real_dates[
            (real_dates["station_id"] == sid) & (real_dates["month"] == month)
        ]["date"]

        if len(candidates) > 0:
            base = candidates.sample(1, random_state=int(rng.integers(0, 2**31))).iloc[0]
            jitter = int(rng.integers(-14, 15))
            synth_date = base + pd.Timedelta(days=jitter)
        else:
            # Fallback: pick a random year 2013-2023, correct month, random day
            year = int(rng.integers(2013, 2024))
            days_in_month = pd.Period(f"{year}-{month:02d}").days_in_month
            day  = int(rng.integers(1, days_in_month + 1))
            synth_date = pd.Timestamp(f"{year}-{month:02d}-{day:02d}")

        synth_date = max(min(synth_date, DATE_MAX), DATE_MIN)
        assigned.append(synth_date)

    return pd.Series(assigned, index=synthetic.index)


# ── Main augmentation ──────────────────────────────────────────────────────────

def augment_parameter(
    src_path: str,
    param_code: str,
    unit: str,
    multiplier: int,
    epochs: int,
    rng: np.random.Generator,
    smoke: bool = False,
) -> pd.DataFrame:
    """Fit CTGAN on one parameter's GFQA data and return augmented DataFrame."""
    try:
        from ctgan import CTGAN
    except ImportError:
        raise ImportError(
            "ctgan is required: pip install ctgan\n"
            "Or install the full SDV suite: pip install sdv"
        )

    df_raw = pd.read_csv(src_path)
    df_raw = df_raw[df_raw["Parameter_Code"] == param_code].copy()
    df_raw["station_id"] = df_raw["Station_ID"]

    if df_raw.empty:
        print(f"    [SKIP] No rows found for code={param_code}")
        return pd.DataFrame()

    features = build_feature_table(df_raw)
    real_dates = df_raw[["station_id"]].copy()
    real_dates["date"]  = pd.to_datetime(df_raw["Sample_Date"])
    real_dates["month"] = real_dates["date"].dt.month.astype(int)

    n_real  = len(features)
    n_synth = (n_real * multiplier) if not smoke else 10

    print(f"    Fitting CTGAN on {n_real} real rows → generating {n_synth} synthetic rows "
          f"({epochs} epochs) ...")

    model = CTGAN(
        epochs=epochs,
        batch_size=min(500, max(50, n_real)),
        verbose=False,
    )
    model.fit(features, discrete_columns=["station_id", "season"])

    synthetic = model.sample(n_synth)

    # Clip values to physically plausible range per parameter
    VALUE_LIMITS = {
        "O2-Dis":  (0.1, 20.0),
        "TP":      (0.001, 5.0),
        "NO3N":    (0.01, 20.0),
        "EC":      (50, 2000),
        "BOD":     (0.5, 50.0),
        "Chl-a":   (0.001, 200.0),
    }
    lo, hi = VALUE_LIMITS.get(param_code, (0, 1e9))
    synthetic["value"] = synthetic["value"].clip(lo, hi)

    synthetic["station_id"] = synthetic["station_id"].astype(str)
    synthetic["month"]      = pd.to_numeric(synthetic["month"], errors="coerce").clip(1, 12)

    # Filter to known stations only
    synthetic = synthetic[synthetic["station_id"].isin(STATION_NAMES)]
    if synthetic.empty:
        print("    [WARN] CTGAN generated no rows for known stations — "
              "try more epochs or a larger dataset.")
        return pd.DataFrame()

    synth_dates = assign_dates(synthetic, real_dates, rng)

    # Build GFQA-format output rows (synthetic)
    synth_rows = pd.DataFrame({
        "Station_ID":     synthetic["station_id"].values,
        "Station_Name":   [STATION_NAMES.get(s, s) for s in synthetic["station_id"].values],
        "Sample_Date":    synth_dates.dt.strftime("%Y-%m-%d").values,
        "Parameter_Code": param_code,
        "Value_Flag":     "",
        "Value":          synthetic["value"].round(4).values,
        "Unit":           unit,
        "Data_Quality":   "Synthetic",
        "source":         "ctgan",
    })

    # Mark originals and combine
    orig_rows = df_raw[[
        "Station_ID", "Station_Name", "Sample_Date", "Parameter_Code",
        "Value_Flag", "Value", "Unit", "Data_Quality"
    ]].copy()
    orig_rows["source"] = "original"

    combined = pd.concat([orig_rows, synth_rows], ignore_index=True)
    combined = combined.sort_values(["Station_ID", "Sample_Date"]).reset_index(drop=True)

    n_orig  = len(orig_rows)
    n_added = len(synth_rows)
    print(f"    Done — {n_orig} original + {n_added} synthetic = {len(combined)} total rows")

    return combined


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CTGAN-based augmentation for Danube WQ measurements"
    )
    parser.add_argument(
        "--src-dir", default=WQ_SRC,
        help="Source GFQA WQ directory (default: data/danube_processed/)",
    )
    parser.add_argument(
        "--out-dir", default=WQ_OUT,
        help="Output directory (default: data/danube_augmented_gan/)",
    )
    parser.add_argument(
        "--multiplier", type=int, default=3,
        help="How many synthetic samples per real sample (default: 3)",
    )
    parser.add_argument(
        "--epochs", type=int, default=200,
        help="CTGAN training epochs per parameter (default: 200)",
    )
    parser.add_argument(
        "--param", nargs="+", default=None,
        help="Restrict to specific WQ files (e.g. dissolved_oxygen.csv)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Quick smoke test: 1 parameter (DO), 5 epochs, 10 synthetic rows",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    files_to_run = WQ_FILES
    if args.smoke:
        files_to_run = [WQ_FILES[0]]     # dissolved_oxygen only
        args.epochs  = 5
        print("[SMOKE TEST] Running DO only, 5 epochs, 10 rows")
    elif args.param:
        files_to_run = [f for f in WQ_FILES if f[0] in args.param]
        if not files_to_run:
            print(f"No matching files for --param {args.param}. "
                  f"Valid choices: {[f[0] for f in WQ_FILES]}")
            return

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\nDanube CTGAN Augmentation")
    print(f"  Source   : {args.src_dir}")
    print(f"  Output   : {args.out_dir}")
    print(f"  Multiplier: {args.multiplier}× | Epochs: {args.epochs}")
    print()

    summary = {}

    for filename, param_code, unit in files_to_run:
        src_path = os.path.join(args.src_dir, filename)
        if not os.path.exists(src_path):
            print(f"  [SKIP] {filename} not found in {args.src_dir}")
            continue

        print(f"  [{param_code}] {filename}")
        try:
            combined = augment_parameter(
                src_path, param_code, unit,
                multiplier=args.multiplier,
                epochs=args.epochs,
                rng=rng,
                smoke=args.smoke,
            )
        except Exception as e:
            print(f"    [ERROR] {e}")
            continue

        if combined.empty:
            continue

        out_path = os.path.join(args.out_dir, filename)
        combined.to_csv(out_path, index=False)
        print(f"    Saved → {out_path}")

        n_orig  = (combined["source"] == "original").sum()
        n_synth = (combined["source"] == "ctgan").sum()
        summary[param_code] = {"original": int(n_orig), "synthetic": int(n_synth)}
        print()

    # Write summary JSON
    summary_path = os.path.join(args.out_dir, "augmentation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary written to {summary_path}")
    print("\nAugmentation complete.")
    print(f"\nTo train with augmented data:")
    print(f"  python train_model.py data/danube_processed danube_gan_v1 "
          f"--model c --danube-wq-dir {args.out_dir}")


if __name__ == "__main__":
    main()
