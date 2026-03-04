"""
llm_attributor.py — LLM-based pollution source attribution for Danube anomalies.

Threshold-routed pipeline:
  • roll_z < LOW_THRESH  (< 2.5) → not an anomaly (filtered out upstream)
  • LOW_THRESH ≤ roll_z < HIGH_THRESH (2.5 – 3.5) → rule-based attribution only
    (already in anomaly_report.json from anomaly_detector.py)
  • roll_z ≥ HIGH_THRESH (≥ 3.5) → route to Claude API for deeper analysis

The LLM receives:
  – The anomaly measurement and its Z-score
  – The rolling baseline and seasonal norm
  – All attributed industrial facilities in the stretch
  – The pollutant mechanism description
  – Multi-station context (how many stations are affected on the same day)

Output: enriched JSON with extra fields per high-severity event:
  llm_primary_source, llm_secondary_factors, llm_confidence,
  llm_confidence_reason, llm_suggested_action, llm_model

Usage:
  python llm_attributor.py                               # process last anomaly report
  python llm_attributor.py --report anomaly_reports/X/anomaly_report.json
  python llm_attributor.py --z-high 4.0                 # only extreme events
  python llm_attributor.py --smoke                       # 1 event, print only
  python llm_attributor.py --run-detector                # run detector first, then enrich

Environment:
  ANTHROPIC_API_KEY must be set (or passed via --api-key)

Dependencies:
  pip install anthropic
"""

import os
import sys
import json
import argparse
import re
from datetime import datetime
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Thresholds ─────────────────────────────────────────────────────────────────
LOW_THRESH  = 2.5    # minimum Z to be an anomaly (detector's gate)
HIGH_THRESH = 3.5    # minimum Z to route to LLM

MODEL_ID = "claude-sonnet-4-6"

# ── Pollutant context (mirrors anomaly_detector.py) ───────────────────────────
PARAM_CONTEXT = {
    "dissolved_oxygen": {
        "direction": "LOW",
        "unit": "mg/L",
        "healthy_range": "6–12 mg/L",
        "hazard": "Aquatic life stress below 5 mg/L; dead zone below 2 mg/L",
    },
    "total_phosphorus": {
        "direction": "HIGH",
        "unit": "mg/L",
        "healthy_range": "0.01–0.4 mg/L",
        "hazard": "Eutrophication and algal blooms above 0.1 mg/L",
    },
    "nitrate_n": {
        "direction": "HIGH",
        "unit": "mg/L",
        "healthy_range": "0.5–6 mg/L",
        "hazard": "Hypoxia risk at high loads; EU WFD limit 50 mg/L as NO3",
    },
    "electrical_conductance": {
        "direction": "HIGH",
        "unit": "µS/cm",
        "healthy_range": "200–600 µS/cm",
        "hazard": "High ion loading; possible heavy metal co-contamination above 1000 µS/cm",
    },
    "chlorophyll_a": {
        "direction": "HIGH",
        "unit": "µg/L",
        "healthy_range": "1–30 µg/L",
        "hazard": "Algal bloom indicator; secondary signal of nutrient enrichment",
    },
    "oxygen_demand": {
        "direction": "HIGH",
        "unit": "mg/L",
        "healthy_range": "1–10 mg/L BOD",
        "hazard": "Organic pollution; BOD > 5 mg/L = moderate; > 10 = severe",
    },
}


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(event: dict) -> str:
    """Build a structured prompt for Claude."""
    param   = event.get("param", "unknown")
    ctx     = PARAM_CONTEXT.get(param, {})
    station = event.get("station_id", "?")
    sname   = event.get("stations_on_same_date", [station])

    # Industry block
    industries = event.get("attributed_industries", [])
    if industries:
        industry_block = "\n".join(
            f"  • {f['name']} (type: {f['type']}, river km {f['river_km']})\n"
            f"    Notes: {f.get('effluent_notes', 'N/A')}"
            for f in industries
        )
    else:
        industry_block = "  None identified in the hardcoded database."

    multi_station_note = (
        f"  Affected stations on the same date: {', '.join(sname)}"
        if len(sname) > 1 else
        f"  Only one station affected on this date ({station})."
    )

    stretch = event.get("stretch_label", "unknown stretch")

    prompt = f"""You are an environmental analyst specialising in Danube River water quality in Serbia.

## ANOMALY DETECTED

- Date        : {event.get('date')}
- Parameter   : {param.replace('_', ' ').title()}
- Direction   : {ctx.get('direction', 'ELEVATED')}
- Station     : {station}
- Stretch     : {stretch}
- Measured    : {event.get('value')} {ctx.get('unit', '')}
- Rolling median (90-day baseline): {event.get('roll_median')} {ctx.get('unit', '')}
- Rolling Z-score : {event.get('roll_z')} σ (threshold: {LOW_THRESH} σ)
- Seasonal mean   : {event.get('seasonal_mean')} {ctx.get('unit', '')}
- Anomaly type    : {event.get('anomaly_type')} (acute = sudden spike; trend = sustained elevation)
- Healthy range   : {ctx.get('healthy_range', 'N/A')}
- Hazard note     : {ctx.get('hazard', 'N/A')}
{multi_station_note}

## INDUSTRIAL SOURCES IN STRETCH

{industry_block}

## YOUR TASK

Analyse this anomaly and respond ONLY with valid JSON (no markdown, no explanation outside JSON).
Use the following structure exactly:

{{
  "primary_source": "<1-2 sentence identification of the most likely pollution source>",
  "secondary_factors": "<brief mention of secondary contributors or 'None identified'>",
  "confidence": "<HIGH | MEDIUM | LOW>",
  "confidence_reason": "<1 sentence explaining the confidence level>",
  "suggested_action": "<1-2 sentences: recommended monitoring or regulatory follow-up>",
  "environmental_impact": "<1 sentence on likely ecological consequence if unaddressed>"
}}

Base your analysis on:
1. The measured value relative to the healthy range and rolling baseline
2. The type of industries in the stretch and their known effluents
3. The anomaly type (acute spill vs sustained trend) and multi-station spread
4. Seasonal context (month {datetime.strptime(str(event.get('date', '2000-01-01')), '%Y-%m-%d').month if event.get('date') else '?'} = {_month_season(event.get('date'))})
"""
    return prompt.strip()


def _month_season(date_str) -> str:
    if not date_str:
        return "unknown"
    try:
        m = int(str(date_str)[5:7])
    except Exception:
        return "unknown"
    return {12:"winter",1:"winter",2:"winter",
            3:"spring",4:"spring",5:"spring",
            6:"summer",7:"summer",8:"summer",
            9:"autumn",10:"autumn",11:"autumn"}.get(m, "unknown")


# ── LLM call ──────────────────────────────────────────────────────────────────

def call_claude(prompt: str, api_key: str) -> dict:
    """Call Claude and parse the JSON response."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic SDK required: pip install anthropic")

    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model=MODEL_ID,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()

    # Strip markdown code fences if the model wraps the JSON
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Return the raw text in a fallback field so it isn't lost
        return {"raw_response": raw, "parse_error": "JSON decode failed"}


# ── Main pipeline ─────────────────────────────────────────────────────────────

def enrich_report(
    events: list[dict],
    api_key: str,
    high_thresh: float = HIGH_THRESH,
    smoke: bool = False,
) -> list[dict]:
    """Enrich high-severity events with LLM attribution.

    Events with roll_z < high_thresh pass through unchanged.
    Events with roll_z ≥ high_thresh get LLM fields added.
    """
    high_events = [e for e in events if abs(e.get("roll_z", 0)) >= high_thresh]
    low_events  = [e for e in events if abs(e.get("roll_z", 0)) < high_thresh]

    print(f"\n  Threshold routing (Z ≥ {high_thresh} → LLM):")
    print(f"    Rule-based only : {len(low_events)} events")
    print(f"    Routed to LLM   : {len(high_events)} events")

    if smoke:
        high_events = high_events[:1]
        print(f"    [SMOKE] Processing only 1 event")

    enriched_high = []
    for i, event in enumerate(high_events):
        param    = event.get("param", "?")
        station  = event.get("station_id", "?")
        date_str = str(event.get("date", "?"))
        roll_z   = event.get("roll_z", 0)

        print(f"\n  [{i+1}/{len(high_events)}] {date_str}  {station}  {param}  Z={roll_z:.1f}σ")

        prompt = build_prompt(event)
        try:
            llm_result = call_claude(prompt, api_key)
        except Exception as e:
            print(f"    [ERROR] Claude API call failed: {e}")
            llm_result = {"error": str(e)}

        enriched = {**event,
                    "llm_primary_source":    llm_result.get("primary_source"),
                    "llm_secondary_factors": llm_result.get("secondary_factors"),
                    "llm_confidence":        llm_result.get("confidence"),
                    "llm_confidence_reason": llm_result.get("confidence_reason"),
                    "llm_suggested_action":  llm_result.get("suggested_action"),
                    "llm_environmental_impact": llm_result.get("environmental_impact"),
                    "llm_model":             MODEL_ID,
                    "llm_raw":               llm_result if "parse_error" in llm_result else None}

        if "primary_source" in llm_result:
            print(f"    Source    : {llm_result['primary_source'][:90]}")
            print(f"    Confidence: {llm_result.get('confidence', '?')} — "
                  f"{llm_result.get('confidence_reason', '')[:70]}")

        enriched_high.append(enriched)

    return low_events + enriched_high


def find_latest_report(base_dir: str) -> str | None:
    """Return the most recently created anomaly_report.json under anomaly_reports/."""
    reports_dir = os.path.join(base_dir, "anomaly_reports")
    if not os.path.isdir(reports_dir):
        return None
    candidates = sorted(
        Path(reports_dir).glob("**/anomaly_report.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(candidates[0]) if candidates else None


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM-enriched attribution for high-severity Danube anomalies"
    )
    parser.add_argument(
        "--report", default=None,
        help="Path to anomaly_report.json (default: latest under anomaly_reports/)",
    )
    parser.add_argument(
        "--z-high", type=float, default=HIGH_THRESH,
        help=f"Z-score threshold for LLM routing (default: {HIGH_THRESH})",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output JSON path (default: <report_dir>/llm_attributed_report.json)",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="Anthropic API key (default: ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Process only the single highest-severity event — do not write file",
    )
    parser.add_argument(
        "--run-detector", action="store_true",
        help="Run anomaly_detector.py first, then enrich its output",
    )
    args = parser.parse_args()

    # ── API key ───────────────────────────────────────────────────────────────
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Anthropic API key required.")
        print("  Set ANTHROPIC_API_KEY environment variable, or use --api-key KEY")
        sys.exit(1)

    # ── Optionally run detector first ─────────────────────────────────────────
    if args.run_detector:
        import subprocess
        ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
        det_out_dir = os.path.join(BASE_DIR, "anomaly_reports", ts)
        print(f"Running anomaly_detector.py → {det_out_dir} ...")
        subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, "anomaly_detector.py"),
             "--output-dir", det_out_dir],
            check=True,
        )
        report_path = os.path.join(det_out_dir, "anomaly_report.json")
    else:
        report_path = args.report or find_latest_report(BASE_DIR)

    if not report_path or not os.path.exists(report_path):
        print(f"ERROR: No anomaly_report.json found.")
        print("  Run anomaly_detector.py first, or use --report PATH or --run-detector")
        sys.exit(1)

    print(f"\nLLM Attributor — Danube WQ Anomalies")
    print(f"  Report    : {report_path}")
    print(f"  Z threshold (LLM): {args.z_high}")
    print(f"  Model     : {MODEL_ID}")

    with open(report_path) as f:
        events = json.load(f)

    print(f"  Loaded {len(events)} anomaly events")

    if not events:
        print("  No events to process.")
        return

    # Sort by Z-score descending so highest severity events are processed first
    events.sort(key=lambda e: abs(e.get("roll_z", 0)), reverse=True)

    enriched = enrich_report(
        events,
        api_key=api_key,
        high_thresh=args.z_high,
        smoke=args.smoke,
    )

    if args.smoke:
        print("\n[SMOKE] Result for highest-severity event:")
        high = [e for e in enriched if e.get("llm_primary_source")]
        if high:
            ev = high[0]
            print(f"  Date      : {ev['date']}")
            print(f"  Station   : {ev['station_id']}")
            print(f"  Parameter : {ev['param']}")
            print(f"  Z-score   : {ev['roll_z']}")
            print(f"  LLM source: {ev.get('llm_primary_source', 'N/A')}")
            print(f"  Confidence: {ev.get('llm_confidence', 'N/A')}")
            print(f"  Action    : {ev.get('llm_suggested_action', 'N/A')}")
        print("\n[SMOKE] File not written.")
        return

    out_path = args.out or os.path.join(
        os.path.dirname(report_path), "llm_attributed_report.json"
    )

    with open(out_path, "w") as f:
        json.dump(enriched, f, indent=2, default=str)

    n_llm = sum(1 for e in enriched if e.get("llm_primary_source"))
    print(f"\n  {n_llm} events enriched by LLM")
    print(f"  Saved → {out_path}")

    # ── CSV summary ────────────────────────────────────────────────────────────
    import pandas as pd
    llm_events = [e for e in enriched if e.get("llm_primary_source")]
    if llm_events:
        rows = []
        for e in llm_events:
            rows.append({
                "date":            e.get("date"),
                "station_id":      e.get("station_id"),
                "parameter":       e.get("param"),
                "value":           e.get("value"),
                "roll_z":          e.get("roll_z"),
                "anomaly_type":    e.get("anomaly_type"),
                "rule_attribution": "; ".join(
                    f["name"] for f in e.get("attributed_industries", [])
                ) or "None",
                "llm_primary_source":    e.get("llm_primary_source"),
                "llm_confidence":        e.get("llm_confidence"),
                "llm_suggested_action":  e.get("llm_suggested_action"),
            })
        csv_path = os.path.join(os.path.dirname(report_path), "llm_attributed_report.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"  CSV    → {csv_path}")


if __name__ == "__main__":
    main()
