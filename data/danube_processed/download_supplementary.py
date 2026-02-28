"""
Download supplementary hydro-meteorological data for the 7 Serbian Danube
mainstem stations.

Sources:
  - Weather:    NASA POWER API (MERRA-2 / ERA5 reanalysis, free, no key)
  - Discharge:  Open-Meteo Flood API (GloFAS reanalysis v4, free, no key)

Date range: 2013-01-01 → 2023-12-31
  (covers all co-measured DO/EC/TP windows)

Outputs (all in the same folder as this script):
  weather.csv      — daily precip, air temp, cloud cover, humidity, radiation
  discharge.csv    — daily river discharge (m³/s) from GloFAS model
"""

import requests
import csv
import os
import time
from datetime import datetime

OUT = os.path.dirname(os.path.abspath(__file__))

STATIONS = {
    'SRB00001': {'name': 'Bezdan',           'lat': 45.8542, 'lon': 18.8586},
    'SRB00040': {'name': 'Bogojevo',         'lat': 45.5291, 'lon': 19.0780},
    'SRB00002': {'name': 'Novi Sad',         'lat': 45.2244, 'lon': 19.8419},
    'SRB00003': {'name': 'Zemun',            'lat': 44.8489, 'lon': 20.4172},
    'SRB00041': {'name': 'Smederevo',        'lat': 44.6960, 'lon': 20.9592},
    'SRB00005': {'name': 'Banatska Palanka', 'lat': 44.8244, 'lon': 21.3450},
    'SRB00006': {'name': 'Tekija',           'lat': 44.6961, 'lon': 22.4113},
}

START_DATE = '2013-01-01'
END_DATE   = '2023-12-31'

# NASA POWER parameter codes → friendly column names
NASA_PARAMS = {
    'PRECTOTCORR':      'precipitation_mm',      # mm/day — runoff proxy
    'T2M':              'temperature_mean_c',     # °C mean — water temp proxy
    'T2M_MAX':          'temperature_max_c',
    'T2M_MIN':          'temperature_min_c',
    'RH2M':             'relative_humidity_pct',  # %
    'WS2M':             'wind_speed_ms',          # m/s
    'CLOUD_AMT':        'cloud_cover_pct',        # % — satellite quality proxy
    'ALLSKY_SFC_SW_DWN':'solar_radiation_mjm2',   # MJ/m²/day — algal growth driver
}


# ── Weather (NASA POWER) ──────────────────────────────────────────────────────

def fetch_weather(lat, lon):
    url = 'https://power.larc.nasa.gov/api/temporal/daily/point'
    params = {
        'parameters': ','.join(NASA_PARAMS.keys()),
        'community':  'AG',
        'longitude':  lon,
        'latitude':   lat,
        'start':      START_DATE.replace('-', ''),
        'end':        END_DATE.replace('-', ''),
        'format':     'JSON',
    }
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    return r.json()


def fetch_discharge(station_id, lat, lon):
    """GloFAS reanalysis via Open-Meteo Flood API.
    Uses corrected river-channel grid coordinates (offset from station coords)
    to snap to the Danube main channel in the GloFAS 0.1° grid.
    """
    # Corrected GloFAS cell coordinates per station (found by scanning ±0.3° grid)
    GLOFAS_COORDS = {
        'SRB00001': (45.5542, 18.9586),
        'SRB00040': (45.2291, 19.2780),
        'SRB00002': (45.2244, 19.7419),
        'SRB00003': (44.7489, 20.6172),
        'SRB00041': (44.7960, 21.1592),
        'SRB00005': (44.8244, 21.3450),
        'SRB00006': (44.4961, 22.1113),
    }
    qlat, qlon = GLOFAS_COORDS.get(station_id, (lat, lon))
    url = 'https://flood-api.open-meteo.com/v1/flood'
    params = {
        'latitude':   qlat,
        'longitude':  qlon,
        'start_date': START_DATE,
        'end_date':   END_DATE,
        'daily':      'river_discharge',
        'models':     'seamless_v4',
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


# ── Write CSVs ────────────────────────────────────────────────────────────────

weather_rows = []
discharge_rows = []

for station_id, info in STATIONS.items():
    name = info['name']
    lat, lon = info['lat'], info['lon']
    print(f"Fetching {name} ({station_id})...")

    # --- weather (NASA POWER) ---
    try:
        wdata = fetch_weather(lat, lon)
        params_data = wdata['properties']['parameter']
        # dates are keys in YYYYMMDD format — normalise to YYYY-MM-DD
        raw_dates = sorted(params_data[list(NASA_PARAMS.keys())[0]].keys())
        for raw_date in raw_dates:
            date_str = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
            row = {
                'Station_ID':   station_id,
                'Station_Name': name,
                'Date':         date_str,
            }
            for nasa_code, col_name in NASA_PARAMS.items():
                val = params_data.get(nasa_code, {}).get(raw_date, '')
                # NASA uses -999 as fill value
                row[col_name] = '' if val == -999 else val
            weather_rows.append(row)
        print(f"  weather: {len(raw_dates)} days OK")
    except Exception as e:
        print(f"  weather FAILED: {e}")

    time.sleep(0.5)   # polite rate limit

    # --- discharge ---
    try:
        ddata = fetch_discharge(station_id, lat, lon)
        daily = ddata.get('daily', {})
        dates = daily.get('time', [])
        vals  = daily.get('river_discharge', [])
        for i, date in enumerate(dates):
            discharge_rows.append({
                'Station_ID':       station_id,
                'Station_Name':     name,
                'Date':             date,
                'river_discharge_m3s': vals[i] if i < len(vals) else '',
            })
        print(f"  discharge: {len(dates)} days OK")
    except Exception as e:
        print(f"  discharge FAILED: {e}")

    time.sleep(0.5)

# ── Save ──────────────────────────────────────────────────────────────────────

weather_path = os.path.join(OUT, 'weather.csv')
weather_cols = ['Station_ID', 'Station_Name', 'Date'] + list(NASA_PARAMS.values())
with open(weather_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=weather_cols)
    writer.writeheader()
    writer.writerows(weather_rows)
print(f"\nSaved: weather.csv  ({len(weather_rows)} rows)")

discharge_path = os.path.join(OUT, 'discharge.csv')
with open(discharge_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['Station_ID','Station_Name','Date','river_discharge_m3s'])
    writer.writeheader()
    writer.writerows(discharge_rows)
print(f"Saved: discharge.csv  ({len(discharge_rows)} rows)")

print('\nDone.')
