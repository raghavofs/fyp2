"""
Extract and clean GFQA_v3 data for 7 Serbian Danube mainstem stations.
Outputs 8 parameter-specific CSVs to the same folder as this script.

Stations (west → east along Danube mainstem through Serbia):
  SRB00001 - Bezdan
  SRB00040 - Bogojevo
  SRB00002 - Novi Sad
  SRB00003 - Zemun (near Belgrade)
  SRB00041 - Smederevo
  SRB00005 - Banatska Palanka
  SRB00006 - Tekija (Iron Gate gorge)
"""

import csv
import os

# ── Config ────────────────────────────────────────────────────────────────────

SRC = os.path.join(os.path.dirname(__file__), '..', 'GFQA_v3')
OUT = os.path.dirname(__file__)

STATIONS = {
    'SRB00001': 'Bezdan',
    'SRB00040': 'Bogojevo',
    'SRB00002': 'Novi Sad',
    'SRB00003': 'Zemun',
    'SRB00041': 'Smederevo',
    'SRB00005': 'Banatska Palanka',
    'SRB00006': 'Tekija',
}

# Quality values to keep (exclude Suspect / Contamination)
GOOD_QUALITY = {'Good', 'Fair', 'Unknown', 'Estimated', 'Pending review'}

# Each entry: output filename, source file, parameter codes to keep
EXTRACTIONS = [
    (
        'dissolved_oxygen.csv',
        'Dissolved_Gas.csv',
        {'O2-Dis', 'O2-Dis-Sat'},
    ),
    (
        'electrical_conductance.csv',
        'Electrical_Conductance.csv',
        {'EC'},
    ),
    (
        'phosphorus.csv',
        'Phosphorus.csv',
        {'TP', 'DRP', 'TDP', 'TIP', 'TRP'},
    ),
    (
        'nitrogen_other.csv',
        'Other_Nitrogen.csv',
        {'TN', 'NH4N', 'NH3N', 'TKN', 'TON'},
    ),
    (
        'nitrogen_oxidized.csv',
        'Oxidized_Nitrogen.csv',
        {'NO3N', 'NO2N', 'NOxN'},
    ),
    (
        'chlorophyll.csv',
        'Pigment.csv',
        {'Chl-a'},
    ),
    (
        'oxygen_demand.csv',
        'Oxygen_Demand.csv',
        {'BOD', 'COD'},
    ),
    (
        'potassium.csv',
        'Potassium.csv',
        {'K-Dis', 'K-Tot'},
    ),
]

OUTPUT_HEADER = [
    'Station_ID',
    'Station_Name',
    'Sample_Date',
    'Parameter_Code',
    'Value_Flag',
    'Value',
    'Unit',
    'Data_Quality',
]

# ── Extraction ────────────────────────────────────────────────────────────────

for out_name, src_name, param_codes in EXTRACTIONS:
    src_path = os.path.join(SRC, src_name)
    out_path = os.path.join(OUT, out_name)

    rows_written = 0
    rows_skipped_quality = 0

    with open(src_path, 'r', encoding='utf-8', errors='replace') as fin, \
         open(out_path, 'w', newline='', encoding='utf-8') as fout:

        reader = csv.reader(fin)
        writer = csv.writer(fout)
        writer.writerow(OUTPUT_HEADER)

        src_header = next(reader)
        # Column indices in source
        # GEMS Station Number,Sample Date,Sample Time,Depth,Parameter Code,
        # Analysis Method Code,Value Flags,Value,Unit,Data Quality,...
        col = {h: i for i, h in enumerate(src_header)}
        i_station  = col['GEMS Station Number']
        i_date     = col['Sample Date']
        i_param    = col['Parameter Code']
        i_flag     = col['Value Flags']
        i_value    = col['Value']
        i_unit     = col['Unit']
        i_quality  = col['Data Quality']

        for row in reader:
            if len(row) <= i_quality:
                continue
            station = row[i_station]
            if station not in STATIONS:
                continue
            param = row[i_param]
            if param not in param_codes:
                continue
            quality = row[i_quality].strip()
            if quality not in GOOD_QUALITY:
                rows_skipped_quality += 1
                continue

            writer.writerow([
                station,
                STATIONS[station],
                row[i_date],
                param,
                row[i_flag].strip(),
                row[i_value].strip(),
                row[i_unit].strip(),
                quality,
            ])
            rows_written += 1

    print(f'{out_name:<35}  {rows_written:>6} rows  (skipped {rows_skipped_quality} suspect/contaminated)')

print('\nDone.')
