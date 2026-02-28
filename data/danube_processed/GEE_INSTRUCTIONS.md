# Google Earth Engine — Satellite Imagery Instructions
## Danube Mainstem (Serbia), 7 Monitoring Stations

---

## Station Coordinates & Buffer

| Station ID  | Name              |   Latitude |  Longitude | Buffer |
|-------------|-------------------|------------|------------|--------|
| SRB00001    | Bezdan            |  45.8542   |  18.8586   | 500 m  |
| SRB00040    | Bogojevo          |  45.5291   |  19.0780   | 500 m  |
| SRB00002    | Novi Sad          |  45.2244   |  19.8419   | 500 m  |
| SRB00003    | Zemun             |  44.8489   |  20.4172   | 500 m  |
| SRB00041    | Smederevo         |  44.6960   |  20.9592   | 500 m  |
| SRB00005    | Banatska Palanka  |  44.8244   |  21.3450   | 500 m  |
| SRB00006    | Tekija            |  44.6961   |  22.4113   | 500 m  |

> The Danube is 400–900 m wide in Serbia. A 500 m circular buffer centred
> on each station coordinate will capture the main channel.
> Increase to 750 m if cloud masking is aggressive and you lose too many pixels.

---

## Date Range

| Purpose                          | Start Date   | End Date     |
|----------------------------------|--------------|--------------|
| Core model training (DO/EC/TP)   | 2013-01-01   | 2023-12-31   |
| With 30-day look-back window     | 2012-12-01   | 2023-12-31   |
| Full measurement record (N/BOD)  | 2000-01-01   | 2023-12-31   |

**Recommended: 2012-12-01 → 2023-12-31**
This covers the 30-day temporal window before the first 2013 measurement.

---

## Satellite Collections to Use

### Primary: Sentinel-2 (2015-06-23 → present)
- Collection: `COPERNICUS/S2_SR_HARMONIZED`  (surface reflectance, cloud-masked)
- Revisit: ~5 days
- Spatial resolution: 10–20 m
- **Use for: 2015-06-23 → 2023-12-31**

### Secondary: Landsat 8 OLI (2013-04-11 → present)
- Collection: `LANDSAT/LC08/C02/T1_L2`  (Tier 1, Level-2 surface reflectance)
- Revisit: ~16 days
- Spatial resolution: 30 m
- **Use for: 2013-04-11 → 2023-12-31** (gap-fill where Sentinel-2 unavailable)
- **Required for: 2012-12-01 → 2015-06-22** (no Sentinel-2 before this date)

> Note: Landsat 7 ETM+ (pre-2013) has scan-line corrector failure artefacts
> (diagonal stripes) from 2003 onwards. Avoid unless you extend to pre-2013.

---

## Spectral Bands to Extract

### Sentinel-2 bands
| Band  | Name         | Wavelength | Resolution |
|-------|--------------|------------|------------|
| B2    | Blue         | 490 nm     | 10 m       |
| B3    | Green        | 560 nm     | 10 m       |
| B4    | Red          | 665 nm     | 10 m       |
| B5    | Red Edge 1   | 705 nm     | 20 m       |
| B6    | Red Edge 2   | 740 nm     | 20 m       |
| B7    | Red Edge 3   | 783 nm     | 20 m       |
| B8    | NIR          | 842 nm     | 10 m       |
| B11   | SWIR 1       | 1610 nm    | 20 m       |
| B12   | SWIR 2       | 2190 nm    | 20 m       |

### Landsat 8 equivalent bands
| Band  | Name         | Wavelength | Notes              |
|-------|--------------|------------|--------------------|
| SR_B2 | Blue         | 482 nm     |                    |
| SR_B3 | Green        | 562 nm     |                    |
| SR_B4 | Red          | 655 nm     |                    |
| SR_B5 | NIR          | 865 nm     |                    |
| SR_B6 | SWIR 1       | 1609 nm    |                    |
| SR_B7 | SWIR 2       | 2201 nm    |                    |

> Landsat 8 has no Red Edge bands. For the spatial (GA-RF) stream,
> you can either skip Red Edge for Landsat-8 dates or impute with NaN.

---

## Derived Indices to Compute in GEE
These can be computed directly in GEE before export to reduce download size.

| Index | Formula                           | Relevance                         |
|-------|-----------------------------------|-----------------------------------|
| NDVI  | (NIR - Red) / (NIR + Red)         | Riparian vegetation, algal growth |
| NDWI  | (Green - NIR) / (Green + NIR)     | Open water extent                 |
| MNDWI | (Green - SWIR1) / (Green + SWIR1) | Water quality, turbidity          |
| NDTI  | (Red - Green) / (Red + Green)     | Turbidity / suspended sediment    |
| FAI   | NIR - (Red + (SWIR1-Red)*0.4)     | Floating algae index              |
| EVI   | 2.5*(NIR-Red)/(NIR+6*Red-7.5*Blue+1) | Algal biomass                  |

---

## Recommended GEE Export Format

Export one row per (station, image_date) with:
- `station_id`  (e.g. `SRB00001`)
- `image_date`  (YYYY-MM-DD)
- `satellite`   (`S2` or `L8`)
- One column per spectral band (mean reflectance within buffer)
- One column per derived index (mean within buffer)
- `valid_pixels` (count of unmasked pixels in buffer — quality indicator)
- `cloud_fraction` (fraction of buffer masked as cloud)

Export to Google Drive as CSV, one file per station, or one combined file.

---

## Measurement Dates Reference Files

Per-station date lists (for targeted image filtering):
- `measurement_dates_SRB00001.txt`  — 222 dates
- `measurement_dates_SRB00040.txt`  — 211 dates
- `measurement_dates_SRB00002.txt`  — 225 dates
- `measurement_dates_SRB00003.txt`  — 219 dates
- `measurement_dates_SRB00041.txt`  — 219 dates
- `measurement_dates_SRB00005.txt`  — 212 dates
- `measurement_dates_SRB00006.txt`  — 203 dates
- `all_measurement_dates.txt`       — 1,206 unique dates (union)

You do NOT need to limit GEE to only these exact dates.
Request the full 2012-12-01 → 2023-12-31 continuous time series.
The model will pair each water quality measurement with the nearest
cloud-free image within ±3 days, or take a 30-day composite window.
