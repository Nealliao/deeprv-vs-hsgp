# Data Sources

All data is from official, citable European sources. Cases and population are
from the same statistical system (ECDC uses Eurostat population for its rates),
so the exposure offset is internally consistent.

## 1. Measles cases — ECDC

- **Source**: ECDC Surveillance Atlas of Infectious Diseases.
- **Indicator**: "Reported confirmed cases", monthly, by country, unit = N (count).
- **Raw file**: `ECDC_surveillance_data_Measles.csv` (HealthTopic=Measles).
- **Raw extent**: 1999-01 to 2026-04, 328 months, 30 countries + an
  `EUEEAUK_21` EU/EEA/UK aggregate row (dropped).
- **Window used**: **2010-01 to 2019-12** (120 months). Pre-COVID — the
  2020-2022 near-zero period is an exogenous shock (control measures +
  reporting disruption) a smooth GP cannot model. Gives **29 countries** (LI
  has no data in this window) × 120 months; 0.9% missing cells (masked out of
  the likelihood).
- **Character**: 54% zeros, overdispersion var/mean ≈ 1134 → Negative Binomial.

## 2. Population (exposure) — Eurostat

- **Source**: Eurostat `demo_pjan` — "Population on 1 January by age and sex"
  (sex=Total, age=Total).
- **Why Eurostat**: it is the same population denominator ECDC uses to compute
  incidence rates, so the exposure offset E_c is consistent and citable.
- **How obtained**: Eurostat dissemination REST API (JSON-stat), saved to
  `data/eurostat_population_2010_2019.csv`. Query (per year 2010-2019):
  ```
  https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/demo_pjan?format=JSON&lang=EN&sex=T&age=TOTAL&time=YYYY
  ```
- **Coverage**: all 29 countries present, yearly 2010-2019 (time-varying
  exposure, not a fixed year). 1 missing cell filled by neighbour-year ffill/bfill.

## 3. Country coordinates

- **Approximate geometric centroids** (lon/lat), hardcoded in script 110.
  Low impact on a smooth spatial GP. For a fully citable map, swap for Eurostat
  GISCO / Natural Earth country centroids.
- Projected with an **equirectangular (cos-latitude) correction** so Euclidean
  distance approximates great-circle distance: raw lon/lat correlates 0.94 with
  true great-circle distance, the corrected projection 0.98 (raw over-stretches
  east-west and squashes north-south). Then centred and scaled so the spatial
  span ≈ 1 (comparable to the synthetic benchmark, so the spatial prior carries
  over). Resulting geometry: min nearest-neighbour 0.083, span 1.33.

## Citation notes for the thesis

- Cases: cite ECDC Surveillance Atlas (with access date).
- Population: cite Eurostat `demo_pjan` (with extraction date / DOI from the API
  response `extension.annotation`).
- Coordinates: state they are approximate centroids (or replace with GISCO).
