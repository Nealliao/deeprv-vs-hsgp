"""
110_prepare_ecdc_measles_panel.py
================================================================================
Prepare the ECDC measles panel (2010-2019) for the joint-9HP HSGP benchmark.

Input:  ~/Desktop/ECDC_surveillance_data_Measles.csv
        (ECDC surveillance: "Reported confirmed cases", monthly, by country)
Output: data/processed/ecdc_measles_panel.csv

Steps:
  - keep HealthTopic=Measles, drop the EUEEAUK_21 aggregate row
  - window 2010-01 .. 2019-12 (120 months, pre-COVID, avoids 2020-22 anomaly)
  - parse NumValue ('-' -> missing); keep an `observed` flag
  - attach country centroid (lon/lat) + 2015 population (exposure offset)
  - standardise coordinates so the spatial span ~ 1 (comparable to the synthetic
    benchmark, so the Betancourt spatial prior logic carries over)
  - build contiguous state_index (0..S-1) and time_index (0..119)

Population: Eurostat demo_pjan (population on 1 January, total), yearly
2010-2019, downloaded via the Eurostat REST API and saved to
data/processed/eurostat_population_2010_2019.csv. This is the same source
ECDC uses for its incidence rates, so the exposure offset E_c is consistent
and citable. Country centroids (lon/lat) are approximate geometric centres
(low impact on a smooth spatial GP); swap for GISCO centroids if needed.
"""

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import numpy as np
import pandas as pd

CSV_IN  = Path.home() / "Desktop" / "ECDC_surveillance_data_Measles.csv"
POP_CSV = PROJECT_ROOT / "data" / "processed" / "eurostat_population_2010_2019.csv"
OUT     = PROJECT_ROOT / "data" / "processed" / "ecdc_measles_panel.csv"

YEAR_LO, YEAR_HI = 2010, 2019

# country code -> (lat, lon) approximate geometric centroid (deg).
# Population comes from Eurostat (POP_CSV), not from here.
META = {
    "AT": (47.5, 14.5),  "BE": (50.8, 4.5),  "BG": (42.7, 25.3), "CY": (35.1, 33.4),
    "CZ": (49.8, 15.5),  "DE": (51.2, 10.4), "DK": (56.0, 9.5),  "EE": (58.6, 25.0),
    "EL": (39.0, 22.0),  "ES": (40.0, -3.7), "FI": (64.0, 26.0), "FR": (46.6, 2.2),
    "HR": (45.1, 15.2),  "HU": (47.2, 19.5), "IE": (53.4, -8.0), "IS": (64.9, -19.0),
    "IT": (42.8, 12.8),  "LI": (47.2, 9.5),  "LT": (55.2, 23.9), "LU": (49.8, 6.1),
    "LV": (56.9, 24.6),  "MT": (35.9, 14.4), "NL": (52.1, 5.3),  "NO": (60.5, 8.5),
    "PL": (51.9, 19.1),  "PT": (39.6, -8.0), "RO": (45.9, 25.0), "SE": (62.0, 15.0),
    "SI": (46.1, 14.8),  "SK": (48.7, 19.5),
}


def main():
    d = pd.read_csv(CSV_IN)
    d = d[d["RegionCode"] != "EUEEAUK_21"].copy()
    d["cases"] = pd.to_numeric(d["NumValue"], errors="coerce")   # '-' -> NaN
    d["year"]  = d["Time"].str[:4].astype(int)
    d["month_of_year"] = d["Time"].str[5:7].astype(int)
    d = d[(d["year"] >= YEAR_LO) & (d["year"] <= YEAR_HI)].copy()

    # full country × month grid (so missing months are explicit)
    countries = sorted(c for c in d["RegionCode"].unique() if c in META)
    months = sorted(d["Time"].unique())
    name_map = d.drop_duplicates("RegionCode").set_index("RegionCode")["RegionName"].to_dict()
    grid = pd.MultiIndex.from_product([countries, months], names=["RegionCode", "Time"])
    panel = d.set_index(["RegionCode", "Time"]).reindex(grid).reset_index()

    panel["observed"] = panel["cases"].notna().astype(int)
    panel["state_index"] = panel["RegionCode"].map({c: i for i, c in enumerate(countries)})
    panel["time_index"]  = panel["Time"].map({m: i for i, m in enumerate(months)})
    panel["country_name"] = panel["RegionCode"].map(name_map)
    panel["month"] = pd.to_datetime(panel["Time"] + "-01")
    panel["month_of_year"] = panel["month"].dt.month

    # exposure: Eurostat yearly population (time-varying offset)
    panel["year"] = panel["month"].dt.year
    pop = pd.read_csv(POP_CSV).sort_values(["RegionCode", "year"])
    pop["population"] = pop.groupby("RegionCode")["population"].transform(
        lambda s: s.ffill().bfill())          # fill the 1 missing cell from neighbours
    panel = panel.merge(pop, on=["RegionCode", "year"], how="left")
    panel["log_population"] = np.log(panel["population"])

    # coordinates (approximate centroids)
    lat = panel["RegionCode"].map(lambda c: META[c][0])
    lon = panel["RegionCode"].map(lambda c: META[c][1])

    # Project lon/lat to a plane with an equirectangular (cos-latitude) correction
    # so Euclidean distance approximates great-circle distance (corr 0.94 -> 0.98).
    # Without it, 1 deg of longitude (~111 km * cos(lat)) is treated as 1 deg of
    # latitude, over-stretching east-west and squashing north-south distances.
    lat0 = lat.mean()
    x = (lon - lon.mean()) * np.cos(np.deg2rad(lat0))
    y = (lat - lat.mean())
    scale = max(x.max() - x.min(), y.max() - y.min())   # span ~1, keep aspect ratio
    panel["x_coord"] = x / scale
    panel["y_coord"] = y / scale

    panel = panel.sort_values(["state_index", "time_index"]).reset_index(drop=True)
    cols = ["state_index", "RegionCode", "country_name", "time_index", "month",
            "month_of_year", "cases", "observed", "population", "log_population",
            "x_coord", "y_coord"]
    panel = panel[cols]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT, index=False)

    S = panel["state_index"].nunique(); T = panel["time_index"].nunique()
    obs = panel[panel["observed"] == 1]
    print(f"[saved] {OUT}")
    print(f"  countries S={S}  months T={T}  cells={len(panel)}  missing={int((panel['observed']==0).sum())}")
    print(f"  cases (observed): mean={obs['cases'].mean():.1f}  median={obs['cases'].median():.0f}  "
          f"max={obs['cases'].max():.0f}  zero_frac={(obs['cases']==0).mean():.3f}")
    print(f"  overdispersion var/mean={obs['cases'].var()/obs['cases'].mean():.0f}  -> use Negative Binomial")


if __name__ == "__main__":
    main()
