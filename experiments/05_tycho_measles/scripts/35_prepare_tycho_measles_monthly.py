from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TYCHO_ZIP = RAW_DIR / "ProjectTycho_Level2_v1.1.0_0.zip"

START_DATE = "1935-01-01"
END_DATE = "2001-12-31"


STATE_CENTROIDS = {
    "AL": ("Alabama", 32.806671, -86.791130),
    "AZ": ("Arizona", 33.729759, -111.431221),
    "AR": ("Arkansas", 34.969704, -92.373123),
    "CA": ("California", 36.116203, -119.681564),
    "CO": ("Colorado", 39.059811, -105.311104),
    "CT": ("Connecticut", 41.597782, -72.755371),
    "DE": ("Delaware", 39.318523, -75.507141),
    "DC": ("District of Columbia", 38.897438, -77.026817),
    "FL": ("Florida", 27.766279, -81.686783),
    "GA": ("Georgia", 33.040619, -83.643074),
    "ID": ("Idaho", 44.240459, -114.478828),
    "IL": ("Illinois", 40.349457, -88.986137),
    "IN": ("Indiana", 39.849426, -86.258278),
    "IA": ("Iowa", 42.011539, -93.210526),
    "KS": ("Kansas", 38.526600, -96.726486),
    "KY": ("Kentucky", 37.668140, -84.670067),
    "LA": ("Louisiana", 31.169546, -91.867805),
    "ME": ("Maine", 44.693947, -69.381927),
    "MD": ("Maryland", 39.063946, -76.802101),
    "MA": ("Massachusetts", 42.230171, -71.530106),
    "MI": ("Michigan", 43.326618, -84.536095),
    "MN": ("Minnesota", 45.694454, -93.900192),
    "MS": ("Mississippi", 32.741646, -89.678696),
    "MO": ("Missouri", 38.456085, -92.288368),
    "MT": ("Montana", 46.921925, -110.454353),
    "NE": ("Nebraska", 41.125370, -98.268082),
    "NV": ("Nevada", 38.313515, -117.055374),
    "NH": ("New Hampshire", 43.452492, -71.563896),
    "NJ": ("New Jersey", 40.298904, -74.521011),
    "NM": ("New Mexico", 34.840515, -106.248482),
    "NY": ("New York", 42.165726, -74.948051),
    "NC": ("North Carolina", 35.630066, -79.806419),
    "ND": ("North Dakota", 47.528912, -99.784012),
    "OH": ("Ohio", 40.388783, -82.764915),
    "OK": ("Oklahoma", 35.565342, -96.928917),
    "OR": ("Oregon", 44.572021, -122.070938),
    "PA": ("Pennsylvania", 40.590752, -77.209755),
    "RI": ("Rhode Island", 41.680893, -71.511780),
    "SC": ("South Carolina", 33.856892, -80.945007),
    "SD": ("South Dakota", 44.299782, -99.438828),
    "TN": ("Tennessee", 35.747845, -86.692345),
    "TX": ("Texas", 31.054487, -97.563461),
    "UT": ("Utah", 40.150032, -111.862434),
    "VT": ("Vermont", 44.045876, -72.710686),
    "VA": ("Virginia", 37.769337, -78.169968),
    "WA": ("Washington", 47.400902, -121.490494),
    "WV": ("West Virginia", 38.491226, -80.954453),
    "WI": ("Wisconsin", 44.268543, -89.616508),
    "WY": ("Wyoming", 42.755966, -107.302490),
}


def read_state_measles_rows():
    frames = []
    with zipfile.ZipFile(TYCHO_ZIP) as archive:
        csv_name = archive.namelist()[0]
        with archive.open(csv_name) as file:
            for chunk in pd.read_csv(file, chunksize=200_000, low_memory=False):
                chunk.columns = [column.strip() for column in chunk.columns]
                for column in ["country", "state", "loc_type", "disease", "event"]:
                    chunk[column] = chunk[column].astype(str).str.strip()

                mask = (
                    (chunk["country"] == "US")
                    & (chunk["loc_type"] == "STATE")
                    & (chunk["event"] == "CASES")
                    & (chunk["disease"].str.upper() == "MEASLES")
                    & (chunk["state"].isin(STATE_CENTROIDS))
                )
                if not mask.any():
                    continue

                keep = chunk.loc[
                    mask,
                    ["epi_week", "state", "loc", "number", "from_date", "to_date", "url"],
                ].copy()
                keep["cases"] = pd.to_numeric(keep["number"], errors="coerce").fillna(0.0)
                keep["week_start"] = pd.to_datetime(keep["from_date"], errors="coerce")
                keep["week_end"] = pd.to_datetime(keep["to_date"], errors="coerce")
                keep = keep[
                    (keep["week_start"] >= START_DATE)
                    & (keep["week_start"] <= END_DATE)
                ]
                frames.append(keep.drop(columns=["number", "from_date", "to_date"]))

    if not frames:
        raise ValueError("No Project Tycho state level measles case rows were found.")
    return pd.concat(frames, ignore_index=True)


def make_regions():
    records = []
    for index, (state, (name, latitude, longitude)) in enumerate(sorted(STATE_CENTROIDS.items())):
        records.append(
            {
                "state_index": index,
                "state": state,
                "state_name": name,
                "latitude": latitude,
                "longitude": longitude,
            }
        )
    regions = pd.DataFrame(records)
    regions["x_coord"] = (regions["longitude"] - regions["longitude"].min()) / (
        regions["longitude"].max() - regions["longitude"].min()
    )
    regions["y_coord"] = (regions["latitude"] - regions["latitude"].min()) / (
        regions["latitude"].max() - regions["latitude"].min()
    )
    return regions


def build_monthly_panel(weekly, regions):
    weekly["month"] = weekly["week_start"].dt.to_period("M").dt.to_timestamp()
    monthly_observed = (
        weekly.groupby(["state", "month"], as_index=False)
        .agg(cases=("cases", "sum"), reporting_weeks=("epi_week", "nunique"))
    )

    states = pd.DataFrame({"state": sorted(STATE_CENTROIDS)})
    months = pd.DataFrame({"month": pd.date_range(START_DATE, END_DATE, freq="MS")})
    balanced = states.merge(months, how="cross")
    panel = balanced.merge(monthly_observed, on=["state", "month"], how="left")
    panel["observed_report"] = panel["cases"].notna().astype(int)
    panel["cases"] = panel["cases"].fillna(0.0).astype(int)
    panel["reporting_weeks"] = panel["reporting_weeks"].fillna(0).astype(int)
    panel = panel.merge(regions, on="state", how="left")

    panel = panel.sort_values(["state_index", "month"]).reset_index(drop=True)
    first_month = panel["month"].min()
    panel["time_index"] = (
        (panel["month"].dt.year - first_month.year) * 12
        + (panel["month"].dt.month - first_month.month)
    )
    panel["year"] = panel["month"].dt.year
    panel["month_of_year"] = panel["month"].dt.month

    state_totals = panel.groupby("state")["cases"].transform("sum")
    month_totals = panel.groupby("month")["cases"].transform("sum")
    grand_total = panel["cases"].sum()
    expected = state_totals * month_totals / grand_total
    panel["expected_cases_separable"] = expected.clip(lower=1e-3)
    panel["log_expected_cases"] = np.log(panel["expected_cases_separable"])
    panel["post_vaccine"] = (panel["month"] >= pd.Timestamp("1963-01-01")).astype(int)
    return panel


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    weekly = read_state_measles_rows()
    regions = make_regions()
    panel = build_monthly_panel(weekly, regions)

    weekly_path = PROCESSED_DIR / "tycho_measles_state_weekly.csv"
    panel_path = PROCESSED_DIR / "tycho_measles_state_monthly_panel.csv"
    regions_path = PROCESSED_DIR / "tycho_measles_state_regions.csv"
    summary_path = PROCESSED_DIR / "tycho_measles_state_monthly_summary.csv"

    summary = pd.DataFrame(
        [
            {
                "start_month": panel["month"].min().strftime("%Y-%m"),
                "end_month": panel["month"].max().strftime("%Y-%m"),
                "n_states": panel["state"].nunique(),
                "n_months": panel["month"].nunique(),
                "n_observations": len(panel),
                "total_cases": int(panel["cases"].sum()),
                "mean_monthly_cases": float(panel["cases"].mean()),
                "zero_fraction": float((panel["cases"] == 0).mean()),
                "no_report_fraction": float((panel["observed_report"] == 0).mean()),
                "mean_reporting_weeks": float(panel["reporting_weeks"].mean()),
                "pre_vaccine_cases": int(panel.loc[panel["post_vaccine"] == 0, "cases"].sum()),
                "post_vaccine_cases": int(panel.loc[panel["post_vaccine"] == 1, "cases"].sum()),
            }
        ]
    )

    weekly.to_csv(weekly_path, index=False)
    panel.to_csv(panel_path, index=False)
    regions.to_csv(regions_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(summary.round(4).to_string(index=False))
    print(weekly_path)
    print(panel_path)
    print(regions_path)
    print(summary_path)


if __name__ == "__main__":
    main()
