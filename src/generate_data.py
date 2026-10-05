"""
generate_data.py
----------------
Stage 1: Synthetic data generation for the Aquapurity prototype.

ALL DATA IS SYNTHETIC. Generated with a fixed seed so results are
reproducible. Columns mimic CGWB-style well logs, IMD-style rainfall,
and Bhuvan-style terrain / land-use layers for two Tamil Nadu districts:
  - Thanjavur  (alluvial / deltaic aquifer)
  - Pudukkottai (hard-rock plain aquifer)

Do NOT interpret any outputs as real groundwater data.
"""

import math
import os
import random
import sys

import numpy as np
import pandas as pd

# Windows terminals default to the cp1252 code page, which cannot encode the
# box-drawing / arrow characters used in our printed summaries. Force UTF-8
# output so the script runs and prints cleanly on any machine.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── fixed seed everywhere ──────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── output directory ───────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  District meta
# ──────────────────────────────────────────────────────────────────────────────
DISTRICTS = {
    "Thanjavur": {
        "lat_range": (10.60, 10.95),
        "lon_range": (79.10, 79.55),
        "aquifer_types": ["Alluvial", "Deltaic"],
        "n_wells": 60,
        "n_sites": 40,
        # Thanjavur gets heavier NE monsoon rainfall (Oct-Dec)
        "rain_base_mm": 130,
        "rain_ne_peak_mm": 280,
        # mean depth below ground; alluvial wells are shallower
        "wl_mean_m": 4.5,
        # slow declining trend (m/year) – stress indicator
        "wl_trend_m_per_year": 0.18,
    },
    "Pudukkottai": {
        "lat_range": (10.20, 10.55),
        "lon_range": (78.65, 79.10),
        "aquifer_types": ["Crystalline", "Weathered"],
        "n_wells": 60,
        "n_sites": 40,
        "rain_base_mm": 80,
        "rain_ne_peak_mm": 180,
        "wl_mean_m": 8.0,
        "wl_trend_m_per_year": 0.30,
    },
}

# ── 5 years of monthly timestamps ─────────────────────────────────────────────
START = "2018-01"
MONTHS = 60
DATES = pd.date_range(start=START, periods=MONTHS, freq="MS")


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Helper: seasonal rainfall signal (NE monsoon Oct-Dec)
# ──────────────────────────────────────────────────────────────────────────────
def monthly_rainfall(meta: dict, n_months: int, rng: np.random.Generator) -> np.ndarray:
    """
    Returns an (n_months,) array of mm/month rainfall.
    Month index 0 = January 2018.
    NE monsoon peaks Oct (9), Nov (10), Dec (11).
    """
    rain = np.full(n_months, meta["rain_base_mm"], dtype=float)
    for i in range(n_months):
        month = (i % 12)  # 0-based
        if month in (9, 10, 11):                          # Oct, Nov, Dec
            rain[i] += meta["rain_ne_peak_mm"] * rng.uniform(0.7, 1.0)
        elif month in (5, 6, 7):                          # SW monsoon (Jun-Aug, weaker)
            rain[i] += meta["rain_base_mm"] * rng.uniform(0.3, 0.8)
        rain[i] += rng.normal(0, 10)                      # measurement noise
        rain[i] = max(0, rain[i])
    return rain


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Helper: groundwater level time-series per well
# ──────────────────────────────────────────────────────────────────────────────
def well_wl_series(
    meta: dict,
    rainfall: np.ndarray,
    n_months: int,
    rng: np.random.Generator,
    missing_frac: float = 0.03,
) -> np.ndarray:
    """
    Generates water-level (m below ground) time series.

    The level responds inversely to rainfall with a 1-2 month lag
    (higher rain → shallower depth → lower value), plus a slow rising
    trend representing long-term depletion, plus well-specific noise.
    A small fraction of values are set to NaN (missing observations).
    """
    wl = np.zeros(n_months, dtype=float)
    base = meta["wl_mean_m"] + rng.normal(0, 1.0)       # well-specific offset
    trend_monthly = meta["wl_trend_m_per_year"] / 12.0

    for i in range(n_months):
        # seasonal component: rainfall 1-month lag reduces depth
        lag_idx = max(0, i - 1)
        rain_effect = -rainfall[lag_idx] / 300.0          # normalised recharge
        seasonal = 0.8 * math.sin(2 * math.pi * (i % 12) / 12.0)  # annual cycle
        noise = rng.normal(0, 0.25)
        wl[i] = base + trend_monthly * i + rain_effect + seasonal + noise
        wl[i] = max(0.5, wl[i])                           # physically bounded

    # introduce missing values
    miss_idx = rng.choice(n_months, size=int(n_months * missing_frac), replace=False)
    wl[miss_idx] = np.nan
    return wl


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Static features for wells / candidate sites
# ──────────────────────────────────────────────────────────────────────────────
LAND_USE_CLASSES = ["Agricultural", "Scrubland", "Waterbody", "Settlement", "Forest"]

def static_features(district: str, rng: np.random.Generator) -> dict:
    """
    Returns a dict of static hydro-geological / terrain features.
    Pudukkottai (hard rock) gets a lineament_density column that
    matters for fracture-controlled recharge.
    """
    feats: dict = {
        # soil permeability index 1 (low) – 10 (high)
        "soil_permeability": round(rng.uniform(2, 9), 2),
        # slope in degrees; flatter = better recharge
        "slope_deg": round(rng.uniform(0.1, 8.0), 2),
        # land use class
        "land_use": rng.choice(LAND_USE_CLASSES),
        # distance to nearest river/tank (km)
        "dist_to_water_km": round(rng.uniform(0.1, 12.0), 2),
        # whether existing recharge structure is nearby (binary)
        "existing_structure": int(rng.random() < 0.15),
    }
    if district == "Pudukkottai":
        # lineament density (km/km²) – fracture network proxy
        feats["lineament_density"] = round(rng.uniform(0.1, 3.5), 2)
    else:
        feats["lineament_density"] = np.nan   # not applicable for alluvial

    return feats


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Constraint flag for candidate sites
# ──────────────────────────────────────────────────────────────────────────────
def constraint_flag(feats: dict, rng: np.random.Generator) -> dict:
    """
    Simple exclusion rules (checked later in constraints.py):
      - protected  : Forest land use (wildlife sanctuary / reserve)
      - built_up   : Settlement land use
      - saline     : high depth + low permeability proxy (random for prototype)
    Returns flags as integers (1 = constrained, 0 = eligible).
    """
    return {
        "constraint_protected": int(feats["land_use"] == "Forest"),
        "constraint_built_up":  int(feats["land_use"] == "Settlement"),
        "constraint_saline":    int(rng.random() < 0.08),   # 8% random saline flag
    }


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Generate wells
# ──────────────────────────────────────────────────────────────────────────────
def generate_wells() -> pd.DataFrame:
    """Generate ~60 wells per district with static features."""
    rng = np.random.default_rng(SEED)
    rows = []
    for district, meta in DISTRICTS.items():
        aq_types = meta["aquifer_types"]
        n = meta["n_wells"]
        lats = rng.uniform(*meta["lat_range"], size=n)
        lons = rng.uniform(*meta["lon_range"], size=n)
        for i in range(n):
            well_id = f"W_{district[:3].upper()}_{i+1:03d}"
            feats = static_features(district, rng)
            rows.append(
                {
                    "well_id": well_id,
                    "district": district,
                    "lat": round(float(lats[i]), 5),
                    "lon": round(float(lons[i]), 5),
                    "aquifer_type": rng.choice(aq_types),
                    **feats,
                }
            )
    df = pd.DataFrame(rows)
    print(f"[Wells] {len(df)} wells generated "
          f"({df.groupby('district').size().to_dict()})")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Generate monthly water-level and rainfall time series
# ──────────────────────────────────────────────────────────────────────────────
def generate_timeseries(wells: pd.DataFrame):
    """
    Returns:
      wl_df   : long-form DataFrame  [well_id, date, water_level_m]
      rain_df : long-form DataFrame  [district, date, rainfall_mm]
    """
    rng = np.random.default_rng(SEED + 1)
    wl_rows = []
    rain_rows = []

    for district, meta in DISTRICTS.items():
        # district-level monthly rainfall
        rain = monthly_rainfall(meta, MONTHS, rng)
        for j, d in enumerate(DATES):
            rain_rows.append(
                {"district": district, "date": d.strftime("%Y-%m"), "rainfall_mm": round(rain[j], 1)}
            )

        # per-well water level driven by rainfall
        dist_wells = wells[wells["district"] == district]
        for _, well in dist_wells.iterrows():
            wl = well_wl_series(meta, rain, MONTHS, rng)
            for j, d in enumerate(DATES):
                wl_rows.append(
                    {
                        "well_id": well["well_id"],
                        "date": d.strftime("%Y-%m"),
                        "water_level_m": round(float(wl[j]), 3) if not np.isnan(wl[j]) else None,
                    }
                )

    wl_df = pd.DataFrame(wl_rows)
    rain_df = pd.DataFrame(rain_rows)
    missing_pct = wl_df["water_level_m"].isna().mean() * 100
    print(f"[WL series] {len(wl_df)} rows | {missing_pct:.1f}% missing values")
    print(f"[Rainfall]  {len(rain_df)} rows")
    return wl_df, rain_df


# ──────────────────────────────────────────────────────────────────────────────
# 8.  Generate candidate recharge sites
# ──────────────────────────────────────────────────────────────────────────────
def generate_sites() -> pd.DataFrame:
    """
    ~40 candidate recharge sites per district.
    Same static features as wells plus constraint flags.
    """
    rng = np.random.default_rng(SEED + 2)
    rows = []
    for district, meta in DISTRICTS.items():
        n = meta["n_sites"]
        lats = rng.uniform(*meta["lat_range"], size=n)
        lons = rng.uniform(*meta["lon_range"], size=n)
        for i in range(n):
            site_id = f"S_{district[:3].upper()}_{i+1:03d}"
            feats = static_features(district, rng)
            flags = constraint_flag(feats, rng)
            rows.append(
                {
                    "site_id": site_id,
                    "district": district,
                    "lat": round(float(lats[i]), 5),
                    "lon": round(float(lons[i]), 5),
                    **feats,
                    **flags,
                }
            )
    df = pd.DataFrame(rows)
    constrained = (
        df[["constraint_protected", "constraint_built_up", "constraint_saline"]]
        .any(axis=1)
        .sum()
    )
    print(
        f"[Sites] {len(df)} candidate sites | {constrained} constrained "
        f"({df.groupby('district').size().to_dict()})"
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 9.  Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Aquapurity – Synthetic Data Generator  [SYNTHETIC DATA]")
    print("=" * 60)

    wells = generate_wells()
    wl_df, rain_df = generate_timeseries(wells)
    sites = generate_sites()

    # ── save CSVs ──────────────────────────────────────────────────────────────
    wells.to_csv(os.path.join(DATA_DIR, "wells.csv"), index=False)
    wl_df.to_csv(os.path.join(DATA_DIR, "water_levels.csv"), index=False)
    rain_df.to_csv(os.path.join(DATA_DIR, "rainfall.csv"), index=False)
    sites.to_csv(os.path.join(DATA_DIR, "sites.csv"), index=False)

    print("\n[Saved] data/wells.csv, water_levels.csv, rainfall.csv, sites.csv")

    # ── summary ────────────────────────────────────────────────────────────────
    print("\n-- Column summary " + "-"*40)
    for name, df in [
        ("wells.csv", wells),
        ("water_levels.csv", wl_df),
        ("rainfall.csv", rain_df),
        ("sites.csv", sites),
    ]:
        print(f"  {name:25s}: {df.shape[0]} rows x {df.shape[1]} cols")
        print(f"    columns: {list(df.columns)}")

    print("\n[NOTE] All data is SYNTHETIC - for demo purposes only.")


if __name__ == "__main__":
    main()
