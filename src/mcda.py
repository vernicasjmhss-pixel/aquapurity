"""
mcda.py
-------
Stage 3a: Multi-Criteria Decision Analysis for recharge site prioritisation.

Steps:
  1. Define AHP pairwise comparison matrices per district.
     - Thanjavur  (alluvial): soil permeability and distance-to-water weighted higher.
     - Pudukkottai (hard-rock): lineament density weighted higher.
  2. Compute AHP weights and the Consistency Ratio (CR).
     Reject any matrix with CR >= 0.10 (Saaty threshold).
  3. Run TOPSIS on eligible sites using AHP weights + forecast stress.
  4. Return a ranked DataFrame.

References: Saaty (1980) AHP; Hwang & Yoon (1981) TOPSIS.
"""

import math
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

# Force UTF-8 stdout: prints use ✓/✗ markers that cp1252 consoles reject.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Saaty Random Index (RI) for n=2..10 ──────────────────────────────────────
# Used to compute the Consistency Ratio CR = CI / RI
RANDOM_INDEX = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
                6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

# ── criteria names (column keys) ──────────────────────────────────────────────
# "forecast_stress" is appended at runtime from model output
CRITERIA_BASE = [
    "soil_permeability",     # higher = better recharge potential
    "slope_deg",             # lower  = better (flatter land)
    "dist_to_water_km",      # lower  = better (closer to water body)
    "lineament_density",     # higher = better (for hard rock only)
    "forecast_stress",       # higher forecast decline = more urgent
]

# Benefit criteria (+1) vs cost criteria (-1)
BENEFIT = {
    "soil_permeability": 1,
    "slope_deg": -1,            # cost: lower slope preferred
    "dist_to_water_km": -1,     # cost: closer preferred
    "lineament_density": 1,
    "forecast_stress": 1,       # sites in stressed zones get priority
}


# ──────────────────────────────────────────────────────────────────────────────
# AHP pairwise matrices
# ──────────────────────────────────────────────────────────────────────────────
# Scale: 1=equal, 3=moderate, 5=strong, 7=very strong, 9=extreme importance
# Row/col order: [soil_perm, slope, dist_water, lineament, forecast_stress]

AHP_MATRICES = {
    # Thanjavur: alluvial – soil permeability most important; lineament less relevant
    "Thanjavur": np.array([
        [1,    3,    3,    5,    2   ],   # soil_permeability
        [1/3,  1,    2,    4,    1/2 ],   # slope_deg
        [1/3,  1/2,  1,    3,    1/2 ],   # dist_to_water_km
        [1/5,  1/4,  1/3,  1,    1/5 ],   # lineament_density (less relevant)
        [1/2,  2,    2,    5,    1   ],   # forecast_stress
    ], dtype=float),

    # Pudukkottai: hard-rock – lineament density most important for fracture recharge
    "Pudukkottai": np.array([
        [1,    2,    2,    1/3,  2   ],   # soil_permeability
        [1/2,  1,    2,    1/4,  1/2 ],   # slope_deg
        [1/2,  1/2,  1,    1/4,  1/2 ],   # dist_to_water_km
        [3,    4,    4,    1,    3   ],   # lineament_density (KEY for hard rock)
        [1/2,  2,    2,    1/3,  1   ],   # forecast_stress
    ], dtype=float),
}


# ──────────────────────────────────────────────────────────────────────────────
# AHP helper functions
# ──────────────────────────────────────────────────────────────────────────────
def compute_ahp_weights(matrix: np.ndarray, district: str) -> tuple[np.ndarray, float]:
    """
    Compute AHP priority weights and Consistency Ratio.

    Returns
    -------
    weights : (n,) normalised priority vector
    CR      : consistency ratio (should be < 0.10)
    """
    n = matrix.shape[0]

    # normalise each column then average rows → priority vector
    col_sums = matrix.sum(axis=0)
    norm_matrix = matrix / col_sums
    weights = norm_matrix.mean(axis=1)

    # consistency check
    # lambda_max = mean of (A @ w / w)
    Aw = matrix @ weights
    lambda_max = np.mean(Aw / weights)
    CI = (lambda_max - n) / (n - 1)
    RI = RANDOM_INDEX.get(n, 1.49)
    CR = CI / RI if RI > 0 else 0.0

    print(f"  [{district}] AHP weights: "
          + " | ".join(f"{c}={w:.3f}" for c, w in zip(CRITERIA_BASE, weights)))
    print(f"  [{district}] CR={CR:.4f} ({'✓ OK' if CR < 0.10 else '✗ FAIL – matrix needs revision'})")

    if CR >= 0.10:
        raise ValueError(
            f"AHP consistency ratio {CR:.4f} ≥ 0.10 for {district}. "
            "Revise the pairwise comparison matrix."
        )
    return weights, CR


# ──────────────────────────────────────────────────────────────────────────────
# TOPSIS
# ──────────────────────────────────────────────────────────────────────────────
def topsis(
    data: pd.DataFrame,
    criteria: list[str],
    weights: np.ndarray,
    benefit: dict,
) -> np.ndarray:
    """
    TOPSIS ranking.

    Parameters
    ----------
    data     : DataFrame with criteria columns
    criteria : ordered list of criterion names
    weights  : AHP weight vector (same order as criteria)
    benefit  : dict {criterion: +1 or -1}

    Returns
    -------
    scores : (n,) array of TOPSIS scores ∈ [0,1]; higher = better
    """
    X = data[criteria].values.astype(float)
    n, m = X.shape

    # Step 1: normalise (vector norm)
    norms = np.sqrt((X ** 2).sum(axis=0)) + 1e-9
    X_norm = X / norms

    # Step 2: weighted normalised matrix
    V = X_norm * weights

    # Step 3: ideal best (A+) and worst (A-) per criterion
    A_pos = np.where(
        np.array([benefit[c] for c in criteria]) == 1,
        V.max(axis=0),   # benefit: maximum is best
        V.min(axis=0),   # cost:    minimum is best
    )
    A_neg = np.where(
        np.array([benefit[c] for c in criteria]) == 1,
        V.min(axis=0),
        V.max(axis=0),
    )

    # Step 4: Euclidean distances
    d_pos = np.sqrt(((V - A_pos) ** 2).sum(axis=1))
    d_neg = np.sqrt(((V - A_neg) ** 2).sum(axis=1))

    # Step 5: closeness coefficient
    scores = d_neg / (d_pos + d_neg + 1e-9)
    return scores


# ──────────────────────────────────────────────────────────────────────────────
# Structure-type recommendation
# ──────────────────────────────────────────────────────────────────────────────
def recommend_structure(row: pd.Series, district: str) -> str:
    """
    Simple rule-based structure recommendation.

    Rules (each explained):
    - Percolation pond : low slope (<2°) + high soil permeability (>6)
        → flat land absorbs water; good for alluvial plains (Thanjavur).
    - Recharge shaft   : high lineament density (>1.5) in hard-rock district
        → shafts inject water directly into fractures (Pudukkottai).
    - Check dam        : moderate slope (2–5°) + close to water (<3 km)
        → dams slow surface runoff near stream channels.
    - Percolation pond (default): catches all remaining eligible sites.
    """
    slope = row.get("slope_deg", 5)
    perm  = row.get("soil_permeability", 5)
    lden  = row.get("lineament_density", 0) or 0
    dist  = row.get("dist_to_water_km", 5)

    if district == "Pudukkottai" and lden > 1.5:
        # fracture-controlled aquifer: shafts work best
        return "Recharge Shaft"
    elif slope < 2.0 and perm > 6.0:
        # flat + permeable → percolation pond ideal
        return "Percolation Pond"
    elif 2.0 <= slope <= 5.0 and dist < 3.0:
        # moderate slope near stream → check dam
        return "Check Dam"
    else:
        # default fallback
        return "Percolation Pond"


# ──────────────────────────────────────────────────────────────────────────────
# Main ranking function
# ──────────────────────────────────────────────────────────────────────────────
def rank_sites(
    sites: pd.DataFrame,
    district: str,
    forecast_stress: Optional[pd.Series] = None,
    structure_filter: Optional[str] = None,
    exclude_constrained: bool = True,
) -> pd.DataFrame:
    """
    Rank candidate sites for a district using AHP + TOPSIS.

    Parameters
    ----------
    sites               : sites DataFrame (from data/sites.csv)
    district            : 'Thanjavur' or 'Pudukkottai'
    forecast_stress     : optional Series indexed by site_id (avg stress score)
    structure_filter    : filter to one structure type after ranking
    exclude_constrained : if True, remove constrained sites before ranking

    Returns
    -------
    ranked DataFrame with added columns: topsis_score, rank, recommended_structure
    """
    df = sites[sites["district"] == district].copy()

    # ── Step 1: exclude constrained sites ────────────────────────────────────
    if exclude_constrained:
        constrained_mask = (
            (df["constraint_protected"] == 1)
            | (df["constraint_built_up"] == 1)
            | (df["constraint_saline"] == 1)
        )
        df = df[~constrained_mask].copy()
        print(f"  [{district}] {len(df)} sites remain after constraint screening")
    else:
        print(f"  [{district}] {len(df)} sites (constraints NOT excluded)")

    if len(df) == 0:
        return df

    # ── Step 2: add forecast stress ───────────────────────────────────────────
    if forecast_stress is not None:
        df["forecast_stress"] = df["site_id"].map(forecast_stress).fillna(
            forecast_stress.mean()
        )
    else:
        # if no model output provided, use a synthetic stress proxy
        # (higher water level = more stress = more urgent)
        df["forecast_stress"] = np.random.default_rng(42).uniform(0.2, 1.0, len(df))

    # ── Step 3: fill lineament_density NaN for Thanjavur with 0 ──────────────
    df["lineament_density"] = df["lineament_density"].fillna(0.0)

    # ── Step 4: compute AHP weights ───────────────────────────────────────────
    weights, cr = compute_ahp_weights(AHP_MATRICES[district], district)

    # ── Step 5: TOPSIS ────────────────────────────────────────────────────────
    scores = topsis(df, CRITERIA_BASE, weights, BENEFIT)
    df["topsis_score"] = np.round(scores, 4)
    df["rank"] = df["topsis_score"].rank(ascending=False, method="min").astype(int)

    # ── Step 6: recommend structure ───────────────────────────────────────────
    df["recommended_structure"] = df.apply(
        lambda r: recommend_structure(r, district), axis=1
    )

    # ── Step 7: optional structure filter ─────────────────────────────────────
    if structure_filter and structure_filter != "All":
        df = df[df["recommended_structure"] == structure_filter]

    df = df.sort_values("topsis_score", ascending=False).reset_index(drop=True)
    return df


# ── CLI demo ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    sites = pd.read_csv(os.path.join(DATA_DIR, "sites.csv"))

    for dist in ["Thanjavur", "Pudukkottai"]:
        print(f"\n{'='*50}\n  {dist}\n{'='*50}")
        ranked = rank_sites(sites, dist, exclude_constrained=True)
        print(ranked[["site_id", "topsis_score", "rank", "recommended_structure"]].head(10))
