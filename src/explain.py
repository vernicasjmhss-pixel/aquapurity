"""
explain.py
----------
Stage 4: Explainability for the STGNN forecasts and TOPSIS site scores.

Well forecast explainability:
  - Primary:  Integrated Gradients (Captum library).
  - Fallback: Gradient × Input (manual PyTorch) if Captum is unavailable.
  Returns per-feature, per-timestep attributions + convergence check.

Site explainability:
  - Returns the weighted criterion contributions that produced the TOPSIS score.
  - No model needed; purely based on AHP weights × normalised criterion values.
"""

import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import torch

# Force UTF-8 stdout so log lines print cleanly on cp1252 Windows consoles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from model import STGNN

# ── check if Captum is installed ──────────────────────────────────────────────
try:
    from captum.attr import IntegratedGradients
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    print("[Explain] Captum not found – will use gradient × input fallback.")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

FEATURE_NAMES = ["water_level", "rainfall"]
HORIZONS = [1, 3, 6]


# ──────────────────────────────────────────────────────────────────────────────
# Load helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_model_artifacts(district: str):
    """Load saved model checkpoint for a district."""
    path = os.path.join(DATA_DIR, f"model_{district.lower()}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No model found at {path}. Run train.py first."
        )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = STGNN(n_nodes=ckpt["n_nodes"], n_features=2,
                  gcn_out=16, gru_hidden=32, n_horizons=3)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    A_hat = torch.tensor(ckpt["A_hat"])
    return model, A_hat, ckpt


def load_well_window(district: str, well_id: str, ckpt: dict,
                     window: int = 12) -> torch.Tensor:
    """
    Build the last 12-month input window for a specific well.
    Returns (1, T, N, 2) tensor.  N = all district wells (GNN needs full graph).
    """
    wl_raw  = pd.read_csv(os.path.join(DATA_DIR, "water_levels.csv"))
    rain_raw = pd.read_csv(os.path.join(DATA_DIR, "rainfall.csv"))

    well_ids = ckpt["well_ids"]
    stats    = ckpt["stats"]

    wl_pivot = (
        wl_raw[wl_raw["well_id"].isin(well_ids)]
        .pivot(index="date", columns="well_id", values="water_level_m")
        [well_ids]
        .sort_index()
        .interpolate(method="linear", axis=0)
        .ffill()
        .bfill()
    )
    rain_s = (
        rain_raw[rain_raw["district"] == district]
        .sort_values("date")
        .set_index("date")["rainfall_mm"]
    )

    # last `window` months
    wl_mat = wl_pivot.values.astype(np.float32)[-window:]   # (W, N)
    rain_v = rain_s.values.astype(np.float32)[-window:]     # (W,)

    # normalise
    wl_norm   = (wl_mat - stats["wl_mean"]) / stats["wl_std"]
    rain_norm = (rain_v - stats["rain_mean"]) / stats["rain_std"]
    rain_broad = np.repeat(rain_norm[:, np.newaxis], len(well_ids), axis=1)

    X = np.stack([wl_norm, rain_broad], axis=-1)        # (W, N, 2)
    return torch.tensor(X[np.newaxis], dtype=torch.float32)  # (1, W, N, 2)


# ──────────────────────────────────────────────────────────────────────────────
# Wrapper for Captum: the model needs a callable that takes X and returns
# a scalar (or vector) we're attributing to.
# ──────────────────────────────────────────────────────────────────────────────
class _ModelWrapper(torch.nn.Module):
    """
    Wraps STGNN so Captum can treat the input tensor X as the sole argument.
    A_hat and node_idx are fixed at construction time.
    Output: scalar – prediction for the target node and horizon.
    """

    def __init__(self, model: STGNN, A_hat: torch.Tensor,
                 node_idx: int, horizon_idx: int):
        super().__init__()
        self.model = model
        self.A_hat = A_hat
        self.node_idx = node_idx
        self.horizon_idx = horizon_idx

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, T, N, 2)
        out = self.model(self.A_hat, X)          # (B, N, H)
        return out[:, self.node_idx, self.horizon_idx].unsqueeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# Integrated Gradients (Captum)
# ──────────────────────────────────────────────────────────────────────────────
def explain_with_ig(
    model: STGNN,
    A_hat: torch.Tensor,
    X: torch.Tensor,
    node_idx: int,
    horizon_idx: int,
    n_steps: int = 50,
) -> tuple[np.ndarray, float]:
    """
    Run Integrated Gradients for the given node and horizon.

    Returns
    -------
    attrs      : (T, 2) attribution array per [timestep, feature]
    convergence: delta value (approximation error); should be near 0
    """
    wrapper = _ModelWrapper(model, A_hat, node_idx, horizon_idx)
    ig = IntegratedGradients(wrapper)

    baseline = torch.zeros_like(X)           # zero baseline (no signal)
    attrs, delta = ig.attribute(
        X,
        baselines=baseline,
        n_steps=n_steps,
        return_convergence_delta=True,
    )
    # attrs shape: (1, T, N, 2) – take the target node and squeeze batch
    node_attrs = attrs[0, :, node_idx, :].detach().numpy()  # (T, 2)
    convergence = float(delta.abs().mean().item())
    return node_attrs, convergence


# ──────────────────────────────────────────────────────────────────────────────
# Gradient × Input fallback
# ──────────────────────────────────────────────────────────────────────────────
def explain_with_grad_input(
    model: STGNN,
    A_hat: torch.Tensor,
    X: torch.Tensor,
    node_idx: int,
    horizon_idx: int,
) -> np.ndarray:
    """
    Gradient × Input attribution (no Captum required).

    Computes ∂output/∂X multiplied element-wise by X.
    Returns (T, 2) attribution array.
    """
    X_req = X.detach().requires_grad_(True)
    out = model(A_hat, X_req)                # (1, N, H)
    scalar = out[0, node_idx, horizon_idx]
    scalar.backward()

    grad = X_req.grad                        # (1, T, N, 2)
    attrs = (grad * X_req).detach()          # gradient × input
    node_attrs = attrs[0, :, node_idx, :].numpy()  # (T, 2)
    return node_attrs


# ──────────────────────────────────────────────────────────────────────────────
# Public API: explain a well forecast
# ──────────────────────────────────────────────────────────────────────────────
def explain_well(
    district: str,
    well_id: str,
    horizon: int = 1,
) -> dict:
    """
    Return attribution dict for a given well and forecast horizon.

    Parameters
    ----------
    district : 'Thanjavur' or 'Pudukkottai'
    well_id  : e.g. 'W_THA_001'
    horizon  : 1, 3, or 6

    Returns
    -------
    {
      method        : 'IntegratedGradients' or 'GradientXInput',
      well_id       : ...,
      horizon       : ...,
      convergence   : float or null,
      attributions  : [{"timestep": t, "water_level": v, "rainfall": v}, ...],
      top_features  : [{"label": ..., "value": ...}, ...]  ← top 5 by abs value
    }
    """
    model, A_hat, ckpt = load_model_artifacts(district)
    well_ids = ckpt["well_ids"]

    if well_id not in well_ids:
        raise ValueError(f"well_id '{well_id}' not found in {district} model.")

    node_idx = well_ids.index(well_id)
    horizon_idx = HORIZONS.index(horizon)

    X = load_well_window(district, well_id, ckpt)

    method = "IntegratedGradients"
    convergence = None

    if CAPTUM_AVAILABLE:
        try:
            attrs, convergence = explain_with_ig(
                model, A_hat, X, node_idx, horizon_idx
            )
        except Exception as e:
            print(f"[Explain] Captum IG failed ({e}); using fallback.")
            attrs = explain_with_grad_input(model, A_hat, X, node_idx, horizon_idx)
            method = "GradientXInput"
    else:
        attrs = explain_with_grad_input(model, A_hat, X, node_idx, horizon_idx)
        method = "GradientXInput"

    T = attrs.shape[0]
    attributions = [
        {
            "timestep": t + 1,
            "water_level": round(float(attrs[t, 0]), 6),
            "rainfall":    round(float(attrs[t, 1]), 6),
        }
        for t in range(T)
    ]

    # top 5 (timestep, feature) combinations by absolute attribution
    flat = [(abs(attrs[t, f]), t, FEATURE_NAMES[f], attrs[t, f])
            for t in range(T) for f in range(2)]
    flat.sort(reverse=True)
    top_features = [
        {"label": f"t-{T-t} {feat}", "value": round(float(val), 6)}
        for _, t, feat, val in flat[:5]
    ]

    return {
        "method": method,
        "well_id": well_id,
        "horizon": horizon,
        "convergence": round(convergence, 6) if convergence is not None else None,
        "attributions": attributions,
        "top_features": top_features,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Public API: explain a site's TOPSIS score
# ──────────────────────────────────────────────────────────────────────────────
def explain_site(site_id: str, ranked_sites: pd.DataFrame,
                 site_row: Optional[pd.DataFrame] = None) -> dict:
    """
    Return weighted criterion contributions for a site.

    The criterion columns are min-max normalised across `ranked_sites`, i.e.
    the SAME reference ranking that produced the score, so the bars explain
    the score the caller actually saw.

    Parameters
    ----------
    site_id      : e.g. 'S_THA_003'
    ranked_sites : eligible-site ranking from mcda.rank_sites() (reference set)
    site_row     : optional 1-row DataFrame used when the site is NOT in the
                   reference ranking (screened-out sites). Contributions are
                   still returned, but topsis_score is null because screening
                   happens before scoring.

    Returns
    -------
    {
      method        : 'AHP-TOPSIS weighted criteria',
      site_id       : ...,
      topsis_score  : float or null (null = screened out),
      screened      : bool,
      contributions : [{criterion, raw_value, weighted_contribution}, ...]
    }
    """
    from mcda import AHP_MATRICES, CRITERIA_BASE, compute_ahp_weights

    match = ranked_sites[ranked_sites["site_id"] == site_id]
    if not match.empty:
        row = match.iloc[0]
    elif site_row is not None and len(site_row) > 0:
        # screened-out site: explain it against the eligible reference set
        row = site_row.iloc[0]
    else:
        raise ValueError(f"site_id '{site_id}' not found in ranked sites.")

    district = row["district"]
    weights, _ = compute_ahp_weights(AHP_MATRICES[district], district)

    # NaN/None-safe float (screened sites can carry null criterion values)
    def _num(v) -> float:
        return 0.0 if v is None or pd.isna(v) else float(v)

    criteria_vals = [_num(row.get(c)) for c in CRITERIA_BASE]

    # normalise across the reference set for each criterion
    contributions = []
    for i, crit in enumerate(CRITERIA_BASE):
        col_vals = pd.to_numeric(ranked_sites[crit], errors="coerce").fillna(0).values
        col_range = col_vals.max() - col_vals.min() + 1e-9
        norm_val = (criteria_vals[i] - col_vals.min()) / col_range
        weighted_val = norm_val * weights[i]
        contributions.append({
            "criterion": crit.replace("_", " ").title(),
            "raw_value": round(criteria_vals[i], 4),
            "weighted_contribution": round(float(weighted_val), 4),
        })

    score = row.get("topsis_score", None)
    score = None if score is None or pd.isna(score) else round(float(score), 4)

    structure = row.get("recommended_structure", None)
    if structure is None or pd.isna(structure):
        structure = None

    return {
        "method": "AHP-TOPSIS weighted criteria",
        "site_id": site_id,
        "topsis_score": score,
        "screened": score is None,   # True = excluded by constraint screening
        "recommended_structure": structure,
        "contributions": contributions,
    }


# ── CLI demo ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    district = "Thanjavur"
    ckpt = torch.load(
        os.path.join(DATA_DIR, f"model_{district.lower()}.pt"),
        map_location="cpu", weights_only=False
    )
    first_well = ckpt["well_ids"][0]
    result = explain_well(district, first_well, horizon=1)
    print(json.dumps(result, indent=2))
