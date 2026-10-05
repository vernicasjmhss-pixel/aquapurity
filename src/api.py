"""
api.py
------
Stage 5: FastAPI application for the Aquapurity prototype.

Five endpoints:
  GET  /sites    ?district=&structure=          candidate sites + scores
  GET  /wells    ?district=                     wells + latest water level
  POST /forecast {well_id, horizon}             STGNN forecast + uncertainty
  POST /rank     {district, structure, exclude_constrained}  ranked sites
  POST /explain  {site_id?, well_id?, horizon?} feature attributions

Model is loaded ONCE at startup per district to avoid repeated disk reads.
All data is SYNTHETIC - labelled in every response.

The dashboard in app/ is mounted on the same origin as the API, so one URL
serves both the UI and the JSON endpoints (no hard-coded host, no CORS).

Run locally with:  uvicorn src.api:app --reload
       (or)        python src/api.py
"""

import contextlib
import json
import math
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

# Force UTF-8 stdout before anything prints: uvicorn log lines and helper
# modules use characters (✓, →) that the default Windows cp1252 page rejects.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── make sure src/ is importable even when run from project root ──────────────
SRC_DIR = os.path.dirname(__file__)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from constraints import screen_sites
from explain import explain_site, explain_well
from mcda import rank_sites, recommend_structure
from model import STGNN

DATA_DIR = os.path.join(SRC_DIR, "..", "data")

# ──────────────────────────────────────────────────────────────────────────────
# Module-level state (populated at startup)
# ──────────────────────────────────────────────────────────────────────────────
DISTRICTS = ["Thanjavur", "Pudukkottai"]
HORIZONS  = [1, 3, 6]
WINDOW    = 12

_wells_df = None
_sites_df = None
_wl_df    = None
_rain_df  = None

_models: dict = {}   # district -> STGNN model
_A_hats: dict = {}   # district -> torch.Tensor (N x N)
_ckpts:  dict = {}   # district -> checkpoint dict
_stress: dict = {}   # district -> pd.Series(site_id -> model forecast stress)
_metrics: dict = {}  # validation metrics from data/metrics.json (RMSE per horizon)


def _records(df: pd.DataFrame) -> list:
    """
    DataFrame -> list of plain dicts that are safe to JSON-encode.
    pandas' dict(orient='records') keeps NaN as a float nan, which Starlette
    rejects ('Out of range float values are not JSON compliant'); to_json()
    maps NaN/NaT to JSON null instead.
    """
    return json.loads(df.to_json(orient="records"))


def _load_all():
    """Load CSVs and model checkpoints once at startup."""
    global _wells_df, _sites_df, _wl_df, _rain_df, _metrics

    print("[API] Loading data...")
    _wells_df = pd.read_csv(os.path.join(DATA_DIR, "wells.csv"))
    _sites_df = pd.read_csv(os.path.join(DATA_DIR, "sites.csv"))
    _wl_df    = pd.read_csv(os.path.join(DATA_DIR, "water_levels.csv"))
    _rain_df  = pd.read_csv(os.path.join(DATA_DIR, "rainfall.csv"))

    # screen sites once and cache the result
    _sites_df = screen_sites(_sites_df, verbose=False)
    # the structure rule depends only on the site's own attributes, so compute
    # it once for EVERY site (rank_sites recomputes it for the ranked rows)
    _sites_df["recommended_structure"] = _sites_df.apply(
        lambda r: recommend_structure(r, r["district"]), axis=1
    )

    for dist in DISTRICTS:
        model_path = os.path.join(DATA_DIR, f"model_{dist.lower()}.pt")
        if not os.path.exists(model_path):
            print(f"[API] WARNING - no model for {dist}. Run train.py first.")
            continue
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        model = STGNN(n_nodes=ckpt["n_nodes"], n_features=2,
                      gcn_out=16, gru_hidden=32, n_horizons=3)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        _models[dist] = model
        _A_hats[dist] = torch.tensor(ckpt["A_hat"])
        _ckpts[dist]  = ckpt
        print(f"[API] Model loaded for {dist}")

    # validation metrics (RMSE per horizon) feed the forecast uncertainty band
    metrics_path = os.path.join(DATA_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as fh:
            _metrics = json.load(fh)

    # Stage 3: derive the "forecast stress" TOPSIS criterion from the model
    _compute_forecast_stress()

    print("[API] Ready - all data is SYNTHETIC.")


@contextlib.asynccontextmanager
async def lifespan(application: FastAPI):
    """FastAPI lifespan: load data before serving, clean up on shutdown."""
    _load_all()
    yield
    # nothing to clean up in this prototype


# ──────────────────────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Aquapurity API",
    description=(
        "Explainable Spatiotemporal GNN for Groundwater Recharge Prioritisation. "
        "WARNING: ALL DATA IS SYNTHETIC - for demonstration purposes only."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the local HTML dashboard to call the API without CORS errors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic request / response models
# ──────────────────────────────────────────────────────────────────────────────
class ForecastRequest(BaseModel):
    well_id: str
    horizon: int = 1

    @field_validator("horizon")
    @classmethod
    def horizon_must_be_valid(cls, v):
        if v not in [1, 3, 6]:
            raise ValueError("horizon must be 1, 3, or 6")
        return v


class RankRequest(BaseModel):
    district: str
    structure: Optional[str] = "All"
    exclude_constrained: bool = True

    @field_validator("district")
    @classmethod
    def district_must_be_valid(cls, v):
        if v not in ["Thanjavur", "Pudukkottai"]:
            raise ValueError("district must be 'Thanjavur' or 'Pudukkottai'")
        return v


class ExplainRequest(BaseModel):
    site_id:  Optional[str] = None
    well_id:  Optional[str] = None
    horizon:  int = 1

    @field_validator("horizon")
    @classmethod
    def horizon_must_be_valid(cls, v):
        if v not in [1, 3, 6]:
            raise ValueError("horizon must be 1, 3, or 6")
        return v


# ──────────────────────────────────────────────────────────────────────────────
# Helper: build last-12-month window tensor for a well
# ──────────────────────────────────────────────────────────────────────────────
def _build_window(district: str, well_id: str):
    ckpt     = _ckpts[district]
    well_ids = ckpt["well_ids"]
    stats    = ckpt["stats"]

    wl_pivot = (
        _wl_df[_wl_df["well_id"].isin(well_ids)]
        .pivot(index="date", columns="well_id", values="water_level_m")
        [well_ids]
        .sort_index()
        .interpolate(method="linear", axis=0)
        .ffill()
        .bfill()
    )
    rain_s = (
        _rain_df[_rain_df["district"] == district]
        .sort_values("date")
        .set_index("date")["rainfall_mm"]
    )

    wl_mat  = wl_pivot.values.astype(np.float32)[-WINDOW:]
    rain_v  = rain_s.values.astype(np.float32)[-WINDOW:]

    wl_norm   = (wl_mat   - stats["wl_mean"])   / stats["wl_std"]
    rain_norm = (rain_v   - stats["rain_mean"]) / stats["rain_std"]
    rain_b    = np.repeat(rain_norm[:, np.newaxis], len(well_ids), axis=1)

    X = np.stack([wl_norm, rain_b], axis=-1)
    return torch.tensor(X[np.newaxis], dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: model-derived "forecast stress" for every candidate site
# ──────────────────────────────────────────────────────────────────────────────
def _compute_forecast_stress():
    """
    Stage 3 requires ranking sites with *the forecast stress from the model*.

    For each district we run ONE STGNN forward pass on the latest 12-month
    window, inverse-transform the predicted levels to metres below ground and
    average the +1/+3/+6-month predictions per well. Every candidate site
    inherits the value of its nearest well in the same district, and the result
    is min-max scaled to [0, 1]:  1 = deepest predicted water level = most
    stressed = highest recharge priority.

    The Series is computed once at startup and reused by /sites, /rank and
    /explain so all three endpoints score sites identically.
    """
    for dist in DISTRICTS:
        if dist not in _models:
            continue

        well_ids = _ckpts[dist]["well_ids"]
        stats    = _ckpts[dist]["stats"]

        # one forward pass -> prediction for every well at every horizon
        X = _build_window(dist, well_ids[0])          # (1, T, N, 2)
        with torch.no_grad():
            pred = _models[dist](_A_hats[dist], X)[0].numpy()   # (N, 3)

        # undo the z-score normalisation -> metres below ground
        pred_m = pred * stats["wl_std"][0][:, None] + stats["wl_mean"][0][:, None]
        well_stress = pred_m.mean(axis=1)             # (N,) mean depth per well

        wells = _wells_df[_wells_df["district"] == dist]
        sites = _sites_df[_sites_df["district"] == dist]
        if len(sites) == 0 or len(wells) == 0:
            _stress[dist] = pd.Series(dtype=float)
            continue

        # nearest well for each site (plain lat/lon distance is fine at district scale)
        dlat = sites["lat"].values[:, None] - wells["lat"].values[None, :]
        dlon = sites["lon"].values[:, None] - wells["lon"].values[None, :]
        nearest = (dlat ** 2 + dlon ** 2).argmin(axis=1)

        stress = well_stress[nearest]
        lo, hi = float(stress.min()), float(stress.max())
        scaled = (stress - lo) / (hi - lo + 1e-9)     # min-max -> [0, 1]

        _stress[dist] = pd.Series(scaled, index=sites["site_id"].values)
        print(f"[API] Forecast stress from model ready for {dist} "
              f"({len(sites)} sites, {len(wells)} wells)")


def _validation_rmse(district: str, horizon: int):
    """Model validation RMSE (normalised) for a district+horizon, or None."""
    try:
        return float(_metrics[district][f"h{horizon}"]["model"]["rmse"])
    except (KeyError, TypeError, ValueError):
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Endpoint 1 - GET /sites
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/sites")
def get_sites(
    district: Optional[str] = Query(None, description="Filter by district"),
    structure: Optional[str] = Query(None, description="Filter by structure type"),
):
    """
    Return candidate recharge sites with constraint flags and - for sites that
    pass screening - their TOPSIS score and rank.

    Constrained sites are included as well (score/rank = null) so the map can
    draw them hollow. Screening precedes scoring, so scores are identical here,
    in /rank and in /explain.
    """
    if district and district not in DISTRICTS:
        raise HTTPException(status_code=400, detail="Invalid district")

    results = []
    for dist in ([district] if district else DISTRICTS):
        sub = _sites_df[_sites_df["district"] == dist]
        if sub.empty:
            continue
        try:
            # rank only the eligible sites (screening first, per the pipeline)
            ranked = rank_sites(sub, dist,
                                forecast_stress=_stress.get(dist),
                                exclude_constrained=True)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        cols = ["site_id", "topsis_score", "rank", "forecast_stress"]
        if ranked.empty:                      # nothing eligible in this district
            out = sub.copy()
            for c in cols[1:]:
                out[c] = None
        else:
            # join scores back onto ALL sites (constrained ones stay null)
            out = sub.merge(ranked[cols], on="site_id", how="left")
            out["rank"] = out["rank"].astype("Int64")   # nulls -> JSON null
        results.append(out)

    if not results:
        return {"data": [], "synthetic": True, "count": 0}

    out = pd.concat(results, ignore_index=True)
    if isinstance(structure, str) and structure not in ("", "All"):
        out = out[out["recommended_structure"] == structure]

    return {
        "data": _records(out),
        "synthetic": True,
        "count": len(out),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Endpoint 2 - GET /wells
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/wells")
def get_wells(
    district: Optional[str] = Query(None, description="Filter by district"),
):
    """
    Return wells with their latest observed water level and static features.
    """
    wells = _wells_df.copy()
    if district:
        if district not in DISTRICTS:
            raise HTTPException(status_code=400, detail="Invalid district")
        wells = wells[wells["district"] == district]

    # attach the latest NON-MISSING reading per well (a few synthetic wells
    # have a gap in the final month, which would otherwise surface as NaN)
    latest = (
        _wl_df.dropna(subset=["water_level_m"])
        .sort_values("date")
        .groupby("well_id")
        .tail(1)[["well_id", "date", "water_level_m"]]
    )
    merged = wells.merge(latest, on="well_id", how="left")

    return {
        "data": _records(merged),   # NaN (e.g. Thanjavur lineament) -> JSON null
        "synthetic": True,
        "count": len(merged),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Endpoint 3 - POST /forecast
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/forecast")
def post_forecast(req: ForecastRequest):
    """
    Run the STGNN to forecast water level for a well.

    Returns the point forecast plus a +/-1-sigma uncertainty range
    estimated by adding small Gaussian noise to the input 10 times.
    """
    row = _wells_df[_wells_df["well_id"] == req.well_id]
    if row.empty:
        raise HTTPException(status_code=404,
                            detail=f"well_id '{req.well_id}' not found")
    district = row.iloc[0]["district"]

    if district not in _models:
        raise HTTPException(status_code=503,
                            detail=f"Model for {district} not loaded. Run train.py first.")

    model    = _models[district]
    A_hat    = _A_hats[district]
    ckpt     = _ckpts[district]
    well_ids = ckpt["well_ids"]
    stats    = ckpt["stats"]

    if req.well_id not in well_ids:
        raise HTTPException(status_code=404, detail="well_id not in model graph")

    node_idx    = well_ids.index(req.well_id)
    horizon_idx = HORIZONS.index(req.horizon)

    X = _build_window(district, req.well_id)

    # point forecast
    with torch.no_grad():
        pred = model(A_hat, X)    # (1, N, 3)
    pred_norm = float(pred[0, node_idx, horizon_idx].item())

    # Uncertainty (normalised space) = quadrature sum of
    #   (a) spread of 10 MC forward passes with jittered inputs  (epistemic), and
    #   (b) the model's validation RMSE for this horizon        (measured error).
    # (b) alone would be a static band; (a) alone was ~0.005 m - too narrow to
    # be meaningful. Together the band reflects validation error per horizon.
    samples = []
    for _ in range(10):
        X_noisy = X + torch.randn_like(X) * 0.05
        with torch.no_grad():
            p = model(A_hat, X_noisy)
        samples.append(float(p[0, node_idx, horizon_idx].item()))
    noise_sigma = float(np.std(samples))

    rmse_sigma = _validation_rmse(district, req.horizon) or 0.0
    sigma_norm = math.sqrt(noise_sigma ** 2 + rmse_sigma ** 2)

    # inverse-transform to original scale
    wl_mean = float(stats["wl_mean"][0, node_idx])
    wl_std  = float(stats["wl_std" ][0, node_idx])

    forecast_m = pred_norm * wl_std + wl_mean
    lower_m    = (pred_norm - sigma_norm) * wl_std + wl_mean
    upper_m    = (pred_norm + sigma_norm) * wl_std + wl_mean

    # historical series for the chart (last 24 months)
    wl_hist = (
        _wl_df[_wl_df["well_id"] == req.well_id]
        .sort_values("date")
        .tail(24)
        [["date", "water_level_m"]]
        .dropna()
        .to_dict(orient="records")
    )

    return {
        "well_id":        req.well_id,
        "district":       district,
        "horizon_months": req.horizon,
        "forecast_m":     round(forecast_m, 3),
        "lower_m":        round(lower_m, 3),
        "upper_m":        round(upper_m, 3),
        "history":        wl_hist,
        "synthetic":      True,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Endpoint 4 - POST /rank
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/rank")
def post_rank(req: RankRequest):
    """
    Rank candidate recharge sites using AHP + TOPSIS.

    Screening always precedes scoring: TOPSIS runs on the eligible sites, so a
    site's score never changes when `exclude_constrained` is toggled. With
    exclude_constrained=false the screened-out sites are appended with
    score/rank = null (they are excluded from the recommendation).
    """
    sites_sub = _sites_df[_sites_df["district"] == req.district].copy()

    try:
        scored = rank_sites(
            sites_sub,
            req.district,
            forecast_stress=_stress.get(req.district),
            exclude_constrained=True,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    parts = [scored]
    if not req.exclude_constrained:
        blocked = sites_sub[sites_sub["is_constrained"]].copy()
        blocked["topsis_score"] = np.nan
        blocked["rank"] = np.nan
        blocked["forecast_stress"] = blocked["site_id"].map(
            _stress.get(req.district, {})
        )
        parts.append(blocked)

    ranked = pd.concat(parts, ignore_index=True)
    if isinstance(req.structure, str) and req.structure not in ("", "All"):
        ranked = ranked[ranked["recommended_structure"] == req.structure]
    ranked = ranked.sort_values("topsis_score", ascending=False,
                                na_position="last").reset_index(drop=True)
    ranked["rank"] = ranked["rank"].astype("Int64")

    return {
        "district":            req.district,
        "exclude_constrained": req.exclude_constrained,
        "structure_filter":    req.structure,
        "data":                _records(ranked),
        "count":               len(ranked),
        "synthetic":           True,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Endpoint 5 - POST /explain
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/explain")
def post_explain(req: ExplainRequest):
    """
    Return XAI contributions for a site or a well.

    - If well_id provided: run Integrated Gradients (or fallback) on the STGNN.
    - If site_id provided: decompose the TOPSIS score into criterion contributions.
    """
    if req.well_id:
        row = _wells_df[_wells_df["well_id"] == req.well_id]
        if row.empty:
            raise HTTPException(status_code=404,
                                detail=f"well_id '{req.well_id}' not found")
        district = row.iloc[0]["district"]
        if district not in _models:
            raise HTTPException(status_code=503,
                                detail="Model not loaded. Run train.py first.")
        try:
            result = explain_well(district, req.well_id, horizon=req.horizon)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        result["synthetic"] = True
        return result

    elif req.site_id:
        district_s = _sites_df.loc[
            _sites_df["site_id"] == req.site_id, "district"
        ]
        if district_s.empty:
            raise HTTPException(status_code=404,
                                detail=f"site_id '{req.site_id}' not found")
        district = district_s.iloc[0]

        try:
            # same reference ranking as /rank and /sites -> same scores
            sub = _sites_df[_sites_df["district"] == district].copy()
            sub["forecast_stress"] = sub["site_id"].map(
                _stress.get(district, {})
            )
            scored = rank_sites(
                sub,
                district,
                forecast_stress=_stress.get(district),
                exclude_constrained=True,
            )
            # site_row lets a screened-out site be explained too (score=null)
            result = explain_site(req.site_id, scored,
                                  site_row=sub[sub["site_id"] == req.site_id])
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        result["synthetic"] = True
        return result

    else:
        raise HTTPException(
            status_code=422,
            detail="Provide either site_id or well_id in the request body.",
        )


# ──────────────────────────────────────────────────────────────────────────────
# /health - cheap liveness probe for the hosting platform (Render / Koyeb /
# Cloud Run). Deliberately touches no data so it stays fast even while the
# models are still loading.
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "synthetic": True, "districts": DISTRICTS}


# ──────────────────────────────────────────────────────────────────────────────
# Static dashboard, served from the SAME origin as the API
# ──────────────────────────────────────────────────────────────────────────────
# Mounted last, so every API route defined above wins; everything else
# (/, /index.html, /vendor/*) is served out of app/. Sharing one origin in the
# cloud means index.html needs no hard-coded API URL and there is no CORS
# pre-flight. It also lets the whole prototype run from a single local port:
#     uvicorn src.api:app   ->   http://127.0.0.1:8000
WEB_DIR = os.path.normpath(os.path.join(SRC_DIR, "..", "app"))
if os.path.isdir(WEB_DIR):
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="dashboard")
else:
    print(f"[API] WARNING - dashboard folder not found at {WEB_DIR}; "
          f"serving the API only.")


if __name__ == "__main__":
    # Convenience entry point used by the cloud start command:
    #   python src/api.py
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        log_level="info",
    )
