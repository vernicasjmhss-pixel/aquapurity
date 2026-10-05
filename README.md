# Aquapurity — Explainable Spatiotemporal GNN for Groundwater Recharge Prioritisation

> **College project demo – Tamil Nadu Groundwater Recharge Site Prioritisation**

> ⚠️ **ALL DATA IS FULLY SYNTHETIC.**
> Generated with a fixed random seed (42) to mimic CGWB-style well logs,
> IMD-style rainfall records, and Bhuvan-style terrain layers.
> Results demonstrate the pipeline architecture only and must **not** be
> interpreted as real groundwater findings.

---

## What it does

Aquapurity implements a full end-to-end pipeline for dynamic recharge-site
prioritisation in two Tamil Nadu districts:

| District | Aquifer type | Special factor |
|---|---|---|
| Thanjavur | Alluvial / Deltaic | NE monsoon rainfall, high soil permeability |
| Pudukkottai | Hard-rock crystalline | Lineament density (fracture-controlled recharge) |

**Pipeline stages:**

```
Sense → Harmonise → Forecast → Rank → Explain → Advise → Learn
```

| Module | File | Purpose |
|---|---|---|
| Data Fusion | `src/generate_data.py` | Synthetic wells, rainfall, time-series, sites |
| Graph | `src/graph.py` | k-NN well graph + normalised adjacency |
| Forecasting | `src/model.py` + `src/train.py` | STGNN (GCN→GRU→MLP), multi-horizon |
| Prioritisation | `src/mcda.py` + `src/constraints.py` | AHP + TOPSIS ranking |
| Explainability | `src/explain.py` | Integrated Gradients (Captum) |
| API | `src/api.py` | FastAPI REST backend |
| Dashboard | `app/index.html` | Leaflet GIS frontend |

---

## Folder layout

```
aquapurity/
├── data/               CSV/JSON outputs + trained model checkpoints
├── src/
│   ├── generate_data.py
│   ├── graph.py
│   ├── model.py
│   ├── train.py
│   ├── mcda.py
│   ├── constraints.py
│   ├── explain.py
│   └── api.py
├── app/
│   ├── index.html      Leaflet dashboard (open in browser)
│   └── vendor/         Leaflet 1.9.4 + Chart.js 4.4 bundled locally
├── requirements.txt
└── README.md
```

Frontend note: **no CDN, no internet at runtime.** Leaflet and Chart.js are
bundled in `app/vendor/`, and the basemap is drawn offline as a vector
graticule with district outlines (no remote map tiles).

---

## Setup (one-time)

```bash
# 1. Create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Run (three terminal commands)

### Step 1 – Generate data
```bash
python src/generate_data.py
```
Writes `data/wells.csv`, `data/water_levels.csv`, `data/rainfall.csv`, `data/sites.csv`.

### Step 2 – Train the model
```bash
python src/train.py
```
Trains the STGNN for both districts, prints RMSE/NSE/KGE vs baselines,
saves `data/model_thanjavur.pt` and `data/model_pudukkottai.pt`.

### Step 3 – Start the API
```bash
uvicorn src.api:app --reload
```
API available at **http://127.0.0.1:8000**
Interactive docs at **http://127.0.0.1:8000/docs**

### Step 4 – Open the dashboard
Open `app/index.html` in a browser (double-click or `file://` URL).
It talks to the API at `http://127.0.0.1:8000`, so keep Step 3 running.
A visible **“Synthetic data”** badge and per-popup warnings remind you that
all values are generated, not measured.

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/sites?district=&structure=` | Candidate sites with scores |
| GET | `/wells?district=` | Wells with latest water level |
| POST | `/forecast` | STGNN forecast + uncertainty range |
| POST | `/rank` | AHP+TOPSIS ranked sites |
| POST | `/explain` | Integrated Gradients / TOPSIS attribution |

---

## Architecture overview

```
Input (B, T=12, N, F=2)
        │
     GCN layer   — spatial aggregation via A_hat @ X @ W
        │
     GRU          — temporal sequence encoding (12 months)
        │
     MLP head     — 3-horizon forecast [+1, +3, +6 months]
        │
   Explainability — Integrated Gradients (Captum) or Gradient×Input fallback
```

**MCDA stack:**
1. AHP pairwise comparison → priority weights per district
   (soil weighted higher in Thanjavur, lineament density in Pudukkottai)
2. Consistency Ratio (CR) check; reject if CR ≥ 0.10
3. Constraint screening (protected / built-up / saline)
4. TOPSIS → ordinal ranking, using AHP weights **plus the model-derived
   forecast stress** (each site inherits the predicted water level of its
   nearest well, min-max scaled to 0–1)
5. Rule-based structure recommendation (Percolation Pond / Check Dam / Recharge Shaft)

Screening always precedes scoring, so a site’s score is identical in `/sites`,
`/rank` and `/explain`; screened-out sites are returned with a null score.

---

## Key design decisions

- **Fixed seed (42)** everywhere so every run produces identical synthetic data and model weights.
- **Metrics**: `train.py` prints RMSE / NSE / KGE for the STGNN against
  persistence and linear-trend baselines on the chronological validation split.
  The STGNN wins on RMSE and NSE at every horizon; KGE is reported honestly too.
- **Forecast uncertainty** = quadrature of MC input-perturbation spread and the
  model’s per-horizon validation RMSE (from `metrics.json`).
- **No database** – all state lives in CSV/JSON/`.pt` files under `data/`.
- **CPU-only** – no GPU, no paid API, no internet at runtime.
- Captum IG **falls back** to Gradient×Input automatically if unavailable; the response states which method was used.

---

## Disclaimer

This prototype is created for a college project review.
All groundwater data, well locations, rainfall figures, and terrain attributes
are **synthetically generated** and bear no relation to actual CGWB measurements,
IMD records, or Bhuvan spatial databases. The pipeline is meant to illustrate
the proposed methodology, not to guide real-world recharge decisions.
