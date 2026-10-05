# Aquapurity — Explainable Spatiotemporal GNN for Groundwater Recharge Prioritisation

**▶ Live demo (static snapshot):** <https://vernicasjmhss-pixel.github.io/aquapurity/>

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/vernicasjmhss-pixel/aquapurity)

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
| Static export | `src/export_static.py` | Precomputes the dashboard for GitHub Pages |

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
├── docs/               GitHub Pages static build (generated, committed)
│   ├── index.html      dashboard in static-snapshot mode
│   ├── data/           precomputed API responses
│   └── vendor/         Leaflet + Chart.js
├── requirements.txt
├── Dockerfile             CPU-only image (API + dashboard, one origin)
├── render.yaml            Render Blueprint – one-click free deploy
├── .dockerignore
└── README.md
```

Frontend note: **no CDN, no internet at runtime.** Leaflet and Chart.js are
bundled in `app/vendor/`, and the basemap is drawn offline as a vector
graticule with district outlines (no remote map tiles).

---

## Deploy to the cloud (free, no credit card)

The prototype is ready to run as a container. `Dockerfile` installs the
**CPU-only** PyTorch wheel (the runtime needs roughly 300 MB of RAM, so it
fits a free 512 MB instance), and `src/api.py` serves the dashboard and the
API from the same origin — one public URL, nothing to configure.

### Live demo — GitHub Pages (static snapshot, no account needed)

**<https://vernicasjmhss-pixel.github.io/aquapurity/>**

GitHub Pages can host HTML/JS but not Python, so `docs/` holds a **static
snapshot**: `src/export_static.py` asks a running API for every request the
dashboard can ever make and writes the answers to `docs/data/*.json`. The
dashboard then reads those files instead of calling the API (the static-mode
branch of `apiFetch` in `app/index.html`).

The interface space is small and deterministic (seed 42, CPU-only, no user
input beyond the controls), which is what makes this possible:

| Endpoint | Requests |
|---|---|
| `/wells`, `/sites` | 2 districts each |
| `/rank` | 2 districts × 4 structures × 2 constrain modes = 16 |
| `/forecast` | 120 wells × 3 horizons = 360 |
| `/explain` | 120 wells × 3 horizons + 80 sites = 440 |

The responses are the API's **real output**, not a re-implementation, so the
numbers are identical. Headless reviewing works too: no cold starts, nothing
to keep awake. Rebuild it after any pipeline change:

```bash
uvicorn src.api:app --port 8000 &
python src/export_static.py --api http://127.0.0.1:8000
python -m http.server 8095 -d docs      # preview locally
```

Pages publishes the `docs/` folder from `main`; commit the regenerated
`docs/` to update the live site.

### Render (recommended for the live API)

The quickest route is the button at the top of this README (or this link):

**<https://render.com/deploy?repo=https://github.com/vernicasjmhss-pixel/aquapurity>**

1. Create a free account at <https://render.com> (no card required).
2. Open the link — Render reads `render.yaml`, asks you to confirm, and
   applies the Blueprint.
3. It builds `./Dockerfile` and deploys on the **Free** plan. You get a URL
   like `https://aquapurity.onrender.com`.

The 3-step alternative is **New + → Blueprint → pick the `aquapurity`
repository → Apply**. First build takes a few minutes because the PyTorch
layer is ~500 MB.

<details>
<summary>Any other Docker host (Koyeb, Cloud Run, Fly, a VM …)</summary>

```bash
docker build -t aquapurity .
docker run -p 8000:8000 aquapurity     # -> http://localhost:8000
```

The container listens on `$PORT` (default `8000`) on `0.0.0.0`, exposes
`/health` for platform health checks, and needs no volumes or secrets.
</details>

**Free-tier behaviour to expect:** a free web service is put to sleep after
~15 minutes without traffic, so the first request after a quiet period takes
roughly 30–60 s (loading torch + the models). Every request after that is fast.

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
The API **and** the dashboard are served from this one process:

| URL | What |
|---|---|
| http://127.0.0.1:8000 | Dashboard (the `app/` folder) |
| http://127.0.0.1:8000/docs | Interactive API docs |
| http://127.0.0.1:8000/health | Liveness probe |

### Step 4 – Open the dashboard
Just open **http://127.0.0.1:8000** — no second server needed. The page and
the JSON API share an origin, so there is no host to configure.
A visible **“Synthetic data”** badge and per-popup warnings remind you that
all values are generated, not measured.

> Opening `app/index.html` directly from disk (or from a separate static
> server on port 8090) still works: the page then falls back to
> `http://127.0.0.1:8000` for the API.

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/sites?district=&structure=` | Candidate sites with scores |
| GET | `/wells?district=` | Wells with latest water level |
| POST | `/forecast` | STGNN forecast + uncertainty range |
| POST | `/rank` | AHP+TOPSIS ranked sites |
| POST | `/explain` | Integrated Gradients / TOPSIS attribution |
| GET | `/health` | Liveness probe (used by the host) |
| GET | `/` | Dashboard (`app/index.html`) |

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
