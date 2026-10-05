"""
train.py
--------
Stage 2c: Training loop for the STGNN.

Pipeline:
  1. Load data/wells.csv, water_levels.csv, rainfall.csv.
  2. Per district: build graph, build sliding-window tensors, split chronologically.
  3. Train STGNN with early stopping; save best model.
  4. Evaluate RMSE, NSE, KGE vs a persistence baseline and a linear-trend baseline.
  5. Save metrics.json and per-district model checkpoints.

All random operations use SEED=42 for reproducibility.
"""

import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Force UTF-8 stdout: the metrics table uses box-drawing characters that the
# default Windows cp1252 code page cannot encode (would raise UnicodeEncodeError).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from graph import build_adj_matrix
from model import STGNN

# ── config ────────────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

WINDOW = 12          # months of history used as input
HORIZONS = [1, 3, 6]  # forecast horizons (months ahead)
N_HORIZONS = len(HORIZONS)

EPOCHS = 80
BATCH_SIZE = 16
LR = 1e-3
PATIENCE = 12        # early-stopping patience
TRAIN_FRAC = 0.75    # chronological split

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = DATA_DIR   # save models alongside data for simplicity


# ──────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_data(district: str):
    """
    Returns:
      wells  : DataFrame for the district
      wl_mat : (T, N) numpy array of water levels (NaN filled by linear interp)
      rain_v : (T,)   numpy array of rainfall for the district
      dates  : list of 'YYYY-MM' strings
    """
    wells = pd.read_csv(os.path.join(DATA_DIR, "wells.csv"))
    wl_raw = pd.read_csv(os.path.join(DATA_DIR, "water_levels.csv"))
    rain_raw = pd.read_csv(os.path.join(DATA_DIR, "rainfall.csv"))

    wells = wells[wells["district"] == district].reset_index(drop=True)
    well_ids = wells["well_id"].tolist()

    wl_raw = wl_raw[wl_raw["well_id"].isin(well_ids)]
    # pivot: rows=date, cols=well_id
    wl_pivot = wl_raw.pivot(index="date", columns="well_id", values="water_level_m")
    wl_pivot = wl_pivot[well_ids]               # ensure column order matches wells
    wl_pivot = wl_pivot.sort_index()

    # fill missing values by linear interpolation then forward/back fill
    wl_pivot = wl_pivot.interpolate(method="linear", axis=0).ffill().bfill()

    rain_dist = (
        rain_raw[rain_raw["district"] == district]
        .sort_values("date")
        .set_index("date")["rainfall_mm"]
    )

    dates = wl_pivot.index.tolist()
    wl_mat = wl_pivot.values.astype(np.float32)   # (T, N)
    rain_v = rain_dist.loc[dates].values.astype(np.float32)  # (T,)

    return wells, wl_mat, rain_v, dates


# ──────────────────────────────────────────────────────────────────────────────
# Build sliding-window samples
# ──────────────────────────────────────────────────────────────────────────────
def make_samples(wl_mat: np.ndarray, rain_v: np.ndarray,
                 window: int, horizons: list):
    """
    Creates (X, Y) where:
      X shape : (samples, window, N, 2)   features = [water_level, rainfall]
      Y shape : (samples, N, len(horizons))  targets = future WL
    """
    T, N = wl_mat.shape
    max_h = max(horizons)
    xs, ys = [], []

    # normalise (z-score) per node to help training
    wl_mean = wl_mat.mean(axis=0, keepdims=True)
    wl_std  = wl_mat.std(axis=0, keepdims=True) + 1e-6
    rain_mean = rain_v.mean()
    rain_std  = rain_v.std() + 1e-6

    wl_norm = (wl_mat - wl_mean) / wl_std
    rain_norm = (rain_v - rain_mean) / rain_std

    for t in range(window, T - max_h + 1):
        # X: window of [wl, rain] – rain broadcast to all nodes
        x_wl   = wl_norm[t - window : t]                       # (W, N)
        x_rain = rain_norm[t - window : t, np.newaxis]          # (W, 1)
        x_rain = np.repeat(x_rain, N, axis=1)                   # (W, N)
        X = np.stack([x_wl, x_rain], axis=-1)                   # (W, N, 2)
        xs.append(X)

        # Y: targets at each horizon (normalised)
        y = np.stack(
            [(wl_norm[t + h - 1]) for h in horizons], axis=-1  # (N, H)
        )
        ys.append(y)

    X_arr = np.array(xs, dtype=np.float32)   # (S, W, N, 2)
    Y_arr = np.array(ys, dtype=np.float32)   # (S, N, H)

    # also store stats for inverse transform
    stats = {"wl_mean": wl_mean, "wl_std": wl_std,
             "rain_mean": rain_mean, "rain_std": rain_std}
    return X_arr, Y_arr, stats


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency.  1 = perfect, <0 = worse than mean."""
    num = np.sum((y_true - y_pred) ** 2)
    den = np.sum((y_true - y_true.mean()) ** 2) + 1e-9
    return float(1 - num / den)

def kge(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Kling-Gupta Efficiency.  1 = perfect."""
    r = float(np.corrcoef(y_true.ravel(), y_pred.ravel())[0, 1])
    alpha = y_pred.std() / (y_true.std() + 1e-9)
    beta  = y_pred.mean() / (y_true.mean() + 1e-9)
    return float(1 - math.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


# ──────────────────────────────────────────────────────────────────────────────
# Baselines
# ──────────────────────────────────────────────────────────────────────────────
def persistence_baseline(X: np.ndarray) -> np.ndarray:
    """Last observed water level repeated for all horizons. Shape (S, N, H)."""
    last_wl = X[:, -1, :, 0]                  # (S, N) – last time-step, feature 0
    return np.stack([last_wl] * N_HORIZONS, axis=-1)  # (S, N, H)

def linear_baseline(X: np.ndarray, horizons: list) -> np.ndarray:
    """
    Fit a linear trend over the window and extrapolate to each horizon.
    Shape (S, N, H).
    """
    S, W, N, _ = X.shape
    t = np.arange(W, dtype=np.float32)
    preds = np.zeros((S, N, len(horizons)), dtype=np.float32)
    for s in range(S):
        for n in range(N):
            wl_seq = X[s, :, n, 0]              # (W,)
            # least-squares linear fit
            slope, intercept = np.polyfit(t, wl_seq, 1)
            for hi, h in enumerate(horizons):
                preds[s, n, hi] = slope * (W + h - 1) + intercept
    return preds


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────
def train_district(district: str) -> dict:
    """Train and evaluate STGNN for one district. Returns metrics dict."""
    print(f"\n{'='*55}")
    print(f"  District: {district}")
    print(f"{'='*55}")

    # ── load & prepare ────────────────────────────────────────────────────────
    wells, wl_mat, rain_v, dates = load_data(district)
    N = len(wells)
    A_hat, well_ids = build_adj_matrix(wells)
    A_hat_t = torch.tensor(A_hat)

    X_all, Y_all, stats = make_samples(wl_mat, rain_v, WINDOW, HORIZONS)
    S = len(X_all)
    split = int(S * TRAIN_FRAC)

    X_tr, Y_tr = X_all[:split], Y_all[:split]
    X_val, Y_val = X_all[split:], Y_all[split:]

    print(f"  Samples: {S} total  (train={split}, val={S-split})")
    print(f"  Wells  : {N}")

    # ── datasets & loaders ────────────────────────────────────────────────────
    tr_ds  = TensorDataset(torch.tensor(X_tr), torch.tensor(Y_tr))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(Y_val))
    tr_ld  = DataLoader(tr_ds,  batch_size=BATCH_SIZE, shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ── model ─────────────────────────────────────────────────────────────────
    model = STGNN(n_nodes=N, n_features=2, gcn_out=16, gru_hidden=32,
                  n_horizons=N_HORIZONS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # ── training with early stopping ──────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_ld:
            optimizer.zero_grad()
            pred = model(A_hat_t, xb)   # (B, N, H)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(tr_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_ld:
                pred = model(A_hat_t, xb)
                val_loss += criterion(pred, yb).item() * len(xb)
        val_loss /= len(val_ds)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  "
                  f"train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break

    elapsed = time.time() - t0
    print(f"  Training time: {elapsed:.1f}s")

    # ── evaluate best model ───────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in val_ld:
            pred = model(A_hat_t, xb)
            all_preds.append(pred.numpy())
            all_true.append(yb.numpy())

    y_pred_norm = np.concatenate(all_preds, axis=0)   # (val_S, N, H)
    y_true_norm = np.concatenate(all_true, axis=0)

    # ── baselines on validation set ───────────────────────────────────────────
    pers_pred  = persistence_baseline(X_val)
    linear_pred = linear_baseline(X_val, HORIZONS)

    # ── per-horizon metrics: RMSE, NSE and KGE for model + baselines ─────────
    # Fixed column widths keep the table aligned on every terminal.
    W = [11, 21, 21, 21]                    # column widths (chars)

    def _line(left, fill, mid, right):
        """Draw a horizontal rule with the fixed column widths."""
        return "  " + left + mid.join(fill * w for w in W) + right

    def _row(*cells):
        """Print one table row, each cell centred in its fixed-width column."""
        return "  │" + "│".join(
            f"{c:^{w}s}" for c, w in zip(cells, W)
        ) + "│"

    print("\n" + _line("┌", "─", "┬", "┐"))
    print(_row("Horizon", "STGNN (model)", "Persistence", "Linear trend"))
    print(_row("", "RMSE/NSE/KGE", "RMSE/NSE/KGE", "RMSE/NSE/KGE"))
    print(_line("├", "─", "┼", "┤"))

    horizon_metrics = {}
    for hi, h in enumerate(HORIZONS):
        yt = y_true_norm[:, :, hi].ravel()
        yp = y_pred_norm[:, :, hi].ravel()
        pp = pers_pred  [:, :, hi].ravel()
        lp = linear_pred[:, :, hi].ravel()

        # metrics for the model and for both naive baselines
        m = (rmse(yt, yp),  nse(yt, yp),  kge(yt, yp))
        p = (rmse(yt, pp),  nse(yt, pp),  kge(yt, pp))
        l = (rmse(yt, lp),  nse(yt, lp),  kge(yt, lp))

        fmt = lambda t: "/".join(f"{v:.3f}" for v in t)
        print(_row(f"+{h} months", fmt(m), fmt(p), fmt(l)))

        horizon_metrics[f"h{h}"] = {
            "model":       dict(zip(("rmse", "nse", "kge"),
                                    (round(v, 4) for v in m))),
            "persistence": dict(zip(("rmse", "nse", "kge"),
                                    (round(v, 4) for v in p))),
            "linear":      dict(zip(("rmse", "nse", "kge"),
                                    (round(v, 4) for v in l))),
        }

    print(_line("└", "─", "┴", "┘"))

    # ── save model ────────────────────────────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, f"model_{district.lower()}.pt")
    torch.save(
        {
            "state_dict": best_state,
            "n_nodes": N,
            "well_ids": well_ids,
            "A_hat": A_hat,
            "stats": stats,
            "horizons": HORIZONS,
        },
        model_path,
    )
    print(f"\n  Model saved → {model_path}")

    return {"district": district, "metrics": horizon_metrics}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Aquapurity – STGNN Training  [SYNTHETIC DATA]")
    print("=" * 55)

    all_results = {}
    for district in ["Thanjavur", "Pudukkottai"]:
        result = train_district(district)
        all_results[district] = result["metrics"]

    # ── save metrics.json ─────────────────────────────────────────────────────
    metrics_path = os.path.join(DATA_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Metrics saved] {metrics_path}")
    print("\n[NOTE] All data is SYNTHETIC – metrics demonstrate the pipeline only.")


if __name__ == "__main__":
    main()
