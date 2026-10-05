"""
graph.py
--------
Stage 2a: Builds a k-NN well graph and a normalised adjacency matrix.

Steps:
  1. Project (lat, lon) to a metric coordinate system using an equirectangular
     approximation (sufficient at district scale).
  2. For each well find k=5 nearest neighbours by Euclidean distance.
  3. Compute A_hat = D^{-1/2} (A + I) D^{-1/2}  (symmetric normalised adjacency).
  4. Return a numpy array and the ordered list of well_ids.

No third-party graph libraries – just NumPy and scikit-learn.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Force UTF-8 stdout so printed summaries work on cp1252 Windows consoles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── default k for the k-NN graph ──────────────────────────────────────────────
K_NEIGHBOURS = 5

# ── reference latitude for equirectangular projection ─────────────────────────
REF_LAT_DEG = 10.6          # centre of study area

# metres per degree of latitude (constant)
M_PER_DEG_LAT = 111_132.0
# metres per degree of longitude at reference latitude
M_PER_DEG_LON = M_PER_DEG_LAT * np.cos(np.radians(REF_LAT_DEG))


def latlon_to_xy(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Convert (lat, lon) arrays to metric (x, y) in metres."""
    x = lon * M_PER_DEG_LON
    y = lat * M_PER_DEG_LAT
    return np.column_stack([x, y])


def build_adj_matrix(wells: pd.DataFrame, k: int = K_NEIGHBOURS):
    """
    Build and return:
      A_hat  – normalised adjacency (N × N numpy array, float32)
      well_ids – ordered list of well_ids matching A_hat rows/cols
    """
    well_ids = wells["well_id"].tolist()
    N = len(well_ids)

    # project to metric
    xy = latlon_to_xy(wells["lat"].values, wells["lon"].values)

    # k-NN (k+1 because the first neighbour is the node itself)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree", metric="euclidean")
    nbrs.fit(xy)
    distances, indices = nbrs.kneighbors(xy)

    # build binary adjacency (undirected)
    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j_pos in range(1, k + 1):   # skip self (position 0)
            j = indices[i, j_pos]
            A[i, j] = 1.0
            A[j, i] = 1.0              # undirected: make symmetric

    # add self-loops: A_tilde = A + I
    A_tilde = A + np.eye(N, dtype=np.float32)

    # degree matrix D_tilde
    D = np.diag(A_tilde.sum(axis=1))

    # D^{-1/2}
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))

    # symmetric normalised adjacency A_hat = D^{-1/2} A_tilde D^{-1/2}
    A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt

    print(f"[Graph] N={N} wells, k={k}, "
          f"avg degree={A.sum(axis=1).mean():.2f}, "
          f"A_hat shape={A_hat.shape}")
    return A_hat.astype(np.float32), well_ids


# ── CLI helper: saves adjacency to data/ for inspection ───────────────────────
if __name__ == "__main__":
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    wells = pd.read_csv(os.path.join(DATA_DIR, "wells.csv"))
    for district in wells["district"].unique():
        sub = wells[wells["district"] == district].reset_index(drop=True)
        A_hat, ids = build_adj_matrix(sub)
        out = os.path.join(DATA_DIR, f"adj_{district.lower()}.npy")
        np.save(out, A_hat)
        print(f"  Saved {out}")
