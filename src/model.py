"""
model.py
--------
Stage 2b: Spatiotemporal GNN for multi-horizon groundwater level forecasting.

Architecture (all kept simple for CPU + readability):
  GCN layer  :  A_hat @ X_t @ W_gcn   →  spatial features at each time step
  GRU        :  encodes the T-step sequence of GCN outputs
  MLP head   :  maps GRU hidden state → 3 forecast horizons (1, 3, 6 months)

Input  :  (B, T, N, F)   B=batch, T=window=12, N=wells_per_district, F=features=2
          features = [water_level_m, rainfall_mm]
Output :  (B, N, 3)      predictions at horizons [+1, +3, +6] months

GCN operates on the spatial dim at each time step (applied identically to each
step → the same learned W_gcn is shared across time).
"""

import torch
import torch.nn as nn


class GCNLayer(nn.Module):
    """
    A single Graph Convolutional layer.
    Out = ReLU( A_hat @ X @ W + b )
    A_hat is fixed (precomputed normalised adjacency) – passed as a buffer.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # learnable weight matrix
        self.W = nn.Linear(in_features, out_features, bias=True)
        self.act = nn.ReLU()

    def forward(self, A_hat: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        A_hat : (N, N)  – normalised adjacency (shared across batch)
        X     : (B, N, F) – node features
        Returns (B, N, out_features)
        """
        # message passing: aggregate neighbour features
        agg = torch.matmul(A_hat, X)        # (B, N, F) – or just (N, F)
        return self.act(self.W(agg))


class STGNN(nn.Module):
    """
    Spatiotemporal Graph Neural Network.

    Parameters
    ----------
    n_nodes    : number of wells in the district
    n_features : input features per node (default 2: [wl, rain])
    gcn_out    : output dimension of the GCN layer
    gru_hidden : hidden size of the GRU
    n_horizons : number of forecast steps (default 3: +1, +3, +6 months)
    """

    def __init__(
        self,
        n_nodes: int,
        n_features: int = 2,
        gcn_out: int = 16,
        gru_hidden: int = 32,
        n_horizons: int = 3,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.gcn_out = gcn_out
        self.gru_hidden = gru_hidden

        self.gcn = GCNLayer(n_features, gcn_out)

        # GRU processes the T-length sequence of GCN outputs per node
        # Input at each step: gcn_out features; each node is treated independently
        self.gru = nn.GRU(
            input_size=gcn_out,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,  # (batch, seq, feature)
        )

        # MLP head: maps GRU final hidden → n_horizons predictions
        self.mlp = nn.Sequential(
            nn.Linear(gru_hidden, 16),
            nn.ReLU(),
            nn.Linear(16, n_horizons),
        )

    def forward(
        self, A_hat: torch.Tensor, X: torch.Tensor
    ) -> torch.Tensor:
        """
        A_hat : (N, N)
        X     : (B, T, N, F)
        Returns: (B, N, n_horizons)
        """
        B, T, N, F = X.shape

        # ── Step 1: GCN over each time step ───────────────────────────────────
        # Reshape to (B*T, N, F) to apply GCN in one pass
        X_flat = X.reshape(B * T, N, F)
        gcn_out = self.gcn(A_hat, X_flat)           # (B*T, N, gcn_out)
        gcn_out = gcn_out.reshape(B, T, N, self.gcn_out)

        # ── Step 2: GRU per node ──────────────────────────────────────────────
        # Permute to (B, N, T, gcn_out) then flatten batch+node dims
        gcn_seq = gcn_out.permute(0, 2, 1, 3)       # (B, N, T, gcn_out)
        gcn_seq = gcn_seq.reshape(B * N, T, self.gcn_out)

        _, h_n = self.gru(gcn_seq)                  # h_n: (1, B*N, gru_hidden)
        h_n = h_n.squeeze(0)                         # (B*N, gru_hidden)
        h_n = h_n.reshape(B, N, self.gru_hidden)

        # ── Step 3: MLP head ──────────────────────────────────────────────────
        out = self.mlp(h_n)                          # (B, N, n_horizons)
        return out
