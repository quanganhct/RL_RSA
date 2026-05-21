# models/gnn_encoder.py

"""
=============================================================
EDGE-CENTRIC GNN ENCODER FOR RMSA
=============================================================

Production-grade batched edge-centric GNN.

=============================================================
DESIGN
=============================================================

Tensor contracts are ALWAYS batched:

edge_features:
    [B, E, F]

edge_index:
    [B, 2, M]

Outputs:
    [B, E, H]

where:
    B = batch size
    E = number of edges
    F = edge feature dimension
    H = hidden dimension
    M = number of graph connections

=============================================================
KEY FEATURES
=============================================================

✔ Fully batched PPO-compatible

✔ No rollout special-casing

✔ Vectorized graph batching

✔ Residual message passing

✔ LayerNorm + GELU

✔ GPU efficient

✔ Stable PPO training

✔ Supports duplicated edge_index per batch

=============================================================
"""

import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing


# ============================================================
# EDGE MESSAGE PASSING LAYER
# ============================================================

class EdgeConvLayer(MessagePassing):

    def __init__(
        self,
        hidden_dim,
        dropout=0.1
    ):

        super().__init__(aggr="mean")

        # ----------------------------------------------------
        # MESSAGE NETWORK
        # ----------------------------------------------------

        self.msg_mlp = nn.Sequential(

            nn.Linear(hidden_dim * 2, hidden_dim),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim)
        )

        # ----------------------------------------------------
        # UPDATE NETWORK
        # ----------------------------------------------------

        self.update_mlp = nn.Sequential(

            nn.LayerNorm(hidden_dim),

            nn.Linear(hidden_dim, hidden_dim),

            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim)
        )

        # ----------------------------------------------------
        # RESIDUAL NORM
        # ----------------------------------------------------

        self.norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x,
        edge_index
    ):

        """
        ====================================================
        INPUTS
        ====================================================

        x:
            [B, E, H]

        edge_index:
            [B, 2, M]

        ====================================================
        OUTPUT
        ====================================================

        out:
            [B, E, H]

        ====================================================
        """

        B, E, H = x.shape

        # ----------------------------------------------------
        # FLATTEN FEATURES
        # ----------------------------------------------------

        x_flat = x.reshape(B * E, H)

        # ----------------------------------------------------
        # EDGE INDICES
        # ----------------------------------------------------

        src = edge_index[:, 0]   # [B,M]
        dst = edge_index[:, 1]   # [B,M]

        # ----------------------------------------------------
        # OFFSET EACH GRAPH
        # ----------------------------------------------------

        offsets = (
            torch.arange(
                B,
                device=x.device
            ) * E
        ).view(B, 1)

        src = src + offsets
        dst = dst + offsets

        # ----------------------------------------------------
        # GLOBAL EDGE INDEX
        # ----------------------------------------------------

        edge_index_batch = torch.stack([

            src.reshape(-1),
            dst.reshape(-1)

        ], dim=0)

        # ----------------------------------------------------
        # MESSAGE PASSING
        # ----------------------------------------------------

        out = self.propagate(
            edge_index_batch,
            x=x_flat
        )

        # ----------------------------------------------------
        # RESIDUAL CONNECTION
        # ----------------------------------------------------

        out = self.norm(
            x_flat + self.dropout(out)
        )

        # ----------------------------------------------------
        # RESHAPE BACK
        # ----------------------------------------------------

        out = out.view(B, E, H)

        return out

    # ========================================================
    # MESSAGE FUNCTION
    # ========================================================

    def message(
        self,
        x_i,
        x_j
    ):

        """
        x_i:
            destination embeddings

        x_j:
            source embeddings
        """

        msg = torch.cat(
            [x_i, x_j],
            dim=-1
        )

        return self.msg_mlp(msg)

    # ========================================================
    # UPDATE FUNCTION
    # ========================================================

    def update(
        self,
        aggr_out
    ):

        return self.update_mlp(aggr_out)


# ============================================================
# GNN ENCODER
# ============================================================

class GNNEncoder(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=2,#4,
        dropout=0.1
    ):

        super().__init__()

        self.hidden_dim = hidden_dim

        # ----------------------------------------------------
        # INPUT PROJECTION
        # ----------------------------------------------------

        self.input_proj = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),

            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim)
        )

        # ----------------------------------------------------
        # MESSAGE PASSING STACK
        # ----------------------------------------------------

        self.layers = nn.ModuleList([

            EdgeConvLayer(
                hidden_dim=hidden_dim,
                dropout=dropout
            )

            for _ in range(num_layers)

        ])

        # ----------------------------------------------------
        # OUTPUT PROJECTION
        # ----------------------------------------------------

        self.output_proj = nn.Sequential(

            nn.LayerNorm(hidden_dim),

            nn.Linear(hidden_dim, hidden_dim),

            nn.GELU()
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        edge_features,
        edge_index
    ):

        """
        ====================================================
        INPUTS
        ====================================================

        edge_features:
            [B,E,F]

        edge_index:
            [B,2,M]

        ====================================================
        OUTPUT
        ====================================================

        edge_embeddings:
            [B,E,H]

        ====================================================
        """

        # ----------------------------------------------------
        # SAFETY CHECKS
        # ----------------------------------------------------

        assert edge_features.dim() == 3, (
            f"edge_features must be [B,E,F], "
            f"got {edge_features.shape}"
        )

        assert edge_index.dim() == 3, (
            f"edge_index must be [B,2,M], "
            f"got {edge_index.shape}"
        )

        # ----------------------------------------------------
        # INPUT PROJECTION
        # ----------------------------------------------------

        x = self.input_proj(edge_features)

        # ----------------------------------------------------
        # MESSAGE PASSING
        # ----------------------------------------------------

        for layer in self.layers:

            x = layer(
                x,
                edge_index
            )

        # ----------------------------------------------------
        # OUTPUT PROJECTION
        # ----------------------------------------------------

        x = self.output_proj(x)

        return x