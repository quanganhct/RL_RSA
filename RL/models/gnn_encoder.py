# models/gnn_encoder.py

"""
=============================================================
EDGE-CENTRIC GNN ENCODER FOR RMSA
=============================================================

Production-grade edge-aware GNN encoder.

=============================================================
KEY FEATURES
=============================================================

✔ Edge-centric representation

✔ Supports BOTH:
    - rollout inference
    - PPO batched training

✔ Residual message passing

✔ LayerNorm + GELU

✔ Batched graph support WITHOUT PyG batching object

✔ GPU friendly

✔ Stable PPO training

=============================================================
INPUTS
=============================================================

edge_features:
    rollout : [E,F]
    training: [B,E,F]

edge_index:
    rollout : [2, M]
    training: [B,2,M]

where:
    E = number of edges
    M = number of adjacency connections

=============================================================
OUTPUTS
=============================================================

edge_embeddings:
    rollout : [E,H]
    training: [B,E,H]

=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing


# ============================================================
# EDGE MESSAGE PASSING
# ============================================================

class EdgeConvLayer(MessagePassing):

    def __init__(
        self,
        hidden_dim,
        dropout=0.1
    ):

        super().__init__(aggr="mean")

        self.msg_mlp = nn.Sequential(

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim)
        )

        self.update_mlp = nn.Sequential(

            nn.LayerNorm(hidden_dim),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim)
        )

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

        out = self.propagate(
            edge_index,
            x=x
        )

        out = self.norm(
            x + self.dropout(out)
        )

        return out

    # ========================================================
    # MESSAGE
    # ========================================================

    def message(
        self,
        x_i,
        x_j
    ):

        msg = torch.cat(
            [x_i, x_j],
            dim=-1
        )

        return self.msg_mlp(msg)

    # ========================================================
    # UPDATE
    # ========================================================

    def update(
        self,
        aggr_out
    ):

        return self.update_mlp(aggr_out)


# ============================================================
# EDGE-CENTRIC GNN
# ============================================================

class GNNEncoder(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=4,
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
                hidden_dim,
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
    # SINGLE GRAPH FORWARD
    # ========================================================

    def forward_single(
        self,
        edge_features,
        edge_index
    ):

        """
        edge_features : [E,F]
        edge_index    : [2,M]
        """
        # print(f'edge_features = {edge_features.shape}')
        x = self.input_proj(edge_features)

        for layer in self.layers:

            x = layer(
                x,
                edge_index
            )

        x = self.output_proj(x)

        return x

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        edge_features,
        edge_index
    ):

        """
        Supports BOTH:

            rollout:
                edge_features = [E,F]
                edge_index    = [2,M]

            training:
                edge_features = [B,E,F]
                edge_index    = [B,2,M]
        """

        # ----------------------------------------------------
        # SINGLE SAMPLE SUPPORT
        # ----------------------------------------------------

        single_sample = False

        if edge_features.dim() == 2:

            single_sample = True

            edge_features = edge_features.unsqueeze(0)

            edge_index = edge_index.unsqueeze(0)

        # ----------------------------------------------------
        # SAFETY CHECKS
        # ----------------------------------------------------

        assert edge_features.dim() == 3, \
            f"edge_features must be [B,E,F], got {edge_features.shape}"

        assert edge_index.dim() == 3, \
            f"edge_index must be [B,2,M], got {edge_index.shape}"

        B = edge_features.shape[0]

        outputs = []

        # ----------------------------------------------------
        # PROCESS EACH GRAPH
        # ----------------------------------------------------

        for b in range(B):

            out = self.forward_single(
                edge_features[b],
                edge_index[b]
            )

            outputs.append(out)

        x = torch.stack(outputs, dim=0)

        # ----------------------------------------------------
        # SINGLE SAMPLE FIX
        # ----------------------------------------------------

        if single_sample:

            x = x.squeeze(0)

        return x