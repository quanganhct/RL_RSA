# models/gnn_encoder.py

"""
=============================================================
EDGE-CENTRIC GNN ENCODER FOR RMSA
=============================================================

We model optical network as:

    nodes: optional
    edges: PRIMARY representation

Each edge contains:

    frag_rate_abp
    max_available_block
    occupied_slots
    min_snr
    distance features

=============================================================
ARCHITECTURE
=============================================================

Edge features → Message Passing → Edge Embeddings

We use:

- GAT or GraphSAGE style updates
- residual connections
- layer norm

=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing


# ============================================================
# EDGE MESSAGE PASSING LAYER
# ============================================================

class EdgeConvLayer(MessagePassing):

    def __init__(self, in_dim, out_dim):

        super().__init__(aggr="mean")

        self.mlp = nn.Sequential(

            nn.Linear(in_dim * 2, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)

        )

        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index):

        return self.propagate(
            edge_index,
            x=x
        )

    def message(self, x_i, x_j):

        msg = torch.cat([x_i, x_j], dim=-1)

        return self.mlp(msg)

    def update(self, aggr_out, x):

        return self.norm(aggr_out + x)


# ============================================================
# EDGE GNN ENCODER
# ============================================================

class GNNEncoder(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=4
    ):

        super().__init__()

        self.input_proj = nn.Linear(
            input_dim,
            hidden_dim
        )

        self.layers = nn.ModuleList([

            EdgeConvLayer(
                hidden_dim,
                hidden_dim
            )

            for _ in range(num_layers)

        ])

        self.dropout = nn.Dropout(0.1)

    def forward(self, edge_features, edge_index):

        """
        edge_features: [E, F]
        edge_index: [2, E]
        """

        x = self.input_proj(edge_features)

        for layer in self.layers:

            residual = x

            x = layer(x, edge_index)

            x = F.relu(x + residual)

            x = self.dropout(x)

        return x