# models/critic.py

"""
=============================================================
CRITIC NETWORK FOR RMSA PPO
=============================================================

Goal:
-----

Estimate V(s) where state includes:

    - edge graph (network state)
    - path embedding
    - modulation embedding
    - slot state summary

=============================================================
KEY DESIGN PRINCIPLE
=============================================================

The critic must be GLOBAL + CONTEXTUAL:

    V(s) = f(graph, path, mod, spectrum)

NOT just MLP(path).

=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# GRAPH POOLING (simple global summarization)
# ============================================================

class GraphPooling(nn.Module):

    def forward(self, edge_embeddings):

        """
        edge_embeddings: [E, D]
        """

        return edge_embeddings.mean(dim=0)


# ============================================================
# CRITIC NETWORK
# ============================================================

class Critic(nn.Module):

    def __init__(
        self,
        edge_dim,
        path_dim,
        mod_dim,
        slot_dim,
        hidden_dim=128
    ):

        super().__init__()

        self.graph_pool = GraphPooling()

        # -----------------------------------------------------
        # GRAPH ENCODING
        # -----------------------------------------------------

        self.graph_encoder = nn.Sequential(

            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)

        )

        # -----------------------------------------------------
        # CONTEXT ENCODER
        # -----------------------------------------------------

        self.context_mlp = nn.Sequential(

            nn.Linear(
                hidden_dim * 3,
                hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)

        )

        # -----------------------------------------------------
        # VALUE HEAD
        # -----------------------------------------------------

        self.value_head = nn.Sequential(

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)

        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        edge_embeddings,
        path_embedding=None,
        mod_embedding=None,
        slot_embedding=None
    ):

        """
        Returns scalar value V(s)
        """

        # -----------------------------------------------------
        # GRAPH SUMMARY
        # -----------------------------------------------------

        g = self.graph_pool(edge_embeddings)
        g = self.graph_encoder(g)

        # -----------------------------------------------------
        # SAFE DEFAULTS
        # -----------------------------------------------------

        if path_embedding is None:
            path_embedding = torch.zeros_like(g)

        if mod_embedding is None:
            mod_embedding = torch.zeros_like(g)

        if slot_embedding is None:
            slot_embedding = torch.zeros_like(g)

        # -----------------------------------------------------
        # CONTEXT FUSION
        # -----------------------------------------------------

        x = torch.cat(
            [g, path_embedding, mod_embedding],
            dim=-1
        )

        x = self.context_mlp(x)

        # -----------------------------------------------------
        # VALUE OUTPUT
        # -----------------------------------------------------

        value = self.value_head(x).squeeze(-1)

        return value