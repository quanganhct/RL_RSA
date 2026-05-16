# models/critic.py

"""
=============================================================
HIERARCHICAL PPO CRITIC FOR RMSA
=============================================================

Production-grade critic network for RMSA PPO.

=============================================================
GOAL
=============================================================

Estimate:

    V(s)

where state includes:

    - global network graph state
    - selected path context
    - selected modulation context
    - selected slot context

=============================================================
KEY IMPROVEMENTS
=============================================================

✔ FULL batch support

✔ Supports BOTH:
    - rollout inference
    - PPO training batches

✔ Stable PPO architecture:
    - LayerNorm
    - GELU
    - residual MLPs
    - dropout

✔ Proper graph pooling

✔ Global + local state fusion

✔ Handles optional hierarchical stages:
    - path stage
    - modulation stage
    - slot stage

=============================================================
INPUTS
=============================================================

edge_embeddings:
    rollout : [E,D]
    training: [B,E,D]

path_embedding:
    rollout : [D]
    training: [B,D]

mod_embedding:
    rollout : [D]
    training: [B,D]

slot_embedding:
    rollout : [D]
    training: [B,D]

=============================================================
OUTPUTS
=============================================================

value:
    rollout : scalar
    training: [B]

=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# RESIDUAL BLOCK
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(
        self,
        dim,
        hidden_mult=2,
        dropout=0.1
    ):

        super().__init__()

        hidden = dim * hidden_mult

        self.norm = nn.LayerNorm(dim)

        self.net = nn.Sequential(

            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):

        return x + self.net(
            self.norm(x)
        )


# ============================================================
# GRAPH POOLING
# ============================================================

class GraphPooling(nn.Module):

    """
    Global graph summarization.
    """

    def forward(self, edge_embeddings):

        """
        edge_embeddings:
            rollout : [E,D]
            training: [B,E,D]
        """

        if edge_embeddings.dim() == 2:

            return edge_embeddings.mean(dim=0)

        elif edge_embeddings.dim() == 3:

            return edge_embeddings.mean(dim=1)

        else:

            raise ValueError(
                f"Invalid edge_embeddings shape: {edge_embeddings.shape}"
            )


# ============================================================
# CRITIC
# ============================================================

class Critic(nn.Module):

    def __init__(
        self,
        edge_dim,
        path_dim,
        mod_dim,
        slot_dim=None,
        hidden_dim=128,
        num_layers=3,
        dropout=0.1
    ):

        super().__init__()

        self.hidden_dim = hidden_dim

        self.graph_pool = GraphPooling()

        # ----------------------------------------------------
        # GRAPH ENCODER
        # ----------------------------------------------------

        self.graph_encoder = nn.Sequential(

            nn.Linear(edge_dim, hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim)
        )

        # ----------------------------------------------------
        # PATH ENCODER
        # ----------------------------------------------------

        self.path_encoder = nn.Sequential(

            nn.Linear(path_dim, hidden_dim),
            nn.GELU()
        )

        # ----------------------------------------------------
        # MOD ENCODER
        # ----------------------------------------------------

        self.mod_encoder = nn.Sequential(

            nn.Linear(mod_dim, hidden_dim),
            nn.GELU()
        )

        # ----------------------------------------------------
        # SLOT ENCODER (OPTIONAL)
        # ----------------------------------------------------

        if slot_dim is not None:

            self.slot_encoder = nn.Sequential(

                nn.Linear(slot_dim, hidden_dim),
                nn.GELU()
            )

        else:

            self.slot_encoder = None

        # ----------------------------------------------------
        # GLOBAL FUSION
        # ----------------------------------------------------

        fusion_dim = hidden_dim * 4

        self.fusion = nn.Sequential(

            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU()
        )

        # ----------------------------------------------------
        # RESIDUAL PROCESSING
        # ----------------------------------------------------

        self.blocks = nn.ModuleList([

            ResidualBlock(
                hidden_dim,
                dropout=dropout
            )

            for _ in range(num_layers)

        ])

        # ----------------------------------------------------
        # VALUE HEAD
        # ----------------------------------------------------

        self.value_head = nn.Sequential(

            nn.LayerNorm(hidden_dim),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, 1)
        )

    # ========================================================
    # SAFE DEFAULT
    # ========================================================

    def _safe_zero(
        self,
        ref_tensor
    ):

        return torch.zeros_like(ref_tensor)

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
        =====================================================
        Supports BOTH:

            rollout:
                edge_embeddings = [E,D]

            training:
                edge_embeddings = [B,E,D]
        =====================================================
        """

        # ----------------------------------------------------
        # SINGLE SAMPLE SUPPORT
        # ----------------------------------------------------

        single_sample = False

        if edge_embeddings.dim() == 2:

            single_sample = True

            edge_embeddings = edge_embeddings.unsqueeze(0)

            if path_embedding is not None:
                path_embedding = path_embedding.unsqueeze(0)

            if mod_embedding is not None:
                mod_embedding = mod_embedding.unsqueeze(0)

            if slot_embedding is not None:
                slot_embedding = slot_embedding.unsqueeze(0)

        # ----------------------------------------------------
        # SAFETY CHECKS
        # ----------------------------------------------------

        assert edge_embeddings.dim() == 3, \
            f"edge_embeddings must be [B,E,D], got {edge_embeddings.shape}"

        B = edge_embeddings.shape[0]

        # ----------------------------------------------------
        # GRAPH SUMMARY
        # ----------------------------------------------------

        g = self.graph_pool(
            edge_embeddings
        )

        g = self.graph_encoder(g)

        # [B,H]

        # ----------------------------------------------------
        # SAFE DEFAULTS
        # ----------------------------------------------------

        if path_embedding is None:

            path_embedding = self._safe_zero(g)

        else:

            path_embedding = self.path_encoder(
                path_embedding
            )

        if mod_embedding is None:

            mod_embedding = self._safe_zero(g)

        else:

            mod_embedding = self.mod_encoder(
                mod_embedding
            )

        if slot_embedding is None:

            slot_embedding = self._safe_zero(g)

        else:

            if self.slot_encoder is not None:

                slot_embedding = self.slot_encoder(
                    slot_embedding
                )

        # ----------------------------------------------------
        # GLOBAL FUSION
        # ----------------------------------------------------

        x = torch.cat(

            [
                g,
                path_embedding,
                mod_embedding,
                slot_embedding
            ],

            dim=-1
        )

        x = self.fusion(x)

        # ----------------------------------------------------
        # RESIDUAL PROCESSING
        # ----------------------------------------------------

        for block in self.blocks:

            x = block(x)

        # ----------------------------------------------------
        # VALUE PREDICTION
        # ----------------------------------------------------

        value = self.value_head(x).squeeze(-1)

        # ----------------------------------------------------
        # SINGLE SAMPLE FIX
        # ----------------------------------------------------

        if single_sample:

            value = value.squeeze(0)

        return value