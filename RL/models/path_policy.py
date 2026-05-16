# models/path_policy.py

"""
=============================================================
PATH POLICY FOR RMSA
=============================================================

This module selects ONE path among K candidate paths.

It operates on:

    - path embeddings from PathTransformer
    - optional path-level features
    - path masks

=============================================================
KEY FEATURES
=============================================================

✔ Proper PPO-compatible policy head

✔ Supports:
    - rollout inference
    - PPO batched training

✔ Attention-based global path reasoning

✔ Residual MLP scoring

✔ Proper masking

✔ Stable training:
    - LayerNorm
    - GELU
    - dropout

=============================================================
INPUTS
=============================================================

path_embeddings:
    rollout : [K,D]
    training: [B,K,D]

path_mask:
    rollout : [K]
    training: [B,K]

path_features (optional):
    rollout : [K,F]
    training: [B,K,F]

=============================================================
OUTPUTS
=============================================================

logits:
    rollout : [K]
    training: [B,K]

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
# PATH POLICY
# ============================================================

class PathPolicy(nn.Module):

    def __init__(
        self,
        path_dim,
        hidden_dim=128,
        feature_dim=None,
        num_layers=2,
        num_heads=4,
        dropout=0.1
    ):

        super().__init__()

        self.path_dim = path_dim

        # ----------------------------------------------------
        # OPTIONAL FEATURE FUSION
        # ----------------------------------------------------

        input_dim = path_dim

        if feature_dim is not None:

            input_dim += feature_dim

        self.input_proj = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),
            nn.GELU()
        )

        # ----------------------------------------------------
        # GLOBAL PATH ATTENTION
        # ----------------------------------------------------

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_dim)

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
        # FINAL SCORER
        # ----------------------------------------------------

        self.scorer = nn.Sequential(

            nn.LayerNorm(hidden_dim),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 1)
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        path_embeddings,
        path_mask=None,
        path_features=None
    ):

        """
        =====================================================
        INPUTS
        =====================================================

        path_embeddings:
            rollout : [K,D]
            training: [B,K,D]

        path_mask:
            rollout : [K]
            training: [B,K]

        path_features:
            rollout : [K,F]
            training: [B,K,F]
        """

        # ----------------------------------------------------
        # SINGLE SAMPLE SUPPORT
        # ----------------------------------------------------

        single_sample = False

        if path_embeddings.dim() == 2:

            single_sample = True

            path_embeddings = path_embeddings.unsqueeze(0)

            if path_mask is not None:
                path_mask = path_mask.unsqueeze(0)

            if path_features is not None:
                path_features = path_features.unsqueeze(0)

        # ----------------------------------------------------
        # SAFETY CHECKS
        # ----------------------------------------------------

        assert path_embeddings.dim() == 3, \
            f"path_embeddings must be [B,K,D], got {path_embeddings.shape}"

        B, K, D = path_embeddings.shape

        # ----------------------------------------------------
        # OPTIONAL FEATURE CONCAT
        # ----------------------------------------------------

        if path_features is not None:

            assert path_features.dim() == 3, \
                f"path_features must be [B,K,F], got {path_features.shape}"

            x = torch.cat(
                [path_embeddings, path_features],
                dim=-1
            )

        else:

            x = path_embeddings

        # ----------------------------------------------------
        # INPUT PROJECTION
        # ----------------------------------------------------

        x = self.input_proj(x)

        # ----------------------------------------------------
        # GLOBAL PATH ATTENTION
        # ----------------------------------------------------

        if path_mask is not None:

            attn_mask = ~path_mask.bool()

        else:

            attn_mask = None

        attn_out, _ = self.attn(

            query=x,
            key=x,
            value=x,

            key_padding_mask=attn_mask
        )

        x = self.norm(
            x + attn_out
        )

        # ----------------------------------------------------
        # RESIDUAL BLOCKS
        # ----------------------------------------------------

        for block in self.blocks:

            x = block(x)

        # ----------------------------------------------------
        # SCORING
        # ----------------------------------------------------

        logits = self.scorer(x).squeeze(-1)

        # [B,K]

        # ----------------------------------------------------
        # MASK INVALID PATHS
        # ----------------------------------------------------

        if path_mask is not None:

            logits = logits.masked_fill(
                ~path_mask.bool(),
                -1e9
            )

        # ----------------------------------------------------
        # SINGLE SAMPLE FIX
        # ----------------------------------------------------

        if single_sample:

            logits = logits.squeeze(0)

        return logits