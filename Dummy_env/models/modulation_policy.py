# models/modulation_policy.py

"""
=============================================================
MODULATION POLICY FOR RMSA
=============================================================

This module selects modulation format conditioned on:

    - selected path embedding
    - path-level features (distance, hops, OSNR)
    - network state embedding (optional)

=============================================================
INPUTS
=============================================================

path_embedding:
    [B, D]

path_features:
    [B, F]  (distance, hops, fragmentation, etc.)

mod_mask:
    [B, M]  (valid modulation formats)

=============================================================
OUTPUTS
=============================================================

logits_mod:
    [B, M]

mod_embedding:
    [B, D]

=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# MODULATION POLICY NETWORK
# ============================================================

class ModulationPolicy(nn.Module):

    def __init__(
        self,
        path_dim,
        feature_dim,
        hidden_dim,
        num_modulations
    ):

        super().__init__()

        self.num_modulations = num_modulations

        # -----------------------------------------------------
        # CONTEXT ENCODER
        # -----------------------------------------------------

        self.context_mlp = nn.Sequential(

            nn.Linear(path_dim + feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)

        )

        # -----------------------------------------------------
        # MODULATION EMBEDDING TABLE
        # -----------------------------------------------------

        self.mod_embeddings = nn.Parameter(
            torch.randn(num_modulations, hidden_dim)
        )

        # -----------------------------------------------------
        # SCORING NETWORK
        # -----------------------------------------------------

        self.scorer = nn.Sequential(

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)

        )

        # -----------------------------------------------------
        # VALUE PROJECTION FOR CONDITIONING
        # -----------------------------------------------------

        self.proj = nn.Linear(hidden_dim, hidden_dim)

    # ========================================================
    # FORWARD (TRAINING / PPO RECOMPUTE)
    # ========================================================

    def forward(
        self,
        path_embedding,
        path_features,
        mod_mask=None
    ):

        """
        Returns modulation logits.

        path_embedding: [B, D]
        path_features:  [B, F]
        """

        B = path_embedding.size(0)

        # -----------------------------------------------------
        # CONTEXT
        # -----------------------------------------------------

        context = torch.cat(
            [path_embedding, path_features],
            dim=-1
        )

        context = self.context_mlp(context)  # [B, H]

        context = self.proj(context)

        # -----------------------------------------------------
        # EXPAND MODES
        # -----------------------------------------------------

        mod_emb = self.mod_embeddings  # [M, H]

        context_exp = context.unsqueeze(1).expand(
            B,
            self.num_modulations,
            context.size(-1)
        )

        mod_exp = mod_emb.unsqueeze(0).expand(
            B,
            self.num_modulations,
            mod_emb.size(-1)
        )

        # -----------------------------------------------------
        # PAIRWISE SCORING
        # -----------------------------------------------------

        x = torch.cat(
            [context_exp, mod_exp],
            dim=-1
        )

        logits = self.scorer(x).squeeze(-1)  # [B, M]

        # -----------------------------------------------------
        # MASKING
        # -----------------------------------------------------

        if mod_mask is not None:

            logits = logits.masked_fill(
                ~mod_mask.bool(),
                -1e9
            )

        return logits

    # ========================================================
    # EMBEDDING FOR DOWNSTREAM (SPECTRUM POLICY)
    # ========================================================

    def get_embedding(
        self,
        path_embedding,
        path_features
    ):

        context = torch.cat(
            [path_embedding, path_features],
            dim=-1
        )

        return self.proj(
            self.context_mlp(context)
        )