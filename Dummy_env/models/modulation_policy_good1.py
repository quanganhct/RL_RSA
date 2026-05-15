# models/modulation_policy.py

"""
=============================================================
MODULATION POLICY FOR RMSA
=============================================================

Improved production-grade implementation.

KEY IMPROVEMENTS
=============================================================

✔ Supports BOTH:
    - rollout inference      : [D]
    - PPO batch training     : [B, D]

✔ Better modulation reasoning:
    - FiLM-style conditioning
    - modulation-query attention
    - deeper context encoder

✔ Stable PPO training:
    - LayerNorm
    - GELU
    - Dropout
    - residual MLP structure

✔ Better modulation embeddings:
    - learnable embeddings
    - modulation-specific scoring

✔ Safer masking

✔ Cleaner downstream embedding extraction

=============================================================
INPUTS
=============================================================

path_embedding:
    rollout : [D]
    training: [B, D]

path_features:
    rollout : [F]
    training: [B, F]

mod_mask:
    rollout : [M]
    training: [B, M]

=============================================================
OUTPUTS
=============================================================

logits:
    rollout : [M]
    training: [B, M]

mod_context:
    rollout : [H]
    training: [B, H]

=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# RESIDUAL BLOCK
# ============================================================

class ResidualMLP(nn.Module):

    def __init__(
        self,
        dim,
        hidden_mult=2,
        dropout=0.1
    ):

        super().__init__()

        hidden = dim * hidden_mult

        self.net = nn.Sequential(

            nn.LayerNorm(dim),

            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):

        return x + self.net(x)


# ============================================================
# MODULATION POLICY
# ============================================================

class ModulationPolicy(nn.Module):

    def __init__(
        self,
        path_dim,
        feature_dim,
        hidden_dim,
        num_modulations,
        num_layers=2,
        dropout=0.1
    ):

        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_modulations = num_modulations

        # ----------------------------------------------------
        # CONTEXT ENCODER
        # ----------------------------------------------------

        input_dim = path_dim + feature_dim

        self.input_proj = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),
            nn.GELU()
        )

        self.context_blocks = nn.ModuleList([

            ResidualMLP(
                hidden_dim,
                dropout=dropout
            )

            for _ in range(num_layers)

        ])

        # ----------------------------------------------------
        # MODULATION EMBEDDINGS
        # ----------------------------------------------------

        self.mod_embeddings = nn.Parameter(
            torch.randn(num_modulations, hidden_dim)
        )

        # ----------------------------------------------------
        # QUERY / KEY ATTENTION
        # ----------------------------------------------------

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)

        self.key_proj = nn.Linear(hidden_dim, hidden_dim)

        # ----------------------------------------------------
        # FINAL SCORER
        # ----------------------------------------------------

        self.scorer = nn.Sequential(

            nn.LayerNorm(hidden_dim * 2),

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 1)
        )

        # ----------------------------------------------------
        # OUTPUT PROJECTION
        # ----------------------------------------------------

        self.output_proj = nn.Sequential(

            nn.LayerNorm(hidden_dim),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

    # ========================================================
    # ENCODE CONTEXT
    # ========================================================

    def encode_context(
        self,
        path_embedding,
        path_features
    ):

        x = torch.cat(
            [path_embedding, path_features],
            dim=-1
        )

        x = self.input_proj(x)

        for block in self.context_blocks:

            x = block(x)

        return x

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        path_embedding,
        path_features,
        mod_mask=None
    ):

        """
        Supports BOTH:

            rollout:
                path_embedding = [D]
                path_features  = [F]

            training:
                path_embedding = [B,D]
                path_features  = [B,F]
        """

        # ----------------------------------------------------
        # SINGLE SAMPLE SUPPORT
        # ----------------------------------------------------

        single_sample = False

        if path_embedding.dim() == 1:

            single_sample = True

            path_embedding = path_embedding.unsqueeze(0)
            path_features = path_features.unsqueeze(0)

            if mod_mask is not None:
                mod_mask = mod_mask.unsqueeze(0)

        B = path_embedding.shape[0]

        # ----------------------------------------------------
        # CONTEXT ENCODING
        # ----------------------------------------------------

        context = self.encode_context(
            path_embedding,
            path_features
        )

        # ----------------------------------------------------
        # MODULATION EMBEDDINGS
        # ----------------------------------------------------

        mod_emb = self.mod_embeddings.unsqueeze(0).expand(
            B,
            -1,
            -1
        )

        # [B,M,H]

        # ----------------------------------------------------
        # ATTENTION-STYLE INTERACTION
        # ----------------------------------------------------

        query = self.query_proj(context).unsqueeze(1)

        key = self.key_proj(mod_emb)

        attention = query * key

        # ----------------------------------------------------
        # CONCAT FOR FINAL SCORING
        # ----------------------------------------------------

        context_expand = context.unsqueeze(1).expand(
            -1,
            self.num_modulations,
            -1
        )

        x = torch.cat(
            [attention, context_expand],
            dim=-1
        )

        logits = self.scorer(x).squeeze(-1)

        # ----------------------------------------------------
        # SAFE MASKING
        # ----------------------------------------------------

        if mod_mask is not None:

            logits = logits.masked_fill(
                ~mod_mask.bool(),
                -1e9
            )

        # ----------------------------------------------------
        # SINGLE SAMPLE FIX
        # ----------------------------------------------------

        if single_sample:

            logits = logits.squeeze(0)
            context = context.squeeze(0)

        return logits, context

    # ========================================================
    # GET SELECTED MODULATION EMBEDDING
    # ========================================================

    def get_modulation_embedding(
        self,
        modulation_idx
    ):

        """
        Returns modulation-specific embedding.

        modulation_idx:
            rollout : int
            training: [B]
        """

        return self.mod_embeddings[modulation_idx]

    # ========================================================
    # FULL DOWNSTREAM CONTEXT
    # ========================================================

    def build_spectrum_context(
        self,
        path_embedding,
        path_features,
        modulation_idx
    ):

        """
        Builds context for spectrum policy.
        """

        context = self.encode_context(
            path_embedding,
            path_features
        )

        mod_emb = self.get_modulation_embedding(
            modulation_idx
        )

        return self.output_proj(
            context + mod_emb
        )