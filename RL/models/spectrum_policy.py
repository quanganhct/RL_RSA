# models/spectrum_policy.py

"""
=============================================================
SPECTRUM POLICY FOR RMSA
=============================================================

Production-grade spectrum allocation policy.

This module selects START SLOT positions conditioned on:

    - selected path embedding
    - selected modulation embedding
    - slot-level features

=============================================================
IMPORTANT HIERARCHICAL SEMANTICS
=============================================================

Path and modulation are ALREADY selected.

Therefore this module receives ONLY:

    - selected path embedding
    - selected modulation embedding

=============================================================
KEY IMPROVEMENTS
=============================================================

✔ Supports:
    - rollout inference
    - PPO batched training

✔ Transformer encoder over spectrum slots

✔ Context-conditioned slot reasoning

✔ Stable PPO architecture:
    - LayerNorm
    - GELU
    - residual transformer
    - dropout

✔ Proper masking

✔ Contiguous-block aware design

✔ Safe tensor contracts

=============================================================
INPUTS
=============================================================

slot_features:
    rollout : [S,F]
    training: [B,S,F]

path_embedding:
    rollout : [D]
    training: [B,D]

mod_embedding:
    rollout : [D]
    training: [B,D]

slot_mask:
    rollout : [S]
    training: [B,S]

=============================================================
OUTPUTS
=============================================================

logits:
    rollout : [S]
    training: [B,S]

slot_embeddings:
    rollout : [S,H]
    training: [B,S,H]

=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=4,
        ff_mult=4,
        dropout=0.1
    ):

        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)

        self.ff = nn.Sequential(

            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):

        # ----------------------------------------------------
        # PRENORM ATTENTION
        # ----------------------------------------------------

        h = self.norm1(x)

        attn_out, _ = self.attn(
            h,
            h,
            h,
            key_padding_mask=mask
        )

        x = x + attn_out

        # ----------------------------------------------------
        # FEEDFORWARD
        # ----------------------------------------------------

        h = self.norm2(x)

        x = x + self.ff(h)

        return x


# ============================================================
# SLOT ENCODER
# ============================================================

class SlotEncoder(nn.Module):

    def __init__(
        self,
        slot_dim,
        hidden_dim
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):

        return self.net(x)


# ============================================================
# SPECTRUM POLICY
# ============================================================

class SpectrumPolicy(nn.Module):

    def __init__(
        self,
        slot_dim,
        hidden_dim,
        path_dim,
        mod_dim,
        num_layers=3,
        num_heads=4,
        dropout=0.1
    ):

        super().__init__()

        self.hidden_dim = hidden_dim

        # ----------------------------------------------------
        # SLOT ENCODER
        # ----------------------------------------------------

        self.slot_encoder = SlotEncoder(
            slot_dim,
            hidden_dim
        )

        # ----------------------------------------------------
        # CONTEXT ENCODER
        # ----------------------------------------------------

        self.context_encoder = nn.Sequential(

            nn.Linear(
                path_dim + mod_dim,
                hidden_dim
            ),

            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim)
        )

        # ----------------------------------------------------
        # POSITIONAL EMBEDDING
        # ----------------------------------------------------

        self.pos_embedding = nn.Parameter(
            torch.randn(1, 1024, hidden_dim)
        )

        # ----------------------------------------------------
        # TRANSFORMER STACK
        # ----------------------------------------------------

        self.layers = nn.ModuleList([

            TransformerBlock(
                hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )

            for _ in range(num_layers)

        ])

        # ----------------------------------------------------
        # SLOT SCORER
        # ----------------------------------------------------

        self.scorer = nn.Sequential(

            nn.LayerNorm(hidden_dim * 2),

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 1)
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        slot_features,
        path_embedding,
        mod_embedding,
        slot_mask=None
    ):

        """
        =====================================================
        Supports BOTH:

            rollout:
                slot_features = [S,F]

            training:
                slot_features = [B,S,F]
        =====================================================
        """

        # ----------------------------------------------------
        # SINGLE SAMPLE SUPPORT
        # ----------------------------------------------------

        single_sample = False
        
        # print(f"slot feature = {slot_features.shape}")
        # print(f"path_embedding = {path_embedding.shape}")
        # print(f"mod_embedding = {mod_embedding.shape}")
        

        if slot_features.dim() == 2:

            single_sample = True

            slot_features = slot_features.unsqueeze(0)

            path_embedding = path_embedding.unsqueeze(0)

            mod_embedding = mod_embedding.unsqueeze(0)

            if slot_mask is not None:
                slot_mask = slot_mask.unsqueeze(0)

        # ----------------------------------------------------
        # SAFETY CHECKS
        # ----------------------------------------------------

        assert slot_features.dim() == 3, \
            f"slot_features must be [B,S,F], got {slot_features.shape}"

        assert path_embedding.dim() == 2, \
            f"path_embedding must be [B,D], got {path_embedding.shape}"

        assert mod_embedding.dim() == 2, \
            f"mod_embedding must be [B,D], got {mod_embedding.shape}"

        B, S, _ = slot_features.shape

        # ----------------------------------------------------
        # SLOT ENCODING
        # ----------------------------------------------------

        slots = self.slot_encoder(
            slot_features
        )

        # [B,S,H]

        # ----------------------------------------------------
        # POSITIONAL ENCODING
        # ----------------------------------------------------

        slots = slots + self.pos_embedding[:, :S]

        # ----------------------------------------------------
        # CONTEXT ENCODING
        # ----------------------------------------------------

        context = torch.cat(
            [path_embedding, mod_embedding],
            dim=-1
        )

        context = self.context_encoder(
            context
        )

        # [B,H]

        # ----------------------------------------------------
        # CONDITION SLOT TOKENS
        # ----------------------------------------------------

        context_expand = context.unsqueeze(1).expand(
            B,
            S,
            self.hidden_dim
        )

        slots = slots + context_expand

        # ----------------------------------------------------
        # MASK
        # ----------------------------------------------------

        if slot_mask is not None:

            transformer_mask = ~slot_mask.bool()

        else:

            transformer_mask = None

        # ----------------------------------------------------
        # TRANSFORMER ENCODING
        # ----------------------------------------------------

        x = slots

        for layer in self.layers:

            x = layer(
                x,
                transformer_mask
            )

        # ----------------------------------------------------
        # GLOBAL CONTEXT
        # ----------------------------------------------------

        global_context = x.mean(dim=1)

        global_expand = global_context.unsqueeze(1).expand(
            B,
            S,
            self.hidden_dim
        )

        # ----------------------------------------------------
        # SLOT SCORING
        # ----------------------------------------------------

        score_input = torch.cat(
            [x, global_expand],
            dim=-1
        )

        logits = self.scorer(
            score_input
        ).squeeze(-1)

        # ----------------------------------------------------
        # INVALID SLOT MASKING
        # ----------------------------------------------------

        if slot_mask is not None:

            logits = logits.masked_fill(
                ~slot_mask.bool(),
                -1e9
            )

        # ----------------------------------------------------
        # SINGLE SAMPLE FIX
        # ----------------------------------------------------

        if single_sample:

            logits = logits.squeeze(0)

            x = x.squeeze(0)

        return logits, x

    # ========================================================
    # GET SLOT EMBEDDINGS
    # ========================================================

    def get_slot_embeddings(
        self,
        slot_features
    ):

        return self.slot_encoder(
            slot_features
        )