

# models/spectrum_policy.py

"""
=============================================================
SPECTRUM POLICY FOR RMSA (CONSTRAINED SLOT SELECTION)
=============================================================

Goal:
-----

Select a valid contiguous spectrum block given:

    - path embedding
    - modulation embedding
    - slot-level features (availability, fragmentation, OSNR)

=============================================================
KEY DESIGN CHOICE
=============================================================

We do NOT treat this as flat classification over slots.

Instead we:

    1. Encode slot features
    2. Use attention-conditioned scoring
    3. Score START positions of contiguous blocks

This is critical for scalability.

=============================================================
INPUTS
=============================================================

slot_features:
    [B, S, F]

path_embedding:
    [B, D]

mod_embedding:
    [B, D]

slot_mask:
    [B, S]  (1 = available)

=============================================================
OUTPUTS
=============================================================

logits_slots:
    [B, S]

slot_embedding:
    [B, D]

=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# SLOT ENCODER
# ============================================================

class SlotEncoder(nn.Module):

    def __init__(self, slot_dim, hidden_dim):

        super().__init__()

        self.mlp = nn.Sequential(

            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)

        )

    def forward(self, x):

        return self.mlp(x)


# ============================================================
# CONTEXT ATTENTION OVER SLOTS
# ============================================================

class SlotAttention(nn.Module):

    def __init__(self, hidden_dim, num_heads=4):

        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, slots, context, mask=None):

        """
        slots:   [B, S, H]
        context: [B, H]
        """

        B, S, H = slots.shape

        context = context.unsqueeze(1)  # [B,1,H]

        attn_out, _ = self.attn(
            query=context,
            key=slots,
            value=slots,
            key_padding_mask=~mask if mask is not None else None
        )

        return self.norm(attn_out.squeeze(1))


# ============================================================
# SPECTRUM POLICY
# ============================================================

class SpectrumPolicy(nn.Module):

    def __init__(
        self,
        slot_dim,
        hidden_dim,
        path_dim,
        mod_dim
    ):

        super().__init__()

        self.slot_encoder = SlotEncoder(
            slot_dim,
            hidden_dim
        )

        self.attn = SlotAttention(hidden_dim)

        # -----------------------------------------------------
        # CONTEXT ENCODER (path + mod)
        # -----------------------------------------------------

        self.context_mlp = nn.Sequential(

            nn.Linear(path_dim + mod_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)

        )

        # -----------------------------------------------------
        # SLOT SCORER
        # -----------------------------------------------------

        self.scorer = nn.Sequential(

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
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
        slot_features: [B,S,F]
        """

        B, S, _ = slot_features.shape

        # -----------------------------------------------------
        # ENCODE SLOTS
        # -----------------------------------------------------

        slots = self.slot_encoder(slot_features)  # [B,S,H]

        # -----------------------------------------------------
        # CONTEXT
        # -----------------------------------------------------

        context = torch.cat(
            [path_embedding, mod_embedding],
            dim=-1
        )

        context = self.context_mlp(context)  # [B,H]

        # -----------------------------------------------------
        # SLOT ATTENTION SUMMARY
        # -----------------------------------------------------

        global_slot_context = self.attn(
            slots,
            context,
            mask=slot_mask
        )  # [B,H]

        # -----------------------------------------------------
        # SCORE EACH SLOT
        # -----------------------------------------------------

        context_exp = global_slot_context.unsqueeze(1).expand(
            B, S, global_slot_context.size(-1)
        )

        x = torch.cat(
            [slots, context_exp],
            dim=-1
        )

        logits = self.scorer(x).squeeze(-1)  # [B,S]

        # -----------------------------------------------------
        # MASK INVALID SLOTS
        # -----------------------------------------------------

        if slot_mask is not None:

            logits = logits.masked_fill(
                ~slot_mask.bool(),
                -1e9
            )

        return logits

    # ========================================================
    # EMBEDDING OUTPUT
    # ========================================================

    def get_embedding(
        self,
        slot_features
    ):

        return self.slot_encoder(slot_features)