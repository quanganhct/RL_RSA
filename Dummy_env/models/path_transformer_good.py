# models/path_transformer.py

"""
=============================================================
PATH TRANSFORMER FOR RMSA
=============================================================

Improved production-grade implementation.

KEY IMPROVEMENTS
=============================================================

✔ Supports BOTH:
    - rollout inference      : (K, L)
    - PPO training batches   : (B, K, L)

✔ Stable transformer encoder
    - residuals
    - GELU
    - dropout
    - pre-norm architecture

✔ Proper masked pooling

✔ Path CLS-token support

✔ Safe edge gathering

✔ GPU-friendly

✔ Cleaner tensor contracts

=============================================================
INPUTS
=============================================================

edge_embeddings:
    rollout : [E, D]
    training: [B, E, D]

candidate_paths:
    rollout : [K, L]
    training: [B, K, L]

path_padding_mask:
    rollout : [K, L]
    training: [B, K, L]

=============================================================
OUTPUTS
=============================================================

path_embeddings:
    rollout : [K, D]
    training: [B, K, D]

path_scores:
    rollout : [K]
    training: [B, K]

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
        # PRE-NORM ATTENTION
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
# PATH TRANSFORMER
# ============================================================

class PathTransformer(nn.Module):

    def __init__(
        self,
        edge_dim,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        dropout=0.1,
        use_cls_token=True
    ):

        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_cls_token = use_cls_token

        # ----------------------------------------------------
        # EDGE PROJECTION
        # ----------------------------------------------------

        self.edge_proj = nn.Sequential(

            nn.Linear(edge_dim, hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim)
        )

        # ----------------------------------------------------
        # CLS TOKEN
        # ----------------------------------------------------

        if use_cls_token:

            self.cls_token = nn.Parameter(
                torch.randn(1, 1, hidden_dim)
            )

        # ----------------------------------------------------
        # POSITIONAL EMBEDDING
        # ----------------------------------------------------

        self.pos_embedding = nn.Parameter(
            torch.randn(1, 512, hidden_dim)
        )

        # ----------------------------------------------------
        # TRANSFORMER STACK
        # ----------------------------------------------------

        self.layers = nn.ModuleList([

            TransformerBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )

            for _ in range(num_layers)

        ])

        # ----------------------------------------------------
        # PATH SCORING HEAD
        # ----------------------------------------------------

        self.path_scorer = nn.Sequential(

            nn.LayerNorm(hidden_dim),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, 1)
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        edge_embeddings,
        candidate_paths,
        path_padding_mask=None
    ):

        """
        Supports BOTH:

            rollout:
                edge_embeddings = [E,D]
                candidate_paths = [K,L]

            training:
                edge_embeddings = [B,E,D]
                candidate_paths = [B,K,L]
        """

        # ====================================================
        # SINGLE-SAMPLE SUPPORT
        # ====================================================

        single_sample = False

        if candidate_paths.dim() == 2:

            single_sample = True

            candidate_paths = candidate_paths.unsqueeze(0)

            if path_padding_mask is not None:
                path_padding_mask = path_padding_mask.unsqueeze(0)

            if edge_embeddings.dim() == 2:
                edge_embeddings = edge_embeddings.unsqueeze(0)

        # ====================================================
        # SHAPES
        # ====================================================

        B, K, L = candidate_paths.shape

        _, E, D_edge = edge_embeddings.shape

        # ====================================================
        # GATHER EDGE EMBEDDINGS
        # ====================================================

        # candidate_paths:
        # [B,K,L]

        expanded_paths = candidate_paths.unsqueeze(-1).expand(
            -1, -1, -1, D_edge
        )

        expanded_edges = edge_embeddings.unsqueeze(1).expand(
            -1, K, -1, -1
        )

        path_edges = torch.gather(
            expanded_edges,
            2,
            expanded_paths
        )

        # path_edges:
        # [B,K,L,D]

        # ====================================================
        # FLATTEN FOR TRANSFORMER
        # ====================================================

        x = path_edges.reshape(B * K, L, D_edge)

        # ====================================================
        # PROJECT TO HIDDEN SPACE
        # ====================================================

        x = self.edge_proj(x)

        # ====================================================
        # MASK PREP
        # ====================================================

        if path_padding_mask is not None:

            mask = path_padding_mask.reshape(B * K, L)

        else:

            mask = None

        # ====================================================
        # CLS TOKEN
        # ====================================================

        if self.use_cls_token:

            cls = self.cls_token.expand(B * K, -1, -1)

            x = torch.cat([cls, x], dim=1)

            if mask is not None:

                cls_mask = torch.zeros(
                    (mask.shape[0], 1),
                    device=mask.device,
                    dtype=torch.bool
                )

                mask = torch.cat([cls_mask, mask], dim=1)

        # ====================================================
        # POSITIONAL ENCODING
        # ====================================================

        x = x + self.pos_embedding[:, :x.shape[1]]

        # ====================================================
        # TRANSFORMER ENCODING
        # ====================================================

        for layer in self.layers:

            x = layer(x, mask)

        # ====================================================
        # POOLING
        # ====================================================

        if self.use_cls_token:

            path_embeddings = x[:, 0]

        else:

            if mask is not None:

                valid = (~mask).float().unsqueeze(-1)

                pooled = (x * valid).sum(dim=1)

                denom = valid.sum(dim=1).clamp(min=1.0)

                path_embeddings = pooled / denom

            else:

                path_embeddings = x.mean(dim=1)

        # ====================================================
        # RESHAPE BACK
        # ====================================================

        path_embeddings = path_embeddings.view(
            B,
            K,
            self.hidden_dim
        )

        # ====================================================
        # PATH SCORING
        # ====================================================

        path_scores = self.path_scorer(
            path_embeddings
        ).squeeze(-1)

        # ====================================================
        # SINGLE-SAMPLE OUTPUT FIX
        # ====================================================

        if single_sample:

            path_embeddings = path_embeddings.squeeze(0)

            path_scores = path_scores.squeeze(0)

        return path_embeddings, path_scores
