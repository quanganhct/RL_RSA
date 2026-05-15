# models/path_transformer.py

"""
=============================================================
PATH TRANSFORMER FOR RMSA (K-PATH REASONING MODULE)
=============================================================

Purpose:
--------

Convert K candidate paths into embeddings using:

    - edge embeddings (from GNN)
    - transformer over edge sequences
    - masked attention over padded paths

Each path = sequence of edges.

=============================================================
INPUT FORMAT
=============================================================

edge_embeddings:
    [E, D]

candidate_paths:
    [B, K, L]   (edge indices per path)

path_padding_mask:
    [B, K, L]   True = invalid/pad

=============================================================
OUTPUT
=============================================================

path_embeddings:
    [B, K, D]

global_path_scores (optional):
    [B, K]

=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(self, dim, num_heads=4, ff_mult=4):

        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ff = nn.Sequential(

            nn.Linear(dim, dim * ff_mult),
            nn.ReLU(),
            nn.Linear(dim * ff_mult, dim)

        )

    def forward(self, x, mask=None):

        """
        x: [B*K, L, D]
        mask: [B*K, L] (True = pad)
        """

        attn_out, _ = self.attn(
            x, x, x,
            key_padding_mask=mask
        )

        x = self.norm1(x + attn_out)

        ff_out = self.ff(x)

        x = self.norm2(x + ff_out)

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
        num_heads=4
    ):

        super().__init__()

        self.edge_proj = nn.Linear(
            edge_dim,
            hidden_dim
        )

        self.layers = nn.ModuleList([

            TransformerBlock(
                hidden_dim,
                num_heads=num_heads
            )

            for _ in range(num_layers)

        ])

        self.path_pool = nn.Linear(
            hidden_dim,
            hidden_dim
        )

    def forward(
        self,
        edge_embeddings,
        candidate_paths,
        path_padding_mask=None
    ):

        """
        edge_embeddings:
            [E, D]

        candidate_paths:
            [B, K, L]

        path_padding_mask:
            [B, K, L]
        """ 
       
        
        B, K, L = candidate_paths.shape

        D = edge_embeddings.shape[-1]

        # -----------------------------------------------------
        # GATHER EDGE EMBEDDINGS FOR EACH PATH
        # -----------------------------------------------------

        paths_emb = edge_embeddings[candidate_paths]  # [B,K,L,D]

        # reshape for transformer
        x = paths_emb.view(B * K, L, D)

        if path_padding_mask is not None:

            mask = path_padding_mask.view(B * K, L)

        else:

            mask = None

        # -----------------------------------------------------
        # PROJECT
        # -----------------------------------------------------

        x = self.edge_proj(x)

        # -----------------------------------------------------
        # TRANSFORMER ENCODING
        # -----------------------------------------------------

        for layer in self.layers:

            x = layer(x, mask)

        # -----------------------------------------------------
        # POOLING (mean over valid edges)
        # -----------------------------------------------------

        if mask is not None:

            valid = (~mask).float().unsqueeze(-1)

            x = (x * valid).sum(dim=1) / (
                valid.sum(dim=1).clamp(min=1.0)
            )

        else:

            x = x.mean(dim=1)

        # -----------------------------------------------------
        # RESHAPE BACK TO PATH SPACE
        # -----------------------------------------------------

        path_embeddings = x.view(B, K, -1)

        path_scores = self.path_pool(path_embeddings)

        return path_embeddings, path_scores
