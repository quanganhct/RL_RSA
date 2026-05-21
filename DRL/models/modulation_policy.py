# models/modulation_policy.py

"""
=============================================================
MODULATION POLICY FOR RMSA
=============================================================

Hierarchical modulation-selection policy.

This module assumes:

    PATH ALREADY SELECTED

Therefore it ONLY receives:

    - selected path embedding
    - selected path features

NOT all K candidate paths.

=============================================================
KEY FEATURES
=============================================================

✔ Correct hierarchical semantics

✔ Supports:
    - rollout inference
    - PPO batched training

✔ Stable PPO architecture
    - LayerNorm
    - residual MLP
    - GELU
    - dropout

✔ Modulation-query interaction

✔ Safe masking

✔ Spectrum-context generation

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

context:
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

        return x + self.net(self.norm(x))


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
        # INPUT ENCODER
        # ----------------------------------------------------

        self.input_proj = nn.Sequential(

            nn.Linear(
                path_dim + feature_dim,
                hidden_dim
            ),

            nn.GELU()
        )

        # ----------------------------------------------------
        # CONTEXT PROCESSOR
        # ----------------------------------------------------

        self.context_blocks = nn.ModuleList([

            ResidualBlock(
                hidden_dim,
                dropout=dropout
            )

            for _ in range(num_layers)

        ])

        # ----------------------------------------------------
        # MODULATION EMBEDDINGS
        # ----------------------------------------------------

        self.mod_embeddings = nn.Parameter(
            torch.randn(
                num_modulations,
                hidden_dim
            )
        )

        # ----------------------------------------------------
        # QUERY / KEY INTERACTION
        # ----------------------------------------------------

        self.query_proj = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.key_proj = nn.Linear(
            hidden_dim,
            hidden_dim
        )

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
        # OUTPUT CONTEXT
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
        =====================================================
        INPUT CONTRACT
        =====================================================

        path_embedding:
            rollout : [D]
            training: [B,D]

        path_features:
            rollout : [F]
            training: [B,F]

        IMPORTANT:
            path already selected.
        """

        # ----------------------------------------------------
        # SINGLE SAMPLE SUPPORT
        # ----------------------------------------------------

        
        # print(f"juju path_features = {path_features.shape}")
        # print(f"path_embedding = {path_embedding.shape}")

     

        B = path_embedding.shape[0]

        # ----------------------------------------------------
        # CONTEXT ENCODING
        # ----------------------------------------------------

        context = self.encode_context(
            path_embedding,
            path_features
        )

        # context:
        # [B,H]

        # ----------------------------------------------------
        # MODULATION EMBEDDINGS
        # ----------------------------------------------------

        mod_emb = self.mod_embeddings.unsqueeze(0).expand(
            B,
            self.num_modulations,
            self.hidden_dim
        )

        # [B,M,H]

        # ----------------------------------------------------
        # ATTENTION-STYLE INTERACTION
        # ----------------------------------------------------

        query = self.query_proj(context).unsqueeze(1)

        key = self.key_proj(mod_emb)

        interaction = query * key

        # ----------------------------------------------------
        # CONCATENATE GLOBAL CONTEXT
        # ----------------------------------------------------

        global_context = context.unsqueeze(1).expand(
            B,
            self.num_modulations,
            self.hidden_dim
        )

        x = torch.cat(
            [interaction, global_context],
            dim=-1
        )

        # ----------------------------------------------------
        # SCORE MODULATIONS
        # ----------------------------------------------------

        logits = self.scorer(x).squeeze(-1)

        # [B,M]

        # ----------------------------------------------------
        # MASK INVALID MODULATIONS
        # ----------------------------------------------------
        # assert mod_mask.any(dim=-1).all, 'All modulation actions masked!'
        mask = mod_mask.bool()
        valid = mask.sum(dim=-1) > 0 
        if not torch.all(valid):
            print("Warning all mod actions masked")
            mod_mask = None #[~valid, 0] = 1
        if mod_mask is not None:

            logits = logits.masked_fill(
                ~mod_mask.bool(),
                -1e9
            )

        # ----------------------------------------------------
        # SINGLE SAMPLE FIX
        # ----------------------------------------------------

       

        return logits, context

    # ========================================================
    # GET MODULATION EMBEDDING
    # ========================================================

    def get_modulation_embedding(
        self,
        modulation_idx
    ):

        return self.mod_embeddings[
            modulation_idx
        ]

    # ========================================================
    # BUILD CONTEXT FOR SPECTRUM POLICY
    # ========================================================

    def build_spectrum_context(
        self, modulation_id, context
    ):

        """
        Produces final conditioning vector for
        spectrum allocation policy.
        """
       
        # batch_idx = torch.arange(mod_emb.size(0), device=mod_emb.device)

        mod_emb = self.mod_embeddings[modulation_id] #mod_emb[:, modulation_id, :]
        # context_1 = context[:, modulation_id, :]
        # print(f"mod_emb = {mod_emb.shape} context = {context.shape} ")

        spectrum_context = self.output_proj(
            context + mod_emb
        )

        # if single_sample:

        #     spectrum_context = spectrum_context.squeeze(0)

        return spectrum_context



    def build_spectrum_context_original(
        self,
        path_embedding,
        path_features,
        modulation_idx
    ):

        """
        Produces final conditioning vector for
        spectrum allocation policy.
        """

        single_sample = False

        if path_embedding.dim() == 1:

            single_sample = True

            path_embedding = path_embedding.unsqueeze(0)
            path_features = path_features.unsqueeze(0)

            if not torch.is_tensor(modulation_idx):

                modulation_idx = torch.tensor(
                    [modulation_idx],
                    device=path_embedding.device
                )

        context = self.encode_context(
            path_embedding,
            path_features
        )

        mod_emb = self.get_modulation_embedding(
            modulation_idx
        )

        spectrum_context = self.output_proj(
            context + mod_emb
        )


        return spectrum_context