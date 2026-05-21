"""
==============================================================
HIERARCHICAL GNN + TRANSFORMER POLICY FOR RMSA
==============================================================

Architecture
============

1) EDGE-AWARE GNN
    - 3-5 residual graph layers
    - edge-centric embeddings

2) PATH ENCODER
    - Transformer over path edge embeddings
    - produces path embeddings

3) PATH POLICY
    - 2-layer MLP
    - masked categorical distribution

4) MODULATION POLICY
    - attention conditioned on selected path
    - modulation logits

5) SPECTRUM POLICY
    - Transformer over slots
    - conditioned on path + modulation

==============================================================
Compatible With:
==============================================================

MaskedHierarchicalRMSAEnv

==============================================================
Dependencies
==============================================================

pip install torch torch-geometric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

from torch_geometric.nn import GATConv


# ============================================================
# UTILITIES
# ============================================================

def masked_softmax_logits(logits, mask):

    """
    mask = 1 for valid actions
    mask = 0 for invalid actions
    """

    logits = logits.masked_fill(mask == 0, -1e9)

    return logits


# ============================================================
# RESIDUAL EDGE-AWARE GNN
# ============================================================

class ResidualGNNLayer(nn.Module):

    def __init__(
        self,
        hidden_dim,
        num_heads=4
    ):

        super().__init__()

        self.gnn = GATConv(
            hidden_dim,
            hidden_dim // num_heads,
            heads=num_heads,
            concat=True
        )

        self.norm = nn.LayerNorm(hidden_dim)

        self.ff = nn.Sequential(

            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)

        )

    def forward(self, x, edge_index):

        residual = x

        x = self.gnn(x, edge_index)

        x = self.norm(x + residual)

        residual = x

        x = self.ff(x)

        x = self.norm(x + residual)

        return x


# ============================================================
# EDGE-AWARE GNN ENCODER
# ============================================================

class EdgeAwareGNN(nn.Module):

    """
    Input:
        edge_features [E, Fe]

    Output:
        edge_embeddings [E, D]
    """

    def __init__(
        self,
        edge_feat_dim=4,
        hidden_dim=128,
        num_layers=4
    ):

        super().__init__()

        self.input_proj = nn.Linear(
            edge_feat_dim,
            hidden_dim
        )

        self.layers = nn.ModuleList([

            ResidualGNNLayer(hidden_dim)

            for _ in range(num_layers)

        ])

    def forward(
        self,
        edge_features,
        edge_index
    ):

        x = self.input_proj(edge_features)

        for layer in self.layers:

            x = layer(x, edge_index)

        return x


# ============================================================
# PATH TRANSFORMER ENCODER
# ============================================================

class PathEncoder(nn.Module):

    """
    Aggregates edge embeddings along candidate paths.

    Input:
        edge_embeddings [E, D]
        paths [K, L]

    Output:
        path_embeddings [K, D]
    """

    def __init__(
        self,
        hidden_dim=128,
        num_layers=2,
        num_heads=4
    ):

        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(

            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation="gelu"

        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, hidden_dim)
        )

    def forward(
        self,
        edge_embeddings,
        paths
    ):

        """
        paths: [K, L]
        """

        K, L = paths.shape

        hidden_dim = edge_embeddings.shape[-1]

        path_embeds = []

        for k in range(K):

            path = paths[k]

            valid_mask = path >= 0

            edge_ids = path[valid_mask]

            path_edges = edge_embeddings[edge_ids]

            cls = self.cls_token.expand(
                1,
                -1,
                -1
            )

            tokens = torch.cat([
                cls.squeeze(0),
                path_edges
            ], dim=0)

            tokens = tokens.unsqueeze(0)

            encoded = self.transformer(tokens)

            path_embedding = encoded[0, 0]

            path_embeds.append(path_embedding)

        path_embeds = torch.stack(path_embeds)

        return path_embeds


# ============================================================
# PATH POLICY
# ============================================================

class PathPolicy(nn.Module):

    """
    Select candidate path.
    """

    def __init__(
        self,
        hidden_dim=128
    ):

        super().__init__()

        self.policy = nn.Sequential(

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, 1)

        )

    def forward(
        self,
        path_embeddings,
        action_mask
    ):

        """
        path_embeddings [K, D]
        """

        logits = self.policy(
            path_embeddings
        ).squeeze(-1)

        logits = masked_softmax_logits(
            logits,
            action_mask
        )

        dist = Categorical(logits=logits)

        return dist


# ============================================================
# MODULATION POLICY
# ============================================================

class ModulationPolicy(nn.Module):

    """
    Attention-based modulation selection.
    """

    def __init__(
        self,
        hidden_dim=128,
        num_modulations=4,
        path_feat_dim=10
    ):

        super().__init__()

        self.mod_embeddings = nn.Parameter(

            torch.randn(
                num_modulations,
                hidden_dim
            )
        )

        self.path_proj = nn.Linear(
            hidden_dim + path_feat_dim,
            hidden_dim
        )

        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=4,
            batch_first=True
        )

        self.mlp = nn.Sequential(

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, 1)

        )

    def forward(
        self,
        path_embedding,
        path_features,
        action_mask
    ):

        """
        path_embedding [D]
        path_features [F]
        """

        context = torch.cat([
            path_embedding,
            path_features
        ], dim=-1)

        context = self.path_proj(context)

        query = context.unsqueeze(0).unsqueeze(0)

        keys = self.mod_embeddings.unsqueeze(0)

        attended, _ = self.attention(
            query=query,
            key=keys,
            value=keys
        )

        attended = attended.squeeze(0)

        logits = self.mlp(
            attended
        ).squeeze(-1)

        logits = masked_softmax_logits(
            logits,
            action_mask
        )

        dist = Categorical(logits=logits)

        return dist


# ============================================================
# SLOT TRANSFORMER POLICY
# ============================================================

class SpectrumPolicy(nn.Module):

    """
    Transformer over slots.

    Input:
        slot_features [S, Fs]

    Output:
        slot distribution
    """

    def __init__(
        self,
        slot_feat_dim=2,
        hidden_dim=128,
        num_layers=2,
        num_heads=4
    ):

        super().__init__()

        self.slot_proj = nn.Linear(
            slot_feat_dim,
            hidden_dim
        )

        self.condition_proj = nn.Linear(
            hidden_dim * 2,
            hidden_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(

            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True,
            activation="gelu"

        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.policy_head = nn.Sequential(

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, 1)

        )

    def forward(
        self,
        slot_features,
        path_embedding,
        modulation_embedding,
        action_mask
    ):

        """
        slot_features [S, Fs]
        """

        slots = self.slot_proj(
            slot_features
        )

        condition = torch.cat([
            path_embedding,
            modulation_embedding
        ], dim=-1)

        condition = self.condition_proj(
            condition
        )

        condition = condition.unsqueeze(0)

        slots = slots + condition

        slots = slots.unsqueeze(0)

        encoded = self.transformer(slots)

        encoded = encoded.squeeze(0)

        logits = self.policy_head(
            encoded
        ).squeeze(-1)

        logits = masked_softmax_logits(
            logits,
            action_mask
        )

        dist = Categorical(logits=logits)

        return dist


# ============================================================
# COMPLETE HIERARCHICAL AGENT
# ============================================================

class RMSAAgent(nn.Module):

    def __init__(
        self,
        edge_feat_dim=4,
        hidden_dim=128,
        num_gnn_layers=4,
        num_modulations=4,
        path_feat_dim=10,
        slot_feat_dim=2
    ):

        super().__init__()

        # ---------------------------------------------------
        # SHARED GNN
        # ---------------------------------------------------

        self.gnn = EdgeAwareGNN(

            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers

        )

        # ---------------------------------------------------
        # PATH ENCODER
        # ---------------------------------------------------

        self.path_encoder = PathEncoder(
            hidden_dim=hidden_dim
        )

        # ---------------------------------------------------
        # PATH POLICY
        # ---------------------------------------------------

        self.path_policy = PathPolicy(
            hidden_dim=hidden_dim
        )

        # ---------------------------------------------------
        # MODULATION POLICY
        # ---------------------------------------------------

        self.mod_policy = ModulationPolicy(

            hidden_dim=hidden_dim,
            num_modulations=num_modulations,
            path_feat_dim=path_feat_dim

        )

        # ---------------------------------------------------
        # SLOT POLICY
        # ---------------------------------------------------

        self.slot_policy = SpectrumPolicy(

            slot_feat_dim=slot_feat_dim,
            hidden_dim=hidden_dim

        )

    # ======================================================
    # PATH STAGE
    # ======================================================

    def act_path(self, obs):

        edge_features = torch.FloatTensor(
            obs["edge_features"]
        )

        edge_index = torch.LongTensor(
            obs["edge_index"]
        )

        candidate_paths = torch.LongTensor(
            obs["candidate_paths"]
        )

        action_mask = torch.BoolTensor(
            obs["action_mask"][:candidate_paths.shape[0]]
        )

        edge_embeddings = self.gnn(
            edge_features,
            edge_index
        )

        path_embeddings = self.path_encoder(
            edge_embeddings,
            candidate_paths
        )

        dist = self.path_policy(
            path_embeddings,
            action_mask
        )

        action = dist.sample()

        logprob = dist.log_prob(action)

        return {

            "action": action,
            "logprob": logprob,
            "path_embeddings": path_embeddings
        }

    # ======================================================
    # MODULATION STAGE
    # ======================================================

    def act_modulation(
        self,
        obs,
        selected_path_embedding
    ):

        path_features = torch.FloatTensor(
            obs["path_features"]
        )

        action_mask = torch.BoolTensor(
            obs["action_mask"][:len(
                self.mod_policy.mod_embeddings
            )]
        )

        dist = self.mod_policy(

            selected_path_embedding,
            path_features,
            action_mask

        )

        action = dist.sample()

        logprob = dist.log_prob(action)

        mod_embedding = (
            self.mod_policy.mod_embeddings[action]
        )

        return {

            "action": action,
            "logprob": logprob,
            "mod_embedding": mod_embedding
        }

    # ======================================================
    # SLOT STAGE
    # ======================================================

    def act_slot(
        self,
        obs,
        path_embedding,
        modulation_embedding
    ):

        slot_features = torch.FloatTensor(
            obs["slot_features"]
        )

        action_mask = torch.BoolTensor(
            obs["action_mask"][:slot_features.shape[0]]
        )

        dist = self.slot_policy(

            slot_features,
            path_embedding,
            modulation_embedding,
            action_mask

        )

        action = dist.sample()

        logprob = dist.log_prob(action)

        return {

            "action": action,
            "logprob": logprob
        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    from pprint import pprint

    agent = RMSAAgent()

    print("\nModel created successfully.")