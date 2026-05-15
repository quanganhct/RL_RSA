# models/hierarchical_policy.py the forward_ppo never uses act_path, act_modulation, act_slot and also, the obs["path_features"]
depends on the selected path. same thing with obs["slot_features"] which files needs to be update to account for this this appect?


"""
=============================================================
HIERARCHICAL POLICY FOR RMSA PPO
=============================================================

This module connects:

    GNNEncoder
    PathTransformer
    ModulationPolicy
    SpectrumPolicy
    Critic

into a single PPO-compatible agent.

=============================================================
DECISION FLOW
=============================================================

obs
 ↓
GNN Encoder
 ↓
Path Transformer
 ↓
Path selection
 ↓
Modulation policy (conditioned on path)
 ↓
Spectrum policy (conditioned on path + mod)
 ↓
Reward

=============================================================
"""

import torch
import torch.nn as nn
import torch.distributions as D

from models.gnn_encoder import GNNEncoder
from models.path_transformer import PathTransformer
from models.modulation_policy import ModulationPolicy
from models.spectrum_policy import SpectrumPolicy
from models.critic import Critic


# ============================================================
# HIERARCHICAL POLICY
# ============================================================

class HierarchicalRMSAPolicy(nn.Module):

    def __init__(
        self,
        edge_dim,
        slot_dim,
        path_feature_dim,
        mod_feature_dim,
        num_paths,
        num_mods,
        hidden_dim=128
    ):

        super().__init__()

        # -----------------------------------------------------
        # ENCODERS
        # -----------------------------------------------------

        self.gnn = GNNEncoder(
            input_dim=edge_dim,
            hidden_dim=hidden_dim
        )

        self.path_encoder = PathTransformer(
            edge_dim=hidden_dim,
            hidden_dim=hidden_dim
        )

        self.mod_policy = ModulationPolicy(
            path_dim=hidden_dim,
            feature_dim=path_feature_dim,
            hidden_dim=hidden_dim,
            num_modulations=num_mods
        )

        self.slot_policy = SpectrumPolicy(
            slot_dim=slot_dim,
            hidden_dim=hidden_dim,
            path_dim=hidden_dim,
            mod_dim=hidden_dim
        )

        self.critic = Critic(
            edge_dim=hidden_dim,
            path_dim=hidden_dim,
            mod_dim=hidden_dim,
            slot_dim=hidden_dim,
            hidden_dim=hidden_dim
        )

    # ========================================================
    # PATH SELECTION
    # ========================================================

    def act_path(self, obs):

        edge_emb = self.gnn(
            obs["edge_features"],
            obs["edge_index"]
        )

        path_emb, _ = self.path_encoder(
            edge_emb,
            obs["candidate_paths"],
            obs.get("path_mask", None)
        )

        # simple policy head: learned projection
        logits = path_emb.mean(dim=-1)

        dist = D.Categorical(logits=logits)

        action = dist.sample()

        return {
            "action": action.item(),
            "logprob": dist.log_prob(action),
            "path_embedding": path_emb[0, action]
        }

    # ========================================================
    # MODULATION SELECTION
    # ========================================================

    def act_modulation(self, obs, path_embedding):

        logits = self.mod_policy(
            path_embedding.unsqueeze(0),
            obs["path_features"].unsqueeze(0),
            obs.get("mod_mask", None)
        )

        dist = D.Categorical(logits=logits)

        action = dist.sample()

        mod_emb = self.mod_policy.get_embedding(
            path_embedding.unsqueeze(0),
            obs["path_features"].unsqueeze(0)
        )

        return {
            "action": action.item(),
            "logprob": dist.log_prob(action),
            "mod_embedding": mod_emb.squeeze(0)
        }

    # ========================================================
    # SPECTRUM SELECTION
    # ========================================================

    def act_slot(self, obs, path_embedding, mod_embedding):

        logits = self.slot_policy(
            obs["slot_features"].unsqueeze(0),
            path_embedding.unsqueeze(0),
            mod_embedding.unsqueeze(0),
            obs.get("slot_mask", None)
        )

        dist = D.Categorical(logits=logits)

        action = dist.sample()

        return {
            "action": action.item(),
            "logprob": dist.log_prob(action)
        }

    # ========================================================
    # PPO FORWARD PASS (RECOMPUTATION PATH)
    # ========================================================

    def forward_ppo(self, batch):

        """
        Used during PPO update:
        recomputes logits + value for all stages.
        """

        B = len(batch["path_actions"])

        # ----------------------------
        # EDGE ENCODING
        # ----------------------------

        edge_emb = self.gnn(
            batch["observations"][0]["edge_features"],
            batch["observations"][0]["edge_index"]
        )

        # ----------------------------
        # PATH ENCODING
        # ----------------------------

        path_emb, _ = self.path_encoder(
            edge_emb,
            batch["observations"][0]["candidate_paths"]
        )

        path_actions = batch["path_actions"]

        selected_path_emb = path_emb[
            torch.arange(B),
            path_actions
        ]

        # ----------------------------
        # MODULATION
        # ----------------------------

        mod_logits = self.mod_policy(
            selected_path_emb,
            batch["observations"][0]["path_features"],
            batch["mod_masks"]
        )

        # embedding for spectrum
        mod_emb = self.mod_policy.get_embedding(
            selected_path_emb,
            batch["observations"][0]["path_features"]
        )

        # ----------------------------
        # SPECTRUM
        # ----------------------------

        slot_logits = self.slot_policy(
            batch["observations"][0]["slot_features"],
            selected_path_emb,
            mod_emb,
            batch["slot_masks"]
        )

        # ----------------------------
        # VALUE
        # ----------------------------

        value = self.critic(
            edge_emb,
            selected_path_emb,
            mod_emb
        )

        return {

            "logits_path": path_emb.mean(dim=-1),
            "logits_mod": mod_logits,
            "logits_slot": slot_logits,
            "value": value
        }

    # ========================================================
    # CRITIC ACCESSOR
    # ========================================================

    def critic(self, emb):

        return self.critic.forward(
            emb
        )