import torch
import torch.nn as nn
import torch.distributions as D

from DRL.models.gnn_encoder import GNNEncoder
from DRL.models.path_transformer import PathTransformer
from DRL.models.path_policy import PathPolicy
from DRL.models.modulation_policy import ModulationPolicy
from DRL.models.spectrum_policy import SpectrumPolicy
from DRL.models.critic import Critic


# ============================================================
# SHARED ENCODER
# ============================================================

class SharedEncoder(nn.Module):

    def __init__(self, edge_dim, hidden_dim):
        super().__init__()

        self.gnn = GNNEncoder(
            input_dim=edge_dim,
            hidden_dim=hidden_dim
        )

    def forward(self, obs):
        # print(obs['edge_features'])
        return self.gnn(obs['edge_features'],
                        obs['edge_index']
            
        )


# ============================================================
# HIERARCHICAL RMSA POLICY (CLEAN PPO VERSION)
# ============================================================

class HierarchicalRMSAPolicy(nn.Module):

    def __init__(
        self,
        edge_dim,
        slot_dim,
        path_feature_dim,
        num_paths,
        num_mods,
        hidden_dim=128
    ):
        super().__init__()

        # -------------------------
        # ENCODER
        # -------------------------

        self.encoder = SharedEncoder(edge_dim, hidden_dim)

        self.path_encoder = PathTransformer(
            edge_dim=hidden_dim,
            hidden_dim=hidden_dim
        )

        self.path_policy = PathPolicy(
            path_dim=hidden_dim,
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

        self.num_paths = num_paths

    # ============================================================
    # PATH ACTION
    # ============================================================

    def act_path(self, obs):

        edge_emb = self.encoder(obs)

        path_emb, _ = self.path_encoder(
            edge_emb,
            obs["candidate_paths"]
        )

        logits = self.path_policy(
            path_emb,
            obs["action_masks"]["path"],
            obs["candidate_paths_features"]
        )
        try:
            dist = D.Categorical(logits=logits)
        except Exception as e:
            print("Edge emb", edge_emb)
            print("obs candidate path", obs["candidate_paths"])
            print("Path emb", path_emb)
            print("Action mask path", obs["action_masks"]["path"])
            raise e
        
        batch_idx = torch.arange(path_emb.size(0), device=path_emb.device)

        action = dist.sample()
        logprob = dist.log_prob(action)

        return action, logprob, {
            "edge_emb": edge_emb,
            "path_emb": path_emb,
            "selected_path_emb": path_emb[batch_idx, action]
        }

    # ============================================================
    # MODULATION ACTION
    # ============================================================

    def act_modulation(self, obs, path_emb):

        logits, context  = self.mod_policy(
            path_emb,
            obs["path_features"],
            obs["action_masks"]["mod"]
        )

        dist = D.Categorical(logits=logits)

        action = dist.sample()
        logprob = dist.log_prob(action)

        selected_mod_emb = self.mod_policy.build_spectrum_context(
            action,
            context
        )

        return action, logprob, selected_mod_emb

    # ============================================================
    # SLOT ACTION
    # ============================================================

    def act_slot(self, obs, cache):
        path_emb = cache["selected_path_emb"] #.unsqueeze(0)
        mod_emb = cache["selected_mod_emb"]

        logits, _ = self.slot_policy(
            obs["mod_features"],
            path_emb,
            mod_emb,
            obs["action_masks"]["slot"]
        )

        dist = D.Categorical(logits=logits)

        action = dist.sample()
        logprob = dist.log_prob(action)

        return action, logprob

    # ============================================================
    # VALUE FUNCTION (CRITIC)
    # ============================================================
    @torch.no_grad()
    def evaluate_value(self, obs):
    
        # =========================================================
        # EDGE ENCODING
        # =========================================================
        # print(obs)
        edge_emb = self.encoder(obs)
    
        path_emb, _ = self.path_encoder(
            edge_emb,
            obs["candidate_paths"]
        )
    
        path_logits = self.path_policy(
            path_emb,
            obs["action_masks"]["path"],
            obs["candidate_paths_features"]
        )
    
        path_action = path_logits.argmax(dim=-1)
    
        batch_idx = torch.arange(path_emb.size(0), device=path_emb.device)
    
        selected_path_emb = path_emb[batch_idx, path_action]
        # selected_path_emb = path_emb[path_action]
        # print(f"path_emb = {path_emb.shape} selected_path_emb = {selected_path_emb.shape}")
    
        # =========================================================
        # MODULATION
        # =========================================================
    
        mod_logits, mod_context = self.mod_policy(
            selected_path_emb,
            obs["path_features"],
            obs["action_masks"]["mod"]
        )
    
        mod_action = mod_logits.argmax(dim=-1)
    
        mod_emb = self.mod_policy.build_spectrum_context(
            mod_action,
            mod_context
        )
    
        # =========================================================
        # CRITIC VALUE
        # =========================================================
    
        value = self.critic(
            edge_emb,
            selected_path_emb,
            mod_emb
        )
    
        return value.squeeze(-1)
    
    def evaluate_value_old(self, cache):

        # edge_emb = self.encoder(obs)

        # path_emb, _ = self.path_encoder(
        #     edge_emb,
        #     obs["candidate_paths"]
        # )
        
        

        # pooled_path = path_emb.mean(dim=1)

        # zero_mod = torch.zeros_like(pooled_path)
        
        # value = self.critic(
        #     edge_emb,
        #     pooled_path,
        #     zero_mod
        # )
        
        edge_emb = cache["edge_emb"]
        path_emb = cache["selected_path_emb"]#.unsqueeze(0)
        mod_emb = cache["selected_mod_emb"]#.unsqueeze(0)

        value = self.critic(
            edge_emb,
            path_emb,
            mod_emb
        )

        

        return value.squeeze(-1)

    # ============================================================
    # PPO FORWARD PASS (RECOMPUTE ONLY)
    # ============================================================

    def forward_ppo(self, batch):

      

        B = batch["path_actions"].shape[0]

        # -------------------------
        # ENCODING
        # -------------------------
        

        edge_emb = self.encoder(batch)

        path_emb, _ = self.path_encoder(
            edge_emb,
            batch["candidate_paths"]
        )

        # -------------------------
        # PATH LOGITS
        # -------------------------

        path_logits = self.path_policy(
            path_emb,
            batch["action_masks"]["path"],
            batch["candidate_paths_features"]
        )

        path_actions = batch["path_actions"]

        batch_idx = torch.arange(
            B,
            device=path_actions.device
        )

        selected_path_emb = path_emb[
            batch_idx,
            path_actions
        ]

        selected_path_features = batch["path_features"]

        # -------------------------
        # MODULATION LOGITS
        # -------------------------

        mod_logits, mod_context = self.mod_policy(
            selected_path_emb,
            selected_path_features,
            batch["action_masks"]["mod"]
        )

        mod_actions = batch["mod_actions"]

        mod_emb = self.mod_policy.build_spectrum_context(
            mod_actions,
            mod_context
        )

        # -------------------------
        # SLOT LOGITS
        # -------------------------

        slot_logits, _ = self.slot_policy(
            batch["mod_features"],
            selected_path_emb,
            mod_emb,
            batch["action_masks"]["slot"]
        )

        # -------------------------
        # VALUE
        # -------------------------

        value = self.critic(
            edge_emb,
            selected_path_emb,
            mod_emb
        )

        return {
            "logits_path": path_logits,
            "logits_mod": mod_logits,
            "logits_slot": slot_logits,
            "value": value.squeeze(-1),
            
            
        }