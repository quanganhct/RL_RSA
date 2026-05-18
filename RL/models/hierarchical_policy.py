import torch
import torch.nn as nn
import torch.distributions as D

from RL.models.gnn_encoder import GNNEncoder
from RL.models.path_transformer import PathTransformer
from RL.models.path_policy import PathPolicy
from RL.models.modulation_policy import ModulationPolicy
from RL.models.spectrum_policy import SpectrumPolicy
from RL.models.critic import Critic


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
        # print(f"edge_features {obs['edge_features'].shape}  edge_features {obs['edge_index'].shape} ")

        return self.gnn(
            obs["edge_features"],
            obs["edge_index"]
        )


# ============================================================
# HIERARCHICAL RMSA POLICY
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
        # ENCODERS
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
    # PATH SELECTION
    # ============================================================

    def act_path_old(self, obs):

        edge_emb = self.encoder(obs)

        path_emb, _ = self.path_encoder(
            edge_emb,
            obs["candidate_paths"]
        )
        
        # print(f'path_embed jojo = {path_emb.shape}')

        logits = path_emb.mean(dim=-1)

        mask = obs["action_masks"]["path"]

        logits = logits.masked_fill(~mask.bool(), -1e9)

        dist = D.Categorical(logits=logits)

        action = dist.sample()

        logprob = dist.log_prob(action)

        return action.item(), logprob, {
            "edge_emb": edge_emb,
            "path_emb": path_emb,
            "selected_path_emb": path_emb[action]
        }


    def act_path(self, obs):

        edge_emb = self.encoder(obs)

        path_emb, _ = self.path_encoder(
            edge_emb,
            obs["candidate_paths"]
        )
        
        # print(f'path_embed jojo = {path_emb.shape}')
        mask = obs["action_masks"]["path"]

        logits = self.path_policy(path_emb, mask) #path_emb.mean(dim=-1)

        

        # logits = logits.masked_fill(~mask.bool(), -1e9)

        dist = D.Categorical(logits=logits)

        action = dist.sample()

        logprob = dist.log_prob(action)

        return action.item(), logprob, {
            "edge_emb": edge_emb,
            "path_emb": path_emb,
            "selected_path_emb": path_emb[action]
        }

    # ============================================================
    # MODULATION SELECTION
    # ============================================================

    def act_modulation(self, obs, cache):

        path_emb = cache["selected_path_emb"]#.unsqueeze(0)
        # print(f"path_emb = {path_emb.shape} selected_path_emb {path_emb.shape}")

        logits, context = self.mod_policy(
            path_emb,
            obs["path_features"],#.unsqueeze(0),
            obs["action_masks"]["mod"]
        )
        # print(f"logits = {logits} ")

        dist = D.Categorical(logits=logits)

        action = dist.sample()

        logprob = dist.log_prob(action)

        mod_emb = self.mod_policy.build_spectrum_context(action, context)
            

        return action.item(), logprob, mod_emb#.squeeze(0)

    # ============================================================
    # SLOT SELECTION
    # ============================================================

    def act_slot(self, obs, cache):

        path_emb = cache["selected_path_emb"]#.unsqueeze(0)
        mod_emb = cache["selected_mod_emb"]#.unsqueeze(0)

        logits, slot_emb = self.slot_policy(
            obs["mod_features"],#.unsqueeze(0),
            path_emb,
            mod_emb,
            obs["action_masks"]["slot"]
        )

        dist = D.Categorical(logits=logits)

        action = dist.sample()

        logprob = dist.log_prob(action)

        return action.item(), logprob

    # ============================================================
    # CRITIC
    # ============================================================

    def value(self, obs, cache):

        edge_emb = cache["edge_emb"]
        path_emb = cache["selected_path_emb"]#.unsqueeze(0)
        mod_emb = cache["selected_mod_emb"]#.unsqueeze(0)

        return self.critic(
            edge_emb,
            path_emb,
            mod_emb
        )

    # ============================================================
    # PPO RECOMPUTE (FIXED - NO RE-SAMPLING)
    # ============================================================

    def forward_ppo(self, batch):

        obs = batch["obs"]
    
        B = batch["path_actions"].shape[0]
    
        # =======================================================
        # EDGE ENCODER
        # =======================================================
    
        edge_emb = self.encoder(obs)
    
        # =======================================================
        # PATH ENCODER
        # =======================================================
    
        path_emb, _ = self.path_encoder(
            edge_emb,
            obs["candidate_paths"]
        )
    
        # =======================================================
        # PATH LOGITS
        # =======================================================
    
        path_logits = self.path_policy(
            path_emb,
            obs["action_masks"]["path"]
        )
    
        # =======================================================
        # SELECT PATH EMBEDDINGS
        # =======================================================
    
        path_actions = batch["path_actions"]
    
        batch_idx = torch.arange(
            B,
            device=path_actions.device
        )
    
        selected_path_emb = path_emb[
            batch_idx,
            path_actions
        ]
        # print(f"selected_path_emb = {selected_path_emb.shape}")
        # print(f"selected_path_features = {obs['path_features'].shape}")
        
        
        # IMPORTANT:
        # path_features should be [B,K,F]
        selected_path_features = obs["path_features"]
    
        # selected_path_features = obs["path_features"][
        #     batch_idx,
        #     path_actions
        # ]
    
        # =======================================================
        # MODULATION LOGITS
        # =======================================================
    
        mod_logits, mod_context = self.mod_policy(
            selected_path_emb,
            selected_path_features,
            obs["action_masks"]["mod"]
        )
    
        # =======================================================
        # MODULATION EMBEDDING
        # =======================================================
    
        mod_actions = batch["mod_actions"]
    
        mod_emb =  self.mod_policy.build_spectrum_context(mod_actions, mod_context)
    
        # =======================================================
        # SLOT LOGITS
        # =======================================================
    
        slot_logits, slot_emb = self.slot_policy(
            obs["mod_features"],
            selected_path_emb,
            mod_emb,
            obs["action_masks"]["slot"]
        )
    
        # =======================================================
        # VALUE FUNCTION
        # =======================================================
    
        value = self.critic(
            edge_emb,
            selected_path_emb,
            mod_emb
        )
    
        return {
    
            "logits_path": path_logits,
    
            "logits_mod": mod_logits,
    
            "logits_slot": slot_logits,
    
            "value": value
        }
    

    # def yforward_ppo_old(self, batch):

    #     obs = batch["obs"]

    #     edge_emb = self.encoder(obs)

    #     path_emb, _ = self.path_encoder(
    #         edge_emb,
    #         obs["candidate_paths"]
    #     )
        
  

    #     path_actions = batch["path_actions"]
    #     mod_actions = batch["mod_actions"]

    #     selected_path_emb = path_emb[
    #         torch.arange(len(path_actions)),
    #         path_actions
    #     ]
    #     print(f"path_emb = {path_emb.shape} selected_path_emb {selected_path_emb.shape}")

    #     # -------------------------
    #     # MOD LOGITS
    #     # -------------------------

    #     mod_logits = self.mod_policy(
    #         selected_path_emb,
    #         obs["path_features"],
    #         obs["action_masks"]["mod"]
    #     )

    #     mod_emb = self.mod_policy.get_embedding(
    #         selected_path_emb,
    #         obs["path_features"]
    #     )

    #     # -------------------------
    #     # SLOT LOGITS
    #     # -------------------------

    #     slot_logits = self.slot_policy(
    #         obs["slot_features"],
    #         selected_path_emb,
    #         mod_emb,
    #         obs["action_masks"]["slot"]
    #     )

    #     # -------------------------
    #     # VALUE
    #     # -------------------------

    #     value = self.critic(
    #         edge_emb,
    #         selected_path_emb,
    #         mod_emb
    #     )

    #     return {
    #         "logits_path": path_emb.mean(dim=-1),
    #         "logits_mod": mod_logits,
    #         "logits_slot": slot_logits,
    #         "value": value
    #     }