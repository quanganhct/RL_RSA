# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 11:56:39 2026

@author: Momo
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
"""
Hierarchical GNN Policy for RMSA (Toy Research Prototype)

Implements the formulation:

π(a|X) =
π_path(p|X) *
π_mod(m|p,X) *
π_spec(f|p,m,X)

Components
----------
1. Graph Neural Network encoder
2. Path agent
3. Modulation agent
4. Spectrum agent (Conv1D)
5. Hierarchical policy
6. Toy RMSA environment
7. Policy gradient training loop
"""




# ==========================================================
# 1. GRAPH NEURAL NETWORK ENCODER
# ==========================================================
# Encodes graph topology and link attributes into embeddings
#
# Input
# -----
# node_feat : [N, d_v]
# edge_feat : [E, d_e]
# edge_index : [2, E]
#
# Output
# ------
# node_embeddings : [N, d_h]
# edge_embeddings : [E, d_h]
#
# where
# N = number of nodes
# E = number of edges
# d_h = hidden dimension
# ==========================================================

class GNNEncoder(nn.Module):

    def __init__(self, node_dim, edge_dim, hidden_dim):

        super().__init__()

        # Project raw node features to hidden dimension
        # node_feat [N,d_v] → [N,d_h]
        self.node_proj = nn.Linear(node_dim, hidden_dim)

        # Project edge features to hidden dimension
        # edge_feat [E,d_e] → [E,d_h]
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        # Message computation
        # concat(node,edge) → hidden
        self.message = nn.Linear(hidden_dim * 2, hidden_dim)


        # Node update (GRU aggregation)
        self.update = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, node_feat, edge_index, edge_feat):

        # Initial node embeddings
        h = self.node_proj(node_feat)         # [N, d_h]

        # Edge embeddings
        e = self.edge_proj(edge_feat)         # [E, d_h]

        src, dst = edge_index                 # [E], [E]

        # Message passing iterations with 3 hops messages
        for _ in range(3):

            # Construct messages from source node + edge
            m = torch.cat([h[src], e], dim=-1)   # [E, 2*d_h]

            m = self.message(m)                  # [E, d_h]

            # Aggregate messages to destination nodes
            agg = torch.zeros_like(h)            # [N, d_h]
            agg.index_add_(0, dst, m)

            # Update node embeddings
            h = self.update(agg, h)              # [N, d_h]

        return h, e
    
    


# ==========================================================
# 2. PATH AGENT
# ==========================================================
#
# Implements:
#
# z_p = Aggregate({h_e : e ∈ p})
#
# π_path(p|X) = Softmax(W_p z_p)
#
# Inputs
# ------
# edge_emb : [E, d_h]
# paths : list of candidate paths
# path_mask : [K]
#
# Output
# ------
# p : selected path index
# logprob : log π_path
# z_p : embedding of chosen path
#
# where
# K = number of candidate paths
# ==========================================================

class PathAgent(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, edge_emb, paths, path_mask, deterministic):

        path_embeddings = []

        # Compute embedding for each candidate path
        for p in paths:

            # p contains edge indices
            # edge_emb[p] → [|p|, d_h]

            emb = edge_emb[p].mean(dim=0)   # Aggregate edges → [d_h]

            path_embeddings.append(emb)

        z = torch.stack(path_embeddings)    # [K, d_h]

        logits = self.linear(z).squeeze()   # [K]

        # Apply path feasibility mask
        logits = logits.masked_fill(path_mask == 0, -1e9)

        dist = Categorical(logits=logits)
        
        if deterministic:
            p = torch.argmax(logits, dim=-1)
        else:
            p = dist.sample()               # scalar

       

        logprob = dist.log_prob(p)

        return p, logprob, z[p]


# ==========================================================
# 3. MODULATION AGENT
# ==========================================================
#
# Implements
#
# φ_p = [path length, Σ NLI, min OSNR]
#
# π_mod(m|p,X) = Softmax(W_m [z_p || φ_p])
#
# Input
# -----
# z_p : [d_h]
# phi_p : [3]
# mod_mask : [M]
#
# Output
# ------
# m : modulation index
# logprob
#
# where
# M = number of modulation formats
# ==========================================================

class ModulationAgent(nn.Module):

    def __init__(self, hidden_dim, num_mod):

        super().__init__()

        self.linear = nn.Linear(hidden_dim + 3, num_mod)

    def forward(self, z_p, phi_p, mod_mask, deterministic):

        x = torch.cat([z_p, phi_p], dim=-1)  # [d_h + 3]

        logits = self.linear(x)              # [M]

        logits = logits.masked_fill(mod_mask == 0, -1e9)

        dist = Categorical(logits=logits)
        
        if deterministic:
            m = torch.argmax(logits, dim=-1)
        else:
            m = dist.sample()
       

        return m, dist.log_prob(m)


# ==========================================================
# 4. SPECTRUM AGENT
# ==========================================================
#
# Implements
#
# S_p = ∩ s_e
#
# h_f = Conv1D(S_p)
#
# π_spec(f|p,m,X) = MaskedSoftmax(W_f h_f)
#
# Input
# -----
# spectrum : [F]
# slots_required : scalar
#
# Output
# ------
# slot_start : starting slot
#
# where
# F = number of spectrum slots
# ==========================================================

class SpectrumAgent(nn.Module):

    def __init__(self, num_slots):

        super().__init__()

        self.conv1 = nn.Conv1d(1, 8, 5, padding=2)
        self.conv2 = nn.Conv1d(8, 1, 1)

        self.num_slots = num_slots

    def forward(self, spectrum, spec_mask, deterministic):

        # spectrum : [F]
        # print(f"spectrum = {spectrum.shape}")
        x = spectrum.unsqueeze(0).unsqueeze(0)  # [1,1,F]

        h = F.relu(self.conv1(x))                # [1,8,F]
        # print(f"h = {h.shape}")
        h = self.conv2(h).squeeze()              # [F]
        
        
        logits = h.masked_fill(spec_mask == 0, -1e9)

        dist = Categorical(logits=logits)

        if deterministic:
            slot = torch.argmax(logits, dim=-1)
        else:
            slot = dist.sample()

        return slot, dist.log_prob(slot)


# ==========================================================
# 5. HIERARCHICAL POLICY
# ==========================================================
#
# Combines the three agents
#
# Output
# ------
# (path, modulation, slot)
# log π(a|X)
# ==========================================================

class HierarchicalPolicy(nn.Module):

    def __init__(self, node_dim,
                 edge_dim,
                 hidden_dim,
                 num_mod,
                 num_slots,
                 deterministic=False):

        super().__init__()
        self.deterministic = deterministic

        self.encoder = GNNEncoder(node_dim, edge_dim, hidden_dim)

        self.path_agent = PathAgent(hidden_dim)

        self.mod_agent = ModulationAgent(hidden_dim, num_mod)

        self.spec_agent = SpectrumAgent(num_slots)
        
        # Value function head (A2C)
        self.value_head = nn.Linear(hidden_dim,1)

    def forward(self, state):

        node_feat = state["node_feat"]          # [N, d_v]
        edge_feat = state["edge_feat"]          # [E, d_e]
        edge_index = state["edge_index"]        # [2,E]

        paths = state["paths"]                  # list of paths
        path_mask = state["path_mask"]          # [K]

        spectrum = state["path_spectrum"]       # [K,F]

        path_features = state["path_features"]  # [K,3]

        mod_masks = state["mod_masks"]          # [K,M]

        spec_masks = state["spec_masks"]          # [K,M,F]

        # GNN encoding
        h_nodes, h_edges = self.encoder(node_feat, edge_index, edge_feat)
        
        
        # global graph embedding for value
        g = h_nodes.mean(dim=0)

        value = self.value_head(g).squeeze()

        # PATH
        p, logp_p, z_p = self.path_agent(h_edges, 
                                         paths, 
                                         path_mask,
                                         self.deterministic )

        # MODULATION
        phi_p = path_features[p]                # [3]

        m, logp_m = self.mod_agent(z_p, phi_p,
                                   mod_masks[p], 
                                   self.deterministic )

        # SLOT
        spec_mask = spec_masks[p,m]
        
        f, logp_f = self.spec_agent(
            spectrum[p],
            spec_mask,
            self.deterministic 
        )

        logprob = logp_p + logp_m + logp_f

        return (p, f, m), logprob, value

    
    def mask_logits(self, logits, mask):

        return logits.masked_fill(mask == 0, -1e9)

