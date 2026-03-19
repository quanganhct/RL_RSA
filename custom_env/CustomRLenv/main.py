# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 22:29:50 2026

@author: Momo
"""

import gym
from optical_rl_gym.envs.rmsa_env import RMSAEnv
from optical_rl_gym.utils import evaluate_heuristic
from CustomRLenv.model import HierarchicalPolicy
from CustomRLenv.CustomRMSAEnv import  CustomRMSAEnv
import pickle
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt


import torch.optim as optim

from collections import defaultdict


# ==========================================================
# 1 - GAE ADVANTAGE
# ==========================================================

def compute_gae(rewards,values,dones,gamma=0.99,lam=0.95):

    advantages=[]
    gae=0
    next_value=0

    for t in reversed(range(len(rewards))):

        delta=rewards[t]+gamma*next_value*(1-dones[t])-values[t]

        gae=delta+gamma*lam*(1-dones[t])*gae

        advantages.insert(0,gae)

        next_value=values[t]

    returns=[a+v for a,v in zip(advantages,values)]

    return torch.tensor(advantages),torch.tensor(returns)


# ==========================================================
# 2 - PPO UPDATE
# ==========================================================

def ppo_update(model,optimizer,logprobs_old,logprobs_new,
               advantages,returns,values,clip=0.2):

    ratio=torch.exp(logprobs_new-logprobs_old)

    surr1=ratio*advantages
    surr2=torch.clamp(ratio,1-clip,1+clip)*advantages

    policy_loss=-torch.min(surr1,surr2).mean()

    value_loss=(returns-values).pow(2).mean()

    loss=policy_loss+0.5*value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
# -----------------------------
# 3 - Training logger
# -----------------------------
class TrainingLogger2:
    def __init__(self, env):
        self.stats = defaultdict(list)
        self.env  = env
    def log_step(self, reward, blocked, impairment, fragmentation):
        self.stats["reward"].append(reward)
        self.stats["blocked"].append(blocked)
        self.stats["impairment"].append(impairment)
        self.stats["fragmentation"].append(fragmentation)

    def log_episode(self):
        # Compute episode-level summaries
        self.stats["episode_reward"].append(sum(self.stats["reward"][-self.env.episode_length:]))
        self.stats["episode_blocking"].append(np.mean(self.stats["blocked"][-self.env.episode_length:]))
        self.stats["episode_impairment"].append(np.mean(self.stats["impairment"][-self.env.episode_length:]))
        self.stats["episode_fragmentation"].append(np.mean(self.stats["fragmentation"][-self.env.episode_length:]))


class TrainingLogger:
    def __init__(self, env):
        self.stats = defaultdict(list)
        self.env  = env
    def log_step(self, reward, service_blocking_rate, bit_rate_blocking_rate, avg_link_utilization):
        self.stats["reward"].append(reward)
        self.stats["service_blocking_rate"].append(service_blocking_rate)
        self.stats["bit_rate_blocking_rate"].append(bit_rate_blocking_rate)
        self.stats["avg_link_utilization"].append(avg_link_utilization)

    def log_episode(self):
        # Compute episode-level summaries
        self.stats["episode_reward"].append(sum(self.stats["reward"][-self.env.episode_length:]))
        self.stats["episode_service_blocking_rate"].append(np.mean(self.stats["service_blocking_rate"][-self.env.episode_length:]))
        self.stats["episode_bit_rate_blocking_rate"].append(np.mean(self.stats["bit_rate_blocking_rate"][-self.env.episode_length:]))
        self.stats["episode_avg_link_utilization"].append(np.mean(self.stats["avg_link_utilization"][-self.env.episode_length:]))


# ==========================================================
# 4. TRAINING LOOP
# ==========================================================
from utils import get_topology
def train():
    
    load = 120  # Traffic load, measured in Erlangs
    seed = 20  # Seed of environment
    episodes = 100 # Number of episodes per execution
    episode_length = 300  # Episode Length

    with open(f'optical_rl/examples/topologies/nsfnet_chen_5-paths_6-modulations.h5', 'rb') as f:
       topology = pickle.load(f)

    # topology = get_topology()

    # Environment arguments for the simulation
    env_args = dict(topology=topology, 
                    seed=seed, 
                    allow_rejection=True, 
                    load=load, 
                    mean_service_holding_time=25,
                   episode_length=episode_length, 
                   num_spectrum_resources=100)

    # -----------------------------
    # Environment setup
    # -----------------------------
    env = CustomRMSAEnv(**env_args)



    # Get environment info
    num_nodes = env.num_nodes
    num_edges = env.topology.number_of_edges()
    num_paths = env.k_paths
    num_modulations =  len(env.topology.graph['modulations'])
    num_spectra = env.num_spectrum_resources



    # -----------------------------
    # RL agent setup
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    policy = HierarchicalPolicy(
        node_dim=env.get_node_features().shape[1],         # degree, betweenness
        edge_dim=env.get_edge_features().shape[1],
        hidden_dim=128,
        # num_paths=num_paths,
        num_mod=num_modulations,
        num_slots=num_spectra
    ).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

   
    logger = TrainingLogger(env)
    
    
    # Build edge_index
    graph_nx = env.topology
    edges =  [tuple(map(int, t)) for t in graph_nx.edges()]
    if len(edges) == 0:  # avoid empty graph
        edge_index = torch.zeros((2,0), dtype=torch.long).to(device)
    else:
        # nodes are from 1 to num_nodes 
        # we do -1 so that it is now from 0 to num_nodes - 1
        # the library can be updated so that nodes are directly from 0
        edge_index = torch.tensor(edges, dtype=torch.long).T - 1  # [2, num_edges]
        
    
    # -----------------------------
    # Training loop (mini-example)
    # -----------------------------
    
    for ep in range(episodes):

        state = env.customreset(False)
        done = False

        logps=[]
        values=[]
        rewards=[]
        dones=[]

        while not done:
            
            # Convert features to torch tensors
            node_features = torch.tensor(state["node_features"], 
                                         dtype=torch.float32).to(device)
            edge_features = torch.tensor(state["edge_features"], 
                                         dtype=torch.float32).to(device)
            demand_embedding = torch.tensor(state["demand_embedding"], 
                                            dtype=torch.float32).unsqueeze(0).to(device)
            path_impairments   = torch.tensor(state['candidate_paths_impairment'], 
                                              dtype=torch.float32).to(device)

            # candidate_paths = torch.tensor(state['candidate_paths'],
            #                                dtype=torch.long).to(device)
            
            candidate_paths = state['candidate_paths']
            path_spectrum = state['path_spectrum']
                                           
            

            
            # Masks (binary tensors)
            path_mask, mod_mask, spec_mask = state["masks"]
            
            path_spectrum = torch.tensor(path_spectrum, dtype=torch.float32).to(device)
            path_mask = torch.tensor(path_mask, dtype=torch.bool).to(device)
            mod_mask = torch.tensor(mod_mask, dtype=torch.bool).to(device)
            spec_mask = torch.tensor(spec_mask, dtype=torch.bool).to(device)
            
            
            
            current_state = dict(
                node_feat=node_features,
                edge_feat=edge_features,
                edge_index=edge_index,
                paths=candidate_paths,
                path_mask=path_mask,
                path_spectrum=path_spectrum,
                path_features=path_impairments,
                mod_masks=mod_mask,
                spec_masks=spec_mask
            )

            
            # Get action + probabilities + value
            action, logprob, value = policy(current_state)
            
            # print(f"action = {action}")

            # Step environment
            next_state, reward, done, info = env.step(action)

            logps.append(logprob)
            values.append(value.item())
            rewards.append(reward)
            dones.append(done)

            
            # new state 
            state = next_state
            
            # Logging
            logger.log_step(reward, 
                            info['service_blocking_rate'],
                            info['bit_rate_blocking_rate'],
                            info['avg_link_utilization']
                            )

        logps = torch.stack(logps)

        advantages, returns = compute_gae(rewards, values, dones)

        
#------------------------------------------------------------
#------------------------------------------------------------
        # recompute logprobs with current policy
        new_logps = []
        new_values = []

        state = env.customreset(False)

        done=False
        
        while not done:
            
            # Convert features to torch tensors
            node_features = torch.tensor(state["node_features"], 
                                         dtype=torch.float32).to(device)
            edge_features = torch.tensor(state["edge_features"], 
                                         dtype=torch.float32).to(device)
            demand_embedding = torch.tensor(state["demand_embedding"], 
                                            dtype=torch.float32).unsqueeze(0).to(device)
            path_impairments   = torch.tensor(state['candidate_paths_impairment'], 
                                              dtype=torch.float32).to(device)

            # candidate_paths = torch.tensor(state['candidate_paths'],
            #                                dtype=torch.long).to(device)
            
            candidate_paths = state['candidate_paths']
            path_spectrum = state['path_spectrum']
                                           
            

            
            # Masks (binary tensors)
            
            path_mask, mod_mask, spec_mask = state["masks"]
            
            path_spectrum = torch.tensor(path_spectrum, dtype=torch.float32).to(device)
            path_mask = torch.tensor(path_mask, dtype=torch.bool).to(device)
            mod_mask = torch.tensor(mod_mask, dtype=torch.bool).to(device)
            spec_mask = torch.tensor(spec_mask, dtype=torch.bool).to(device)
            
            
            
            current_state = dict(
                node_feat=node_features,
                edge_feat=edge_features,
                edge_index=edge_index,
                paths=candidate_paths,
                path_mask=path_mask,
                path_spectrum=path_spectrum,
                path_features=path_impairments,
                mod_masks=mod_mask,
                spec_masks=spec_mask
            )

            
            # Get action + probabilities + value
            action, logprob, value = policy(current_state)
            # Step environment
            next_state, reward, done, info = env.step(action)
            
            new_logps.append(logprob)
            new_values.append(value)
            
            state = next_state
            
        new_logps=torch.stack(new_logps)
        new_values=torch.stack(new_values)
        
        
        loss=ppo_update(
            policy,
            optimizer,
            logps.detach(),
            new_logps,
            advantages,
            returns,
            new_values
        )
            
            
            
        
        # End of episode logging
        logger.log_episode()
        print(f"Episode {ep+1}: Reward={logger.stats['episode_reward'][-1]:.2f}, "
              f"service_blocking_rate={logger.stats['episode_service_blocking_rate'][-1]:.2f}, "
              f"bit_rate_blocking_rate={logger.stats['episode_bit_rate_blocking_rate'][-1]:.2f}, "
              f"avg_link_utilization={logger.stats['episode_avg_link_utilization'][-1]:.2f}")
        
        print(
            f"Episode {ep+1} | Reward {sum(rewards):.3f} | Loss {loss:.4f}"
        )
        
    return logger, env

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    print("\nHierarchical GNN RMSA Example\n")

    logger, env = train()
    
    plt.plot(logger.stats["episode_reward"], label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()

    plt.plot(logger.stats["episode_service_blocking_rate"], label="Episode service_blocking_rate")
    plt.xlabel("Episode")
    plt.ylabel("service_blocking_rate")
    plt.legend()
    plt.show()

    plt.plot(logger.stats["episode_bit_rate_blocking_rate"], label="Episode bit_rate_blocking_rate")
    plt.xlabel("Episode")
    plt.ylabel("bit_rate_blocking_rate")
    plt.legend()
    plt.show()

    plt.plot(logger.stats["episode_avg_link_utilization"], label="Episode avg_link_utilization")
    plt.xlabel("Episode")
    plt.ylabel("avg_link_utilization")
    plt.legend()
    plt.show()
    
    # average reward
    window = 10
    
    avg_reward = [
        np.mean(logger.stats["episode_reward"][i:i+window])
        for i in range(0, len(logger.stats["episode_reward"]) -window + 1,
                       window)]
    x_ax = np.arange(len(avg_reward))*window + window
    
    plt.plot(x_ax, avg_reward, label=f"Average rewards over {window} episodes")
    plt.xlabel("Episode")
    plt.ylabel(f"Average rewards per {window} episodes")
    plt.legend()
    plt.show()