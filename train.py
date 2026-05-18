# train.py

import torch
import numpy as np
import random

from custom_env.optical_rl_gym.envs.rmsa_env  import RMSAEnv
from custom_env.optical_rl_gym.utils import evaluate_heuristic
from custom_env.CustomRLenv.model import HierarchicalPolicy
from custom_env.CustomRLenv.CustomRMSAEnv import  CustomRMSAEnv
from custom_env.CustomRLenv.utils import transform_graph
from env import constant
from custom_env.CustomRLenv.utils import get_topology
import pickle
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt


import torch.optim as optim

from collections import defaultdict



from RL.models.hierarchical_policy import HierarchicalRMSAPolicy

from RL.ppo.rollout_worker import RolloutWorker
from RL.ppo.buffer import HierarchicalReplayBuffer
from RL.ppo.trainer import PPOTrainer

from RL.utils.batching import batch_episodes
from RL.utils.logging import Logger


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
DEVICE = "cpu"
print(f"\nUsing device: {DEVICE}\n")


# ============================================================
# CONFIG
# ============================================================

# DATA LOADING PARAMS
LOAD = 300  # Traffic load, measured in Erlangs
EPISODES = 100 # Number of episodes per execution
EPISODE_LENGTH= 1000 
MEAN_SERVICE_HOLDING_TIME = 200
NUM_SPECTRUM_RESOURCES = 380


# TRAINING PARAMS
NUM_ITERATIONS = 1000
EPISODES_PER_ITERATION = 8
LR = 3e-4
CLIP_EPS = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01

# NEURAL NWTWORK ARCHITECTURE PARAMS
HIDDEN_DIM =128


# ============================================================
# ENVIRONMENT
# ============================================================

# Topology
topology = get_topology(f'./data/germany/sndlib_germany.txt', 'Germany', sndformat=True, alpha=1)
# topology = get_topology('./data/nsf/nsfnet_chen.txt', 'NSFNET')

# Environment arguments for the simulation
env_args = dict(topology=topology, 
                seed=SEED, 
                allow_rejection=True, 
                load=LOAD, 
                mean_service_holding_time=MEAN_SERVICE_HOLDING_TIME,
                episode_length=EPISODE_LENGTH, 
                num_spectrum_resources=NUM_SPECTRUM_RESOURCES,
                bit_rates = constant.bit_rates,
                bit_rate_selection="discrete",)

    
# Environment setup
env = CustomRMSAEnv(**env_args)

obs = env.customreset(False)
a = env.step_path(obs, 0)
b = env.step_modulation(obs, 0)
next_state, reward, done, info = env.step(0)


# ============================================================
# MODEL
# ============================================================

# Get environment info
num_nodes = env.num_nodes
num_edges = env.topology.number_of_edges()
num_paths = env.max_num_path
num_mods =  len(env.topology.graph['modulations'])
num_spectra = env.num_spectrum_resources

edge_dim = 4 # env.topology.number_of_edges()
path_feature_dim= 1 + 1 + 2 * len(env.topology.graph['modulations'])
slot_dim =  2 


policy = HierarchicalRMSAPolicy(
    edge_dim=edge_dim,
    slot_dim=slot_dim,
    path_feature_dim=path_feature_dim,
    num_paths=num_paths,
    num_mods=num_mods,
    hidden_dim=HIDDEN_DIM
).to(DEVICE)


# ============================================================
# PPO COMPONENTS
# ============================================================

worker = RolloutWorker(
    env=env,
    policy=policy,
    device=DEVICE
)

buffer = HierarchicalReplayBuffer()


 
trainer = PPOTrainer(
    policy=policy,
    logger=Logger(),
    lr=LR,
    clip_eps=CLIP_EPS,
    value_coef=VALUE_COEF,
    entropy_coef=ENTROPY_COEF,
    device=DEVICE
)


# ============================================================
# TRAIN LOOP
# ============================================================

print("Starting training...\n")

for iteration in range(NUM_ITERATIONS):

    # --------------------------------------------------------
    # CLEAR BUFFER
    # --------------------------------------------------------

    buffer.clear()

    # --------------------------------------------------------
    # COLLECT ROLLOUTS
    # --------------------------------------------------------

    episode_rewards = []
    episode_service_blocking_rate = []
    episode_bit_rate_blocking_rate = []
    episode_avg_link_utilization = []
    

    for _ in range(EPISODES_PER_ITERATION):

        episode = worker.collect_episode()

        buffer.add_episode(episode)

        episode_rewards.append(sum(episode["rewards"]))
        episode_service_blocking_rate.append(episode["service_blocking_rate"])
        episode_bit_rate_blocking_rate.append(episode["bit_rate_blocking_rate"])
        episode_avg_link_utilization.append(episode["avg_link_utilization"])
        
        print(f"Reward={sum(episode['rewards']):.4f} | "
              f"service_blocking_rate = {episode['service_blocking_rate']:.4f} | "
              f"bit_rate_blocking_rate = {episode['bit_rate_blocking_rate']:.4f} | ")

    # --------------------------------------------------------
    # BUILD TRAINING BATCH
    # --------------------------------------------------------

    batch = batch_episodes(buffer.episodes)

    # --------------------------------------------------------
    # TRAIN PPO
    # --------------------------------------------------------

    stats = trainer.train_step(batch)

    # --------------------------------------------------------
    # LOGGING
    # --------------------------------------------------------

    mean_reward = np.mean(episode_rewards)
    mean_service_blocking_rate = np.mean(episode_service_blocking_rate)
    mean_bit_rate_blocking_rate = np.mean(episode_bit_rate_blocking_rate)
    mean_avg_link_utilization = np.mean(episode_avg_link_utilization)

    print(
        f"[Iter {iteration:04d}] "
        f"Reward={mean_reward:.4f} | "
        f"Loss={stats['loss_total']:.4f} | "
        f"PathEnt={stats['entropy_path']:.4f} | "
        f"ModEnt={stats['entropy_mod']:.4f} | "
        f"SlotEnt={stats['entropy_slot']:.4f}"
    )
    
    print(f"service_blocking_rate = {mean_service_blocking_rate:.4f} | "
          f"bit_rate_blocking_rate = {mean_bit_rate_blocking_rate:.4f} | "
          f"avg_link_utilization = {mean_avg_link_utilization:.2f}")
    # --------------------------------------------------------
    # OPTIONAL CHECKPOINT
    # --------------------------------------------------------

    if iteration % 1000000 == 0:

        checkpoint = {
            "iteration": iteration,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict()
        }

        torch.save(
            checkpoint,
            f"checkpoint_{iteration}.pt"
        )

        print(f"Checkpoint saved at iteration {iteration}")


print("\nTraining complete.\n")