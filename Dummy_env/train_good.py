# train.py

import torch
import numpy as np
import random

from env.dummy_rmsaenv_good import DummyRMSAEnv

from models.hierarchical_policy_good import HierarchicalRMSAPolicy

from ppo.rollout_worker_good_with_ids import RolloutWorker
from ppo.buffer_good import HierarchicalReplayBuffer
from ppo.trainer_good_with_log import PPOTrainer

from utils.batching_good import batch_episodes
from utils.logging_good import Logger


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

print(f"\nUsing device: {DEVICE}\n")


# ============================================================
# CONFIG
# ============================================================

NUM_ITERATIONS = 1000
EPISODES_PER_ITERATION = 8

LR = 3e-4

CLIP_EPS = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01


# ============================================================
# ENVIRONMENT
# ============================================================

env = DummyRMSAEnv()


# ============================================================
# MODEL
# ============================================================

policy = HierarchicalRMSAPolicy(
    edge_dim=4,
    slot_dim=3,
    path_feature_dim=10,
    num_paths=5,
    num_mods=6,
    hidden_dim=128
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

    for _ in range(EPISODES_PER_ITERATION):

        episode = worker.collect_episode()

        buffer.add_episode(episode)

        episode_rewards.append(sum(episode["rewards"]))

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

    print(
        f"[Iter {iteration:04d}] "
        f"Reward={mean_reward:.4f} | "
        f"Loss={stats['loss_total']:.4f} | "
        f"PathEnt={stats['entropy_path']:.4f} | "
        f"ModEnt={stats['entropy_mod']:.4f} | "
        f"SlotEnt={stats['entropy_slot']:.4f}"
    )

    # --------------------------------------------------------
    # OPTIONAL CHECKPOINT
    # --------------------------------------------------------

    if iteration % 100 == 0:

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