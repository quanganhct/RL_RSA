# train.py

"""
=============================================================
FINAL RMSA PPO TRAINING ENTRY POINT
=============================================================

This script runs:

    Hierarchical RMSA PPO agent
    with real environment plugged in externally.

ONLY missing component:
    → real RMSA simulator environment

=============================================================
"""

import torch

from ppo.trainer_good import PPOTrainer
from utils.checkpoint import resume_if_available

from models.hierarchical_policy_good import HierarchicalRMSAPolicy
from env.dummy_rmsaenv_good import DummyRMSAEnv

# ============================================================
# MAIN
# ============================================================

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------
    # ENVIRONMENT (USER PROVIDED)
    # --------------------------------------------------------

    env = DummyRMSAEnv()

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    policy = HierarchicalRMSAPolicy(

        edge_dim=4,          # edge features
        slot_dim=3,          # slot features
        path_feature_dim=10, # path metadata
        # mod_feature_dim=0,   # optional internal
        num_paths=10,
        num_mods=6,
        hidden_dim=128

    ).to(device)

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=3e-4
    )

    # --------------------------------------------------------
    # PPO TRAINER
    # --------------------------------------------------------

    trainer = PPOTrainer(

        env=env,
        policy=policy,
        optimizer=optimizer,
        device=device,

        rollout_episodes=8,
        ppo_epochs=4,
        minibatch_size=32,

        gamma=0.99,
        lam=0.95,

        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01

    )

    # --------------------------------------------------------
    # RESUME CHECKPOINT
    # --------------------------------------------------------

    checkpoint_path = "checkpoints/rmsa_ppo.pt"

    policy, optimizer, start_iter, _ = resume_if_available(
        checkpoint_path,
        policy,
        optimizer,
        map_location=device
    )

    # --------------------------------------------------------
    # TRAIN LOOP
    # --------------------------------------------------------

    trainer.train(

        total_iters=1000,
        log_interval=10,
        save_path=checkpoint_path

    )




# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()

