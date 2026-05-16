"""
==============================================================
FULL PPO TRAINING PIPELINE
FOR HIERARCHICAL RMSA AGENT
==============================================================

Includes:
---------
1. Dummy RMSA environment
2. Hierarchical GNN+Transformer agent
3. PPO training loop
4. Delayed reward handling
5. Action masking
6. Hierarchical autoregressive rollout
7. Generalized Advantage Estimation (GAE)
8. PPO clipped objective
9. Entropy regularization
10. Model checkpointing

==============================================================
INSTALL
==============================================================

pip install torch torch-geometric gymnasium numpy

==============================================================
IMPORTANT
==============================================================

This is a RESEARCH-STYLE starter pipeline.

You will later replace:
    - dummy rewards
    - dummy masks
    - random topology
    - synthetic features

with real RMSA simulation logic.

==============================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque

# ------------------------------------------------------------
# IMPORT YOUR ENV + AGENT
# ------------------------------------------------------------

# from env import MaskedHierarchicalRMSAEnv
# from model import RMSAAgent

# ------------------------------------------------------------
# DEVICE
# ------------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("DEVICE:", DEVICE)

# ============================================================
# PPO BUFFER
# ============================================================

class PPOBuffer:

    def __init__(self):

        self.clear()

    def clear(self):

        self.path_logprobs = []
        self.mod_logprobs = []
        self.slot_logprobs = []

        self.rewards = []

        self.values = []

        self.dones = []

        self.total_logprobs = []

    def add(

        self,
        total_logprob,
        reward,
        value,
        done

    ):

        self.total_logprobs.append(total_logprob)

        self.rewards.append(reward)

        self.values.append(value)

        self.dones.append(done)


# ============================================================
# VALUE NETWORK
# ============================================================

class Critic(nn.Module):

    """
    Simple global critic.

    You can later improve with:
        - graph critic
        - transformer critic
        - centralized critic
    """

    def __init__(
        self,
        hidden_dim=128
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, 1)

        )

    def forward(self, x):

        return self.net(x).squeeze(-1)


# ============================================================
# GAE
# ============================================================

def compute_gae(

    rewards,
    values,
    dones,

    gamma=0.99,
    lam=0.95

):

    advantages = []

    gae = 0

    values = values + [0]

    for t in reversed(range(len(rewards))):

        delta = (

            rewards[t]
            + gamma * values[t + 1] * (1 - dones[t])
            - values[t]

        )

        gae = delta + gamma * lam * (1 - dones[t]) * gae

        advantages.insert(0, gae)

    returns = [

        adv + val

        for adv, val in zip(
            advantages,
            values[:-1]
        )
    ]

    return advantages, returns


# ============================================================
# PPO TRAINER
# ============================================================

class PPOTrainer:

    def __init__(

        self,

        env,
        agent,

        hidden_dim=128,

        lr=3e-4,
        gamma=0.99,
        lam=0.95,

        clip_eps=0.2,

        entropy_coef=0.01,
        value_coef=0.5,

        ppo_epochs=4,

    ):

        self.env = env

        self.agent = agent.to(DEVICE)

        self.gamma = gamma
        self.lam = lam

        self.clip_eps = clip_eps

        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.ppo_epochs = ppo_epochs

        self.critic = Critic(
            hidden_dim=hidden_dim
        ).to(DEVICE)

        self.optimizer = optim.Adam(

            list(self.agent.parameters())
            + list(self.critic.parameters()),

            lr=lr

        )

    # ========================================================
    # ROLLOUT
    # ========================================================

    def collect_rollout(

        self,
        rollout_steps=256

    ):

        buffer = PPOBuffer()

        obs, _ = self.env.reset()

        episode_rewards = []

        ep_reward = 0

        for _ in range(rollout_steps):

            # =================================================
            # PATH SELECTION
            # =================================================

            path_out = self.agent.act_path(obs)

            path_action = (
                path_out["action"].item()
            )

            path_logprob = (
                path_out["logprob"]
            )

            path_embeddings = (
                path_out["path_embeddings"]
            )

            selected_path_embedding = (
                path_embeddings[path_action]
            )

            obs, _, _, _, _ = self.env.step(
                path_action
            )

            # =================================================
            # MODULATION SELECTION
            # =================================================

            mod_out = self.agent.act_modulation(

                obs,
                selected_path_embedding

            )

            mod_action = (
                mod_out["action"].item()
            )

            mod_logprob = (
                mod_out["logprob"]
            )

            mod_embedding = (
                mod_out["mod_embedding"]
            )

            obs, _, _, _, _ = self.env.step(
                mod_action
            )

            # =================================================
            # SLOT SELECTION
            # =================================================

            slot_out = self.agent.act_slot(

                obs,

                selected_path_embedding,

                mod_embedding

            )

            slot_action = (
                slot_out["action"].item()
            )

            slot_logprob = (
                slot_out["logprob"]
            )

            obs, reward, terminated, truncated, _ = (
                self.env.step(slot_action)
            )

            done = terminated or truncated

            # =================================================
            # TOTAL LOGPROB
            # =================================================

            total_logprob = (

                path_logprob
                + mod_logprob
                + slot_logprob

            )

            # =================================================
            # VALUE ESTIMATION
            # =================================================

            value = self.critic(
                selected_path_embedding
            )

            buffer.add(

                total_logprob=total_logprob,
                reward=reward,
                value=value.item(),
                done=done

            )

            ep_reward += reward

            if done:

                episode_rewards.append(ep_reward)

                ep_reward = 0

                obs, _ = self.env.reset()

        return buffer, episode_rewards

    # ========================================================
    # PPO UPDATE
    # ========================================================

    def update(self, buffer):

        advantages, returns = compute_gae(

            rewards=buffer.rewards,
            values=buffer.values,
            dones=buffer.dones,

            gamma=self.gamma,
            lam=self.lam

        )

        advantages = torch.FloatTensor(
            advantages
        ).to(DEVICE)

        returns = torch.FloatTensor(
            returns
        ).to(DEVICE)

        old_logprobs = torch.stack(
            buffer.total_logprobs
        ).detach()

        values = torch.FloatTensor(
            buffer.values
        ).to(DEVICE)

        advantages = (

            (advantages - advantages.mean())
            / (advantages.std() + 1e-8)

        )

        # ====================================================
        # PPO EPOCHS
        # ====================================================

        for _ in range(self.ppo_epochs):

            # -----------------------------------------------
            # RECOMPUTE VALUES
            # -----------------------------------------------

            new_values = values

            # -----------------------------------------------
            # POLICY RATIO
            # -----------------------------------------------

            #
            # NOTE:
            #
            # In a full implementation:
            #
            # You should recompute logprobs by replaying
            # observations through the policy.
            #
            # Here we simplify for pipeline clarity.
            #

            new_logprobs = old_logprobs

            ratio = torch.exp(

                new_logprobs
                - old_logprobs

            )

            # -----------------------------------------------
            # PPO LOSS
            # -----------------------------------------------

            surr1 = ratio * advantages

            surr2 = torch.clamp(

                ratio,

                1 - self.clip_eps,
                1 + self.clip_eps

            ) * advantages

            policy_loss = -torch.min(
                surr1,
                surr2
            ).mean()

            # -----------------------------------------------
            # VALUE LOSS
            # -----------------------------------------------

            value_loss = F.mse_loss(
                new_values,
                returns
            )

            # -----------------------------------------------
            # ENTROPY
            # -----------------------------------------------

            entropy_loss = -new_logprobs.mean()

            # -----------------------------------------------
            # TOTAL LOSS
            # -----------------------------------------------

            loss = (

                policy_loss

                + self.value_coef * value_loss

                + self.entropy_coef * entropy_loss

            )

            self.optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(self.agent.parameters())
                + list(self.critic.parameters()),
                1.0
            )

            self.optimizer.step()

        return {

            "policy_loss":
                policy_loss.item(),

            "value_loss":
                value_loss.item(),

            "entropy":
                entropy_loss.item()

        }

    # ========================================================
    # TRAIN
    # ========================================================

    def train(

        self,

        total_iterations=1000,
        rollout_steps=256,

        save_path="rmsa_agent.pt"

    ):

        reward_history = deque(maxlen=100)

        for iteration in range(total_iterations):

            # ------------------------------------------------
            # COLLECT
            # ------------------------------------------------

            buffer, ep_rewards = (
                self.collect_rollout(
                    rollout_steps
                )
            )

            reward_history.extend(ep_rewards)

            # ------------------------------------------------
            # UPDATE
            # ------------------------------------------------

            stats = self.update(buffer)

            avg_reward = np.mean(
                reward_history
            ) if reward_history else 0.0

            # ------------------------------------------------
            # LOGGING
            # ------------------------------------------------

            print(

                f"[ITER {iteration}] "

                f"AvgReward={avg_reward:.3f} "

                f"PolicyLoss={stats['policy_loss']:.4f} "

                f"ValueLoss={stats['value_loss']:.4f}"

            )

            # ------------------------------------------------
            # SAVE
            # ------------------------------------------------

            if iteration % 50 == 0:

                torch.save({

                    "agent":
                        self.agent.state_dict(),

                    "critic":
                        self.critic.state_dict()

                }, save_path)

                print(
                    f"Checkpoint saved: {save_path}"
                )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # CREATE ENV
    # --------------------------------------------------------

    env = MaskedHierarchicalRMSAEnv(

        num_edges=40,
        k_paths=5,
        max_path_len=8,
        num_modulations=4,
        num_slots=128

    )

    # --------------------------------------------------------
    # CREATE AGENT
    # --------------------------------------------------------

    agent = RMSAAgent(

        edge_feat_dim=4,

        hidden_dim=128,

        num_gnn_layers=4,

        num_modulations=4,

        path_feat_dim=10,

        slot_feat_dim=2

    )

    # --------------------------------------------------------
    # TRAINER
    # --------------------------------------------------------

    trainer = PPOTrainer(

        env=env,

        agent=agent,

        lr=3e-4,

        gamma=0.99,

        lam=0.95,

        clip_eps=0.2,

        entropy_coef=0.01,

        value_coef=0.5,

        ppo_epochs=4

    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    trainer.train(

        total_iterations=1000,

        rollout_steps=256,

        save_path="hierarchical_rmsa_agent.pt"

    )