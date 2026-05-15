# ppo/trainer.py

"""
=============================================================
PRODUCTION PPO TRAINER FOR RMSA
=============================================================

This trainer orchestrates:

1. Rollout collection (hierarchical RMSA)
2. Advantage computation (GAE)
3. PPO minibatch optimization
4. Gradient clipping
5. Logging + diagnostics
6. Checkpointing
7. Multi-epoch PPO updates

=============================================================
KEY DESIGN PRINCIPLE
=============================================================

We assume:

    rollout_worker → buffer → gae → losses → update

Everything is fully recomputed each PPO epoch.

=============================================================

env
  ↓
rollout_worker
  ↓
buffer
  ↓
trainer
  ↓
forward_ppo()
"""

import torch
import numpy as np

from ppo.rollout_worker import RolloutWorker
from ppo.gae import compute_advantages_rmsa
from ppo.losses import compute_ppo_loss

from utils.checkpoint import save_checkpoint
from utils.logging import Logger


# ============================================================
# PPO TRAINER
# ============================================================

class PPOTrainer:

    def __init__(
        self,
        env,
        policy,
        optimizer,
        device="cpu",

        # PPO hyperparams
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,

        # training config
        rollout_episodes=8,
        ppo_epochs=4,
        minibatch_size=32,

        max_grad_norm=0.5
    ):

        self.env = env
        self.policy = policy.to(device)
        self.optimizer = optimizer

        self.device = device

        self.gamma = gamma
        self.lam = lam

        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.rollout_episodes = rollout_episodes
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size

        self.max_grad_norm = max_grad_norm

        self.worker = RolloutWorker(env, policy, device)

        self.logger = Logger()

    # ========================================================
    # COLLECT EXPERIENCE
    # ========================================================

    def collect(self):

        buffer = self.worker.run_batch(
            num_episodes=self.rollout_episodes
        )

        data = buffer.get_flat()

        advantages, returns = compute_advantages_rmsa(

            rewards=data["rewards"],
            values=data["values"],
            dones=data["dones"]

        )

        data["advantages"] = advantages
        data["returns"] = returns

        return data

    # ========================================================
    # TRAIN STEP
    # ========================================================

    def update(self, data):

        n = len(data["rewards"])

        indices = np.arange(n)

        metrics = {}

        for epoch in range(self.ppo_epochs):

            np.random.shuffle(indices)

            for start in range(0, n, self.minibatch_size):

                end = start + self.minibatch_size
                mb_idx = indices[start:end]

                batch = self._make_batch(data, mb_idx)

                self.optimizer.zero_grad()

                outputs = self.policy.forward_ppo(batch)

                loss_dict = compute_ppo_loss(

                    model_outputs=outputs,
                    batch=batch,
                    advantages=batch["advantages"],
                    returns=batch["returns"],

                    clip_eps=self.clip_eps,
                    value_coef=self.value_coef,
                    entropy_coef=self.entropy_coef
                )

                loss = loss_dict["loss"]

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.max_grad_norm
                )

                self.optimizer.step()

                # logging
                for k, v in loss_dict.items():
                    metrics[k] = metrics.get(k, 0) + v.item()

        for k in metrics:
            metrics[k] /= (self.ppo_epochs * (n // self.minibatch_size + 1))

        return metrics

    # ========================================================
    # MINIBATCH CREATION
    # ========================================================

    def _make_batch(self, data, idx):

        def select(x):
            return [x[i] for i in idx]

        return {

            # observations
            "observations": select(data["observations"]),

            # actions
            "path_actions": torch.tensor(
                select(data["path_actions"])
            ),

            "mod_actions": torch.tensor(
                select(data["mod_actions"])
            ),

            "slot_actions": torch.tensor(
                select(data["slot_actions"])
            ),

            # masks
            "path_masks": torch.tensor(
                select(data["path_masks"])
            ),

            "mod_masks": torch.tensor(
                select(data["mod_masks"])
            ),

            "slot_masks": torch.tensor(
                select(data["slot_masks"])
            ),

            # logprobs
            "old_path_logprobs": torch.tensor(
                select(data["old_path_logprobs"])
            ),

            "old_mod_logprobs": torch.tensor(
                select(data["old_mod_logprobs"])
            ),

            "old_slot_logprobs": torch.tensor(
                select(data["old_slot_logprobs"])
            ),

            # value
            "values": torch.tensor(
                select(data["values"])
            ),

            # advantages / returns
            "advantages": torch.tensor(
                select(data["advantages"])
            ),

            "returns": torch.tensor(
                select(data["returns"])
            )
        }

    # ========================================================
    # TRAIN LOOP
    # ========================================================

    def train(
        self,
        total_iters=1000,
        log_interval=10,
        save_path="rmsa_ppo.pt"
    ):

        for it in range(total_iters):

            data = self.collect()

            metrics = self.update(data)

            self.logger.log(it, metrics)

            if it % log_interval == 0:

                print(
                    f"[ITER {it}] "
                    f"loss={metrics['loss']:.4f} "
                    f"value={metrics['value_loss']:.4f} "
                    f"entropy={metrics['entropy']:.4f}"
                )

            if it % 50 == 0:

                save_checkpoint(

                    path=save_path,
                    model=self.policy,
                    optimizer=self.optimizer,
                    iteration=it

                )