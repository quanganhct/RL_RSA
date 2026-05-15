import torch
import torch.optim as optim

from ppo.losses_good import compute_hierarchical_loss
from ppo.gae_good import compute_hierarchical_gae


class PPOTrainer:

    def __init__(
        self,
        policy,
        logger=None,              # ✅ ADD LOGGER
        lr=3e-4,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        device="cpu"
    ):

        self.policy = policy
        self.device = device
        self.logger = logger      # ✅ STORE LOGGER

        self.optimizer = optim.Adam(policy.parameters(), lr=lr)

        self.global_step = 0     # ✅ IMPORTANT FOR TB

        self.config = {
            "clip_eps": clip_eps,
            "value_coef": value_coef,
            "entropy_coef": entropy_coef
        }

    # ============================================================
    # TRAIN STEP
    # ============================================================

    def train_step(self, batch):

        self.global_step += 1

        # --------------------------------------------------------
        # MOVE TO DEVICE
        # --------------------------------------------------------
        batch = self._to_device(batch)

        # --------------------------------------------------------
        # VALUE ESTIMATION (OLD POLICY)
        # --------------------------------------------------------
        with torch.no_grad():
            outputs = self.policy.forward_ppo(batch)
            values = outputs["value"].squeeze()

        # --------------------------------------------------------
        # GAE
        # --------------------------------------------------------
        advantages, returns = compute_hierarchical_gae(
            rewards=batch["rewards"],
            values=values,
            dones=batch["dones"],
            stage_ids=batch["stage_ids"]
        )

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --------------------------------------------------------
        # FORWARD PASS (NEW POLICY)
        # --------------------------------------------------------
        outputs = self.policy.forward_ppo(batch)

        # --------------------------------------------------------
        # LOSS
        # --------------------------------------------------------
        loss, stats = compute_hierarchical_loss(
            outputs=outputs,
            batch=batch,
            advantages={
                "path": advantages,
                "mod": advantages,
                "slot": advantages
            },
            returns=returns,
            config=self.config
        )

        # ========================================================
        # 🔥 LOGGING (MAIN ADDITION)
        # ========================================================

        if self.logger is not None:

            # 1. core PPO stats
            self.logger.log(self.global_step, stats)

            # 2. PPO diagnostics (if present in stats)
            self.logger.log_ppo_diagnostics(
                kl=stats.get("kl"),
                clip_fraction=stats.get("clip_fraction"),
                grad_norm=stats.get("grad_norm"),
                entropy_path=stats.get("entropy_path"),
                entropy_mod=stats.get("entropy_mod"),
                entropy_slot=stats.get("entropy_slot"),
            )

            # 3. RMSA metrics (optional if provided)
            if "rmsa" in stats:
                self.logger.log_rmsa_metrics(**stats["rmsa"])

            # 4. advantage diagnostics (VERY IMPORTANT for debugging collapse)
            self.logger.log_rmsa_metrics(
                path_success=None,
                mod_success=None,
                slot_success=None,
                spectrum_utilization=None,
                fragmentation=None,
                blocking_rate=None
            )

            # 5. quick scalar debug signals
            self.logger.log_action_distribution(
                "advantage_mean",
                advantages.mean().item()
            )

        # --------------------------------------------------------
        # BACKPROP
        # --------------------------------------------------------
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            max_norm=0.5
        )

        self.optimizer.step()

        return stats

    # ============================================================
    # DEVICE HANDLING
    # ============================================================

    def _to_device(self, batch):

        def recursive_move(obj):

            if torch.is_tensor(obj):
                return obj.to(self.device)

            if isinstance(obj, dict):
                return {k: recursive_move(v) for k, v in obj.items()}

            if isinstance(obj, list):
                return [recursive_move(v) for v in obj]

            return obj

        return recursive_move(batch)