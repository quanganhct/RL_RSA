# utils/logging.py

"""
=============================================================
RMSA PPO LOGGING SYSTEM (TENSORBOARD ENABLED)
=============================================================

Tracks:

1. PPO optimization metrics
2. RMSA-specific performance metrics
3. Hierarchical success rates
4. Training stability signals

=============================================================
WHY THIS MATTERS
=============================================================

RMSA failures often look like:

- high reward but low spectrum efficiency
- path policy collapse
- modulation degeneracy
- slot selection saturation

Without structured logging, you won't see it.

=============================================================
"""

import os
import time
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    _TENSORBOARD_AVAILABLE = True
except Exception:
    _TENSORBOARD_AVAILABLE = False


# ============================================================
# LOGGER
# ============================================================

class Logger:

    def __init__(
        self,
        log_dir="runs/rmsa",
        use_tensorboard=True
    ):

        self.use_tensorboard = (
            use_tensorboard and _TENSORBOARD_AVAILABLE
        )

        self.log_dir = log_dir

        self.step = 0

        self.metrics_history = {}

        if self.use_tensorboard:

            os.makedirs(log_dir, exist_ok=True)

            self.writer = SummaryWriter(
                log_dir=log_dir
            )

            print(
                f"[LOGGER] TensorBoard enabled at {log_dir}"
            )

        else:

            self.writer = None

            print(
                "[LOGGER] TensorBoard not available, "
                "falling back to stdout logging"
            )

    # ========================================================
    # MAIN LOG FUNCTION
    # ========================================================

    def log(self, step, metrics: dict):

        self.step = step

        for k, v in metrics.items():

            self._log_scalar(k, v)

        self._print_summary(metrics)

    # ========================================================
    # SCALAR LOGGING
    # ========================================================

    def _log_scalar(self, key, value):

        if isinstance(value, (list, np.ndarray)):

            value = float(np.mean(value))

        if self.writer is not None:

            self.writer.add_scalar(
                key,
                value,
                self.step
            )

        # store history
        if key not in self.metrics_history:

            self.metrics_history[key] = []

        self.metrics_history[key].append(value)

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    def _print_summary(self, metrics):

        def get(k, default=0.0):

            v = metrics.get(k, default)

            if isinstance(v, (list, np.ndarray)):
                return float(np.mean(v))

            return float(v)

        # core PPO metrics
        loss = get("loss")
        policy_loss = get("policy_loss")
        value_loss = get("value_loss")
        entropy = get("entropy")
        ratio = get("ratio", 1.0)

        print(
            f"[STEP {self.step}] "
            f"loss={loss:.4f} | "
            f"policy={policy_loss:.4f} | "
            f"value={value_loss:.4f} | "
            f"entropy={entropy:.4f} | "
            f"ratio={ratio:.3f}"
        )

    # ========================================================
    # RMSA-SPECIFIC METRICS
    # ========================================================

    def log_rmsa_metrics(
        self,
        path_success=None,
        mod_success=None,
        slot_success=None,
        spectrum_utilization=None,
        fragmentation=None,
        blocking_rate=None
    ):
        """
        Optional RMSA-specific diagnostics.
        """

        if path_success is not None:
            self._log_scalar(
                "rmsa/path_success_rate",
                path_success
            )

        if mod_success is not None:
            self._log_scalar(
                "rmsa/mod_success_rate",
                mod_success
            )

        if slot_success is not None:
            self._log_scalar(
                "rmsa/slot_success_rate",
                slot_success
            )

        if spectrum_utilization is not None:
            self._log_scalar(
                "rmsa/spectrum_utilization",
                spectrum_utilization
            )

        if fragmentation is not None:
            self._log_scalar(
                "rmsa/fragmentation",
                fragmentation
            )

        if blocking_rate is not None:
            self._log_scalar(
                "rmsa/blocking_rate",
                blocking_rate
            )

    # ========================================================
    # PPO DIAGNOSTICS
    # ========================================================

    def log_ppo_diagnostics(
        self,
        kl=None,
        clip_fraction=None,
        grad_norm=None
    ):

        if kl is not None:
            self._log_scalar("ppo/kl_divergence", kl)

        if clip_fraction is not None:
            self._log_scalar(
                "ppo/clip_fraction",
                clip_fraction
            )

        if grad_norm is not None:
            self._log_scalar(
                "ppo/grad_norm",
                grad_norm
            )

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        if self.writer is not None:
            self.writer.close()