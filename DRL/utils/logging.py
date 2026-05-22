# utils/logging.py

"""
=============================================================
RMSA PPO LOGGING SYSTEM (IMPROVED)
=============================================================

Key upgrades:
✔ safer type handling
✔ structured RMSA logging
✔ distribution monitoring
✔ PPO stability diagnostics
✔ avoids silent metric corruption
=============================================================
"""

import os
import numpy as np
import logging

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
        use_tensorboard=True,
        smoothing=0.0
    ):

        self.use_tensorboard = use_tensorboard and _TENSORBOARD_AVAILABLE
        self.log_dir = log_dir
        self.smoothing = smoothing

        self.step = 0
        self.metrics_history = {}

        if self.use_tensorboard:

            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)

            print(f"[LOGGER] TensorBoard enabled: {log_dir}")

        else:
            self.writer = None
            print("[LOGGER] TensorBoard disabled")

        self.logger = logging.getLogger(name='main')

    def set_log_file(self, filename, folder_name):
        logging.basicConfig(
                    level=logging.NOTSET,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='w', force=True
                )

        format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler('%s/%s'%(folder_name, filename), 'w')
        file_handler.setFormatter(format)

        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        self.logger.addHandler(file_handler)

    def logger_close(self):
        handlers = self.logger.handlers[:]
        for handler in handlers:
            self.logger.removeHandler(handler)
            handler.close()


    # ========================================================
    # MAIN ENTRY
    # ========================================================

    def log(self, step, metrics: dict):
        self.step = step

        for k, v in metrics.items():
            self._log_scalar(k, v)

        self._print_summary(metrics)
        self.logger.info(f"[STEP {step}]")
        self.log_dict(metrics)

    # ========================================================
    # SAFE SCALAR LOGGING
    # ========================================================

    def _to_scalar(self, value):

        if value is None:
            return None

        if isinstance(value, (list, tuple, np.ndarray)):

            arr = np.array(value, dtype=np.float32)

            if arr.size == 0:
                return None

            return float(arr.mean())

        if np.isscalar(value):
            return float(value)

        return None

    def _log_scalar(self, key, value):

        value = self._to_scalar(value)

        if value is None:
            return

        # smoothing (optional EMA-like behavior)
        if self.smoothing > 0:

            prev = self.metrics_history.get(key, [value])[-1]

            value = self.smoothing * prev + (1 - self.smoothing) * value

        if self.writer is not None:
            self.writer.add_scalar(key, value, self.step)

        self.metrics_history.setdefault(key, []).append(value)

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    def _get(self, metrics, key, default=0.0):
        v = metrics.get(key, default)
        v = self._to_scalar(v)
        return default if v is None else v

    def _print_summary(self, metrics):
        s = f"[STEP {self.step}] "
        for k, v in metrics.items():
            value = self._to_scalar(v)
            s += f" | {k}={value:.4f}"
        print(s)

    def log_dict(self, metrics: dict):
        s = ""
        first = True
        for k, v in metrics.items():
            value = self._to_scalar(v)
            if first:
                s += f"{k}={value:.4f}"
                first = False
            else:
                s += f" | {k}={value:.4f}"
        print(s)
        self.logger.info(s)
    
    def log_str(self, s:str):
        print(s)
        self.logger.info(s)

    # ========================================================
    # RMSA METRICS
    # ========================================================

    def log_rmsa_metrics(
        self,
        path_success=None,
        mod_success=None,
        slot_success=None,
        spectrum_utilization=None,
        fragmentation=None,
        blocking_rate=None,
        traffic_load=None
    ):

        self._log_scalar("rmsa/path_success_rate", path_success)
        self._log_scalar("rmsa/mod_success_rate", mod_success)
        self._log_scalar("rmsa/slot_success_rate", slot_success)
        self._log_scalar("rmsa/spectrum_utilization", spectrum_utilization)
        self._log_scalar("rmsa/fragmentation", fragmentation)
        self._log_scalar("rmsa/blocking_rate", blocking_rate)
        self._log_scalar("rmsa/traffic_load", traffic_load)

    # ========================================================
    # PPO DIAGNOSTICS
    # ========================================================

    def log_ppo_diagnostics(
        self,
        kl=None,
        clip_fraction=None,
        grad_norm=None,
        entropy_path=None,
        entropy_mod=None,
        entropy_slot=None
    ):

        self._log_scalar("ppo/kl_divergence", kl)
        self._log_scalar("ppo/clip_fraction", clip_fraction)
        self._log_scalar("ppo/grad_norm", grad_norm)

        # important for RMSA collapse detection
        self._log_scalar("entropy/path", entropy_path)
        self._log_scalar("entropy/mod", entropy_mod)
        self._log_scalar("entropy/slot", entropy_slot)

    # ========================================================
    # DISTRIBUTION LOGGING (VERY IMPORTANT)
    # ========================================================

    def log_action_distribution(self, key, probs):

        """
        Detect policy collapse (very important for RMSA)
        """

        probs = self._to_scalar(probs)

        if probs is None:
            return

        self._log_scalar(f"dist/{key}_mean", np.mean(probs))
        self._log_scalar(f"dist/{key}_min", np.min(probs))
        self._log_scalar(f"dist/{key}_max", np.max(probs))

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):
        if self.writer is not None:
            self.writer.close()