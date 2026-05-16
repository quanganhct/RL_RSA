# inference/deploy_agent.py

"""
=============================================================
RMSA DEPLOYMENT AGENT
=============================================================

Production deployment wrapper for hierarchical RMSA PPO agent.

=============================================================
PURPOSE
=============================================================

This module provides:

    ✔ checkpoint loading
    ✔ device management
    ✔ deterministic inference
    ✔ real-time request handling
    ✔ batched inference support
    ✔ latency-safe no_grad execution

=============================================================
USAGE
=============================================================

agent = RMSADeploymentAgent(
    checkpoint_path="checkpoints/best.pt",
    device="cuda"
)

action = agent.act(obs)

=============================================================
"""

import time
import torch
import numpy as np

from models.hierarchical_policy import HierarchicalPolicy


# ============================================================
# DEPLOYMENT AGENT
# ============================================================

class RMSADeploymentAgent:

    def __init__(
        self,
        checkpoint_path,
        model_config,
        device="cpu",
        deterministic=True
    ):

        self.device = torch.device(device)

        self.deterministic = deterministic

        # ----------------------------------------------------
        # BUILD POLICY
        # ----------------------------------------------------

        self.policy = HierarchicalPolicy(
            **model_config
        ).to(self.device)

        # ----------------------------------------------------
        # LOAD CHECKPOINT
        # ----------------------------------------------------

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device
        )

        if "model_state_dict" in checkpoint:

            self.policy.load_state_dict(
                checkpoint["model_state_dict"]
            )

        else:

            self.policy.load_state_dict(
                checkpoint
            )

        self.policy.eval()

        print(
            f"[Deployment] Loaded checkpoint from:"
            f" {checkpoint_path}"
        )

    # ========================================================
    # SINGLE REQUEST INFERENCE
    # ========================================================

    @torch.no_grad()
    def act(self, obs):

        """
        =====================================================
        INPUT
        =====================================================

        obs:
            environment observation dict

        =====================================================
        OUTPUT
        =====================================================

        {
            "path": int,
            "modulation": int,
            "slot": int
        }
        """

        start = time.time()

        obs = self._to_device(obs)

        action_dict = self.policy.act(
            obs,
            deterministic=self.deterministic
        )

        latency_ms = (
            time.time() - start
        ) * 1000

        action_dict["latency_ms"] = latency_ms

        return action_dict

    # ========================================================
    # BATCH INFERENCE
    # ========================================================

    @torch.no_grad()
    def act_batch(self, batch_obs):

        """
        =====================================================
        INPUT
        =====================================================

        batch_obs:
            batched observation dict

        =====================================================
        OUTPUT
        =====================================================

        {
            "path": [B]
            "modulation": [B]
            "slot": [B]
        }
        """

        start = time.time()

        batch_obs = self._to_device(batch_obs)

        actions = self.policy.act(
            batch_obs,
            deterministic=self.deterministic
        )

        latency_ms = (
            time.time() - start
        ) * 1000

        actions["latency_ms"] = latency_ms

        return actions

    # ========================================================
    # GREEDY DEPLOYMENT LOOP
    # ========================================================

    @torch.no_grad()
    def run_live(self, env):

        """
        =====================================================
        Runs continuously on streaming RMSA requests.
        =====================================================
        """

        obs = env.reset()

        done = False

        total_reward = 0

        while not done:

            action = self.act(obs)

            obs, reward, done, info = env.step(
                action
            )

            total_reward += reward

            print(
                f"[Deploy] "
                f"path={action['path']} | "
                f"mod={action['modulation']} | "
                f"slot={action['slot']} | "
                f"reward={reward:.2f} | "
                f"latency={action['latency_ms']:.2f}ms"
            )

        return total_reward

    # ========================================================
    # DEVICE UTILS
    # ========================================================

    def _to_device(self, obj):

        if torch.is_tensor(obj):

            return obj.to(self.device)

        elif isinstance(obj, dict):

            return {
                k: self._to_device(v)
                for k, v in obj.items()
            }

        elif isinstance(obj, list):

            return [
                self._to_device(v)
                for v in obj
            ]

        else:

            return obj

    # ========================================================
    # EXPORT TORCHSCRIPT (OPTIONAL)
    # ========================================================

    def export_torchscript(
        self,
        save_path="policy_ts.pt"
    ):

        """
        Optional deployment optimization.
        """

        scripted = torch.jit.script(
            self.policy
        )

        scripted.save(save_path)

        print(
            f"[Deployment] TorchScript saved:"
            f" {save_path}"
        )

    # ========================================================
    # BENCHMARK
    # ========================================================

    @torch.no_grad()
    def benchmark(
        self,
        sample_obs,
        num_runs=100
    ):

        """
        Measure inference latency.
        """

        sample_obs = self._to_device(sample_obs)

        latencies = []

        for _ in range(num_runs):

            start = time.time()

            _ = self.policy.act(
                sample_obs,
                deterministic=True
            )

            latency_ms = (
                time.time() - start
            ) * 1000

            latencies.append(latency_ms)

        print(
            f"[Benchmark] "
            f"Mean latency: {np.mean(latencies):.3f} ms | "
            f"P95: {np.percentile(latencies,95):.3f} ms | "
            f"P99: {np.percentile(latencies,99):.3f} ms"
        )

        return {

            "mean_ms": np.mean(latencies),
            "p95_ms": np.percentile(latencies,95),
            "p99_ms": np.percentile(latencies,99)
        }