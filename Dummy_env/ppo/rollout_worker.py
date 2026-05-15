# ppo/rollout_worker.py

"""
=============================================================
ROLLOUT WORKER FOR HIERARCHICAL RMSA PPO
=============================================================

This module is responsible for:

1. Interacting with REAL RMSA simulator
2. Executing hierarchical policy:
       path -> modulation -> slot
3. Collecting full trajectories
4. Storing PPO-compatible transitions
5. Handling action masking at each stage
6. Producing rollout buffers for PPO update

=============================================================
KEY DESIGN REQUIREMENT
=============================================================

The environment is STEP-DEPENDENT:

    step(path)
    step(modulation)
    step(slot) -> reward

So rollout worker MUST preserve state across stages.

=============================================================
"""

import numpy as np
import torch

from ppo.buffer import RMSATransition, RMSAPPORolloutBuffer


# ============================================================
# ROLLOUT WORKER
# ============================================================

class RolloutWorker:

    def __init__(
        self,
        env,
        policy,
        device="cpu"
    ):

        self.env = env
        self.policy = policy
        self.device = device

        self.buffer = RMSAPPORolloutBuffer()

    # ========================================================
    # SINGLE EPISODE ROLLOUT
    # ========================================================

    def run_episode(self):

        """
        Executes ONE RMSA request episode.
        """

        obs = self.env.reset()

        done = False

        while not done:

            # ------------------------------------------------
            # STAGE 1: PATH SELECTION
            # ------------------------------------------------

            path_out = self.policy.act_path(obs)

            path_action = int(path_out["action"])

            path_logprob = path_out["logprob"].item()

            path_embedding = path_out["path_embedding"]

            path_mask = obs["action_masks"]["path"]

            # environment transitions after path
            obs_after_path, _ = self.env.step_path(path_action)

            # ------------------------------------------------
            # STAGE 2: MODULATION SELECTION
            # ------------------------------------------------

            mod_out = self.policy.act_modulation(
                obs_after_path,
                path_embedding
            )

            mod_action = int(mod_out["action"])

            mod_logprob = mod_out["logprob"].item()

            mod_embedding = mod_out["mod_embedding"]

            mod_mask = obs_after_path["action_masks"]["mod"]

            obs_after_mod, _ = self.env.step_modulation(
                mod_action
            )

            # ------------------------------------------------
            # STAGE 3: SLOT SELECTION
            # ------------------------------------------------

            slot_out = self.policy.act_slot(
                obs_after_mod,
                path_embedding,
                mod_embedding
            )

            slot_action = int(slot_out["action"])

            slot_logprob = slot_out["logprob"].item()

            slot_mask = obs_after_mod["action_masks"]["slot"]

            # FINAL STEP RETURNS REWARD
            obs_after_slot, reward, done, info = self.env.step_slot(
                slot_action
            )

            # ------------------------------------------------
            # VALUE ESTIMATE (optional centralized critic)
            # ------------------------------------------------

            value = self.policy.critic(
                path_embedding
            ).item()

            # ------------------------------------------------
            # STORE TRANSITION
            # ------------------------------------------------

            transition = RMSATransition(

                # observations (store ONLY path-level root obs)
                obs_path=obs,
                obs_mod=obs_after_path,
                obs_slot=obs_after_mod,

                # actions
                path_action=path_action,
                mod_action=mod_action,
                slot_action=slot_action,

                # masks
                path_mask=path_mask,
                mod_mask=mod_mask,
                slot_mask=slot_mask,

                # logprobs
                path_logprob=path_logprob,
                mod_logprob=mod_logprob,
                slot_logprob=slot_logprob,

                # value
                value=value,

                # reward
                reward=reward,
                done=done
            )

            self.buffer.add(transition)

            # ------------------------------------------------
            # UPDATE STATE
            # ------------------------------------------------

            obs = obs_after_slot

        return self.buffer


    # ========================================================
    # MULTI-EPISODE ROLLOUT
    # ========================================================

    def run_batch(
        self,
        num_episodes=8
    ):

        """
        Collect multiple RMSA episodes.
        """

        self.buffer.reset()

        for _ in range(num_episodes):

            self.run_episode()

        self.buffer.finalize()

        return self.buffer


# ============================================================
# VECTORISED VERSION (OPTIONAL PRODUCTION SPEEDUP)
# ============================================================

class VectorizedRolloutWorker:

    """
    Runs multiple RMSA environments in parallel.

    Used for real training scale (recommended 8–64 envs).
    """

    def __init__(
        self,
        envs,
        policy,
        device="cpu"
    ):

        self.envs = envs
        self.policy = policy
        self.device = device

        self.num_envs = len(envs)

        self.buffers = [
            RMSAPPORolloutBuffer()
            for _ in range(self.num_envs)
        ]

    def run(self):

        obs = [env.reset() for env in self.envs]

        dones = [False] * self.num_envs

        while not all(dones):

            for i, env in enumerate(self.envs):

                if dones[i]:
                    continue

                # -----------------------------
                # PATH
                # -----------------------------
                path_out = self.policy.act_path(obs[i])

                a1 = int(path_out["action"])
                lp1 = path_out["logprob"].item()
                emb = path_out["path_embedding"]

                obs1, _ = env.step_path(a1)

                # -----------------------------
                # MOD
                # -----------------------------
                mod_out = self.policy.act_modulation(
                    obs1,
                    emb
                )

                a2 = int(mod_out["action"])
                lp2 = mod_out["logprob"].item()
                emb2 = mod_out["mod_embedding"]

                obs2, _ = env.step_modulation(a2)

                # -----------------------------
                # SLOT
                # -----------------------------
                slot_out = self.policy.act_slot(
                    obs2,
                    emb,
                    emb2
                )

                a3 = int(slot_out["action"])
                lp3 = slot_out["logprob"].item()

                obs3, reward, done, _ = env.step_slot(a3)

                value = self.policy.critic(emb).item()

                transition = RMSATransition(

                    obs_path=obs[i],
                    obs_mod=obs1,
                    obs_slot=obs2,

                    path_action=a1,
                    mod_action=a2,
                    slot_action=a3,

                    path_mask=obs[i]["action_masks"]["path"],
                    mod_mask=obs1["action_masks"]["mod"],
                    slot_mask=obs2["action_masks"]["slot"],

                    path_logprob=lp1,
                    mod_logprob=lp2,
                    slot_logprob=lp3,

                    value=value,

                    reward=reward,
                    done=done
                )

                self.buffers[i].add(transition)

                obs[i] = obs3
                dones[i] = done

        # finalize all buffers
        for b in self.buffers:
            b.finalize()

        return self.buffers