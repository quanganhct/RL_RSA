# inference/evaluator.py

"""
=============================================================
RMSA POLICY EVALUATOR
=============================================================

Production-grade evaluator for hierarchical RMSA PPO agents.

=============================================================
PURPOSE
=============================================================

Runs deterministic or stochastic evaluation on:

    - trained PPO agent
    - real RMSA simulator
    - dummy environment

Collects:

    - reward
    - blocking probability
    - episode statistics
    - path/modulation/slot distributions

=============================================================
USAGE
=============================================================

evaluator = Evaluator(
    env,
    policy,
    device="cuda"
)

stats = evaluator.evaluate(
    num_episodes=100
)

=============================================================
"""

import time
import torch
import numpy as np
from collections import defaultdict


# ============================================================
# EVALUATOR
# ============================================================

class Evaluator:

    def __init__(
        self,
        env,
        policy,
        device="cpu"
    ):

        self.env = env
        self.policy = policy
        self.device = device

    # ========================================================
    # EVALUATE
    # ========================================================

    @torch.no_grad()
    def evaluate(
        self,
        num_episodes=10,
        deterministic=True,
        render=False
    ):

        self.policy.eval()

        # ----------------------------------------------------
        # GLOBAL STATS
        # ----------------------------------------------------

        stats = defaultdict(list)

        total_requests = 0
        total_accepted = 0
        total_blocked = 0

        # ----------------------------------------------------
        # EPISODE LOOP
        # ----------------------------------------------------

        for ep in range(num_episodes):

            obs = self.env.reset()

            done = False

            ep_reward = 0
            ep_steps = 0

            accepted = 0
            blocked = 0

            start_time = time.time()

            # ------------------------------------------------
            # ENV LOOP
            # ------------------------------------------------

            while not done:

                # --------------------------------------------
                # POLICY ACTION
                # --------------------------------------------

                action_dict = self.policy.act(
                    obs,
                    deterministic=deterministic
                )

                # --------------------------------------------
                # ENV STEP
                # --------------------------------------------

                next_obs, reward, done, info = self.env.step(
                    action_dict
                )

                obs = next_obs

                ep_reward += reward
                ep_steps += 1

                # --------------------------------------------
                # REQUEST STATS
                # --------------------------------------------

                accepted += int(info.get("accepted", 0))
                blocked += int(info.get("blocked", 0))

                # --------------------------------------------
                # ACTION STATS
                # --------------------------------------------

                stats["path_actions"].append(
                    int(action_dict["path"])
                )

                stats["mod_actions"].append(
                    int(action_dict["modulation"])
                )

                stats["slot_actions"].append(
                    int(action_dict["slot"])
                )

                # --------------------------------------------
                # OPTIONAL RENDER
                # --------------------------------------------

                if render:

                    self._render_step(
                        ep,
                        ep_steps,
                        action_dict,
                        reward,
                        info
                    )

            # ------------------------------------------------
            # EPISODE METRICS
            # ------------------------------------------------

            episode_time = time.time() - start_time

            blocking_prob = blocked / max(
                accepted + blocked,
                1
            )

            stats["episode_reward"].append(ep_reward)

            stats["episode_length"].append(ep_steps)

            stats["blocking_probability"].append(
                blocking_prob
            )

            stats["accepted_requests"].append(
                accepted
            )

            stats["blocked_requests"].append(
                blocked
            )

            stats["episode_time"].append(
                episode_time
            )

            total_requests += accepted + blocked
            total_accepted += accepted
            total_blocked += blocked

            print(
                f"[Eval] "
                f"Episode={ep+1}/{num_episodes} | "
                f"Reward={ep_reward:.2f} | "
                f"Blocking={blocking_prob:.4f} | "
                f"Steps={ep_steps}"
            )

        # ----------------------------------------------------
        # GLOBAL SUMMARY
        # ----------------------------------------------------

        final_stats = {

            "mean_reward":
                np.mean(stats["episode_reward"]),

            "std_reward":
                np.std(stats["episode_reward"]),

            "mean_episode_length":
                np.mean(stats["episode_length"]),

            "mean_blocking_probability":
                np.mean(stats["blocking_probability"]),

            "global_blocking_probability":
                total_blocked / max(total_requests, 1),

            "acceptance_ratio":
                total_accepted / max(total_requests, 1),

            "mean_episode_time":
                np.mean(stats["episode_time"]),

            "num_episodes":
                num_episodes,

            "total_requests":
                total_requests,

            "total_accepted":
                total_accepted,

            "total_blocked":
                total_blocked
        }

        # ----------------------------------------------------
        # ACTION HISTOGRAMS
        # ----------------------------------------------------

        final_stats["path_action_histogram"] = \
            self._histogram(stats["path_actions"])

        final_stats["mod_action_histogram"] = \
            self._histogram(stats["mod_actions"])

        final_stats["slot_action_histogram"] = \
            self._histogram(stats["slot_actions"])

        return final_stats

    # ========================================================
    # HISTOGRAM
    # ========================================================

    def _histogram(self, values):

        hist = defaultdict(int)

        for v in values:

            hist[int(v)] += 1

        return dict(hist)

    # ========================================================
    # RENDER
    # ========================================================

    def _render_step(
        self,
        episode,
        step,
        action_dict,
        reward,
        info
    ):

        print(
            f"[Episode {episode}] "
            f"Step={step} | "
            f"Path={action_dict['path']} | "
            f"Mod={action_dict['modulation']} | "
            f"Slot={action_dict['slot']} | "
            f"Reward={reward:.2f}"
        )