import torch


class RolloutWorker:
    """
    ============================================================
    HIERARCHICAL RMSA ROLLOUT WORKER (PPO-CORRECT)
    ============================================================

    Executes:

        path → env.step_path()
        mod  → env.step_modulation()
        slot → env.step_slot()

    Stores full trajectory for PPO recomputation.
    """

    def __init__(self, env, policy, device="cpu"):

        self.env = env
        self.policy = policy
        self.device = device

    # ============================================================
    # SINGLE EPISODE ROLLOUT
    # ============================================================

    def collect_episode(self):

        obs = self.env.reset()

        trajectory = {

            "obs": [],
            "path_actions": [],
            "mod_actions": [],
            "slot_actions": [],

            "path_logprobs": [],
            "mod_logprobs": [],
            "slot_logprobs": [],

            "rewards": [],
            "dones": [],

            "cache": []
        }

        done = False

        while not done:

            # =====================================================
            # 1. PATH SELECTION
            # =====================================================

            path_action, path_logprob, cache = self.policy.act_path(obs)

            obs, _ = self.env.step_path(path_action)

            # =====================================================
            # 2. MODULATION SELECTION
            # =====================================================

            mod_action, mod_logprob, mod_emb = self.policy.act_modulation(
                obs,
                cache
            )

            cache["mod_embedding"] = mod_emb

            obs, _ = self.env.step_modulation(mod_action)

            # =====================================================
            # 3. SPECTRUM SELECTION
            # =====================================================

            slot_action, slot_logprob = self.policy.act_slot(
                obs,
                cache
            )

            obs, reward, done, _ = self.env.step_slot(slot_action)

            # =====================================================
            # STORE TRANSITION
            # =====================================================

            trajectory["obs"].append(obs)

            trajectory["path_actions"].append(path_action)
            trajectory["mod_actions"].append(mod_action)
            trajectory["slot_actions"].append(slot_action)

            trajectory["path_logprobs"].append(path_logprob)
            trajectory["mod_logprobs"].append(mod_logprob)
            trajectory["slot_logprobs"].append(slot_logprob)

            trajectory["rewards"].append(reward)
            trajectory["dones"].append(done)

            trajectory["cache"].append(cache)

        return trajectory

    # ============================================================
    # MULTI-EPISODE COLLECTION
    # ============================================================

    def collect_batch(self, num_episodes=8):

        batch = []

        for _ in range(num_episodes):

            episode = self.collect_episode()

            batch.append(episode)

        return self._stack_batch(batch)

    # ============================================================
    # STACK BATCHES FOR PPO
    # ============================================================

    def _stack_batch(self, batch):

        def stack(key):

            return [item for ep in batch for item in ep[key]]

        return {

            "obs": stack("obs"),

            "path_actions": torch.tensor(stack("path_actions")),
            "mod_actions": torch.tensor(stack("mod_actions")),
            "slot_actions": torch.tensor(stack("slot_actions")),

            "path_logprobs": torch.tensor(stack("path_logprobs")),
            "mod_logprobs": torch.tensor(stack("mod_logprobs")),
            "slot_logprobs": torch.tensor(stack("slot_logprobs")),

            "rewards": torch.tensor(stack("rewards")),
            "dones": torch.tensor(stack("dones")),

            "cache": batch  # keep episode structure for debugging
        }