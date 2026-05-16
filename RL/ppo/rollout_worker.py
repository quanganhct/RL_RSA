import torch


class RolloutWorker:
    """
    ============================================================
    HIERARCHICAL RMSA ROLLOUT WORKER (WITH STAGE IDS)
    ============================================================

    Adds:

        stage_ids:
            0 → path
            1 → modulation
            2 → slot

    This is REQUIRED for hierarchical GAE + loss separation.
    """

    def __init__(self, env, policy, device="cpu"):

        self.env = env
        self.policy = policy
        self.device = device

    # ============================================================
    # SINGLE EPISODE
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

            "stage_ids": [],   # 🔥 IMPORTANT FIX

            "cache": []
        }

        done = False
        

        while not done:

            # =====================================================
            # STAGE 0: PATH SELECTION
            # =====================================================
            print(f"edge_features shape = {obs['edge_features'].shape}")
            print(f"edge_index shape = {obs['edge_index'].shape}")
            print(f"candidate_paths shape = {obs['candidate_paths'].shape}")

            path_action, path_logprob, cache = self.policy.act_path(obs)
            print(f"path_action  = {path_action}")

            obs, _ = self.env.step_path(path_action)
            print(f"path_features shape = {obs['path_features'].shape}")
 

            trajectory["stage_ids"].append(0)

            # =====================================================
            # STAGE 1: MODULATION SELECTION
            # =====================================================

            mod_action, mod_logprob, mod_emb = self.policy.act_modulation(
                obs,
                cache
            )

            cache["selected_mod_emb"] = mod_emb

            obs, _ = self.env.step_modulation(mod_action)

            trajectory["stage_ids"].append(1)

            # =====================================================
            # STAGE 2: SLOT SELECTION (REWARD)
            # =====================================================

            slot_action, slot_logprob = self.policy.act_slot(
                obs,
                cache
            )

            obs, reward, done, _ = self.env.step_slot(slot_action)

            trajectory["stage_ids"].append(2)

            # =====================================================
            # STORE TRANSITION DATA
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
    # STACK BATCH
    # ============================================================

    def _stack_batch(self, batch):

        def flatten(key):

            out = []

            for ep in batch:
                out.extend(ep[key])

            return torch.tensor(out) if isinstance(out[0], (int, float)) else out

        return {

            "obs": flatten("obs"),

            "path_actions": torch.tensor(flatten("path_actions")),
            "mod_actions": torch.tensor(flatten("mod_actions")),
            "slot_actions": torch.tensor(flatten("slot_actions")),

            "path_logprobs": torch.tensor(flatten("path_logprobs")),
            "mod_logprobs": torch.tensor(flatten("mod_logprobs")),
            "slot_logprobs": torch.tensor(flatten("slot_logprobs")),

            "rewards": torch.tensor(flatten("rewards")),
            "dones": torch.tensor(flatten("dones")),

            "stage_ids": torch.tensor(flatten("stage_ids")),  # 🔥 FIXED

            "episodes": batch
        }