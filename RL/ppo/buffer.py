import torch


class HierarchicalReplayBuffer:
    """
    ============================================================
    HIERARCHICAL PPO BUFFER FOR RMSA
    ============================================================

    Stores trajectories from:

        path → modulation → slot

    Key property:
        DOES NOT flatten hierarchy incorrectly.

    Maintains stage-aligned structure for PPO recomputation.

    ============================================================
    """

    def __init__(self):

        self.reset()

    # ============================================================
    # RESET BUFFER
    # ============================================================

    def reset(self):

        self.episodes = []

    # ============================================================
    # ADD EPISODE
    # ============================================================

    def add_episode(self, episode):

        """
        episode comes from rollout_worker:

        {
            obs,
            path_actions,
            mod_actions,
            slot_actions,
            path_logprobs,
            mod_logprobs,
            slot_logprobs,
            rewards,
            dones,
            cache
        }
        """

        self.episodes.append(episode)

    # ============================================================
    # FLATTEN FOR PPO TRAINING
    # ============================================================

    def get_batch(self):

        """
        Converts hierarchical episodes into PPO batch.

        IMPORTANT:
            Keeps stage alignment intact.
        """

        batch = {

            # -------------------------
            # OBSERVATIONS
            # -------------------------
            "obs": self._flatten("obs"),

            # -------------------------
            # ACTIONS (STAGED)
            # -------------------------
            "path_actions": self._flatten("path_actions"),
            "mod_actions": self._flatten("mod_actions"),
            "slot_actions": self._flatten("slot_actions"),

            # -------------------------
            # LOGPROBS (STAGED)
            # -------------------------
            "path_logprobs": self._flatten("path_logprobs"),
            "mod_logprobs": self._flatten("mod_logprobs"),
            "slot_logprobs": self._flatten("slot_logprobs"),

            # -------------------------
            # REWARDS / DONE
            # -------------------------
            "rewards": self._flatten("rewards"),
            "dones": self._flatten("dones"),

            # -------------------------
            # EPISODE BOUNDARIES
            # -------------------------
            "episode_lengths": [len(ep["rewards"]) for ep in self.episodes],

            # -------------------------
            # RAW EPISODES (debug / diagnostics)
            # -------------------------
            "episodes": self.episodes
        }

        return batch

    # ============================================================
    # FLATTEN HELPER
    # ============================================================

    def _flatten(self, key):

        out = []

        for ep in self.episodes:

            out.extend(ep[key])

        # convert to tensor if numeric
        if isinstance(out[0], (int, float, torch.Tensor)):

            return torch.tensor(out)

        return out

    # ============================================================
    # CLEAR AFTER UPDATE
    # ============================================================

    def clear(self):

        self.reset()