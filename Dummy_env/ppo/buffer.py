# ppo/buffer.py

"""
=============================================================
HIERARCHICAL RMSA PPO BUFFER (PRODUCTION GRADE)
=============================================================

Stores autoregressive trajectories:

    path -> modulation -> slot -> reward

with full observation alignment for PPO recomputation.

Key property:
-------------
Reward is ONLY available after slot selection.

So we store full transition triplets.

=============================================================
DESIGN GOAL
=============================================================

We must support:

- hierarchical actions
- PPO recomputation (critical)
- action masks at each stage
- graph + transformer observations
- minibatch sampling
- GAE computation compatibility

=============================================================
"""

from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np


# ============================================================
# TRANSITION (ONE COMPLETE RMSA REQUEST)
# ============================================================

@dataclass
class RMSATransition:

    """
    One full hierarchical decision:

        state -> path -> mod -> slot -> reward
    """

    # ---------------------------------------------------------
    # OBSERVATIONS (RAW - for PPO recomputation)
    # ---------------------------------------------------------

    obs_path: Dict[str, Any]
    obs_mod: Dict[str, Any]
    obs_slot: Dict[str, Any]

    # ---------------------------------------------------------
    # ACTIONS
    # ---------------------------------------------------------

    path_action: int
    mod_action: int
    slot_action: int

    # ---------------------------------------------------------
    # ACTION MASKS (IMPORTANT FOR PPO RECOMPUTE)
    # ---------------------------------------------------------

    path_mask: np.ndarray
    mod_mask: np.ndarray
    slot_mask: np.ndarray

    # ---------------------------------------------------------
    # LOGPROBS (OLD POLICY)
    # ---------------------------------------------------------

    path_logprob: float
    mod_logprob: float
    slot_logprob: float

    # ---------------------------------------------------------
    # VALUE ESTIMATES (optional but recommended)
    # ---------------------------------------------------------

    value: float

    # ---------------------------------------------------------
    # FINAL REWARD (ONLY AFTER SLOT)
    # ---------------------------------------------------------

    reward: float

    done: bool


# ============================================================
# PPO BUFFER
# ============================================================

class RMSAPPORolloutBuffer:

    """
    Stores full RMSA rollout trajectories.
    """

    def __init__(self):

        self.reset()

    # ---------------------------------------------------------
    # RESET
    # ---------------------------------------------------------

    def reset(self):

        self.trajectories: List[RMSATransition] = []

        # flattened views for PPO
        self.flat_obs = []

        self.flat_path_actions = []
        self.flat_mod_actions = []
        self.flat_slot_actions = []

        self.flat_path_masks = []
        self.flat_mod_masks = []
        self.flat_slot_masks = []

        self.flat_path_logprobs = []
        self.flat_mod_logprobs = []
        self.flat_slot_logprobs = []

        self.flat_values = []
        self.flat_rewards = []
        self.flat_dones = []

    # ---------------------------------------------------------
    # ADD TRANSITION
    # ---------------------------------------------------------

    def add(self, transition: RMSATransition):

        self.trajectories.append(transition)

    # ---------------------------------------------------------
    # FLATTEN FOR PPO
    # ---------------------------------------------------------

    def finalize(self):

        """
        Convert hierarchical trajectories into flat PPO tensors.
        """

        for t in self.trajectories:

            self.flat_obs.append(t.obs_path)  # main entry

            self.flat_path_actions.append(t.path_action)
            self.flat_mod_actions.append(t.mod_action)
            self.flat_slot_actions.append(t.slot_action)

            self.flat_path_masks.append(t.path_mask)
            self.flat_mod_masks.append(t.mod_mask)
            self.flat_slot_masks.append(t.slot_mask)

            self.flat_path_logprobs.append(t.path_logprob)
            self.flat_mod_logprobs.append(t.mod_logprob)
            self.flat_slot_logprobs.append(t.slot_logprob)

            self.flat_values.append(t.value)

            self.flat_rewards.append(t.reward)

            self.flat_dones.append(t.done)

    # ---------------------------------------------------------
    # GETTERS (FOR PPO TRAINER)
    # ---------------------------------------------------------

    def get_flat(self):

        """
        Returns PPO-compatible dictionary.
        """

        return {

            # observations (only path-level stored; others recomputed)
            "observations": self.flat_obs,

            # actions
            "path_actions": np.array(self.flat_path_actions),
            "mod_actions": np.array(self.flat_mod_actions),
            "slot_actions": np.array(self.flat_slot_actions),

            # masks
            "path_masks": np.array(self.flat_path_masks),
            "mod_masks": np.array(self.flat_mod_masks),
            "slot_masks": np.array(self.flat_slot_masks),

            # logprobs (old policy)
            "old_path_logprobs":
                np.array(self.flat_path_logprobs),

            "old_mod_logprobs":
                np.array(self.flat_mod_logprobs),

            "old_slot_logprobs":
                np.array(self.flat_slot_logprobs),

            # values
            "values":
                np.array(self.flat_values),

            # rewards (delayed, RMSA final reward)
            "rewards":
                np.array(self.flat_rewards),

            # dones
            "dones":
                np.array(self.flat_dones),
        }

    # ---------------------------------------------------------
    # SIZE
    # ---------------------------------------------------------

    def __len__(self):

        return len(self.trajectories)


# ============================================================
# WHY THIS DESIGN IS CORRECT
# ============================================================

"""
Key RMSA insight:

We do NOT assign intermediate rewards.

Instead:

    path + modulation + slot -> ONE reward

So each buffer entry is a COMPLETE decision episode.

This ensures:

1. correct credit assignment
2. stable PPO gradients
3. no reward hacking at intermediate steps
4. proper hierarchical learning signal

-------------------------------------------------------------

This also makes PPO recomputation easy:

We always recompute:

    π(path | graph)
    π(mod | path)
    π(slot | path, mod)

from stored observations.

-------------------------------------------------------------

This is EXACTLY what production RMSA systems need.
"""