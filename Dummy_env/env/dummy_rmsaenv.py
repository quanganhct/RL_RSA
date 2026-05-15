
"""
============================================================
DUMMY RMSA ENVIRONMENT (FULL PIPELINE VALIDATION)
============================================================

Purpose:
--------
- Validate model + PPO + rollout integration
- Ensure tensor shapes are correct
- Ensure hierarchical flow is stable
- Simulate spectrum + path decision process

NOT a real physics simulator.
ONLY for system integration testing.

============================================================
"""

import torch
import random
import numpy as np


class DummyRMSAEnv:

    def __init__(self):

        self.num_edges = 20
        self.num_paths = 5
        self.max_path_len = 6
        self.num_slots = 64
        self.num_mods = 6

        self.reset()

    # ----------------------------------------------------
    # RESET
    # ----------------------------------------------------

    def reset(self):

        self.t = 0

        return self._obs()

    # ----------------------------------------------------
    # PATH STEP
    # ----------------------------------------------------

    def step_path(self, action):

        self.selected_path = int(action)

        return self._obs(), {}

    # ----------------------------------------------------
    # MOD STEP
    # ----------------------------------------------------

    def step_modulation(self, action):

        self.selected_mod = int(action)

        return self._obs(), {}

    # ----------------------------------------------------
    # SLOT STEP (REWARD HERE)
    # ----------------------------------------------------

    def step_slot(self, action):

        self.selected_slot = int(action)

        # ------------------------------------------------
        # SIMPLE SYNTHETIC FEASIBILITY RULE
        # ------------------------------------------------

        base_success = 0.5

        # pretend congestion penalty
        congestion_penalty = random.random() * 0.2

        # modulation penalty (higher mod index = harder)
        mod_penalty = self.selected_mod * 0.03

        # slot position penalty
        slot_penalty = (self.selected_slot / self.num_slots) * 0.1

        reward = base_success - congestion_penalty - mod_penalty - slot_penalty

        reward = float(np.clip(reward, 0.0, 1.0))

        done = True

        return self._obs(), reward, done, {}

    # ----------------------------------------------------
    # OBSERVATION GENERATION
    # ----------------------------------------------------

    def _obs(self):

        # -----------------------------
        # EDGE FEATURES
        # -----------------------------

        edge_features = torch.randn(
            self.num_edges, 4
        )

        edge_index = torch.randint(
            0,
            self.num_edges,
            (2, self.num_edges * 2)
        )

        # -----------------------------
        # PATH FEATURES
        # -----------------------------

        path_features = torch.randn(10)

        candidate_paths = torch.randint(
            0,
            self.num_edges,
            (self.num_paths, self.max_path_len)
        )

        # -----------------------------
        # SLOT FEATURES
        # -----------------------------

        slot_features = torch.randn(
            self.num_slots,
            3
        )

        # make some slots “bad” randomly
        slot_mask = torch.rand(
            self.num_slots
        ) > 0.2

        # -----------------------------
        # ACTION MASKS
        # -----------------------------

        path_mask = torch.ones(self.num_paths)

        mod_mask = torch.ones(self.num_mods)

        # randomly disable some mods (simulate OSNR constraints)
        mod_mask[4:] = 0

        return {

            "edge_features": edge_features,
            "edge_index": edge_index,

            "candidate_paths": candidate_paths,

            "path_features": path_features,

            "slot_features": slot_features,

            "action_masks": {
                "path": path_mask,
                "mod": mod_mask,
                "slot": slot_mask
            }
        }

# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    env = DummyRMSAEnv()

    obs, info = env.reset()

    print("\n============================")
    print("STAGE:", obs["stage"])
    print("============================")

    # ------------------------------------------------------
    # PATH SELECTION
    # ------------------------------------------------------

    path_mask = obs["action_mask"][:env.k_paths]

    print("\nPATH MASK:")
    print(path_mask)

    valid_paths = np.where(path_mask == 1)[0]

    path_action = np.random.choice(valid_paths)

    print("\nSELECTED PATH:", path_action)

    obs, reward, term, trunc, info = env.step(
        path_action
    )

    # ------------------------------------------------------
    # MODULATION SELECTION
    # ------------------------------------------------------

    mod_mask = obs["action_mask"][
        :env.num_modulations
    ]

    print("\nMODULATION MASK:")
    print(mod_mask)

    valid_mods = np.where(mod_mask == 1)[0]

    mod_action = np.random.choice(valid_mods)

    print("\nSELECTED MOD:", mod_action)

    obs, reward, term, trunc, info = env.step(
        mod_action
    )

    # ------------------------------------------------------
    # SLOT SELECTION
    # ------------------------------------------------------

    slot_mask = obs["action_mask"][
        :env.num_slots
    ]

    print("\nNUM VALID SLOTS:",
          slot_mask.sum())

    valid_slots = np.where(slot_mask == 1)[0]

    slot_action = np.random.choice(valid_slots)

    print("\nSELECTED SLOT:", slot_action)

    obs, reward, term, trunc, info = env.step(
        slot_action
    )

    print("\nFINAL REWARD:", reward)

    print("\nBACK TO STAGE:",
          obs["stage"])