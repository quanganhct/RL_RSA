# -*- coding: utf-8 -*-
"""
Created on Wed May 13 11:10:33 2026

@author: Momo
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MaskedHierarchicalRMSAEnv(gym.Env):
    """
    ==========================================================
    Hierarchical RMSA Environment with Action Masking
    ==========================================================

    Sequential Decision Process
    ---------------------------
    Stage 0 -> Select Path
    Stage 1 -> Select Modulation
    Stage 2 -> Select Starting Slot

    Reward:
        ONLY after slot selection.

    ==========================================================
    OBSERVATIONS
    ==========================================================

    STAGE 0 -> PATH SELECTION
    -------------------------

    edge_features : [E, 4]

        0 -> frag_rate_abp
        1 -> max_available_block
        2 -> num_path_occupied
        3 -> min_osnr

    candidate_paths : [K, L]

    action_mask : [K]

    ==========================================================
    STAGE 1 -> MODULATION SELECTION
    ==========================================================

    path_features : [2 + 2*M]

        geographic_distance
        num_slots_per_modulation[M]
        num_hops
        fragmentation_per_modulation[M]

    action_mask : [M]

    ==========================================================
    STAGE 2 -> SLOT SELECTION
    ==========================================================

    slot_features : [S, 2]

        slot_score
        min_gap_osnr

    action_mask : [S]

    ==========================================================
    IMPORTANT
    ==========================================================

    Invalid actions are masked.

    The agent MUST sample only valid actions.

    During PPO training:
        logits[mask == 0] = -1e9
    """

    metadata = {"render_modes": []}

    # ======================================================
    # INIT
    # ======================================================

    def __init__(
        self,
        num_edges=40,
        k_paths=5,
        max_path_len=8,
        num_modulations=4,
        num_slots=128,
        episode_length=100,
        seed=42,
    ):

        super().__init__()

        self.rng = np.random.default_rng(seed)

        self.num_edges = num_edges
        self.k_paths = k_paths
        self.max_path_len = max_path_len
        self.num_modulations = num_modulations
        self.num_slots = num_slots
        self.episode_length = episode_length

        # --------------------------------------------------
        # STAGES
        # --------------------------------------------------

        self.STAGE_PATH = 0
        self.STAGE_MODULATION = 1
        self.STAGE_SLOT = 2

        self.stage = self.STAGE_PATH

        # --------------------------------------------------
        # INTERNAL MEMORY
        # --------------------------------------------------

        self.selected_path = None
        self.selected_modulation = None

        self.current_request = 0

        # --------------------------------------------------
        # GRAPH
        # --------------------------------------------------

        self.edge_index = self._generate_graph()

        # --------------------------------------------------
        # ACTION SPACE
        # --------------------------------------------------

        self.action_space = spaces.Discrete(
            max(
                self.k_paths,
                self.num_modulations,
                self.num_slots,
            )
        )

        # --------------------------------------------------
        # OBS SPACE
        # --------------------------------------------------

        self.observation_space = spaces.Dict({

            "stage": spaces.Discrete(3),

            # graph
            "edge_features": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.num_edges, 4),
                dtype=np.float32,
            ),

            "edge_index": spaces.Box(
                low=0,
                high=self.num_edges,
                shape=self.edge_index.shape,
                dtype=np.int64,
            ),

            "candidate_paths": spaces.Box(
                low=-1,
                high=self.num_edges,
                shape=(self.k_paths, self.max_path_len),
                dtype=np.int64,
            ),

            # path-level
            "path_features": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(2 + 2*self.num_modulations,),
                dtype=np.float32,
            ),

            # slot-level
            "slot_features": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.num_slots, 2),
                dtype=np.float32,
            ),

            # masks
            "action_mask": spaces.Box(
                low=0,
                high=1,
                shape=(max(
                    self.k_paths,
                    self.num_modulations,
                    self.num_slots
                ),),
                dtype=np.int8,
            ),

            # selected decisions
            "selected_path": spaces.Discrete(
                self.k_paths
            ),

            "selected_modulation": spaces.Discrete(
                self.num_modulations
            ),
        })

    # ======================================================
    # RESET
    # ======================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.stage = self.STAGE_PATH

        self.selected_path = None
        self.selected_modulation = None

        self.current_request = 0

        obs = self._build_observation()

        info = {}

        return obs, info

    # ======================================================
    # STEP
    # ======================================================

    def step(self, action):

        reward = 0.0
        terminated = False
        truncated = False

        obs = self._build_observation()

        mask = obs["action_mask"]

        # --------------------------------------------------
        # INVALID ACTION
        # --------------------------------------------------

        if mask[action] == 0:

            reward = -5.0

            info = {
                "invalid_action": True
            }

            return obs, reward, terminated, truncated, info

        # ==================================================
        # STAGE 0 -> PATH
        # ==================================================

        if self.stage == self.STAGE_PATH:

            self.selected_path = int(action)

            self.stage = self.STAGE_MODULATION

            reward = 0.0

        # ==================================================
        # STAGE 1 -> MODULATION
        # ==================================================

        elif self.stage == self.STAGE_MODULATION:

            self.selected_modulation = int(action)

            self.stage = self.STAGE_SLOT

            reward = 0.0

        # ==================================================
        # STAGE 2 -> SLOT
        # ==================================================

        elif self.stage == self.STAGE_SLOT:

            selected_slot = int(action)

            reward = self._compute_final_reward(
                self.selected_path,
                self.selected_modulation,
                selected_slot
            )

            self.current_request += 1

            if self.current_request >= self.episode_length:
                truncated = True

            # restart hierarchy
            self.stage = self.STAGE_PATH

            self.selected_path = None
            self.selected_modulation = None

        obs = self._build_observation()

        info = {
            "invalid_action": False,
            "stage": self.stage,
            "selected_path": self.selected_path,
            "selected_modulation": self.selected_modulation,
        }

        return obs, reward, terminated, truncated, info

    # ======================================================
    # BUILD OBSERVATION
    # ======================================================

    def _build_observation(self):

        obs = {

            "stage": self.stage,

            "edge_features": np.zeros(
                (self.num_edges, 4),
                dtype=np.float32
            ),

            "edge_index": self.edge_index,

            "candidate_paths": np.full(
                (self.k_paths, self.max_path_len),
                -1,
                dtype=np.int64
            ),

            "path_features": np.zeros(
                (2 + 2*self.num_modulations,),
                dtype=np.float32
            ),

            "slot_features": np.zeros(
                (self.num_slots, 2),
                dtype=np.float32
            ),

            "action_mask": np.zeros(
                max(
                    self.k_paths,
                    self.num_modulations,
                    self.num_slots
                ),
                dtype=np.int8
            ),

            "selected_path":
                self.selected_path
                if self.selected_path is not None
                else 0,

            "selected_modulation":
                self.selected_modulation
                if self.selected_modulation is not None
                else 0,
        }

        # ==================================================
        # PATH SELECTION
        # ==================================================

        if self.stage == self.STAGE_PATH:

            obs["edge_features"] = (
                self._generate_edge_features()
            )

            obs["candidate_paths"] = (
                self._generate_candidate_paths()
            )

            path_mask = self._generate_path_mask()

            obs["action_mask"][:self.k_paths] = (
                path_mask
            )

        # ==================================================
        # MODULATION SELECTION
        # ==================================================

        elif self.stage == self.STAGE_MODULATION:

            obs["path_features"] = (
                self._generate_path_features(
                    self.selected_path
                )
            )

            mod_mask = (
                self._generate_modulation_mask(
                    self.selected_path
                )
            )

            obs["action_mask"][
                :self.num_modulations
            ] = mod_mask

        # ==================================================
        # SLOT SELECTION
        # ==================================================

        elif self.stage == self.STAGE_SLOT:

            obs["slot_features"] = (
                self._generate_slot_features(
                    self.selected_path,
                    self.selected_modulation
                )
            )

            slot_mask = (
                self._generate_slot_mask(
                    self.selected_path,
                    self.selected_modulation
                )
            )

            obs["action_mask"][
                :self.num_slots
            ] = slot_mask

        return obs

    # ======================================================
    # GRAPH
    # ======================================================

    def _generate_graph(self):

        num_conn = self.num_edges * 2

        src = self.rng.integers(
            0,
            self.num_edges,
            size=num_conn
        )

        dst = self.rng.integers(
            0,
            self.num_edges,
            size=num_conn
        )

        return np.stack([src, dst], axis=0)

    # ======================================================
    # EDGE FEATURES
    # ======================================================

    def _generate_edge_features(self):

        return np.stack([

            self.rng.uniform(
                0,
                1,
                self.num_edges
            ),

            self.rng.uniform(
                0,
                1,
                self.num_edges
            ),

            self.rng.uniform(
                0,
                1,
                self.num_edges
            ),

            self.rng.uniform(
                0,
                1,
                self.num_edges
            ),

        ], axis=1).astype(np.float32)

    # ======================================================
    # CANDIDATE PATHS
    # ======================================================

    def _generate_candidate_paths(self):

        paths = np.full(
            (self.k_paths, self.max_path_len),
            -1,
            dtype=np.int64
        )

        for k in range(self.k_paths):

            length = self.rng.integers(
                2,
                self.max_path_len + 1
            )

            edges = self.rng.choice(
                self.num_edges,
                size=length,
                replace=False
            )

            paths[k, :length] = edges

        return paths

    # ======================================================
    # PATH FEATURES
    # ======================================================

    def _generate_path_features(
        self,
        path_id
    ):

        geographic_distance = self.rng.uniform(
            0,
            1,
            1
        )

        num_slots_per_modulation = (
            self.rng.uniform(
                0,
                1,
                self.num_modulations
            )
        )

        num_hops = self.rng.uniform(
            0,
            1,
            1
        )

        fragmentation_abp = (
            self.rng.uniform(
                0,
                1,
                self.num_modulations
            )
        )

        return np.concatenate([

            geographic_distance,

            num_slots_per_modulation,

            num_hops,

            fragmentation_abp

        ]).astype(np.float32)

    # ======================================================
    # SLOT FEATURES
    # ======================================================

    def _generate_slot_features(
        self,
        path_id,
        modulation_id
    ):

        slot_score = self.rng.uniform(
            0,
            1,
            self.num_slots
        )

        min_gap_osnr = self.rng.uniform(
            0,
            1,
            self.num_slots
        )

        return np.stack([
            slot_score,
            min_gap_osnr
        ], axis=1).astype(np.float32)

    # ======================================================
    # ACTION MASKS
    # ======================================================

    def _generate_path_mask(self):

        mask = np.zeros(
            self.k_paths,
            dtype=np.int8
        )

        num_valid = self.rng.integers(
            1,
            self.k_paths + 1
        )

        valid_paths = self.rng.choice(
            self.k_paths,
            size=num_valid,
            replace=False
        )

        mask[valid_paths] = 1

        return mask

    def _generate_modulation_mask(
        self,
        path_id
    ):

        mask = np.zeros(
            self.num_modulations,
            dtype=np.int8
        )

        num_valid = self.rng.integers(
            1,
            self.num_modulations + 1
        )

        valid_mods = self.rng.choice(
            self.num_modulations,
            size=num_valid,
            replace=False
        )

        mask[valid_mods] = 1

        return mask

    def _generate_slot_mask(
        self,
        path_id,
        modulation_id
    ):

        mask = np.zeros(
            self.num_slots,
            dtype=np.int8
        )

        num_valid = self.rng.integers(
            self.num_slots // 8,
            self.num_slots // 2
        )

        valid_slots = self.rng.choice(
            self.num_slots,
            size=num_valid,
            replace=False
        )

        mask[valid_slots] = 1

        return mask

    # ======================================================
    # FINAL REWARD
    # ======================================================

    def _compute_final_reward(
        self,
        path,
        modulation,
        slot
    ):

        reward = 0.0

        # prefer lower path index
        reward += 1.0 / (1 + path)

        # prefer higher modulation
        reward += modulation * 0.5

        # prefer lower slot index
        reward -= slot / self.num_slots

        return float(reward)

    # ======================================================
    # RENDER
    # ======================================================

    def render(self):
        pass


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    env = MaskedHierarchicalRMSAEnv()

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