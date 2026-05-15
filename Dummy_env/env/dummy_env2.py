# -*- coding: utf-8 -*-
"""
Created on Wed May 13 11:04:25 2026

@author: Momo
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class HierarchicalDummyRMSAEnv(gym.Env):
    """
    Hierarchical RMSA Environment
    =============================

    Sequential decision process:

        Stage 0 -> Select Path
        Stage 1 -> Select Modulation
        Stage 2 -> Select Starting Slot

    This environment is intentionally dummy/synthetic.
    It generates observations with proper shapes and
    dependencies for future GNN + Transformer training.

    ---------------------------------------------------
    OBSERVATION FLOW
    ---------------------------------------------------

    STAGE 0 (PATH SELECTION)
    ------------------------
    Observation:
        edge_features
        edge_index
        candidate_paths

    Action:
        select path_id

    STAGE 1 (MODULATION SELECTION)
    ------------------------------
    Observation:
        selected_path
        path_features
        path_embedding_input

    Action:
        select modulation_id

    STAGE 2 (SLOT SELECTION)
    ------------------------
    Observation:
        selected_path
        selected_modulation
        slot_features

    Action:
        select start_slot

    ---------------------------------------------------
    FEATURE DEFINITIONS
    ---------------------------------------------------

    EDGE FEATURES
    -------------
    [E, 4]

    0 -> frag_rate_abp
    1 -> max_available_block
    2 -> num_path_occupied
    3 -> min_osnr

    PATH FEATURES
    -------------
    [2 + 2*M]

    geographic_distance          -> 1
    num_slots_per_modulation     -> M
    num_hops                     -> 1
    fragmentation_per_modulation -> M

    SLOT FEATURES
    -------------
    [S, 2]

    slot_score
    min_gap_osnr
    """

    metadata = {"render_modes": []}

    # =====================================================
    # INIT
    # =====================================================

    def __init__(
        self,
        num_edges=40,
        k_paths=5,
        max_path_len=8,
        num_modulations=4,
        num_slots=128,
        episode_length=50,
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

        # -------------------------------------------------
        # GRAPH
        # -------------------------------------------------

        self.edge_index = self._generate_graph()

        # -------------------------------------------------
        # STAGES
        # -------------------------------------------------

        self.STAGE_PATH = 0
        self.STAGE_MODULATION = 1
        self.STAGE_SLOT = 2

        self.stage = self.STAGE_PATH

        # -------------------------------------------------
        # INTERNAL MEMORY
        # -------------------------------------------------

        self.selected_path = None
        self.selected_modulation = None

        self.current_step = 0

        # -------------------------------------------------
        # ACTION SPACE
        # -------------------------------------------------

        # Dynamic interpretation based on stage
        #
        # stage 0 -> path_id
        # stage 1 -> modulation_id
        # stage 2 -> slot_id

        self.action_space = spaces.Discrete(
            max(
                self.k_paths,
                self.num_modulations,
                self.num_slots
            )
        )

        # -------------------------------------------------
        # OBSERVATION SPACE
        # -------------------------------------------------

        self.observation_space = spaces.Dict({

            "stage": spaces.Discrete(3),

            # =================================================
            # PATH SELECTION OBS
            # =================================================

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

            # =================================================
            # MODULATION SELECTION OBS
            # =================================================

            "selected_path": spaces.Discrete(self.k_paths),

            "path_features": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(2 + 2 * self.num_modulations,),
                dtype=np.float32,
            ),

            # =================================================
            # SLOT SELECTION OBS
            # =================================================

            "selected_modulation": spaces.Discrete(
                self.num_modulations
            ),

            "slot_features": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.num_slots, 2),
                dtype=np.float32,
            ),

            # =================================================
            # OPTIONAL ACTION MASKS
            # =================================================

            "valid_actions": spaces.Box(
                low=0,
                high=1,
                shape=(max(
                    self.k_paths,
                    self.num_modulations,
                    self.num_slots
                ),),
                dtype=np.int8,
            )
        })

    # =====================================================
    # RESET
    # =====================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.current_step = 0

        self.stage = self.STAGE_PATH

        self.selected_path = None
        self.selected_modulation = None

        obs = self._get_observation()

        info = {}

        return obs, info

    # =====================================================
    # STEP
    # =====================================================

    def step(self, action):

        reward = 0.0
        terminated = False
        truncated = False

        # =================================================
        # STAGE 0 -> PATH SELECTION
        # =================================================

        if self.stage == self.STAGE_PATH:

            if action >= self.k_paths:
                reward = -10.0
            else:
                self.selected_path = int(action)

                reward = 1.0

                self.stage = self.STAGE_MODULATION

        # =================================================
        # STAGE 1 -> MODULATION SELECTION
        # =================================================

        elif self.stage == self.STAGE_MODULATION:

            if action >= self.num_modulations:
                reward = -10.0
            else:
                self.selected_modulation = int(action)

                reward = 1.0

                self.stage = self.STAGE_SLOT

        # =================================================
        # STAGE 2 -> SLOT SELECTION
        # =================================================

        elif self.stage == self.STAGE_SLOT:

            if action >= self.num_slots:
                reward = -10.0
            else:

                slot = int(action)

                reward = self._compute_final_reward(
                    self.selected_path,
                    self.selected_modulation,
                    slot
                )

                # episode complete
                self.current_step += 1

                if self.current_step >= self.episode_length:
                    truncated = True

                # restart hierarchy for next request
                self.stage = self.STAGE_PATH
                self.selected_path = None
                self.selected_modulation = None

        obs = self._get_observation()

        info = {
            "stage": self.stage,
            "selected_path": self.selected_path,
            "selected_modulation": self.selected_modulation,
        }

        return obs, reward, terminated, truncated, info

    # =====================================================
    # OBSERVATION GENERATION
    # =====================================================

    def _get_observation(self):

        obs = {
            "stage": self.stage,

            # path selection data
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

            # modulation selection data
            "selected_path": 0,

            "path_features": np.zeros(
                (2 + 2 * self.num_modulations,),
                dtype=np.float32
            ),

            # slot selection data
            "selected_modulation": 0,

            "slot_features": np.zeros(
                (self.num_slots, 2),
                dtype=np.float32
            ),

            "valid_actions": np.zeros(
                (
                    max(
                        self.k_paths,
                        self.num_modulations,
                        self.num_slots
                    ),
                ),
                dtype=np.int8
            )
        }

        # =================================================
        # STAGE 0
        # =================================================

        if self.stage == self.STAGE_PATH:

            obs["edge_features"] = (
                self._generate_edge_features()
            )

            obs["candidate_paths"] = (
                self._generate_candidate_paths()
            )

            obs["valid_actions"][:self.k_paths] = 1

        # =================================================
        # STAGE 1
        # =================================================

        elif self.stage == self.STAGE_MODULATION:

            obs["selected_path"] = self.selected_path

            obs["path_features"] = (
                self._generate_path_features(
                    self.selected_path
                )
            )

            valid_mods = self._generate_valid_modulations()

            obs["valid_actions"][
                :self.num_modulations
            ] = valid_mods

        # =================================================
        # STAGE 2
        # =================================================

        elif self.stage == self.STAGE_SLOT:

            obs["selected_path"] = self.selected_path

            obs["selected_modulation"] = (
                self.selected_modulation
            )

            obs["slot_features"] = (
                self._generate_slot_features(
                    self.selected_path,
                    self.selected_modulation
                )
            )

            valid_slots = self._generate_valid_slots()

            obs["valid_actions"][
                :self.num_slots
            ] = valid_slots

        return obs

    # =====================================================
    # EDGE FEATURES
    # =====================================================

    def _generate_edge_features(self):

        frag_rate_abp = self.rng.uniform(
            0,
            1,
            self.num_edges
        )

        max_available_block = self.rng.uniform(
            0,
            1,
            self.num_edges
        )

        num_path_occupied = self.rng.uniform(
            0,
            1,
            self.num_edges
        )

        min_osnr = self.rng.uniform(
            0,
            1,
            self.num_edges
        )

        edge_features = np.stack([
            frag_rate_abp,
            max_available_block,
            num_path_occupied,
            min_osnr,
        ], axis=1)

        return edge_features.astype(np.float32)

    # =====================================================
    # PATHS
    # =====================================================

    def _generate_candidate_paths(self):

        paths = np.full(
            (self.k_paths, self.max_path_len),
            -1,
            dtype=np.int64,
        )

        for k in range(self.k_paths):

            path_len = self.rng.integers(
                2,
                self.max_path_len + 1
            )

            edge_ids = self.rng.choice(
                self.num_edges,
                size=path_len,
                replace=False
            )

            paths[k, :path_len] = edge_ids

        return paths

    # =====================================================
    # PATH FEATURES
    # =====================================================

    def _generate_path_features(self, path_id):

        geographic_distance = self.rng.uniform(
            0,
            1,
            1
        )

        num_slots_per_mod = self.rng.uniform(
            0,
            1,
            self.num_modulations
        )

        num_hops = self.rng.uniform(
            0,
            1,
            1
        )

        fragmentation_per_mod = self.rng.uniform(
            0,
            1,
            self.num_modulations
        )

        features = np.concatenate([
            geographic_distance,
            num_slots_per_mod,
            num_hops,
            fragmentation_per_mod,
        ])

        return features.astype(np.float32)

    # =====================================================
    # SLOT FEATURES
    # =====================================================

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

        slot_features = np.stack([
            slot_score,
            min_gap_osnr
        ], axis=1)

        return slot_features.astype(np.float32)

    # =====================================================
    # VALID ACTIONS
    # =====================================================

    def _generate_valid_modulations(self):

        mask = np.zeros(
            self.num_modulations,
            dtype=np.int8
        )

        num_valid = self.rng.integers(
            1,
            self.num_modulations + 1
        )

        valid_ids = self.rng.choice(
            self.num_modulations,
            size=num_valid,
            replace=False
        )

        mask[valid_ids] = 1

        return mask

    def _generate_valid_slots(self):

        mask = np.zeros(
            self.num_slots,
            dtype=np.int8
        )

        num_valid = self.rng.integers(
            self.num_slots // 4,
            self.num_slots // 2
        )

        valid_slots = self.rng.choice(
            self.num_slots,
            size=num_valid,
            replace=False
        )

        mask[valid_slots] = 1

        return mask

    # =====================================================
    # GRAPH
    # =====================================================

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

    # =====================================================
    # REWARD
    # =====================================================

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

        # prefer lower slots
        reward -= slot / self.num_slots

        return float(reward)

    # =====================================================
    # RENDER
    # =====================================================

    def render(self):
        pass


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    env = HierarchicalDummyRMSAEnv()

    obs, info = env.reset()

    print("\n===== RESET =====")
    print("Stage:", obs["stage"])

    # -----------------------------------------------------
    # PATH SELECTION
    # -----------------------------------------------------

    print("\nPATH OBS")
    print("edge_features:",
          obs["edge_features"].shape)

    print("candidate_paths:",
          obs["candidate_paths"].shape)

    path_action = 2

    obs, reward, term, trunc, info = env.step(
        path_action
    )

    print("\nSelected Path:", path_action)

    # -----------------------------------------------------
    # MODULATION SELECTION
    # -----------------------------------------------------

    print("\nMODULATION OBS")
    print("path_features:",
          obs["path_features"].shape)

    mod_action = 1

    obs, reward, term, trunc, info = env.step(
        mod_action
    )

    print("\nSelected Modulation:", mod_action)

    # -----------------------------------------------------
    # SLOT SELECTION
    # -----------------------------------------------------

    print("\nSLOT OBS")
    print("slot_features:",
          obs["slot_features"].shape)

    slot_action = 15

    obs, reward, term, trunc, info = env.step(
        slot_action
    )

    print("\nSelected Slot:", slot_action)

    print("\nReward:", reward)

    print("\nBack to Stage:", obs["stage"])