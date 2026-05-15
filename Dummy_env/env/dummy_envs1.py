"""
Dummy RMSA Gym Environment
--------------------------

Goal:
- Path selection (choose among K precomputed paths)
- Modulation selection (choose among M modulations)
- Starting-slot selection

This environment DOES NOT simulate full optical physics.
It only generates structured observations with the feature
shapes required for your future GNN + Transformer agent.

Observation structure
=====================

1) Edge-level graph features (for GNN path encoder)
---------------------------------------------------
edge_features: shape = [num_edges, 4]

Features:
0 -> frag_rate_abp
1 -> max_available_block
2 -> num_path_occupied
3 -> min_osnr

edge_index: graph connectivity for GNN
shape = [2, num_connections]

2) Path candidates
------------------
k candidate paths.
Each path is a list of edge ids.

paths: shape = [k_paths, max_path_len]

Padding value = -1

3) Path-level features for modulation selection
-----------------------------------------------
path_features: shape = [k_paths, path_feature_dim]

Features:
- geographic_distance               -> 1
- num_slots_per_modulation          -> M
- num_hops                          -> 1
- fragmentation_abp_per_modulation  -> M

Total dimension:
1 + M + 1 + M = 2 + 2M

4) Modulation/slot features for slot selection
----------------------------------------------
slot_features: shape = [M, num_slots, 2]

Features:
0 -> slot_score
1 -> min_gap_osnr

Action space
============

MultiDiscrete:
[path_id, modulation_id, start_slot]

Reward
======
Dummy reward for now.
You will later replace with:
- blocking reward
- spectrum efficiency
- fragmentation penalties
- QoT constraints
etc.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DummyRMSAEnv(gym.Env):

    metadata = {"render_modes": []}

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

        self.num_edges = num_edges
        self.k_paths = k_paths
        self.max_path_len = max_path_len
        self.num_modulations = num_modulations
        self.num_slots = num_slots
        self.episode_length = episode_length

        self.rng = np.random.default_rng(seed)

        # ---------------------------------------------------
        # GRAPH TOPOLOGY (dummy)
        # ---------------------------------------------------

        self.edge_index = self._generate_random_graph()

        # ---------------------------------------------------
        # OBSERVATION SPACE
        # ---------------------------------------------------

        self.observation_space = spaces.Dict({

            # Edge features for GNN
            "edge_features": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(num_edges, 4),
                dtype=np.float32,
            ),

            # Graph connectivity
            "edge_index": spaces.Box(
                low=0,
                high=num_edges,
                shape=self.edge_index.shape,
                dtype=np.int64,
            ),

            # Candidate paths
            "paths": spaces.Box(
                low=-1,
                high=num_edges,
                shape=(k_paths, max_path_len),
                dtype=np.int64,
            ),

            # Path features
            "path_features": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(k_paths, 2 + 2 * num_modulations),
                dtype=np.float32,
            ),

            # Slot features
            "slot_features": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(num_modulations, num_slots, 2),
                dtype=np.float32,
            ),
        })

        # ---------------------------------------------------
        # ACTION SPACE
        # ---------------------------------------------------

        self.action_space = spaces.MultiDiscrete([
            k_paths,
            num_modulations,
            num_slots,
        ])

        self.current_step = 0

    # =======================================================
    # RESET
    # =======================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.current_step = 0

        obs = self._generate_observation()

        info = {}

        return obs, info

    # =======================================================
    # STEP
    # =======================================================

    def step(self, action):

        path_id, modulation_id, start_slot = action

        self.current_step += 1

        # ---------------------------------------------------
        # DUMMY REWARD
        # ---------------------------------------------------

        reward = self._compute_dummy_reward(
            path_id,
            modulation_id,
            start_slot
        )

        terminated = False
        truncated = self.current_step >= self.episode_length

        obs = self._generate_observation()

        info = {
            "selected_path": int(path_id),
            "selected_modulation": int(modulation_id),
            "selected_slot": int(start_slot),
        }

        return obs, reward, terminated, truncated, info

    # =======================================================
    # OBSERVATION GENERATION
    # =======================================================

    def _generate_observation(self):

        edge_features = self._generate_edge_features()

        paths = self._generate_candidate_paths()

        path_features = self._generate_path_features()

        slot_features = self._generate_slot_features()

        obs = {
            "edge_features": edge_features.astype(np.float32),
            "edge_index": self.edge_index.astype(np.int64),
            "paths": paths.astype(np.int64),
            "path_features": path_features.astype(np.float32),
            "slot_features": slot_features.astype(np.float32),
        }

        return obs

    # =======================================================
    # EDGE FEATURES
    # =======================================================

    def _generate_edge_features(self):

        """
        Features per edge:

        0 -> frag_rate_abp
        1 -> max_available_block
        2 -> num_path_occupied
        3 -> min_osnr
        """

        frag_rate_abp = self.rng.uniform(0, 1, self.num_edges)

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

        return edge_features

    # =======================================================
    # PATH GENERATION
    # =======================================================

    def _generate_candidate_paths(self):

        """
        paths shape:
        [k_paths, max_path_len]

        padded with -1
        """

        paths = np.full(
            (self.k_paths, self.max_path_len),
            fill_value=-1,
            dtype=np.int64,
        )

        for k in range(self.k_paths):

            path_len = self.rng.integers(
                low=2,
                high=self.max_path_len + 1
            )

            edge_ids = self.rng.choice(
                self.num_edges,
                size=path_len,
                replace=False
            )

            paths[k, :path_len] = edge_ids

        return paths

    # =======================================================
    # PATH FEATURES
    # =======================================================

    def _generate_path_features(self):

        """
        Per path:

        geographic_distance -> 1
        num_slots_per_modulation -> M
        num_hops -> 1
        fragmentation_abp_per_mod -> M

        Total = 2 + 2M
        """

        features = []

        for _ in range(self.k_paths):

            geographic_distance = self.rng.uniform(0, 1, 1)

            num_slots_per_modulation = self.rng.uniform(
                0,
                1,
                self.num_modulations
            )

            num_hops = self.rng.uniform(0, 1, 1)

            fragmentation_abp = self.rng.uniform(
                0,
                1,
                self.num_modulations
            )

            feat = np.concatenate([
                geographic_distance,
                num_slots_per_modulation,
                num_hops,
                fragmentation_abp,
            ])

            features.append(feat)

        return np.array(features)

    # =======================================================
    # SLOT FEATURES
    # =======================================================

    def _generate_slot_features(self):

        """
        shape:
        [M, num_slots, 2]

        Features:
        0 -> slot_score
        1 -> min_gap_osnr
        """

        slot_score = self.rng.uniform(
            0,
            1,
            (self.num_modulations, self.num_slots)
        )

        min_gap_osnr = self.rng.uniform(
            0,
            1,
            (self.num_modulations, self.num_slots)
        )

        slot_features = np.stack([
            slot_score,
            min_gap_osnr,
        ], axis=-1)

        return slot_features

    # =======================================================
    # RANDOM GRAPH
    # =======================================================

    def _generate_random_graph(self):

        """
        Generates dummy edge_index for GNN.
        """

        num_connections = self.num_edges * 2

        src = self.rng.integers(
            0,
            self.num_edges,
            size=num_connections
        )

        dst = self.rng.integers(
            0,
            self.num_edges,
            size=num_connections
        )

        edge_index = np.stack([src, dst], axis=0)

        return edge_index

    # =======================================================
    # REWARD
    # =======================================================

    def _compute_dummy_reward(
        self,
        path_id,
        modulation_id,
        start_slot
    ):

        """
        Temporary reward.

        Encourages:
        - lower path index
        - higher modulation
        - lower slot index
        """

        reward = (
            1.0
            - 0.1 * path_id
            + 0.2 * modulation_id
            - 0.001 * start_slot
        )

        return float(reward)

    # =======================================================
    # RENDER
    # =======================================================

    def render(self):
        pass


# ===========================================================
# TEST
# ===========================================================

if __name__ == "__main__":

    env = DummyRMSAEnv()

    obs, info = env.reset()

    print("\n=== OBSERVATION SHAPES ===")

    for k, v in obs.items():
        print(k, v.shape)

    action = env.action_space.sample()

    next_obs, reward, terminated, truncated, info = env.step(action)

    print("\nSample action:", action)
    print("Reward:", reward)