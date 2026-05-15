import torch
import numpy as np
import random


class DummyRMSAEnv:
    """
    ============================================================
    HIERARCHICAL RMSA DUMMY ENV (STATEFUL + CONSISTENT)
    ============================================================

    This environment simulates:

        S0 → path selection → S1
           → modulation → S2
           → slot selection → reward

    KEY PROPERTY:
        All features depend on internal state transitions.

    ============================================================
    """

    def __init__(self):

        # -----------------------------
        # GRAPH / NETWORK SIZE
        # -----------------------------
        self.num_edges = 20
        self.num_nodes = 12

        # -----------------------------
        # DECISION SPACE
        # -----------------------------
        self.num_paths = 5
        self.max_path_len = 6
        self.num_mods = 6
        self.num_slots = 64

        # -----------------------------
        # INTERNAL STATE
        # -----------------------------
        self.reset()

    # ============================================================
    # RESET
    # ============================================================

    def reset(self):

        self.t = 0

        self.selected_path = None
        self.selected_mod = None

        self._build_graph_state()

        return self._obs()

    # ============================================================
    # GRAPH STATE (SIMULATED DYNAMIC NETWORK)
    # ============================================================

    def _build_graph_state(self):

        # edge congestion / fragmentation proxy
        self.edge_load = torch.rand(self.num_edges)

        self.edge_features = torch.randn(self.num_edges, 4)

        self.edge_index = torch.randint(
            0,
            self.num_edges,
            (2, self.num_edges * 2)
        )

    # ============================================================
    # PATH FEATURES (DEPEND ON STATE + ACTION)
    # ============================================================

    def compute_path_features(self, path_id):

        """
        Features depend on:
        - edge congestion
        - random structural metrics
        """

        distance = random.random()
        hops = random.randint(1, 6)

        frag = float(self.edge_load.mean().item())

        return torch.tensor([
            distance,
            hops / 6.0,
            frag,
            path_id / self.num_paths,
            random.random(),
            random.random(),
            random.random(),
            random.random(),
            random.random(),
            random.random()
        ], dtype=torch.float)

    # ============================================================
    # SLOT FEATURES (DEPEND ON PATH + MOD)
    # ============================================================

    def compute_slot_features(self, path_id, mod_id):

        base = torch.rand(self.num_slots, 3)

        # simulate fragmentation worsening with higher mod index
        noise = (mod_id + 1) * 0.05

        base[:, 0] += noise  # "fragmentation proxy"
        base[:, 1] -= noise  # "availability proxy"

        return base

    # ============================================================
    # PATH STEP
    # ============================================================

    def step_path(self, action):

        self.selected_path = int(action)

        obs = self._obs()

        return obs, {}

    # ============================================================
    # MODULATION STEP
    # ============================================================

    def step_modulation(self, action):

        self.selected_mod = int(action)

        obs = self._obs()

        return obs, {}

    # ============================================================
    # SLOT STEP (REWARD COMPUTED HERE)
    # ============================================================

    def step_slot(self, action):

        slot = int(action)

        # -----------------------------
        # SIMPLE FEASIBILITY MODEL
        # -----------------------------

        base = 0.6

        # path quality effect
        path_penalty = 0.05 * (self.selected_path or 0)

        # modulation difficulty
        mod_penalty = 0.03 * (self.selected_mod or 0)

        # slot position penalty
        slot_penalty = (slot / self.num_slots) * 0.1

        # congestion penalty
        congestion = float(self.edge_load.mean().item()) * 0.2

        reward = base - path_penalty - mod_penalty - slot_penalty - congestion

        reward = float(np.clip(reward, 0.0, 1.0))

        done = True

        self.t += 1

        return self._obs(), reward, done, {}

    # ============================================================
    # OBSERVATION (STATE DEPENDENT)
    # ============================================================

    def _obs(self):

        # -----------------------------
        # PATH CANDIDATES [should be list of list as size might differ]
        # -----------------------------

        candidate_paths = torch.randint(
            0,
            self.num_edges,
            (self.num_paths, self.max_path_len)
        )

        # -----------------------------
        # SLOT FEATURES (DEPEND ON STATE)
        # -----------------------------

        if self.selected_path is None:
            slot_features = torch.randn(self.num_slots, 3)
        else:
            slot_features = self.compute_slot_features(
                self.selected_path,
                self.selected_mod or 0
            )

        # -----------------------------
        # ACTION MASKS (STATE DEPENDENT)
        # -----------------------------

        path_mask = torch.ones(self.num_paths)

        mod_mask = torch.ones(self.num_mods)

        # simulate OSNR constraint
        if self.selected_path is not None:
            mod_mask[-2:] = 0

        slot_mask = torch.rand(self.num_slots) > 0.2

        # -----------------------------
        # PATH FEATURES (IMPORTANT FIX)
        # -----------------------------

        if self.selected_path is None:
            path_features = torch.zeros(10)
        else:
            path_features = self.compute_path_features(
                self.selected_path
            )

        return {

            # GRAPH STATE
            "edge_features": self.edge_features,
            "edge_index": self.edge_index,

            # PATH STRUCTURE
            "candidate_paths": candidate_paths,
            "path_features": path_features,

            # SLOT STRUCTURE
            "slot_features": slot_features,

            # MASKS
            "action_masks": {
                "path": path_mask,
                "mod": mod_mask,
                "slot": slot_mask
            }
        }