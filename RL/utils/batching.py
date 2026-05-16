# utils/batching.py

import torch


# ============================================================
# GENERIC HELPERS
# ============================================================

def flatten_list(list_of_lists):

    out = []

    for sublist in list_of_lists:
        out.extend(sublist)

    return out


def to_tensor(x, dtype=torch.float32):

    if torch.is_tensor(x):
        return x

    return torch.tensor(x, dtype=dtype)


# ============================================================
# OBSERVATION BATCHING
# ============================================================

def batch_observations(obs_list):
    """
    ============================================================
    Converts:

        List[obs]

    into fully batched tensors.

    Each obs contains:

        edge_features      : (E, F)
        edge_index         : (2, E)
        candidate_paths    : (K, L)
        path_features      : (D,)
        slot_features      : (S, Fslot)

        action_masks:
            path           : (K,)
            mod            : (M,)
            slot           : (S,)

    OUTPUT:

        edge_features      : (B, E, F)
        edge_index         : (B, 2, E)
        candidate_paths    : (B, K, L)
        path_features      : (B, D)
        slot_features      : (B, S, Fslot)

    ============================================================
    """

    # --------------------------------------------------------
    # GRAPH FEATURES
    # --------------------------------------------------------

    edge_features = torch.stack([
        to_tensor(obs["edge_features"])
        for obs in obs_list
    ], dim=0)

    edge_index = torch.stack([
        to_tensor(obs["edge_index"], dtype=torch.long)
        for obs in obs_list
    ], dim=0)

    # --------------------------------------------------------
    # PATH FEATURES
    # --------------------------------------------------------

    candidate_paths = torch.stack([
        to_tensor(obs["candidate_paths"], dtype=torch.long)
        for obs in obs_list
    ], dim=0)

    path_features = torch.stack([
        to_tensor(obs["path_features"])
        for obs in obs_list
    ], dim=0)

    # --------------------------------------------------------
    # SLOT FEATURES
    # --------------------------------------------------------

    slot_features = torch.stack([
        to_tensor(obs["slot_features"])
        for obs in obs_list
    ], dim=0)

    # --------------------------------------------------------
    # ACTION MASKS
    # --------------------------------------------------------

    path_mask = torch.stack([
        to_tensor(obs["action_masks"]["path"], dtype=torch.bool)
        for obs in obs_list
    ], dim=0)

    mod_mask = torch.stack([
        to_tensor(obs["action_masks"]["mod"], dtype=torch.bool)
        for obs in obs_list
    ], dim=0)

    slot_mask = torch.stack([
        to_tensor(obs["action_masks"]["slot"], dtype=torch.bool)
        for obs in obs_list
    ], dim=0)

    # --------------------------------------------------------
    # FINAL BATCH
    # --------------------------------------------------------

    batch = {

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

    return batch


# ============================================================
# FULL PPO BATCHING
# ============================================================

def batch_episodes(episodes):
    """
    ============================================================
    Converts list of PPO episodes into training batch.

    IMPORTANT:
        Produces batch-first tensors compatible with:
            - hierarchical_policy.forward_ppo()
            - PPO trainer
            - GAE
            - losses

    ============================================================
    """

    # --------------------------------------------------------
    # FLATTEN OBSERVATIONS
    # --------------------------------------------------------

    all_obs = flatten_list([
        ep["obs"] for ep in episodes
    ])

    obs_batch = batch_observations(all_obs)

    # --------------------------------------------------------
    # ACTIONS
    # --------------------------------------------------------

    path_actions = torch.tensor(
        flatten_list([ep["path_actions"] for ep in episodes]),
        dtype=torch.long
    )

    mod_actions = torch.tensor(
        flatten_list([ep["mod_actions"] for ep in episodes]),
        dtype=torch.long
    )

    slot_actions = torch.tensor(
        flatten_list([ep["slot_actions"] for ep in episodes]),
        dtype=torch.long
    )

    # --------------------------------------------------------
    # LOGPROBS
    # --------------------------------------------------------

    path_logprobs = torch.tensor(
        flatten_list([ep["path_logprobs"] for ep in episodes]),
        dtype=torch.float32
    )

    mod_logprobs = torch.tensor(
        flatten_list([ep["mod_logprobs"] for ep in episodes]),
        dtype=torch.float32
    )

    slot_logprobs = torch.tensor(
        flatten_list([ep["slot_logprobs"] for ep in episodes]),
        dtype=torch.float32
    )

    # --------------------------------------------------------
    # REWARDS / DONES
    # --------------------------------------------------------

    rewards = torch.tensor(
        flatten_list([ep["rewards"] for ep in episodes]),
        dtype=torch.float32
    )

    dones = torch.tensor(
        flatten_list([ep["dones"] for ep in episodes]),
        dtype=torch.float32
    )

    # --------------------------------------------------------
    # STAGE IDS
    # --------------------------------------------------------

    stage_ids = torch.tensor(
        flatten_list([ep["stage_ids"] for ep in episodes]),
        dtype=torch.long
    )

    # --------------------------------------------------------
    # FINAL PPO BATCH
    # --------------------------------------------------------

    batch = {

        "obs": obs_batch,

        "path_actions": path_actions,
        "mod_actions": mod_actions,
        "slot_actions": slot_actions,

        "path_logprobs": path_logprobs,
        "mod_logprobs": mod_logprobs,
        "slot_logprobs": slot_logprobs,

        "rewards": rewards,
        "dones": dones,

        "stage_ids": stage_ids
    }

    return batch