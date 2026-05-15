# -*- coding: utf-8 -*-
"""
Created on Wed May 13 11:36:50 2026

@author: Momo
"""

# utils/batching.py

"""
=============================================================
BATCHING UTILITIES FOR RMSA
=============================================================

This file handles batching for:

1. Edge-centric graph observations
2. Variable-length candidate paths
3. Path transformer padding
4. Slot feature batching
5. PPO minibatch collation

IMPORTANT
=========
This RMSA formulation is EDGE-CENTRIC.

We do NOT care primarily about node features.

The GNN operates over edge features:
    edge_features [E, Fe]

where edges represent optical links.

=============================================================
SUPPORTED COMPONENTS
=============================================================

- Edge graph batching
- Candidate path batching
- Transformer padding masks
- Slot feature batching
- PPO rollout collation
- Hierarchical observation collation

=============================================================
DEPENDENCIES
=============================================================

pip install torch torch-geometric
"""

from dataclasses import dataclass
from typing import List, Dict, Any

import torch
import numpy as np

from torch.nn.utils.rnn import pad_sequence

from torch_geometric.data import Data
from torch_geometric.data import Batch


# ===========================================================
# GRAPH BATCH CONTAINER
# ===========================================================

@dataclass
class GraphBatch:

    """
    Batched edge-centric graph.

    Attributes
    ----------
    edge_features : Tensor
        [total_edges, Fe]

    edge_index : Tensor
        [2, total_connections]

    batch : Tensor
        graph assignment per edge
    """

    edge_features: torch.Tensor
    edge_index: torch.Tensor
    batch: torch.Tensor


# ===========================================================
# PATH BATCH CONTAINER
# ===========================================================

@dataclass
class PathBatch:

    """
    Candidate path batch.

    Attributes
    ----------
    paths : Tensor
        [B, K, L]

    padding_mask : BoolTensor
        [B, K, L]

    lengths : Tensor
        [B, K]
    """

    paths: torch.Tensor
    padding_mask: torch.Tensor
    lengths: torch.Tensor


# ===========================================================
# SLOT BATCH CONTAINER
# ===========================================================

@dataclass
class SlotBatch:

    """
    Batched slot features.

    slot_features:
        [B, S, Fs]
    """

    slot_features: torch.Tensor


# ===========================================================
# PPO BATCH CONTAINER
# ===========================================================

@dataclass
class PPOMiniBatch:

    """
    Production PPO minibatch.
    """

    observations: Dict[str, Any]

    path_actions: torch.Tensor
    mod_actions: torch.Tensor
    slot_actions: torch.Tensor

    old_logprobs: torch.Tensor

    returns: torch.Tensor
    advantages: torch.Tensor

    rewards: torch.Tensor

    dones: torch.Tensor


# ===========================================================
# BUILD EDGE GRAPH
# ===========================================================

def build_edge_graph(
    edge_features,
    edge_index
):

    """
    Build PyG graph using edge-centric features.

    IMPORTANT
    =========
    We treat EDGES as graph entities.

    In this formulation:
        x = edge_features
    """

    data = Data(

        x=torch.FloatTensor(edge_features),

        edge_index=torch.LongTensor(edge_index)

    )

    return data


# ===========================================================
# BATCH EDGE GRAPHS
# ===========================================================

def batch_edge_graphs(
    observations: List[Dict]
):

    """
    Batch multiple RMSA graphs.

    Parameters
    ----------
    observations : list of env observations

    Returns
    -------
    GraphBatch
    """

    pyg_graphs = []

    for obs in observations:

        graph = build_edge_graph(

            obs["edge_features"],
            obs["edge_index"]

        )

        pyg_graphs.append(graph)

    batch = Batch.from_data_list(
        pyg_graphs
    )

    return GraphBatch(

        edge_features=batch.x,
        edge_index=batch.edge_index,
        batch=batch.batch

    )


# ===========================================================
# PAD PATHS
# ===========================================================

def pad_candidate_paths(
    path_tensors: List[torch.Tensor],
    pad_value: int = -1
):

    """
    Pad variable-length candidate paths.

    Input
    -----
    list of:
        [K, L_i]

    Returns
    -------
    padded_paths:
        [B, K, Lmax]
    """

    max_len = max([
        p.shape[-1]
        for p in path_tensors
    ])

    padded = []

    for p in path_tensors:

        K, L = p.shape

        if L < max_len:

            pad = torch.full(

                (K, max_len - L),

                pad_value,

                dtype=p.dtype

            )

            p = torch.cat([
                p,
                pad
            ], dim=-1)

        padded.append(p)

    return torch.stack(padded)


# ===========================================================
# BUILD PATH MASKS
# ===========================================================

def build_path_padding_mask(
    padded_paths,
    pad_value=-1
):

    """
    True = padding

    Shape:
        [B, K, L]
    """

    return padded_paths == pad_value


# ===========================================================
# BUILD PATH LENGTHS
# ===========================================================

def build_path_lengths(
    padded_paths,
    pad_value=-1
):

    """
    Compute valid path lengths.

    Shape:
        [B, K]
    """

    return (
        padded_paths != pad_value
    ).sum(dim=-1)


# ===========================================================
# BATCH PATHS
# ===========================================================

def batch_candidate_paths(
    observations: List[Dict]
):

    """
    Batch candidate paths.

    Returns
    -------
    PathBatch
    """

    paths = [

        torch.LongTensor(
            obs["candidate_paths"]
        )

        for obs in observations

    ]

    padded_paths = pad_candidate_paths(
        paths
    )

    padding_mask = build_path_padding_mask(
        padded_paths
    )

    lengths = build_path_lengths(
        padded_paths
    )

    return PathBatch(

        paths=padded_paths,
        padding_mask=padding_mask,
        lengths=lengths

    )


# ===========================================================
# BATCH SLOT FEATURES
# ===========================================================

def batch_slot_features(
    observations: List[Dict]
):

    """
    Batch slot tensors.

    Output:
        [B, S, Fs]
    """

    slot_features = torch.stack([

        torch.FloatTensor(
            obs["slot_features"]
        )

        for obs in observations

    ])

    return SlotBatch(
        slot_features=slot_features
    )


# ===========================================================
# BATCH ACTION MASKS
# ===========================================================

def batch_action_masks(
    observations: List[Dict]
):

    """
    Batch action masks.

    Returns:
        [B, A]
    """

    return torch.stack([

        torch.BoolTensor(
            obs["action_mask"]
        )

        for obs in observations

    ])


# ===========================================================
# COLLATE HIERARCHICAL OBS
# ===========================================================

def collate_hierarchical_obs(
    observations: List[Dict]
):

    """
    Production hierarchical collation.

    Returns fully batched tensors
    for PPO updates.

    Returns
    -------
    dict
    """

    graph_batch = batch_edge_graphs(
        observations
    )

    path_batch = batch_candidate_paths(
        observations
    )

    slot_batch = batch_slot_features(
        observations
    )

    action_masks = batch_action_masks(
        observations
    )

    stages = torch.LongTensor([

        obs["stage"]

        for obs in observations

    ])

    path_features = torch.stack([

        torch.FloatTensor(
            obs["path_features"]
        )

        for obs in observations

    ])

    selected_paths = torch.LongTensor([

        obs["selected_path"]

        for obs in observations

    ])

    selected_modulations = torch.LongTensor([

        obs["selected_modulation"]

        for obs in observations

    ])

    return {

        # graph
        "graph": graph_batch,

        # paths
        "paths": path_batch,

        # slots
        "slots": slot_batch,

        # masks
        "action_masks": action_masks,

        # stage info
        "stages": stages,

        # path-level features
        "path_features": path_features,

        # selected decisions
        "selected_paths": selected_paths,
        "selected_modulations":
            selected_modulations

    }


# ===========================================================
# PPO MINIBATCH COLLATOR
# ===========================================================

def build_ppo_minibatches(

    rollout_data,
    minibatch_size

):

    """
    Build PPO minibatches.

    rollout_data:
        list of transitions
    """

    num_samples = len(
        rollout_data["advantages"]
    )

    indices = np.arange(num_samples)

    np.random.shuffle(indices)

    minibatches = []

    for start in range(
        0,
        num_samples,
        minibatch_size
    ):

        end = start + minibatch_size

        mb_idx = indices[start:end]

        obs_batch = [

            rollout_data["observations"][i]

            for i in mb_idx

        ]

        collated_obs = (
            collate_hierarchical_obs(
                obs_batch
            )
        )

        minibatch = PPOMiniBatch(

            observations=collated_obs,

            path_actions=torch.LongTensor([

                rollout_data["path_actions"][i]

                for i in mb_idx

            ]),

            mod_actions=torch.LongTensor([

                rollout_data["mod_actions"][i]

                for i in mb_idx

            ]),

            slot_actions=torch.LongTensor([

                rollout_data["slot_actions"][i]

                for i in mb_idx

            ]),

            old_logprobs=torch.FloatTensor([

                rollout_data["old_logprobs"][i]

                for i in mb_idx

            ]),

            returns=torch.FloatTensor([

                rollout_data["returns"][i]

                for i in mb_idx

            ]),

            advantages=torch.FloatTensor([

                rollout_data["advantages"][i]

                for i in mb_idx

            ]),

            rewards=torch.FloatTensor([

                rollout_data["rewards"][i]

                for i in mb_idx

            ]),

            dones=torch.FloatTensor([

                rollout_data["dones"][i]

                for i in mb_idx

            ])

        )

        minibatches.append(minibatch)

    return minibatches


# ===========================================================
# DEVICE TRANSFER
# ===========================================================

def move_batch_to_device(
    batch,
    device
):

    """
    Recursively move nested tensors to device.
    """

    if isinstance(batch, torch.Tensor):

        return batch.to(device)

    elif isinstance(batch, dict):

        return {
            k: move_batch_to_device(v, device)
            for k, v in batch.items()
        }

    elif hasattr(batch, "__dict__"):

        for key, value in vars(batch).items():

            setattr(

                batch,
                key,

                move_batch_to_device(
                    value,
                    device
                )

            )

        return batch

    else:

        return batch


# ===========================================================
# TEST
# ===========================================================

if __name__ == "__main__":

    B = 4

    observations = []

    for _ in range(B):

        obs = {

            "stage": 0,

            "edge_features":
                np.random.rand(40, 4),

            "edge_index":
                np.random.randint(
                    0,
                    40,
                    (2, 80)
                ),

            "candidate_paths":
                np.random.randint(
                    -1,
                    40,
                    (5, 8)
                ),

            "slot_features":
                np.random.rand(
                    128,
                    2
                ),

            "action_mask":
                np.ones(128),

            "path_features":
                np.random.rand(10),

            "selected_path": 0,
            "selected_modulation": 0
        }

        observations.append(obs)

    batch = collate_hierarchical_obs(
        observations
    )

    print("\nGRAPH EDGES:")
    print(batch["graph"].edge_features.shape)

    print("\nPATHS:")
    print(batch["paths"].paths.shape)

    print("\nSLOTS:")
    print(batch["slots"].slot_features.shape)