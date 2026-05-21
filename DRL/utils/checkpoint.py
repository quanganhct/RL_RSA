# utils/checkpoint.py

"""
=============================================================
CHECKPOINTING SYSTEM FOR RMSA PPO TRAINING
=============================================================

Supports:

1. Model saving / loading
2. Optimizer state preservation
3. Training metadata tracking
4. Resume-safe reproducibility
5. Versioned checkpoints

=============================================================
WHY THIS MATTERS FOR RMSA
=============================================================

RMSA training is:

- long horizon
- stochastic
- sensitive to initialization
- sensitive to traffic distribution

So checkpoints MUST fully restore training state.

=============================================================
"""

import os
import torch


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    iteration,
    extra=None
):
    """
    Save full training state.
    """

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    checkpoint = {

        # model parameters
        "model_state_dict": model.state_dict(),

        # optimizer state
        "optimizer_state_dict": optimizer.state_dict(),

        # training metadata
        "iteration": iteration,

        # optional metadata
        "extra": extra if extra is not None else {}

    }

    torch.save(checkpoint, path)

    print(f"[CHECKPOINT] Saved to {path}")


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(
    path,
    model,
    optimizer=None,
    map_location="cpu"
):
    """
    Load checkpoint and restore training state.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location=map_location
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None and "optimizer_state_dict" in checkpoint:

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    iteration = checkpoint.get("iteration", 0)

    extra = checkpoint.get("extra", {})

    print(f"[CHECKPOINT] Loaded from {path}")

    return model, optimizer, iteration, extra


# ============================================================
# SAFE RESUME CHECK
# ============================================================

def resume_if_available(
    path,
    model,
    optimizer=None,
    map_location="cpu"
):
    """
    Resume training if checkpoint exists.
    """

    if os.path.exists(path):

        return load_checkpoint(
            path,
            model,
            optimizer,
            map_location
        )

    print("[CHECKPOINT] No checkpoint found, starting fresh")

    return model, optimizer, 0, {}