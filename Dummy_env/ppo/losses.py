# ppo/losses.py

"""
=============================================================
HIERARCHICAL PPO LOSSES FOR RMSA
=============================================================

This module implements:

1. Hierarchical policy loss:
      π(path) + π(mod|path) + π(slot|path,mod)

2. Value function loss (clipped)

3. Entropy regularization per stage

4. Mask-aware logprob computation

5. Stable PPO ratio computation

=============================================================
CRITICAL DESIGN RULE
=============================================================

We ALWAYS recompute log-probabilities.

We NEVER trust stored logprobs.

Because:
    - masking changes over time
    - spectrum state changes
    - path feasibility changes

=============================================================
"""

import torch
import torch.nn.functional as F

from utils.masking import (
    masked_logprob,
    masked_entropy
)


# ============================================================
# PPO CLIP FUNCTION
# ============================================================

def clip_ratio(ratio, eps=0.2):
    return torch.clamp(ratio, 1 - eps, 1 + eps)


# ============================================================
# HIERARCHICAL POLICY LOSS
# ============================================================

def compute_policy_loss(
    logprob_path,
    logprob_mod,
    logprob_slot,
    old_logprob_path,
    old_logprob_mod,
    old_logprob_slot,
    advantages,
    clip_eps=0.2
):
    """
    Hierarchical PPO loss.

    total logprob:
        π = πp + πm + πs
    """

    new_logprob = (
        logprob_path
        + logprob_mod
        + logprob_slot
    )

    old_logprob = (
        old_logprob_path
        + old_logprob_mod
        + old_logprob_slot
    )

    ratio = torch.exp(new_logprob - old_logprob)

    clipped_ratio = clip_ratio(ratio, clip_eps)

    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages

    loss = -torch.min(surr1, surr2).mean()

    return loss, ratio


# ============================================================
# VALUE LOSS (CLIPPED PPO STYLE)
# ============================================================

def compute_value_loss(
    value_pred,
    value_old,
    returns,
    clip_eps=0.2
):
    """
    PPO value clipping (very important for RMSA stability).
    """

    value_pred_clipped = value_old + torch.clamp(
        value_pred - value_old,
        -clip_eps,
        clip_eps
    )

    loss1 = (value_pred - returns) ** 2
    loss2 = (value_pred_clipped - returns) ** 2

    return 0.5 * torch.max(loss1, loss2).mean()


# ============================================================
# ENTROPY LOSS (HIERARCHICAL)
# ============================================================

def compute_entropy_loss(
    logits_path,
    logits_mod,
    logits_slot,
    mask_path,
    mask_mod,
    mask_slot,
    entropy_coef=0.01
):
    """
    Encourages exploration at all decision levels.
    """

    entropy_path = masked_entropy(logits_path, mask_path)
    entropy_mod = masked_entropy(logits_mod, mask_mod)
    entropy_slot = masked_entropy(logits_slot, mask_slot)

    entropy = (
        entropy_path
        + entropy_mod
        + entropy_slot
    ).mean()

    return -entropy_coef * entropy


# ============================================================
# FULL PPO LOSS (MAIN ENTRY POINT)
# ============================================================

def compute_ppo_loss(
    model_outputs,
    batch,
    advantages,
    returns,
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.01
):
    """
    Full RMSA PPO loss.

    model_outputs MUST contain:

        path:
            logits_path
            logprob_path
            value_path (or global value)

        mod:
            logits_mod
            logprob_mod

        slot:
            logits_slot
            logprob_slot

    batch contains:
        old_logprobs
        masks
        etc.
    """

    # --------------------------------------------------------
    # UNPACK MODEL OUTPUTS
    # --------------------------------------------------------

    logprob_path = model_outputs["logprob_path"]
    logprob_mod = model_outputs["logprob_mod"]
    logprob_slot = model_outputs["logprob_slot"]

    logits_path = model_outputs["logits_path"]
    logits_mod = model_outputs["logits_mod"]
    logits_slot = model_outputs["logits_slot"]

    value_pred = model_outputs["value"]

    # old policy values (from buffer)
    old_logprob_path = batch["old_path_logprobs"]
    old_logprob_mod = batch["old_mod_logprobs"]
    old_logprob_slot = batch["old_slot_logprobs"]

    value_old = batch["values"]

    # masks
    mask_path = batch["path_masks"]
    mask_mod = batch["mod_masks"]
    mask_slot = batch["slot_masks"]

    # --------------------------------------------------------
    # POLICY LOSS
    # --------------------------------------------------------

    policy_loss, ratio = compute_policy_loss(
        logprob_path,
        logprob_mod,
        logprob_slot,
        old_logprob_path,
        old_logprob_mod,
        old_logprob_slot,
        advantages,
        clip_eps
    )

    # --------------------------------------------------------
    # VALUE LOSS
    # --------------------------------------------------------

    value_loss = compute_value_loss(
        value_pred,
        value_old,
        returns,
        clip_eps
    )

    # --------------------------------------------------------
    # ENTROPY LOSS
    # --------------------------------------------------------

    entropy_loss = compute_entropy_loss(
        logits_path,
        logits_mod,
        logits_slot,
        mask_path,
        mask_mod,
        mask_slot,
        entropy_coef
    )

    # --------------------------------------------------------
    # TOTAL LOSS
    # --------------------------------------------------------

    total_loss = (
        policy_loss
        + value_coef * value_loss
        + entropy_loss
    )

    return {
        "loss": total_loss,
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "entropy": (-entropy_loss).detach(),
        "ratio": ratio.detach().mean()
    }


# ============================================================
# AUXILIARY: DEBUG STABILITY CHECK
# ============================================================

def check_loss_health(metrics):
    """
    Detect common PPO failure modes in RMSA.
    """

    ratio = metrics["ratio"].item()

    if ratio > 2.0:
        print("[WARNING] Policy collapse risk: ratio too high")

    if ratio < 0.5:
        print("[WARNING] Policy under-updating")

    if torch.isnan(metrics["loss"]):
        raise RuntimeError("NaN detected in PPO loss")

