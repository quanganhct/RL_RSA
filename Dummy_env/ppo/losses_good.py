import torch
import torch.nn.functional as F


# ============================================================
# PPO CLIP FUNCTION
# ============================================================

def clipped_surrogate_loss(new_logprob, old_logprob, advantage, clip_eps=0.2):

    ratio = torch.exp(new_logprob - old_logprob)

    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)

    loss = -torch.min(
        ratio * advantage,
        clipped_ratio * advantage
    )

    return loss.mean()


# ============================================================
# ENTROPY BONUS
# ============================================================

def entropy_bonus(logits):

    probs = F.softmax(logits, dim=-1)

    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)

    return entropy.mean()


# ============================================================
# VALUE LOSS
# ============================================================

def value_loss(pred_value, target_value):

    return F.mse_loss(pred_value, target_value)


# ============================================================
# HIERARCHICAL PPO LOSS
# ============================================================

def compute_hierarchical_loss(
    outputs,
    batch,
    advantages,
    returns,
    config
):
    """
    ============================================================
    HIERARCHICAL PPO LOSS FOR RMSA
    ============================================================

    outputs:
        from forward_ppo():
            logits_path
            logits_mod
            logits_slot
            value

    batch:
        stored actions + logprobs

    advantages:
        computed from GAE

    returns:
        discounted returns

    ============================================================
    """

    clip_eps = config.get("clip_eps", 0.2)
    entropy_coef = config.get("entropy_coef", 0.01)
    value_coef = config.get("value_coef", 0.5)

    # ------------------------------------------------------------
    # PATH LOSS
    # ------------------------------------------------------------

    path_loss = clipped_surrogate_loss(
        outputs["logits_path"][torch.arange(len(batch["path_actions"])), batch["path_actions"]],
        batch["path_logprobs"],
        advantages["path"],
        clip_eps
    )

    path_entropy = entropy_bonus(outputs["logits_path"])

    # ------------------------------------------------------------
    # MODULATION LOSS
    # ------------------------------------------------------------

    mod_loss = clipped_surrogate_loss(
        outputs["logits_mod"][torch.arange(len(batch["mod_actions"])), batch["mod_actions"]],
        batch["mod_logprobs"],
        advantages["mod"],
        clip_eps
    )

    mod_entropy = entropy_bonus(outputs["logits_mod"])

    # ------------------------------------------------------------
    # SLOT LOSS
    # ------------------------------------------------------------

    slot_loss = clipped_surrogate_loss(
        outputs["logits_slot"][torch.arange(len(batch["slot_actions"])), batch["slot_actions"]],
        batch["slot_logprobs"],
        advantages["slot"],
        clip_eps
    )

    slot_entropy = entropy_bonus(outputs["logits_slot"])

    # ------------------------------------------------------------
    # VALUE LOSS
    # ------------------------------------------------------------

    v_loss = value_loss(
        outputs["value"],
        returns
    )

    # ------------------------------------------------------------
    # TOTAL LOSS (HIERARCHICAL WEIGHTING)
    # ------------------------------------------------------------

    total_loss = (
        path_loss
        + mod_loss
        + slot_loss
        + value_coef * v_loss
        - entropy_coef * (path_entropy + mod_entropy + slot_entropy)
    )

    # ------------------------------------------------------------
    # DIAGNOSTICS
    # ------------------------------------------------------------

    stats = {

        "loss_total": total_loss.item(),

        "loss_path": path_loss.item(),
        "loss_mod": mod_loss.item(),
        "loss_slot": slot_loss.item(),

        "value_loss": v_loss.item(),

        "entropy_path": path_entropy.item(),
        "entropy_mod": mod_entropy.item(),
        "entropy_slot": slot_entropy.item()
    }

    return total_loss, stats