import torch
import torch.nn.functional as F
import torch.distributions as D

# ============================================================
# PPO CLIPPED LOSS
# ============================================================

def clipped_surrogate_loss(new_logits, old_logprob, actions, advantage, clip_eps=0.2):

    new_dist = D.Categorical(logits=new_logits)    
    new_logprob = new_dist.log_prob(actions)
    
    ratio = torch.exp(new_logprob - old_logprob)

    clipped_ratio = torch.clamp(
        ratio,
        1 - clip_eps,
        1 + clip_eps
    )

    loss = -torch.min(
        ratio * advantage,
        clipped_ratio * advantage
    )

    return loss.mean()


# ============================================================
# ENTROPY BONUS (LOGITS BASED)
# ============================================================

def entropy_bonus(logits):

    probs = F.softmax(logits, dim=-1)

    entropy = -torch.sum(
        probs * torch.log(probs + 1e-8),
        dim=-1
    )

    return entropy.mean()


# ============================================================
# VALUE LOSS
# ============================================================

def value_loss(pred_value, target_value):

    return F.mse_loss(pred_value, target_value)


# ============================================================
# CLEAN HIERARCHICAL PPO LOSS (UPDATED)
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
    CLEAN PPO LOSS FOR RMSA (FIXED ARCHITECTURE)
    ============================================================

    One timestep = one RMSA decision:
        (path, modulation, slot)

    PPO uses:
        stored logprobs vs new logprobs

    ============================================================
    """

    clip_eps = config.get("clip_eps", 0.2)
    entropy_coef = config.get("entropy_coef", 0.01)
    value_coef = config.get("value_coef", 0.5)

    # ============================================================
    # PATH LOSS
    # ============================================================

    path_loss = clipped_surrogate_loss(
        outputs["logits_path"],
        batch["path_logprobs"],
        batch["path_actions"],
        advantages,
        clip_eps
    )

    path_entropy = entropy_bonus(outputs["logits_path"])

    # ============================================================
    # MODULATION LOSS
    # ============================================================

    mod_loss = clipped_surrogate_loss(
        outputs["logits_mod"],
        batch["mod_logprobs"],
        batch["mod_actions"],
        advantages,
        clip_eps
    )

    mod_entropy = entropy_bonus(outputs["logits_mod"])

    # ============================================================
    # SLOT LOSS
    # ============================================================

    slot_loss = clipped_surrogate_loss(
        outputs["logits_slot"],
        batch["slot_logprobs"],
        batch["slot_actions"],
        advantages,
        clip_eps
    )

    slot_entropy = entropy_bonus(outputs["logits_slot"])

    # ============================================================
    # VALUE LOSS
    # ============================================================

    v_loss = value_loss(
        outputs["value"].squeeze(-1),
        returns
    )

    # ============================================================
    # TOTAL LOSS
    # ============================================================

    total_loss = (
        path_loss
        + mod_loss
        + slot_loss
        + value_coef * v_loss
        - entropy_coef * (
            path_entropy
            + mod_entropy
            + slot_entropy
        )
    )

    # ============================================================
    # STATS
    # ============================================================

    stats = {

        "loss_total": total_loss.item(),

        "loss_path": path_loss.item(),
        "loss_mod": mod_loss.item(),
        "loss_slot": slot_loss.item(),

        "value_loss": v_loss.item(),

        "entropy_path": path_entropy.item(),
        "entropy_mod": mod_entropy.item(),
        "entropy_slot": slot_entropy.item(),
    }

    return total_loss, stats