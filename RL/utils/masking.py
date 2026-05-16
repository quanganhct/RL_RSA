import torch


# ============================================================
# BASIC SAFE MASKING UTILS
# ============================================================

def apply_mask(logits, mask, invalid_value=-1e9):
    """
    Applies boolean mask to logits.

    mask:
        1 = valid
        0 = invalid
    """

    mask = mask.to(dtype=torch.bool)

    return logits.masked_fill(~mask, invalid_value)


# ============================================================
# PATH MASKING
# ============================================================

def get_path_mask(env_obs):
    """
    Mask valid paths.

    In real RMSA:
        - path may be invalid due to congestion or disconnection
        - here we simulate via env-provided mask
    """

    return env_obs["action_masks"]["path"]


# ============================================================
# MODULATION MASKING (DEPENDENT ON PATH)
# ============================================================

def get_modulation_mask(env_obs, selected_path=None):
    """
    In real RMSA:
        modulation depends on:
            - path length
            - OSNR
            - distance

    In dummy env:
        mask already precomputed but may depend on state.
    """

    return env_obs["action_masks"]["mod"]


# ============================================================
# SLOT MASKING (DEPENDENT ON PATH + MODULATION)
# ============================================================

def get_slot_mask(env_obs, selected_path=None, selected_mod=None):
    """
    Critical RMSA constraint layer.

    In real system:
        slot feasibility depends on:
            - spectrum continuity
            - guard bands
            - fragmentation
            - modulation bandwidth

    Here:
        delegated to env.
    """

    return env_obs["action_masks"]["slot"]


# ============================================================
# HIERARCHICAL MASK BUILDER
# ============================================================

def build_hierarchical_masks(env_obs, selected_path=None, selected_mod=None):

    return {
        "path": get_path_mask(env_obs),
        "mod": get_modulation_mask(env_obs, selected_path),
        "slot": get_slot_mask(env_obs, selected_path, selected_mod)
    }


# ============================================================
# SAFE LOGIT MASKING FOR ALL STAGES
# ============================================================

def mask_hierarchical_logits(logits_dict, masks):

    """
    Applies masks to all policy heads safely.
    """

    return {
        "path": apply_mask(logits_dict["path"], masks["path"]),
        "mod": apply_mask(logits_dict["mod"], masks["mod"]),
        "slot": apply_mask(logits_dict["slot"], masks["slot"])
    }