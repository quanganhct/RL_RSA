# utils/masking.py

import torch


# ============================================================
# ACTION MASKING
# ============================================================

def apply_action_mask(
    logits: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Apply invalid action masking to logits.

    Parameters
    ----------
    logits : Tensor
        Shape [B, A] or [A]

    mask : Tensor
        Shape same as logits.
        1 = valid
        0 = invalid

    Returns
    -------
    masked_logits : Tensor
    """

    mask = mask.bool()

    min_value = torch.finfo(
        logits.dtype
    ).min

    masked_logits = logits.masked_fill(
        ~mask,
        min_value
    )

    return masked_logits


# ============================================================
# SAFE CATEGORICAL
# ============================================================

def masked_categorical(
    logits: torch.Tensor,
    mask: torch.Tensor
):
    """
    Create categorical distribution
    with invalid actions removed.
    """

    from torch.distributions import Categorical

    masked_logits = apply_action_mask(
        logits,
        mask
    )

    return Categorical(logits=masked_logits)


# ============================================================
# VALIDATE ACTIONS
# ============================================================

def validate_actions(
    actions: torch.Tensor,
    mask: torch.Tensor
):
    """
    Ensure sampled actions are valid.
    """

    valid = mask.gather(
        -1,
        actions.unsqueeze(-1)
    ).squeeze(-1)

    if not torch.all(valid):

        invalid_idx = torch.where(
            valid == 0
        )[0]

        raise RuntimeError(
            f"Invalid actions detected "
            f"at indices {invalid_idx}"
        )


# ============================================================
# TRANSFORMER PADDING MASK
# ============================================================

def build_padding_mask(
    sequences: torch.Tensor,
    pad_value: int = -1
):
    """
    Build transformer padding mask.

    Parameters
    ----------
    sequences : Tensor
        Shape [B, L]

    Returns
    -------
    padding_mask : BoolTensor
        True = padding
    """

    return sequences == pad_value


# ============================================================
# SEQUENCE LENGTHS
# ============================================================

def get_sequence_lengths(
    sequences: torch.Tensor,
    pad_value: int = -1
):
    """
    Compute valid sequence lengths.
    """

    return (sequences != pad_value).sum(
        dim=-1
    )


# ============================================================
# MASKED ENTROPY
# ============================================================

def masked_entropy(
    logits: torch.Tensor,
    mask: torch.Tensor
):
    """
    Stable entropy computation
    under invalid action masking.
    """

    masked_logits = apply_action_mask(
        logits,
        mask
    )

    probs = torch.softmax(
        masked_logits,
        dim=-1
    )

    log_probs = torch.log_softmax(
        masked_logits,
        dim=-1
    )

    entropy = -(
        probs * log_probs
    ).sum(dim=-1)

    return entropy


# ============================================================
# MASKED LOGPROB
# ============================================================

def masked_logprob(
    logits: torch.Tensor,
    actions: torch.Tensor,
    mask: torch.Tensor
):
    """
    Compute stable masked log-probabilities.
    """

    masked_logits = apply_action_mask(
        logits,
        mask
    )

    log_probs = torch.log_softmax(
        masked_logits,
        dim=-1
    )

    return log_probs.gather(
        -1,
        actions.unsqueeze(-1)
    ).squeeze(-1)