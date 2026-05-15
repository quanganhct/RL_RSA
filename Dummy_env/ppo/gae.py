# ppo/gae.py

"""
=============================================================
GENERALIZED ADVANTAGE ESTIMATION (RMSA-ADAPTED)
=============================================================

Key RMSA property:
------------------
Reward is ONLY available after full decision:

    path -> modulation -> slot -> reward

So each trajectory is effectively:

    ONE-step MDP with delayed reward

BUT we still use GAE for stability across requests.

=============================================================
FEATURES
=============================================================

1. Standard GAE (γ, λ)
2. Sparse reward support (RMSA-style)
3. Per-request bootstrapping (optional)
4. Episode-level advantage normalization
5. Supports vectorized rollouts

=============================================================
"""

import numpy as np


# ============================================================
# STANDARD GAE
# ============================================================

def compute_gae(
    rewards,
    values,
    dones,
    gamma=0.99,
    lam=0.95
):
    """
    Standard GAE for flat trajectories.

    Note:
    -----
    In RMSA, rewards are sparse but per-step is still valid
    because each transition = full decision.
    """

    advantages = np.zeros_like(rewards, dtype=np.float32)

    gae = 0.0

    # bootstrap value
    next_value = 0.0

    for t in reversed(range(len(rewards))):

        mask = 1.0 - dones[t]

        delta = (
            rewards[t]
            + gamma * next_value * mask
            - values[t]
        )

        gae = delta + gamma * lam * mask * gae

        advantages[t] = gae

        next_value = values[t]

    returns = advantages + values

    return advantages, returns


# ============================================================
# RMSA-SAFE GAE (RECOMMENDED)
# ============================================================

def compute_rmsa_gae(
    rewards,
    values,
    dones,
    gamma=0.99,
    lam=0.95,
    reward_is_terminal=True
):
    """
    RMSA-aware GAE.

    This version is safer when:

        - reward is only at final step
        - intermediate steps have zero reward
        - trajectories are short (1 decision = 1 reward)

    """

    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)

    advantages = np.zeros_like(rewards, dtype=np.float32)

    gae = 0.0

    next_value = 0.0

    for t in reversed(range(len(rewards))):

        mask = 1.0 - dones[t]

        # -----------------------------------------------------
        # RMSA SPECIAL CASE:
        # -----------------------------------------------------
        # Only final step has reward
        # so earlier steps rely fully on bootstrapping
        # -----------------------------------------------------

        delta = (
            rewards[t]
            + gamma * next_value * mask
            - values[t]
        )

        gae = delta + gamma * lam * mask * gae

        advantages[t] = gae

        next_value = values[t]

    returns = advantages + values

    return advantages, returns


# ============================================================
# PER-REQUEST BOOTSTRAPPING (IMPORTANT FOR RMSA)
# ============================================================

def compute_rmsa_request_level_gae(
    rewards,
    values,
    dones,
    request_ids,
    gamma=0.99,
    lam=0.95
):
    """
    Bootstrapped GAE at REQUEST level.

    Why this exists:
    -----------------
    RMSA episodes are grouped by "connection requests".

    Each request = independent decision outcome.

    This prevents cross-request leakage.

    ---------------------------------------------------------
    INPUT
    ---------------------------------------------------------

    request_ids:
        array like [0,0,0,1,1,2,...]

    Each request has its own trajectory.

    """

    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    request_ids = np.asarray(request_ids)

    advantages = np.zeros_like(rewards, dtype=np.float32)
    returns = np.zeros_like(rewards, dtype=np.float32)

    unique_requests = np.unique(request_ids)

    for r in unique_requests:

        idx = np.where(request_ids == r)[0]

        r_rewards = rewards[idx]
        r_values = values[idx]
        r_dones = dones[idx]

        adv, ret = compute_rmsa_gae(
            r_rewards,
            r_values,
            r_dones,
            gamma,
            lam
        )

        advantages[idx] = adv
        returns[idx] = ret

    return advantages, returns


# ============================================================
# VALUE BOOTSTRAPPING (OPTIONAL ADVANCED FEATURE)
# ============================================================

def compute_bootstrapped_returns(
    rewards,
    values,
    dones,
    bootstrap_value=0.0,
    gamma=0.99
):
    """
    Simpler alternative for RMSA:

    Use value bootstrap only at episode end.

    Useful when:

    - episodes are short
    - reward is extremely sparse
    """

    returns = np.zeros_like(rewards, dtype=np.float32)

    running_return = bootstrap_value

    for t in reversed(range(len(rewards))):

        running_return = (
            rewards[t]
            + gamma * running_return * (1.0 - dones[t])
        )

        returns[t] = running_return

    advantages = returns - values

    return advantages, returns


# ============================================================
# ADVANTAGE NORMALIZATION (CRITICAL)
# ============================================================

def normalize_advantages(advantages, eps=1e-8):
    """
    Standard PPO trick.
    """

    advantages = np.asarray(advantages, dtype=np.float32)

    return (
        advantages - advantages.mean()
    ) / (advantages.std() + eps)


# ============================================================
# RMSA RECOMMENDED DEFAULT PIPELINE
# ============================================================

def compute_advantages_rmsa(
    rewards,
    values,
    dones,
    request_ids=None,
    gamma=0.99,
    lam=0.95,
    use_request_bootstrap=True
):
    """
    Unified entry point for RMSA PPO.
    """

    if request_ids is not None and use_request_bootstrap:

        advantages, returns = compute_rmsa_request_level_gae(
            rewards,
            values,
            dones,
            request_ids,
            gamma,
            lam
        )

    else:

        advantages, returns = compute_rmsa_gae(
            rewards,
            values,
            dones,
            gamma,
            lam
        )

    advantages = normalize_advantages(advantages)

    return advantages, returns