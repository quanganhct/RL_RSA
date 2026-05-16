import torch


# ============================================================
# STANDARD GAE (GENERIC)

# advantages, returns = compute_hierarchical_gae(
#     rewards,
#     values,
#     dones,
#     stage_ids=batch["stage_ids"]
# )
# ============================================================

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):

    """
    Standard Generalized Advantage Estimation.

    rewards: [T]
    values:  [T]
    dones:   [T]
    """

    T = len(rewards)

    advantages = torch.zeros(T)
    last_gae = 0.0

    for t in reversed(range(T)):

        if t == T - 1:
            next_value = 0.0
            next_non_terminal = 1.0 - dones[t]
        else:
            next_value = values[t + 1]
            next_non_terminal = 1.0 - dones[t]

        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]

        last_gae = delta + gamma * lam * next_non_terminal * last_gae

        advantages[t] = last_gae

    returns = advantages + values

    return advantages, returns


# ============================================================
# HIERARCHICAL GAE (RMSA VERSION)
# ============================================================

def compute_hierarchical_gae(
    rewards,
    values,
    dones,
    stage_ids,
    gamma=0.99,
    lam=0.95
):
    """
    ============================================================
    HIERARCHICAL GAE FOR RMSA
    ============================================================

    stage_ids:
        tensor [T] with values:
            0 = path stage
            1 = modulation stage
            2 = slot stage (reward delivered here)

    KEY IDEA:
    --------
    - Only slot stage receives real reward
    - path + mod receive propagated credit

    ============================================================
    """

    T = len(rewards)

    advantages = torch.zeros(T)
    returns = torch.zeros(T)

    last_gae = 0.0

    for t in reversed(range(T)):

        # --------------------------------------------------------
        # BOOTSTRAP VALUE
        # --------------------------------------------------------

        if t == T - 1:
            next_value = 0.0
            non_terminal = 1.0 - dones[t]
        else:
            next_value = values[t + 1]
            non_terminal = 1.0 - dones[t]

        # --------------------------------------------------------
        # STAGE-AWARE REWARD SHAPING
        # --------------------------------------------------------

        reward = rewards[t]

        stage = stage_ids[t]

        # --------------------------------------------------------
        # IMPORTANT RMSA DESIGN CHOICE:
        # --------------------------------------------------------
        # - slot stage gets full reward
        # - path/mod get discounted shaping signal
        # --------------------------------------------------------

        if stage == 0:
            reward *= 0.2   # path influence
        elif stage == 1:
            reward *= 0.5   # modulation influence
        # stage 2 = full reward

        # --------------------------------------------------------
        # TD ERROR
        # --------------------------------------------------------

        delta = reward + gamma * next_value * non_terminal - values[t]

        last_gae = delta + gamma * lam * non_terminal * last_gae

        advantages[t] = last_gae
        returns[t] = advantages[t] + values[t]

    return advantages, returns


# ============================================================
# OPTIONAL: PER-EPISODE BOOTSTRAP VERSION
# ============================================================

def compute_bootstrapped_gae(
    rewards,
    values,
    dones,
    bootstrap_value=0.0,
    gamma=0.99,
    lam=0.95
):

    """
    Used when episodes are truncated (not fully done).
    """

    T = len(rewards)

    advantages = torch.zeros(T)
    last_gae = 0.0

    for t in reversed(range(T)):

        next_value = bootstrap_value if t == T - 1 else values[t + 1]

        non_terminal = 1.0 - dones[t]

        delta = rewards[t] + gamma * next_value * non_terminal - values[t]

        last_gae = delta + gamma * lam * non_terminal * last_gae

        advantages[t] = last_gae

    returns = advantages + values

    return advantages, returns