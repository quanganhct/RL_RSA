import torch


# ============================================================
# STANDARD PPO GAE
# ============================================================



# ============================================================
# GENERALIZED ADVANTAGE ESTIMATION (GAE)
# ============================================================
def compute_gae(
    rewards,
    values,
    dones,
    last_value,
    gamma=0.99,
    gae_lambda=0.95,
    normalize=True
):

    rewards = rewards.view(-1)
    values = values.view(-1)
    dones = dones.view(-1)

    last_value = last_value.view(1)

    device = rewards.device

    T = rewards.shape[0]

    advantages = torch.zeros(
        T,
        dtype=torch.float32,
        device=device
    )

    gae = 0.0

    for t in reversed(range(T)):

        non_terminal = 1.0 - dones[t]

        if t == T - 1:
            next_value = last_value[0]
        else:
            next_value = values[t + 1]

        delta = (
            rewards[t]
            + gamma * next_value * non_terminal
            - values[t]
        )

        gae = (
            delta
            + gamma * gae_lambda * non_terminal * gae
        )

        advantages[t] = gae

    returns = advantages + values

    if normalize:
        advantages = (
            advantages - advantages.mean()
        ) / (
            advantages.std() + 1e-8
        )

    return advantages, returns


def compute_gae2(
    rewards,
    values,
    dones,
    last_value,
    gamma=0.99,
    gae_lambda=0.95,
    normalize=True
):
    """
    ============================================================
    CLEAN PPO GAE
    ============================================================

    Computes:

        advantages
        returns

    Supports:
    ----------
    ✔ truncated rollouts
    ✔ PPO minibatching
    ✔ batched rollout collection
    ✔ stable normalization

    ============================================================
    INPUTS
    ============================================================

    rewards:
        [T]

    values:
        [T]

    dones:
        [T]

    last_value:
        scalar tensor
        critic bootstrap value for final state

    ============================================================
    OUTPUTS
    ============================================================

    advantages:
        [T]

    returns:
        [T]

    ============================================================
    """

    device = rewards.device

    T = rewards.shape[0]

    advantages = torch.zeros(
        T,
        dtype=torch.float32,
        device=device
    )

    gae = 0.0

    # --------------------------------------------------------
    # REVERSE TIME LOOP
    # --------------------------------------------------------

    for t in reversed(range(T)):

        # ----------------------------------------------------
        # TERMINAL MASK
        # ----------------------------------------------------

        non_terminal = 1.0 - dones[t]

        # ----------------------------------------------------
        # BOOTSTRAP VALUE
        # ----------------------------------------------------

        if t == T - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]

        # ----------------------------------------------------
        # TD ERROR
        # ----------------------------------------------------

        delta = (
            rewards[t]
            + gamma * next_value * non_terminal
            - values[t]
        )

        # ----------------------------------------------------
        # GAE RECURSION
        # ----------------------------------------------------

        gae = (
            delta
            + gamma
            * gae_lambda
            * non_terminal
            * gae
        )

        advantages[t] = gae

    # --------------------------------------------------------
    # RETURNS
    # --------------------------------------------------------

    returns = advantages + values

    # --------------------------------------------------------
    # ADVANTAGE NORMALIZATION
    # --------------------------------------------------------

    if normalize:

        advantages = (
            advantages - advantages.mean()
        ) / (
            advantages.std() + 1e-8
        )

    return advantages, returns


def compute_gae1(
    rewards,
    values,
    dones,
    last_value,
    gamma=0.99,
    lam=0.95,
    normalize=True
):
    """
    ============================================================
    GENERALIZED ADVANTAGE ESTIMATION (PPO)
    ============================================================

    Supports:
    ----------
    - truncated rollouts
    - infinite horizon environments
    - PPO rollout buffers

    Parameters
    ----------
    rewards : tensor [T]
    values  : tensor [T]
    dones   : tensor [T]
    last_value : scalar tensor

        Bootstrap critic value for final state.

        IMPORTANT:
            If rollout truncated before terminal state,
            this must be critic(last_obs).

            If truly terminal:
                last_value = 0

    ============================================================
    """

    rewards = rewards.float()
    values = values.float()
    dones = dones.float()

    T = len(rewards)

    advantages = torch.zeros_like(rewards)

    gae = 0.0

    next_value = last_value

    for t in reversed(range(T)):

        non_terminal = 1.0 - dones[t]

        delta = (
            rewards[t]
            + gamma * next_value * non_terminal
            - values[t]
        )

        gae = (
            delta
            + gamma * lam * non_terminal * gae
        )

        advantages[t] = gae

        next_value = values[t]

    returns = advantages + values

    # --------------------------------------------------------
    # ADVANTAGE NORMALIZATION
    # --------------------------------------------------------

    if normalize:

        advantages = (
            (advantages - advantages.mean())
            / (advantages.std() + 1e-8)
        )

    return advantages, returns


# ============================================================
# HIERARCHICAL RMSA GAE
# ============================================================

def compute_hierarchical_gae(
    rewards,
    values,
    dones,
    last_value,
    gamma=0.99,
    lam=0.95,
    path_reward_scale=0.2,
    mod_reward_scale=0.5,
    normalize=True
):
    """
    ============================================================
    HIERARCHICAL GAE FOR RMSA
    ============================================================

    stage_ids:
        0 = path stage
        1 = modulation stage
        2 = slot stage

    Reward shaping:
    ----------------
    path stage:
        reward *= path_reward_scale

    modulation stage:
        reward *= mod_reward_scale

    slot stage:
        full reward

    ============================================================
    """

    rewards = rewards.float()
    values = values.float()
    dones = dones.float()

    T = len(rewards)

    advantages = torch.zeros_like(rewards)

    gae = 0.0

    next_value = last_value

    for t in reversed(range(T)):

        non_terminal = 1.0 - dones[t]


        # TD error

        delta = (
            rewards[t]
            + gamma * next_value * non_terminal
            - values[t]
        )
        
        # GAE recursion 

        gae = (
            delta
            + gamma * lam * non_terminal * gae
        )

        advantages[t] = gae

        next_value = values[t]

    returns = advantages + values

    # --------------------------------------------------------
    # ADVANTAGE NORMALIZATION
    # --------------------------------------------------------

    if normalize:

        advantages = (
            (advantages - advantages.mean())
            / (advantages.std() + 1e-8)
        )

    return advantages, returns