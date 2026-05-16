import torch

from models.hierarchical_policy_good import HierarchicalPolicy
from utils.checkpoint import load_checkpoint
from env.dummy_rmsaenv_good import RMSAEnv


def run_inference():

    device = "cuda"

    # --------------------------------------------------
    # ENV
    # --------------------------------------------------

    env = RMSAEnv()

    # --------------------------------------------------
    # POLICY
    # --------------------------------------------------

    policy = HierarchicalPolicy(...).to(device)

    load_checkpoint(
        model=policy,
        path="checkpoints/latest.pt"
    )

    policy.eval()

    # --------------------------------------------------
    # EPISODE LOOP
    # --------------------------------------------------

    obs = env.reset()

    done = False

    total_reward = 0

    while not done:

        with torch.no_grad():

            action_dict = policy.act(
                obs,
                deterministic=True
            )

        obs, reward, done, info = env.step(action_dict)

        total_reward += reward

    print("reward =", total_reward)