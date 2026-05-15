import torch
import numpy as np

from env.dummy_rmsaenv_good import DummyRMSAEnv
from models.hierarchical_policy_good import HierarchicalRMSAPolicy
from ppo.rollout_worker_good_with_ids import RolloutWorker
from ppo.buffer_good import HierarchicalReplayBuffer
from ppo.trainer_good import PPOTrainer


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cpu"
NUM_EPISODES = 20


# ============================================================
# MAIN TEST
# ============================================================

def run_sanity_test():

    print("\n==============================")
    print("RMSA HIERARCHICAL PPO SANITY TEST")
    print("==============================\n")

    # --------------------------------------------------------
    # INIT ENV + POLICY
    # --------------------------------------------------------

    env = DummyRMSAEnv()

    policy = HierarchicalRMSAPolicy(
        edge_dim=4,
        slot_dim=3,
        path_feature_dim=10,
        num_paths=10,
        num_mods=6
    )

    worker = RolloutWorker(env, policy)

    buffer = HierarchicalReplayBuffer()

    trainer = PPOTrainer(policy)

    # --------------------------------------------------------
    # COLLECT DATA
    # --------------------------------------------------------

    rewards_over_time = []

    print("Collecting rollouts...\n")

    for i in range(NUM_EPISODES):

        episode = worker.collect_episode()

        buffer.add_episode(episode)

        rewards_over_time.append(sum(episode["rewards"]))

        print(f"Episode {i:02d} | Reward: {rewards_over_time[-1]:.4f}")

    batch = buffer.get_batch()

    # --------------------------------------------------------
    # BASIC CONSISTENCY CHECKS
    # --------------------------------------------------------

    print("\nChecking batch consistency...\n")

    assert len(batch["path_actions"]) == len(batch["mod_actions"]) == len(batch["slot_actions"]), \
        "Action length mismatch"

    assert len(batch["stage_ids"]) > 0, "Missing stage_ids"

    assert not torch.isnan(batch["rewards"]).any(), "NaN rewards detected"

    print("✔ Batch structure OK")

    # --------------------------------------------------------
    # POLICY FORWARD CHECK
    # --------------------------------------------------------

    print("\nRunning forward_ppo check...\n")

    with torch.no_grad():
        outputs = policy.forward_ppo(batch)

    for k, v in outputs.items():
        if torch.isnan(v).any():
            raise ValueError(f"NaN detected in {k}")

    print("✔ forward_ppo OK")

    # --------------------------------------------------------
    # TRAIN STEP CHECK
    # --------------------------------------------------------

    print("\nRunning training step...\n")

    stats = trainer.train_step(batch)

    for k, v in stats.items():
        print(f"{k}: {v:.6f}")

    print("\n✔ Training step OK")

    # --------------------------------------------------------
    # LEARNING SIGNAL CHECK
    # --------------------------------------------------------

    print("\nAnalyzing reward signal...\n")

    rewards = np.array(rewards_over_time)

    print(f"Mean reward: {rewards.mean():.4f}")
    print(f"Std reward : {rewards.std():.4f}")
    print(f"Max reward : {rewards.max():.4f}")
    print(f"Min reward : {rewards.min():.4f}")

    # --------------------------------------------------------
    # STAGE DISTRIBUTION CHECK
    # --------------------------------------------------------

    stage_counts = torch.bincount(batch["stage_ids"])

    print("\nStage distribution:")
    print(f"Path stage (0): {stage_counts[0].item()}")
    print(f"Mod  stage (1): {stage_counts[1].item()}")
    print(f"Slot stage (2): {stage_counts[2].item()}")

    # --------------------------------------------------------
    # FINAL HEALTH CHECK
    # --------------------------------------------------------

    print("\n==============================")
    print("SANITY TEST COMPLETE ✔")
    print("==============================\n")


if __name__ == "__main__":

    run_sanity_test()