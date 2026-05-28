# ============================================================
# TRAIN.PY — CLEAN PPO PIPELINE FOR RMSA
# ============================================================

import torch
import numpy as np
import random

from custom_env.CustomRLenv.CustomRMSAEnv import CustomRMSAEnv
from custom_env.CustomRLenv.utils import get_topology
from env import constant

from DRL.models.hierarchical_policy import HierarchicalRMSAPolicy
from DRL.ppo.rollout_worker import RolloutWorker
from DRL.ppo.buffer import HierarchicalRolloutBuffer
from DRL.ppo.trainer import PPOTrainer
from DRL.utils.logging import Logger

import tqdm
import datetime

# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = 'cpu'
print(f"\nUsing device: {DEVICE}\n")


# ============================================================
# CONFIG
# ============================================================

LOAD = 20
EPISODE_LENGTH = 50
MEAN_SERVICE_HOLDING_TIME = 200
NUM_SPECTRUM_RESOURCES = 100

NUM_ITERATIONS = 100
ROLLOUT_SIZE = 50#4096
MINI_BATCH_SIZE = 10

LR = 3e-4
CLIP_EPS = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01

HIDDEN_DIM = 128




# ============================================================
# ENVIRONMENT
# ============================================================

topology = get_topology(
    './data/germany/sndlib_germany.txt',
    'Germany',
    sndformat=True,
    alpha=1
)

now = datetime.datetime.now()
log_filename = now.strftime("%Y-%m-%d_%H-%M-%S")+".txt"
debug_filename = now.strftime("DEBUG_%Y-%m-%d_%H-%M-%S")+".txt"
logger = Logger()
logger.set_log_file(log_filename, debug_filename, 'log')


env_args = dict(
    topology=topology,
    seed=SEED,
    allow_rejection=True,
    load=LOAD,
    mean_service_holding_time=MEAN_SERVICE_HOLDING_TIME,
    episode_length=EPISODE_LENGTH,
    num_spectrum_resources=NUM_SPECTRUM_RESOURCES,
    bit_rates=constant.bit_rates,
    bit_rate_selection="discrete",
)

env = CustomRMSAEnv(**env_args)
env.logger=logger

# ============================================================
# MODEL
# ============================================================

num_paths = env.max_num_path
num_mods = len(env.topology.graph["modulations"])

edge_dim = 4
path_feature_dim = 1 + 1 + 2 * num_mods
slot_dim = 2


policy = HierarchicalRMSAPolicy(
    edge_dim=edge_dim,
    slot_dim=slot_dim,
    path_feature_dim=path_feature_dim,
    num_paths=num_paths,
    num_mods=num_mods,
    hidden_dim=HIDDEN_DIM
).to(DEVICE)


# ============================================================
# PPO COMPONENTS
# ============================================================

worker = RolloutWorker(
    env=env,
    policy=policy,
    device=DEVICE
)

buffer = HierarchicalRolloutBuffer(rollout_size=ROLLOUT_SIZE, 
                                   mini_batch_size=MINI_BATCH_SIZE)

trainer = PPOTrainer(
    policy=policy,
    logger=logger,
    lr=LR,
    mini_batch_size=MINI_BATCH_SIZE,
    clip_eps=CLIP_EPS,
    value_coef=VALUE_COEF,
    entropy_coef=ENTROPY_COEF,
    device=DEVICE
)


# ============================================================
# TRAIN LOOP
# ============================================================

print("Starting training...\n")

for iteration in range(NUM_ITERATIONS):

    buffer.clear()

    # --------------------------------------------------------
    # ROLLOUT COLLECTION (FIXED SIZE PPO)
    # --------------------------------------------------------

    last_value, rollout_info = worker.collect_rollout(buffer)

    # --------------------------------------------------------
    # BUILD BATCH
    # --------------------------------------------------------

    batch = buffer.get_batch()
    # print(batch.keys())

    # --------------------------------------------------------
    # BOOTSTRAP VALUE FOR GAE
    # --------------------------------------------------------

    # with torch.no_grad():
    #     last_value = policy.evaluate_value(batch)

    # --------------------------------------------------------
    # PPO TRAINING STEP
    # --------------------------------------------------------

    stats = trainer.train_step(batch, last_value)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    mean_reward = float(batch["rewards"].mean().item())
    
   

    print(f"[Iter {iteration:04d}] Reward = {mean_reward:.4f} service_blocking_rate = {np.mean(rollout_info['service_blocking_rate']):.4f} |bit_rate_blocking_rate = {np.mean(rollout_info['bit_rate_blocking_rate']):.4f} | avg_link_utilization = {np.mean(rollout_info['avg_link_utilization']):.2f}")
    print("LOSS:", np.mean(stats['loss_total']))
    print("VALUE LOSS", np.mean(stats['value_loss']))
    print("PATH ENTROPY", np.mean(stats['entropy_path']))
    print("MODULATION ENTROPY", np.mean(stats['entropy_mod']))
    print("SLOT ENTROPY", np.mean(stats['entropy_slot']))

    logger.log_str(f"[Iter {iteration:04d}] Reward = {mean_reward:.4f}")
    logger.log_dict(stats)

    # print(
    #     f"[Iter {iteration:04d}] "
    #     f"Reward={mean_reward:.4f} | "
    #     # f"Loss={stats['loss_total']:.4f} | "
    #     f"Value={stats['value_loss']:0.4f} /"
    #     f"PathEnt={stats['entropy_path']:.4f} | "
    #     f"ModEnt={stats['entropy_mod']:.4f} | "
    #     f"SlotEnt={stats['entropy_slot']:.4f}"
    # )

    # print(
    #     f"service_blocking_rate = {np.mean(rollout_info['service_blocking_rate']):.4f} | "
    #     f"bit_rate_blocking_rate = {np.mean(rollout_info['bit_rate_blocking_rate']):.4f} | "
    #     f"avg_link_utilization = {np.mean(rollout_info['avg_link_utilization']):.2f}"
    # )
    
    

    # --------------------------------------------------------
    # CHECKPOINTING
    # --------------------------------------------------------

    # if iteration % 100 == 0:

    #     checkpoint = {
    #         "iteration": iteration,
    #         "model_state_dict": policy.state_dict(),
    #         "optimizer_state_dict": trainer.optimizer.state_dict()
    #     }

    #     torch.save(checkpoint, f"checkpoint_{iteration}.pt")

    #     print(f"Checkpoint saved at iteration {iteration}")

logger.logger_close()
print("\nTraining complete.\n")
