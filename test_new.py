# ============================================================
# TRAIN.PY â€” CLEAN PPO PIPELINE FOR RMSA
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
from DRL.utils.csv_writer import CSVWriter

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

LOAD = 80# 300
EPISODE_LENGTH = 10
MEAN_SERVICE_HOLDING_TIME = 200
NUM_SPECTRUM_RESOURCES = 380#300

NUM_ITERATIONS = 2
ROLLOUT_SIZE = EPISODE_LENGTH#4096
MINI_BATCH_SIZE = 128#64

LR = 3e-4
CLIP_EPS = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01

HIDDEN_DIM = 128




# ============================================================
# ENVIRONMENT
# ============================================================


topology_data = [dict(file_name='./data/usa/backbone.txt', topology_name='USA', sndformat=False, undirected_file=False),
                 dict(file_name='./data/germany/sndlib_germany.txt', topology_name='Germany', sndformat=True),
                 dict(file_name='./data/european/european.txt', topology_name='European', sndformat=False, undirected_file=False),
                 dict(file_name='./data/nsf/nsfnet_chen.txt', topology_name='NSF', sndformat=False, undirected_file=True)]

# topology_data = [dict(file_name='./data/germany/sndlib_germany.txt', topology_name='Germany', sndformat=True),\
#                  dict(file_name='./data/european/european.txt', topology_name='European', sndformat=False, undirected_file=False),\
#                  dict(file_name='./data/nsf/nsfnet_chen.txt', topology_name='NSF', sndformat=False, undirected_file=True),\
#                  dict(file_name='./data/usa/backbone.txt', topology_name='USA', sndformat=False, undirected_file=False)]

# loads = [80, 100, 200, 300, 500]

# topology_data = [dict(file_name='./data/germany/sndlib_germany.txt', topology_name='Germany', sndformat=True)]

loads = [200]


# topology_data = [#dict(file_name='./data/germany/sndlib_germany.txt', topology_name='Germany', sndformat=True),\
#                   dict(file_name='./data/european/european.txt', topology_name='European', sndformat=False, undirected_file=False),\
#                   ]

# loads = [80]

now = datetime.datetime.now()
writer = CSVWriter('Test_load_result_%s.csv'%(now.strftime("%Y-%m-%d_%H-%M-%S")), 'log')

writer.write(['topology_name', 'load', 'Reward', 'service_blocking_rate', 
              'episode_service_blocking_rate',
              'our_service_blocking_rate',
              'bit_rate_blocking_rate',
              'episode_bit_rate_blocking_rate',
              'avg_link_utilization',
              'num_accepted_request',
              'num_total_request',
              'done'])

for arg in topology_data:
    # topology = get_topology(
    #     './data/germany/sndlib_germany.txt',
    #     'Germany',
    #     sndformat=True,
    #     alpha=1
    # )
    
    topology = get_topology(**arg, alpha=1)


    for load in loads:
        now = datetime.datetime.now()
        log_filename = now.strftime("TEST%Y-%m-%d_%H-%M-%S")+".txt"
        debug_filename = now.strftime("TESTDEBUG_%Y-%m-%d_%H-%M-%S")+".txt"
        logger = Logger()
        logger.set_log_file(log_filename, debug_filename, 'log')
        
        
        env_args = dict(
            topology=topology,
            seed=SEED,
            allow_rejection=True,
            load=load,
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
        candidate_feature_dim = 1
        path_feature_dim = 1 + 1 + 2 * num_mods
        slot_dim = 2
        
        
        policy = HierarchicalRMSAPolicy(
            edge_dim=edge_dim,
            slot_dim=slot_dim,
            path_feature_dim=path_feature_dim,
            candidate_feature_dim=candidate_feature_dim,
            num_paths=num_paths,
            num_mods=num_mods,
            hidden_dim=HIDDEN_DIM
        ).to(DEVICE)
        
        chekpoint = torch.load('checkpoint_iter_30_rate_0.12903225806451613.pt', weights_only=True)
        policy.load_state_dict(chekpoint['model_state_dict'])
        # ============================================================
        # PPO COMPONENTS
        # ============================================================
        
        worker = RolloutWorker(
            env=env,
            policy=policy,
            logger=logger,
            device=DEVICE
        )
        
        buffer = HierarchicalRolloutBuffer(rollout_size=ROLLOUT_SIZE, 
                                            mini_batch_size=MINI_BATCH_SIZE,
                                            device=DEVICE)
        
        # trainer = PPOTrainer(
        #     policy=policy,
        #     logger=logger,
        #     lr=LR,
        #     mini_batch_size=MINI_BATCH_SIZE,
        #     clip_eps=CLIP_EPS,
        #     value_coef=VALUE_COEF,
        #     entropy_coef=ENTROPY_COEF,
        #     device=DEVICE
        # )
        
        
        # ============================================================
        # TEST LOOP
        # ============================================================
        
        print("Starting training...\n")
        
        all_reward = []
        all_blocking_service = []
        all_blocking_rate = []
        all_avg_link_utils = []
        
        
        
        
        determistic = True
        best_rate = 2
        for iteration in range(NUM_ITERATIONS):
            print(f"iteration: {iteration} on going topology: {arg['topology_name']} load: {load}")
            
            buffer.clear()
        
            # --------------------------------------------------------
            # ROLLOUT COLLECTION (FIXED SIZE PPO)
            # --------------------------------------------------------
        
            last_value, rollout_info = worker.collect_rollout(buffer, determistic)
        
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
        
            # stats = trainer.train_step(batch, last_value)
        
            # --------------------------------------------------------
            # METRICS
            # --------------------------------------------------------
        
            mean_reward = float(batch["rewards"].mean().item())
            
            to_print = f"[Iter {iteration:04d}] Reward = {mean_reward:.4f}" + \
            f" service_blocking_rate = {rollout_info['service_blocking_rate'][-1]:.4f} |" +\
            f" episode_service_blocking_rate = {rollout_info['episode_service_blocking_rate'][-1]:.4f} |" +\
            f" our_service_blocking_rate = {rollout_info['our_service_blocking_rate'][-1]:.4f} |" +\
            f" bit_rate_blocking_rate = {rollout_info['bit_rate_blocking_rate'][-1]:.4f} | "+\
            f" episode_bit_rate_blocking_rate = {rollout_info['episode_bit_rate_blocking_rate'][-1]:.4f} | "+\
            f" avg_link_utilization = {rollout_info['avg_link_utilization'][-1]:.2f}"
        
            print(to_print)
            # print("LOSS:", np.mean(stats['loss_total']))
            # print("VALUE LOSS", np.mean(stats['value_loss']))
            # print("PATH ENTROPY", np.mean(stats['entropy_path']))
            # print("MODULATION ENTROPY", np.mean(stats['entropy_mod']))
            # print("SLOT ENTROPY", np.mean(stats['entropy_slot']))
        
            # logger.log_str(f"[Iter {iteration:04d}] Reward = {mean_reward:.4f}")
            # logger.log_dict(stats)
            
            all_reward.append(mean_reward)
            all_blocking_service.append(rollout_info['service_blocking_rate'][-1])
            all_blocking_rate.append(rollout_info['bit_rate_blocking_rate'][-1])
            all_avg_link_utils.append(rollout_info['avg_link_utilization'][-1])
        
            writer.write([arg['topology_name'], load,
                          mean_reward, 
                          rollout_info['service_blocking_rate'][-1],
                          rollout_info['episode_service_blocking_rate'][-1],
                          rollout_info['our_service_blocking_rate'][-1],
                          rollout_info['bit_rate_blocking_rate'][-1],
                          rollout_info['episode_bit_rate_blocking_rate'][-1],
                          rollout_info['avg_link_utilization'][-1],
                          rollout_info['num_accepted_request'][-1],
                          rollout_info['num_total_request'][-1]])
           
<<<<<<< HEAD
            
            # print(rollout_info['done']) 
        
=======
            # print(rollout_info['num_accepted_request'])
            # print(rollout_info['num_total_request'])         
>>>>>>> d8f49b12a50dbd48d53ab855adb12d18b147dd61
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
            # new_rate = rollout_info['our_service_blocking_rate'][-1]
            # # if iteration % 100 == 0:
            # if best_rate > new_rate and new_rate != 0:
            #     determistic = True
        
            #     checkpoint = {
            #         "iteration": iteration,
            #         "model_state_dict": policy.state_dict(),
            #         "optimizer_state_dict": trainer.optimizer.state_dict()
            #     }
        
            #     torch.save(checkpoint, f"checkpoint_iter_{iteration}_rate_{new_rate}.pt")
        
            #     print(f"Checkpoint saved at iteration {iteration}")
            #     best_rate = new_rate
            # else:
            #     determistic = False
                
            # if iteration % 5 == 0:
            #     checkpoint = {
            #         "iteration": iteration,
            #         "model_state_dict": policy.state_dict(),
            #         "optimizer_state_dict": trainer.optimizer.state_dict()
            #     }
        
            #     torch.save(checkpoint, f"checkpoint_iter_{iteration}_rate_{new_rate}.pt")
        
            #     print(f"Checkpoint saved at iteration {iteration}")
        
        logger.logger_close()
        print("\nTest complete.\n")
writer.close()




# # Combine the lists into rows using zip
# rows = zip(all_reward, all_blocking_service, all_blocking_rate, all_avg_link_utils)

# with open('output.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     # Optional: write a header row
#     writer.writerow(['Column1', 'Column2', 'Column3'])
#     # Write all rows at once
#     writer.writerows(rows)