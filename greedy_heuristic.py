from config import LOAD, EPISODE_LENGTH, MEAN_SERVICE_HOLDING_TIME, NUM_SPECTRUM_RESOURCES, NUM_ITERATIONS, SEED

from custom_env.CustomRLenv.CustomRMSAEnv import CustomRMSAEnv
from custom_env.CustomRLenv.utils import get_topology, modulations
from env import constant
from DRL.utils.logging import Logger
from DRL.utils.csv_writer import CSVWriter
from custom_env.CustomRLenv.utils import Path, Modulation, Service, compute_number_of_slots
from custom_env.CustomRLenv.osnr import compute_ase_nli

import datetime
import numpy as np
from typing import Collection


def first_fit_heuristic(env:CustomRMSAEnv, request:Service):
    src, dst = request.source, request.destination
    paths:Collection[Path] = env.k_shortest_paths[src, dst]
    request.accepted = False

    for p in range(len(paths)):
        path = paths[p]
        count = 0
        for modulation in reversed(modulations):
            if modulation.spectral_efficiency > path.best_modulation.spectral_efficiency:
                continue
            count += 1
            if count > 2:
                break
            initial_indexes, lengths = env.get_available_blocks(p, modulation)
            slots = compute_number_of_slots(request.bit_rate, modulation)
            path.current_modulation = modulation

            for i in range(len(initial_indexes)):
                initial_slot = initial_indexes[i]
                length = lengths[i]

                request.path = path
                request.initial_slot = initial_slot
                request.number_slots = slots
                request.center_frequency = constant.frequency_start \
                    + constant.frequency_slot_bandwidth * initial_slot \
                    + constant.frequency_slot_bandwidth * (slots / 2.0)
                request.bandwidth = constant.frequency_slot_bandwidth * slots
                request.launch_power = env.launch_power

                osnr, ase, nli = compute_ase_nli(env, request)
                if osnr >= path.current_modulation.minimum_osnr + constant.osnr_margin:
                    env._provision_path(path, initial_slot, slots)
                    request.accepted = True
                    env._add_release(request)
                    break

            if request.accepted:
                break
        if request.accepted:
            break

def first_fit_best_modulation_heuristic(env:CustomRMSAEnv, request:Service):
    src, dst = request.source, request.destination
    paths:Collection[Path] = env.k_shortest_paths[src, dst]
    request.accepted = False

    for p in range(len(paths)):
        path = paths[p]
        modulation = path.best_modulation

        initial_indexes, lengths = env.get_available_blocks(p, modulation)
        slots = compute_number_of_slots(request.bit_rate, modulation)
        path.current_modulation = modulation

        for i in range(len(initial_indexes)):
            initial_slot = initial_indexes[i]
            length = lengths[i]

            request.path = path
            request.initial_slot = initial_slot
            request.number_slots = slots
            request.center_frequency = constant.frequency_start \
                + constant.frequency_slot_bandwidth * initial_slot \
                + constant.frequency_slot_bandwidth * (slots / 2.0)
            request.bandwidth = constant.frequency_slot_bandwidth * slots
            request.launch_power = env.launch_power

            osnr, ase, nli = compute_ase_nli(env, request)
            if osnr >= path.current_modulation.minimum_osnr + constant.osnr_margin:
                env._provision_path(path, initial_slot, slots)
                request.accepted = True
                env._add_release(request)
                break

        if request.accepted:
            break
                
def greedy_algorithm(env:CustomRMSAEnv, writer:CSVWriter):
    env._new_service = False
    accepted_count = 0
    for i in range(EPISODE_LENGTH):
        print("Process request", i)
        env._next_service()
        first_fit_heuristic(env, env.current_service)
        # first_fit_best_modulation_heuristic(env, env.current_service)
        if env.current_service.accepted:
            accepted_count += 1
        env._new_service = False
    writer.write([EPISODE_LENGTH, accepted_count, float(EPISODE_LENGTH-accepted_count)/EPISODE_LENGTH])

    
topology = get_topology(
    './data/germany/sndlib_germany.txt',
    'Germany',
    sndformat=True,
    alpha=1
)

now = datetime.datetime.now()
log_filename = now.strftime("Greedy_%Y-%m-%d_%H-%M-%S")+".txt"
debug_filename = now.strftime("Greedy_DEBUG_%Y-%m-%d_%H-%M-%S")+".txt"
logger = Logger()
logger.set_log_file(log_filename, debug_filename, 'log')

# bitrates = np.arange(25, 101, 5)

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
writer = CSVWriter(now.strftime("Greedy_firstfit_%Y-%m-%d_%H-%M-%S.csv"), 'log')
writer.write(['num_request', 'accepted', 'service_blocking_rate'])
print("Run Greedy Heuristic")
for i in range(NUM_ITERATIONS):
    print("Iteration", i)
    greedy_algorithm(env, writer)

writer.close()

