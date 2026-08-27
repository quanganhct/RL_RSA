from config import LOAD, EPISODE_LENGTH, MEAN_SERVICE_HOLDING_TIME, NUM_SPECTRUM_RESOURCES, NUM_ITERATIONS, SEED

from custom_env.CustomRLenv.CustomRMSAEnv import CustomRMSAEnv
from custom_env.CustomRLenv.utils import get_topology, modulations
from env import constant
from DRL.utils.logging import Logger
from DRL.utils.csv_writer import CSVWriter
from custom_env.CustomRLenv.utils import Path, Modulation, Service, compute_number_of_slots
from custom_env.CustomRLenv.osnr import compute_ase_nli, check_osnr_constraint_of_running_requests
from custom_env.CustomRLenv.return_code import FailedCode

import datetime
import numpy as np
from typing import Collection


def first_fit_heuristic(env:CustomRMSAEnv, request:Service):
    src, dst = request.source, request.destination
    paths:Collection[Path] = env.k_shortest_paths[src, dst]
    request.accepted = False
    request.failed_gap = None
    violating_prev_osnr = False

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

                osnr, ase, nli = compute_ase_nli(env, request, debug=False)
                request.failed_gap = osnr - (path.current_modulation.minimum_osnr + constant.osnr_margin)
                if osnr >= path.current_modulation.minimum_osnr + constant.osnr_margin:
                    check, request.failed_gap, sid_set, dict_nli = check_osnr_constraint_of_running_requests(env, request)
                    env._provision_path(path, initial_slot, slots)
                    request.accepted = True
                    request.return_code = FailedCode.SUCCESS
                    env._add_release(request)

                    if not check:
                        violating_prev_osnr = True
                    break
                    
                else:
                    request.return_code = FailedCode.OSNR
                    
                
                if not request.accepted:
                    request.nli_inf_from = None
                    request.ase_inf = None
                    if request.return_code == FailedCode.PREV_OSNR:
                        print(request.return_code, request.failed_gap)

            if request.accepted:
                break
        if request.accepted:
            break
    return violating_prev_osnr

def first_fit_best_modulation_heuristic(env:CustomRMSAEnv, request:Service):
    src, dst = request.source, request.destination
    paths:Collection[Path] = env.k_shortest_paths[src, dst]
    request.accepted = False

    for p in range(len(paths)):
        path = paths[p]
        modulation = path.best_modulation

        initial_indexes, lengths = env.get_available_blocks(p, modulation)
        slots = compute_number_of_slots(request.bit_rate, modulation) + 1
        path.current_modulation = modulation

        for i in range(len(initial_indexes)):
            initial_slot = initial_indexes[i]
            length = lengths[i]

            if not env.is_path_free(path, initial_slot, slots):
                continue

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

def first_fit_heuristic_modulation_first(env:CustomRMSAEnv, request:Service):
    src, dst = request.source, request.destination
    paths:Collection[Path] = env.k_shortest_paths[src, dst]
    request.accepted = False

    modulation_count = 0

    for mod_count in range(2):
        for p in range(len(paths)):
            path = paths[p]
            
            modulation_count = 0
            for modulation in reversed(modulations):
                if modulation.spectral_efficiency > path.best_modulation.spectral_efficiency:
                    continue
                
                if modulation_count < mod_count:
                    modulation_count += 1
                    continue

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
        if request.accepted:
            break
            
def random_fit(env:CustomRMSAEnv, request:Service):
    src, dst = request.source, request.destination
    paths:Collection[Path] = env.k_shortest_paths[src, dst]
    request.accepted = False

    # for path in paths:



def greedy_algorithm(env:CustomRMSAEnv, iteration):
    env._new_service = False
    accepted_count = 0
    return_val = 0
    for i in range(EPISODE_LENGTH):
        # print("Process request", i)
        env._next_service()
        val = first_fit_heuristic(env, env.current_service)
        return_val += 1 if val else 0
        # first_fit_best_modulation_heuristic(env, env.current_service)
        # first_fit_heuristic_modulation_first(env, env.current_service)
        print(env.current_service.return_code, env.current_service.failed_gap)
        if env.current_service.accepted:
            accepted_count += 1
        env._new_service = False
    
    print(f"[Iteration = {iteration}] Total = {EPISODE_LENGTH} | accepted_count = {accepted_count} | blocking_rate = {float(EPISODE_LENGTH-accepted_count)/EPISODE_LENGTH}")
    return accepted_count, return_val


topology_data = [dict(file_name='./data/european/european.txt', topology_name='European', sndformat=False, undirected_file=False),\
                 dict(file_name='./data/nsf/nsfnet_chen.txt', topology_name='NSF', sndformat=False, undirected_file=True),\
                 dict(file_name='./data/usa/backbone.txt', topology_name='USA', sndformat=False, undirected_file=False),\
                 dict(file_name='./data/germany/sndlib_germany.txt', topology_name='Germany', sndformat=True)]

now = datetime.datetime.now()
writer = CSVWriter(now.strftime("Greedy_firstfit_ksp_%Y-%m-%d_%H-%M-%S.csv"), 'log')
writer.write(['topology_name', 'load', 'num_request', 'accepted', 'service_blocking_rate'])


topology_data = [dict(file_name='./data/germany/sndlib_germany.txt', topology_name='Germany', sndformat=True)]
loads = [80, 200, 500]
loads = [200]

for arg in topology_data:
    topology = get_topology(**arg, alpha=1)

    for load in loads:

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
            load=load,
            mean_service_holding_time=MEAN_SERVICE_HOLDING_TIME,
            episode_length=EPISODE_LENGTH,
            num_spectrum_resources=300,
            bit_rates=constant.bit_rates,
            # bit_rate_probabilities=[0.5, 0.3, 0.2],
            bit_rate_selection="discrete",
        )

        env = CustomRMSAEnv(**env_args)
        env.logger=logger
        
        print("Run Greedy Heuristic")
        return_val = []
        for i in range(2):
            # print("Iteration", i)
            nbaccepted, val = greedy_algorithm(env, i)
            return_val.append(val)
            sbr = float(EPISODE_LENGTH - nbaccepted)/EPISODE_LENGTH
            writer.write([arg['topology_name'], load, EPISODE_LENGTH, nbaccepted, sbr])
            _ = env.customreset(False)

    print("PREV OSNR violated:", return_val)
writer.close()

