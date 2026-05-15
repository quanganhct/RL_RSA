from math import pi

from math import log, exp, asinh, log10, ceil
from env import constant
from custom_env.CustomRLenv.utils import compute_number_of_slots
import numpy as np

from typing import List, Collection

from custom_env.optical_rl_gym.envs.rmsa_env import RMSAEnv, Service, Path
from custom_env.optical_rl_gym.utils import Modulation

# Compute OSNR without writing the OSNR factors into env
def eval_osnr(env: RMSAEnv, current_service: Service):
    # if not current_service.accepted and current_service not in env.topology.graph["running_services"]:
    #     return None, None, None

    beta_2: float = -21.3e-27  
    gamma: float = 1.3e-3  
    h_plank: float = 6.626e-34  
    acc_gsnr: float = 0
    acc_ase: float = 0
    acc_nli: float = 0
    gsnr: float = 0
    ase: float = 0
    nli: float = 0
    l_eff_a: float = 0
    l_eff: float = 0
    phi: float = 0
    sum_phi: float = 0
    power_ase: float = 0
    power_nli_span: float = 0
    phi_modulation_format = np.array((1, 1, 2/3, 17/25, 69/100, 13/21))
    service: Service

    attenuation_normalized = constant.attenuation_db_km / (2 * 10 * np.log10(np.exp(1)) * 1e3)
    noise_figure_normalized = 10 ** (constant.noise_figure_db / 10)

    l_eff_a = 1 / (2 * attenuation_normalized)
    l_eff = (1 - np.exp(-2 * attenuation_normalized * constant.fiber_span * 1e3)) / (2 * attenuation_normalized)
    

    for i in range(len(current_service.path.node_list)-1):
        src, dst = current_service.path.node_list[i], current_service.path.node_list[i+1]
        nb_span = ceil(env.topology[src][dst]["length"] / constant.fiber_span)

        sum_phi = asinh(
                pi ** 2 * \
                abs(beta_2) * \
                (current_service.bandwidth) ** 2 / \
                (4 * attenuation_normalized)
            )
        
        for service in env.topology[src][dst]["running_services"]:
            if service.service_id != current_service.service_id:
                d_frequency = abs(service.center_frequency - current_service.center_frequency)
                phi = np.log(abs(d_frequency + service.bandwidth/2) / \
                             abs(d_frequency - service.bandwidth/2))
                # - \
                #     (phi_modulation_format[service.path.current_modulation.spectral_efficiency - 1] * \
                #         (service.bandwidth / abs(service.center_frequency - current_service.center_frequency)) * \
                #         5 / 3 * (l_eff / (constant.fiber_span * 1e3)))
                
                sum_phi += phi

        power_nli_span = nb_span * (current_service.launch_power / (current_service.bandwidth)) ** 3 * \
            (8 / (27 * pi * abs(beta_2))) * gamma ** 2 * l_eff * sum_phi * current_service.bandwidth
        power_ase = nb_span * h_plank * current_service.center_frequency * \
            (exp(2 * attenuation_normalized * constant.fiber_span * 1e3) - 1) * noise_figure_normalized

        print(current_service.launch_power, power_ase)
        acc_gsnr = acc_gsnr + 1 / (current_service.launch_power / (power_ase + power_nli_span))
        acc_ase = acc_ase + 1 / (current_service.launch_power / power_ase)
        acc_nli = acc_nli + 1 / (current_service.launch_power / power_nli_span)

    gsnr = 10 * np.log10(1 / acc_gsnr)
    ase = 10 * np.log10(1 / acc_ase)
    nli = 10 * np.log10(1 / acc_nli)
    return gsnr, ase, nli

# Compute OSNR and writing OSNR factor into env for the purpose of recomputing later
def compute_ase_nli(env: RMSAEnv, current_service: Service, update_old_service=True):
    # if not current_service.accepted and current_service not in env.topology.graph["running_services"]:
    #     return None, None, None
    
    beta_2: float = -21.3e-27  
    gamma: float = 1.3e-3  
    h_plank: float = 6.626e-34  
    acc_gsnr: float = 0
    acc_ase: float = 0
    acc_nli: float = 0
    gsnr: float = 0
    ase: float = 0
    nli: float = 0
    l_eff_a: float = 0
    l_eff: float = 0
    phi: float = 0
    sum_phi: float = 0
    power_ase: float = 0
    power_nli_span: float = 0
    phi_modulation_format = np.array((1, 1, 2/3, 17/25, 69/100, 13/21))
    service: Service

    attenuation_normalized = constant.attenuation_db_km / (2 * 10 * np.log10(np.exp(1)) * 1e3)
    noise_figure_normalized = 10 ** (constant.noise_figure_db / 10)

    l_eff_a = 1 / (2 * attenuation_normalized)
    l_eff = (1 - np.exp(-2 * attenuation_normalized * constant.fiber_span * 1e3)) / (2 * attenuation_normalized)

    nli_coef = (8 / (27 * pi * abs(beta_2))) * gamma ** 2 * l_eff
    ase_coef = h_plank * current_service.center_frequency * \
            (exp(2 * attenuation_normalized * constant.fiber_span * 1e3) - 1) * noise_figure_normalized

    firstime = False
    if current_service.nli_inf_from is None or current_service.ase_inf is None or \
            current_service.ase_inf == 0:
        
        current_service.nli_inf_from = dict()
        current_service.nli_inf_from[current_service.service_id] = 0
        current_service.ase_inf = 0
        firstime = True
        
    # power_ase = nb_span * h_plank * current_service.center_frequency * \
    #         (exp(2 * attenuation_nor malized * constant.fiber_span * 1e3) - 1) * noise_figure_normalized

    if firstime:
        phi_sci = asinh(pi ** 2 * abs(beta_2) * (current_service.bandwidth) ** 2 / \
                            (4 * attenuation_normalized))
        
        for i in range(len(current_service.path.node_list)-1):
            src, dst = current_service.path.node_list[i], current_service.path.node_list[i+1]
            nb_span = ceil(env.topology[src][dst]["length"] / constant.fiber_span)

            #ASE
            current_service.ase_inf += nb_span * ase_coef
        
            #SCI
            current_service.nli_inf_from[current_service.service_id] += nb_span * \
                                (current_service.launch_power / current_service.bandwidth) ** 3 * \
                                nli_coef * current_service.bandwidth * phi_sci

            for service in env.topology[src][dst]["running_services"]:
                if service.service_id != current_service.service_id:
                    d_frequency = abs(service.center_frequency - current_service.center_frequency)

                    phi_xci = np.log(abs(d_frequency + service.bandwidth/2) / \
                                abs(d_frequency - service.bandwidth/2))

                    if service.service_id not in current_service.nli_inf_from:
                        current_service.nli_inf_from[service.service_id] = 0

                    #XCI
                    current_service.nli_inf_from[service.service_id] += nb_span * \
                                (current_service.launch_power / current_service.bandwidth) ** 3 * \
                                nli_coef * phi_xci * current_service.bandwidth
                    
                    phi_xci = np.log(abs(d_frequency + current_service.bandwidth/2) / \
                                abs(d_frequency - current_service.bandwidth/2))
                    
                    if update_old_service and service.nli_inf_from is not None and current_service.service_id in service.nli_inf_from:
                        service.nli_inf_from[current_service.service_id] += nb_span * \
                                    (service.launch_power / service.bandwidth) ** 3 * \
                                    nli_coef + phi_xci * service.bandwidth

    list_running_service = env.topology.graph["running_services"]
    set_running_service_idx = set([s.service_id for s in list_running_service])
    sid_set = set_running_service_idx.intersection(current_service.nli_inf_from.keys())
    sid_set.add(current_service.service_id)

    power_nli = sum([current_service.nli_inf_from[sid] for sid in sid_set])
    nli = power_nli / current_service.launch_power
    ase = current_service.ase_inf / current_service.launch_power
    osnr = nli + ase

    osnr = 10 * np.log10(1 / osnr)
    ase = 10 * np.log10(1 / ase)
    nli = 10 * np.log10(1 / nli)

    return osnr, ase, nli

# Return min osnr gap, together with service_id
def compute_min_gap_osnr(env: RMSAEnv, new_service: Service, path: Path, modulation: Modulation, \
                         initial_slot: int, running_service: List[Service]):
    beta_2: float = -21.3e-27  
    gamma: float = 1.3e-3  
    h_plank: float = 6.626e-34  
    acc_gsnr: float = 0
    acc_ase: float = 0
    acc_nli: float = 0
    gsnr: float = 0
    ase: float = 0
    nli: float = 0
    l_eff_a: float = 0
    l_eff: float = 0
    phi: float = 0
    sum_phi: float = 0
    power_ase: float = 0
    power_nli_span: float = 0
    phi_modulation_format = np.array((1, 1, 2/3, 17/25, 69/100, 13/21))
    service: Service

    nbslots = compute_number_of_slots(new_service.bit_rate, modulation)
    if not env.is_path_free(path, initial_slot, nbslots):
        raise Exception("Slot conflict exception")
    

    attenuation_normalized = constant.attenuation_db_km / (2 * 10 * np.log10(np.exp(1)) * 1e3)
    noise_figure_normalized = 10 ** (constant.noise_figure_db / 10)

    l_eff_a = 1 / (2 * attenuation_normalized)
    l_eff = (1 - np.exp(-2 * attenuation_normalized * constant.fiber_span * 1e3)) / (2 * attenuation_normalized)

    nli_coef = (8 / (27 * pi * abs(beta_2))) * gamma ** 2 * l_eff
    ase_coef = h_plank * new_service_center_frequency * \
        (exp(2 * attenuation_normalized * constant.fiber_span * 1e3) - 1) * noise_figure_normalized


    new_service_nli_from = dict([(s.service_id, 0) for s in running_service + [new_service]])
    new_service_nli_to = dict([(s.service_id, 0) for s in running_service + [new_service]])
    ase_power = 0

    new_service_center_frequency = constant.frequency_start \
                        + constant.frequency_slot_bandwidth * initial_slot \
                        + constant.frequency_slot_bandwidth * (nbslots / 2.0)
    
    new_service_bandwidth = constant.frequency_slot_bandwidth * nbslots
    
    phi_sci = asinh(pi ** 2 * abs(beta_2) * (new_service_bandwidth) ** 2 / \
                            (4 * attenuation_normalized))

    for i in range(len(new_service.path.node_list)-1):
        src, dst = new_service.path.node_list[i], new_service.path.node_list[i+1]
        nb_span = ceil(env.topology[src][dst]["length"] / constant.fiber_span)

        #ASE
        ase_power += nb_span * ase_coef

        #SCI
        new_service_nli_from[new_service.service_id] += nb_span * \
                            (env.launch_power / new_service_bandwidth) ** 3 * \
                            nli_coef * new_service_bandwidth * phi_sci
        
        for service in running_service:
            if service.service_id != new_service.service_id:
                d_frequency = abs(service.center_frequency - new_service_center_frequency)
                phi_xci = np.log(abs(d_frequency + service.bandwidth/2) / \
                             abs(d_frequency - service.bandwidth/2))
                
                #XCI
                new_service_nli_from[service.service_id] += nb_span * \
                            (env.launch_power / new_service_bandwidth) ** 3 * \
                            nli_coef * phi_xci * new_service_bandwidth
                
                phi_xci = np.log(abs(d_frequency + new_service_bandwidth/2) / \
                                abs(d_frequency - new_service_bandwidth/2))
                    
                new_service_nli_to[service.service_id] += nb_span * \
                            (service.launch_power / service.bandwidth) ** 3 * \
                            nli_coef + phi_xci * service.bandwidth


    list_running_service = env.topology.graph["running_services"]
    set_running_service_idx = set([s.service_id for s in list_running_service])

    result = []
    # result for @new_service
    power_nli = sum([new_service_nli_from[sid] for sid in new_service_nli_from.keys()])
    nli = power_nli / env.launch_power
    ase = ase_power / env.launch_power
    osnr = nli + ase

    osnr = 10 * np.log10(1 / osnr)
    ase = 10 * np.log10(1 / ase)
    nli = 10 * np.log10(1 / nli)

    result.append((osnr - modulation.minimum_osnr, new_service.service_id))

    for service in running_service:
        sid_set = set_running_service_idx.intersection(service.nli_inf_from.keys())
        power_nli = sum([service.nli_inf_from[sid] for sid in sid_set])
        power_nli += new_service_nli_to[service.service_id]

        nli = power_nli / env.launch_power
        ase = service.ase_inf / env.launch_power
        osnr = nli + ase

        osnr = 10 * np.log10(1 / osnr)
        ase = 10 * np.log10(1 / ase)
        nli = 10 * np.log10(1 / nli)

        result.append((osnr - service.path.current_modulation.minimum_osnr, service.service_id))

    min_gap, sid = min(result, key=lambda x: x[0])
    return min_gap, sid