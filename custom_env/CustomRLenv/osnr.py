from math import pi

from math import log, exp, asinh, log10, ceil
from env import constant
from custom_env.CustomRLenv.utils import compute_number_of_slots
import numpy as np

from typing import List, Collection

from custom_env.optical_rl_gym.envs.rmsa_env import RMSAEnv, Service, Path
from custom_env.optical_rl_gym.utils import Modulation
from custom_env.CustomRLenv.utils import modulations, compute_number_of_slots


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
    nli_coef = (8 / (27 * pi * abs(beta_2))) * gamma ** 2 * l_eff

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

        power_nli_span += nb_span * (current_service.launch_power / (current_service.bandwidth)) ** 3 * \
            nli_coef * sum_phi * current_service.bandwidth
        power_ase += nb_span * current_service.bandwidth * h_plank * current_service.center_frequency * \
            (exp(2 * attenuation_normalized * constant.fiber_span * 1e3) - 1) * noise_figure_normalized

        # print(current_service.launch_power, power_ase)
        acc_gsnr = acc_gsnr + 1 / (current_service.launch_power / (power_ase + power_nli_span))
        acc_ase = acc_ase + 1 / (current_service.launch_power / power_ase)
        acc_nli = acc_nli + 1 / (current_service.launch_power / power_nli_span)

    print("G ASE, NLI:", acc_ase, acc_nli)
    gsnr = 10 * np.log10(1 / acc_gsnr)
    ase = 10 * np.log10(1 / acc_ase)
    nli = 10 * np.log10(1 / acc_nli)
    return gsnr, ase, nli

def eval_osnr_in_dark_fiber(service:Service, length: float, modulation: Modulation):
    beta_2: float = -21.3e-27  
    gamma: float = 1.3e-3  
    h_plank: float = 6.626e-34  
    attenuation_normalized = constant.attenuation_db_km / (2 * 10 * np.log10(np.exp(1)) * 1e3)
    l_eff = (1 - np.exp(-2 * attenuation_normalized * constant.fiber_span * 1e3)) / (2 * attenuation_normalized)

    nb_span = ceil(length / constant.fiber_span)


# Compute OSNR and writing OSNR factor into env for the purpose of recomputing later
def compute_ase_nli(env: RMSAEnv, current_service: Service, update_old_service=True, debug=False):
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
    span_power_ase = current_service.bandwidth * h_plank * current_service.center_frequency * \
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

    other_service_first_time = set()
    if firstime:
        phi_sci = asinh(pi ** 2 * abs(beta_2) * (current_service.bandwidth) ** 2 / \
                            (4 * attenuation_normalized))
        
        for i in range(len(current_service.path.node_list)-1):
            src, dst = current_service.path.node_list[i], current_service.path.node_list[i+1]
            nb_span = ceil(env.topology[src][dst]["length"] / constant.fiber_span)

            #ASE
            current_service.ase_inf += nb_span * span_power_ase
        
            #SCI
            current_service.nli_inf_from[current_service.service_id] += nb_span * \
                                (current_service.launch_power / current_service.bandwidth) ** 3 * \
                                nli_coef * current_service.bandwidth * phi_sci

            for service in env.topology[src][dst]["running_services"]:
                if service.service_id != current_service.service_id:

                    if debug and current_service.service_id == 7:
                        print("Compute ASE NLI", current_service.service_id, service.service_id)

                    if update_old_service and current_service.service_id not in service.nli_inf_from:
                        service.nli_inf_from[current_service.service_id] = 0

                    d_frequency = abs(service.center_frequency - current_service.center_frequency)

                    phi_xci = asinh(
                            pi ** 2 * \
                            abs(beta_2) * \
                            l_eff_a * \
                            service.bandwidth * \
                            (  
                                service.center_frequency - \
                                current_service.center_frequency + \
                                (service.bandwidth / 2)
                            )
                        ) - \
                        asinh(
                            pi ** 2 * \
                            abs(beta_2) * \
                            l_eff_a * \
                            service.bandwidth * \
                            (  
                                service.center_frequency - \
                                current_service.center_frequency - \
                                (service.bandwidth / 2)
                            )
                        ) - \
                        (phi_modulation_format[service.path.current_modulation.spectral_efficiency - 1] * \
                        (service.bandwidth / abs(service.center_frequency - current_service.center_frequency)) * \
                        5 / 3 * (l_eff / (constant.fiber_span * 1e3)))

                    if service.service_id not in current_service.nli_inf_from:
                        current_service.nli_inf_from[service.service_id] = 0

                    #XCI
                    current_service.nli_inf_from[service.service_id] += nb_span * \
                                (current_service.launch_power / current_service.bandwidth) ** 3 * \
                                nli_coef * phi_xci * current_service.bandwidth
                    
                    phi_xci = np.log(abs(d_frequency + current_service.bandwidth/2) / \
                                abs(d_frequency - current_service.bandwidth/2))
                    
                    if update_old_service and service.nli_inf_from is not None and current_service.service_id in service.nli_inf_from:
                        if service.service_id not in other_service_first_time:
                            service.nli_inf_from[current_service.service_id] = 0
                            other_service_first_time.add(service.service_id)

                        service.nli_inf_from[current_service.service_id] += nb_span * \
                                    (service.launch_power / service.bandwidth) ** 3 * \
                                    nli_coef * phi_xci * service.bandwidth

    # Compute the nli from only running service. The name @current_service may not be the new generated service
    list_running_service = env.topology.graph["running_services"]
    set_running_service_idx = set([s.service_id for s in list_running_service])
    sid_set = set_running_service_idx.intersection(current_service.nli_inf_from.keys())
    sid_set.add(current_service.service_id)

    power_nli = sum([current_service.nli_inf_from[sid] for sid in sid_set])
    nli = power_nli / current_service.launch_power
    ase = current_service.ase_inf / current_service.launch_power
    osnr = nli + ase

    # print("P ASE, NLI", ase, nli)

    osnr = 10 * np.log10(1 / osnr)
    ase = 10 * np.log10(1 / ase)
    nli = 10 * np.log10(1 / nli)
    return osnr, ase, nli

'''
Check OSNR constraints of running service, with the new service (not provisioned yet) is @new_service
'''
def check_osnr_constraint_of_running_requests(env: RMSAEnv, new_service: Service):
    set_shared_link_service_id = set()
    shared_link_service = dict()
    for i in range(len(new_service.path.node_list)-1):
        src, dst = new_service.path.node_list[i], new_service.path.node_list[i+1]
        list_service:List[Service] = env.topology[src][dst]["running_services"]
        shared_link_service.update([(service.service_id, service) for service in list_service])
        set_shared_link_service_id.update([s.service_id for s in list_service])

    running_service_id = set([service.service_id for service in env.topology.graph["running_services"]])
    service:Service
    for service_id in set_shared_link_service_id:
        service = shared_link_service[service_id]
        sid_set = running_service_id.intersection(service.nli_inf_from.keys())
        sid_set.add(new_service.service_id)

        try:
            power_nli = sum([service.nli_inf_from[sid] for sid in sid_set])
        except Exception as e:
            print("New Service Id:", new_service.service_id)
            print("Check Service:", service_id)
            print("Shared Link Service:", set_shared_link_service_id)
            print("New Service", new_service.path.node_list)
            print("Service:", service.path.node_list)
            raise e
        
        osnr = 10 * np.log10(service.launch_power / (power_nli + service.ase_inf))

        if osnr < service.path.current_modulation.minimum_osnr:
            return False, osnr, sid_set, service.nli_inf_from
    
    return True, None, None, None


# Return min osnr gap of all services that shared link with @path, together with service_id
def compute_min_gap_osnr(env: RMSAEnv, new_service: Service, path: Path, modulation: Modulation, \
                         initial_slot: int, running_service: List[int]):
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
        print("Initial slot %s, nb slot %s, spectrum %s"%(initial_slot, nbslots, env.num_spectrum_resources))
        raise Exception("Not enough slot resource")
    

    attenuation_normalized = constant.attenuation_db_km / (2 * 10 * np.log10(np.exp(1)) * 1e3)
    noise_figure_normalized = 10 ** (constant.noise_figure_db / 10)

    l_eff_a = 1 / (2 * attenuation_normalized)
    l_eff = (1 - np.exp(-2 * attenuation_normalized * constant.fiber_span * 1e3)) / (2 * attenuation_normalized)

    new_service_nli_from = dict([(s, 0) for s in running_service + [new_service.service_id]])
    new_service_nli_to = dict([(s, 0) for s in running_service + [new_service.service_id]])
    ase_power = 0

    new_service_center_frequency = constant.frequency_start \
                        + constant.frequency_slot_bandwidth * initial_slot \
                        + constant.frequency_slot_bandwidth * (nbslots / 2.0)
    
    new_service_bandwidth = constant.frequency_slot_bandwidth * nbslots
    
    nli_coef = (8 / (27 * pi * abs(beta_2))) * gamma ** 2 * l_eff
    span_power_ase = new_service_bandwidth * h_plank * new_service_center_frequency * \
        (exp(2 * attenuation_normalized * constant.fiber_span * 1e3) - 1) * noise_figure_normalized

    phi_sci = asinh(pi ** 2 * abs(beta_2) * (new_service_bandwidth) ** 2 / \
                            (4 * attenuation_normalized))

    for i in range(len(path.node_list)-1):
        src, dst = path.node_list[i], path.node_list[i+1]
        nb_span = ceil(env.topology[src][dst]["length"] / constant.fiber_span)

        #ASE
        ase_power += nb_span * span_power_ase

        #SCI
        new_service_nli_from[new_service.service_id] += nb_span * \
                            (env.launch_power / new_service_bandwidth) ** 3 * \
                            nli_coef * new_service_bandwidth * phi_sci
        
        for service in env.topology.graph["running_services"]:
            if service.service_id not in running_service:
                continue
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
                            nli_coef * phi_xci * service.bandwidth


    list_running_service = env.topology.graph["running_services"]
    set_running_service_idx = set([s.service_id for s in list_running_service])

    result = []
    # result for @new_service
    power_nli = sum([new_service_nli_from[sid] for sid in new_service_nli_from.keys()])
    nli = power_nli / env.launch_power
    ase = ase_power / env.launch_power
    osnr = nli + ase
    # print("OSNR, NLI, ASE:", osnr, nli, ase)

    osnr = 10 * np.log10(1 / osnr)
    ase = 10 * np.log10(1 / ase)
    nli = 10 * np.log10(1 / nli)

    result.append((osnr - modulation.minimum_osnr, new_service.service_id))

    for service in env.topology.graph["running_services"]:
        if service.service_id not in running_service:
            continue
        sid_set = set_running_service_idx.intersection(service.nli_inf_from.keys())
        power_nli = sum([service.nli_inf_from[sid] for sid in sid_set])
        power_nli += new_service_nli_to[service.service_id]

        nli = power_nli / env.launch_power
        ase = service.ase_inf / env.launch_power
        osnr = nli + ase

        # print("OSNR, NLI, ASE:", osnr, nli, ase)
        osnr = 10 * np.log10(1 / osnr)
        ase = 10 * np.log10(1 / ase)
        nli = 10 * np.log10(1 / nli)

        result.append((osnr - service.path.current_modulation.minimum_osnr, service.service_id))
    # print("new_service_nli_to", new_service_nli_to)
    print("Result", result)
    min_gap, sid = min(result, key=lambda x: x[0])
    return min_gap, sid

def compute_max_osnr(launch_power: float, bitrate: list[float], nbspan=1):
    beta_2: float = -21.3e-27  
    gamma: float = 1.3e-3  
    h_plank: float = 6.626e-34  
    l_eff: float = 0
    power_ase: float = 0

    attenuation_normalized = constant.attenuation_db_km / (2 * 10 * np.log10(np.exp(1)) * 1e3)
    noise_figure_normalized = 10 ** (constant.noise_figure_db / 10)

    l_eff = (1 - np.exp(-2 * attenuation_normalized * constant.fiber_span * 1e3)) / (2 * attenuation_normalized)

    nli_coef = (8 / (27 * pi * abs(beta_2))) * gamma ** 2 * l_eff

    result = dict()
    max_osnr = 0
    center_frequency = constant.frequency_start
    #ASE
    for rate in bitrate:
        result[rate] = dict()
        for modulation in modulations:
            nbslot = compute_number_of_slots(rate, modulation)
            bandwidth = nbslot * constant.frequency_slot_bandwidth
            power_ase = nbspan * bandwidth * h_plank * center_frequency * \
                    (exp(2 * attenuation_normalized * constant.fiber_span * 1e3) - 1) * noise_figure_normalized


            phi_sci = asinh(pi ** 2 * abs(beta_2) * (bandwidth) ** 2 / \
                                    (4 * attenuation_normalized))
            
            

            #SCI
            power_sci = nbspan * (launch_power / bandwidth) ** 3 * \
                                nli_coef * bandwidth * phi_sci
            
            osnr = 10 * np.log10(launch_power / (power_ase + power_sci))

            result[rate][modulation.name] = osnr
            max_osnr = max(max_osnr, osnr)

    return max_osnr, result
    

def compute_ase_nli_vectorized(env: RMSAEnv, current_service: Service, update_old_service=True, debug=False):
    # if not current_service.accepted and current_service not in env.topology.graph["running_services"]:
    #     return None, None, None
    
    beta_2: float = -21.3e-27  
    gamma: float = 1.3e-3  
    h_plank: float = 6.626e-34  
    ase: float = 0
    nli: float = 0
    l_eff: float = 0
    phi_modulation_format = np.array((1, 1, 2/3, 17/25, 69/100, 13/21))

    attenuation_normalized = constant.attenuation_db_km / (2 * 10 * np.log10(np.exp(1)) * 1e3)
    noise_figure_normalized = 10 ** (constant.noise_figure_db / 10)

    l_eff = (1 - np.exp(-2 * attenuation_normalized * constant.fiber_span * 1e3)) / (2 * attenuation_normalized)

    nli_coef = (8 / (27 * pi * abs(beta_2))) * gamma ** 2 * l_eff
    span_power_ase = current_service.bandwidth * h_plank * current_service.center_frequency * \
            (exp(2 * attenuation_normalized * constant.fiber_span * 1e3) - 1) * noise_figure_normalized
    
    if current_service.nli_inf_from is None or current_service.ase_inf is None:
        current_service.nli_inf_from = dict()
        current_service.nli_inf_from[current_service.service_id] = 0
        current_service.ase_inf = 0
        
    set_shared_link_service_id = set()
    shared_link_service:dict[(int, Service)] = dict()
    
    list_link = list()
    list_service_on_link = list()
    for i in range(len(current_service.path.node_list)-1):
        src, dst = current_service.path.node_list[i], current_service.path.node_list[i+1]
        list_link.append((src, dst))
        list_service:List[Service] = env.topology[src][dst]["running_services"]
        shared_link_service.update([(service.service_id, service) for service in list_service])
        set_shared_link_service_id.update([s.service_id for s in list_service])
        list_service_on_link.append([s.service_id for s in list_service])

    list_shared_link_service_id = list(set_shared_link_service_id)
    span_array = np.array([ceil(env.topology[src][dst]["length"] / constant.fiber_span) for src, dst in list_link])
    current_service.ase_inf = np.sum(span_array) * span_power_ase

    phi_sci = asinh(pi ** 2 * abs(beta_2) * (current_service.bandwidth) ** 2 / \
                            (4 * attenuation_normalized))
    
    current_service.nli_inf_from[current_service.service_id] = np.sum(span_array) * phi_sci * \
                        (current_service.launch_power / current_service.bandwidth) ** 3 * \
                                nli_coef * current_service.bandwidth
    
    power_nli = current_service.nli_inf_from[current_service.service_id]
    if len(list_shared_link_service_id) > 0:
        array_link = np.array([[1 if sid in list_service_on_link[lindex] else 0 for lindex in range(len(span_array))] \
                    for sid in list_shared_link_service_id])
        
        total_span = np.multiply(span_array, array_link)
        total_span = np.sum(total_span, axis=1)

        d_freq = np.array([abs(shared_link_service[sid].center_frequency - current_service.center_frequency) \
                for sid in list_shared_link_service_id])
        shared_service_bandwidth = np.array([shared_link_service[sid].bandwidth for sid in list_shared_link_service_id])
        phi_to_current = np.log(abs(d_freq + shared_service_bandwidth/2) / abs(d_freq - shared_service_bandwidth/2)) \
                    - 5 / 3 * (l_eff / (constant.fiber_span * 1e3)) \
                    * np.multiply(np.array([phi_modulation_format[shared_link_service[sid].path.current_modulation.spectral_efficiency - 1] for sid in list_shared_link_service_id]), \
                                  np.divide(shared_service_bandwidth, d_freq))
        
        nli_to_current = (current_service.launch_power / (current_service.bandwidth)) ** 3 * nli_coef * current_service.bandwidth \
                        * np.multiply(total_span, phi_to_current)
                
        phi_from_current = np.log(abs(d_freq + current_service.bandwidth/2) / abs(d_freq - current_service.bandwidth/2))
        psd_cube = np.array([(shared_link_service[sid].launch_power/shared_link_service[sid].bandwidth)**3 for sid in list_shared_link_service_id])
        nli_from_current = nli_coef * np.prod([total_span, psd_cube, phi_from_current, shared_service_bandwidth], axis=0)

        for i in range(len(list_shared_link_service_id)):
            sid = list_shared_link_service_id[i]
            shared_link_service[sid].nli_inf_from[current_service.service_id] = nli_from_current[i]
            current_service.nli_inf_from[sid] = nli_to_current[i]

        power_nli = power_nli + np.sum(nli_to_current)

    
    nli = power_nli / current_service.launch_power
    ase = current_service.ase_inf / current_service.launch_power
    osnr = nli + ase

    # print("P ASE, NLI", ase, nli)

    osnr = 10 * np.log10(1 / osnr)
    ase = 10 * np.log10(1 / ase)
    nli = 10 * np.log10(1 / nli)
    return osnr, ase, nli

def compute_min_gap_osnr_vectorized(env: RMSAEnv, new_service: Service, path: Path, modulation: Modulation, \
                         spectrum: List[int]):
    beta_2: float = -21.3e-27  
    gamma: float = 1.3e-3  
    h_plank: float = 6.626e-34  
    ase: float = 0
    nli: float = 0
    l_eff_a: float = 0
    l_eff: float = 0
    phi_modulation_format = np.array((1, 1, 2/3, 17/25, 69/100, 13/21))
    service: Service

    nbslots = compute_number_of_slots(new_service.bit_rate, modulation)
    length = [0 for i in range(len(spectrum))]
    fea_length = [0 for i in range(len(spectrum))]
    for i in range(len(spectrum)-1, -1, -1):
        if i == len(spectrum) - 1:
            length[i] = 1 if spectrum[i] == 1 else 0
        else:
            length[i] = 0 if spectrum[i] == 0 else length[i+1]+1

    fea_length = np.zeros(len(spectrum))
    fea_length = np.maximum(np.array(length)-nbslots+1, fea_length)

    eligible_init_slot_index = np.argwhere(fea_length).flatten()
    # print("eligible", eligible_init_slot_index)
    if len(eligible_init_slot_index) == 0:
        return np.zeros(len(spectrum))

    attenuation_normalized = constant.attenuation_db_km / (2 * 10 * np.log10(np.exp(1)) * 1e3)
    noise_figure_normalized = 10 ** (constant.noise_figure_db / 10)

    l_eff = (1 - np.exp(-2 * attenuation_normalized * constant.fiber_span * 1e3)) / (2 * attenuation_normalized)
    nli_coef = (8 / (27 * pi * abs(beta_2))) * gamma ** 2 * l_eff

    set_shared_link_service_id = set()
    shared_link_service:dict[(int, Service)] = dict()

    list_link = list()
    list_service_on_link = list()
    for i in range(len(path.node_list)-1):
        src, dst = path.node_list[i], path.node_list[i+1]
        list_link.append((src, dst))
        list_service:List[Service] = env.topology[src][dst]["running_services"]
        shared_link_service.update([(service.service_id, service) for service in list_service])
        set_shared_link_service_id.update([s.service_id for s in list_service])
        list_service_on_link.append([s.service_id for s in list_service])

    list_shared_link_service_id = list(set_shared_link_service_id)
    span_array = np.array([ceil(env.topology[src][dst]["length"] / constant.fiber_span) for src, dst in list_link])
    
    #array of eligible center freq
    new_service_center_freq = constant.frequency_start \
                        + constant.frequency_slot_bandwidth * eligible_init_slot_index \
                        + constant.frequency_slot_bandwidth * (nbslots / 2.0)
    
    new_service_bandwidth = constant.frequency_slot_bandwidth * nbslots

    span_power_ase = new_service_bandwidth * h_plank * new_service_center_freq * \
        (exp(2 * attenuation_normalized * constant.fiber_span * 1e3) - 1) * noise_figure_normalized
    
    ase_power = np.sum(span_array) * span_power_ase

    phi_sci = asinh(pi ** 2 * abs(beta_2) * (new_service_bandwidth) ** 2 / \
                            (4 * attenuation_normalized))
        
    sci_power = np.sum(span_array) * phi_sci * \
                (env.launch_power / new_service_bandwidth) ** 3 * \
                nli_coef * new_service_bandwidth

    
    if len(list_shared_link_service_id) == 0:
        nli = sci_power / env.launch_power
        ase = ase_power / env.launch_power
        osnr = nli + ase

        osnr = 10 * np.log10(1 / osnr)
        gap = osnr - modulation.minimum_osnr
        result = np.zeros(len(spectrum))
        result[eligible_init_slot_index] = gap
        # for i in range(len(eligible_init_slot_index)):
        #     index = eligible_init_slot_index[i]
        #     result[index] = gap[i]
        return result
    else:
        array_link = np.array([[1 if sid in list_service_on_link[lindex] else 0 for lindex in range(len(span_array))] \
                    for sid in list_shared_link_service_id])

        total_span = np.multiply(span_array, array_link)
        total_span = np.sum(total_span, axis=1)
        
        d_freq = np.array([np.array([abs(shared_link_service[sid].center_frequency - center_freq) \
                for sid in list_shared_link_service_id]) for center_freq in new_service_center_freq])

        shared_service_bandwidth = np.array([shared_link_service[sid].bandwidth for sid in list_shared_link_service_id])

        print("D_FREQ", np.shape(d_freq), d_freq)
        print("SHARED", np.shape(shared_service_bandwidth), shared_service_bandwidth)

        phi_to_current = np.log(abs(d_freq + shared_service_bandwidth/2) / abs(d_freq - shared_service_bandwidth/2)) \
                    - 5 / 3 * (l_eff / (constant.fiber_span * 1e3)) \
                    * np.multiply(np.array([phi_modulation_format[shared_link_service[sid].path.current_modulation.spectral_efficiency - 1] for sid in list_shared_link_service_id]), \
                                  np.divide(shared_service_bandwidth, d_freq))
        
        
        nli_to_current = (env.launch_power / (new_service_bandwidth)) ** 3 * nli_coef * new_service_bandwidth \
                        * np.multiply(total_span, phi_to_current)

        phi_from_current = np.log(abs(d_freq + new_service_bandwidth/2) / abs(d_freq - new_service_bandwidth/2))
        psd_cube = np.array([(shared_link_service[sid].launch_power/shared_link_service[sid].bandwidth)**3 for sid in list_shared_link_service_id])
        prod = np.prod([total_span, psd_cube, shared_service_bandwidth], axis=0)
        nli_from_current = nli_coef * prod * phi_from_current

        power_current_nli = sci_power + np.sum(nli_to_current, axis=1)
        nli = power_current_nli / env.launch_power
        ase = ase_power / env.launch_power
        osnr = nli + ase

        osnr = 10 * np.log10(1 / osnr)
        gap1 = osnr - modulation.minimum_osnr
        # print("GAP1", np.shape(gap1), gap1)

        list_running_service = env.topology.graph["running_services"]
        set_running_service_idx = set([s.service_id for s in list_running_service])

        _nli_power = np.array([sum([v if k in set_running_service_idx else 0 for k,v in shared_link_service[sid].nli_inf_from.items()]) \
              if shared_link_service[sid].nli_inf_from is not None else 0 for sid in list_shared_link_service_id])
        _ase_power = np.array([shared_link_service[sid].ase_inf for sid in list_shared_link_service_id])

        shared_service_noise_power = _nli_power + _ase_power

        
        shared_service_noise_power = shared_service_noise_power + nli_from_current
        osnr = 10 * np.log10(env.launch_power / shared_service_noise_power)
        gap2 = osnr - np.array([shared_link_service[sid].path.current_modulation.minimum_osnr for sid in list_shared_link_service_id])   
        # print("GAP2", np.shape(gap2), gap2)
        min_gap = np.minimum(gap1[:,None], gap2).min(axis=1)
        
        # g = np.column_stack((gap1, gap2))
        # print("GAP", g)
        result = np.zeros(len(spectrum))
        result[eligible_init_slot_index] = min_gap
        return result