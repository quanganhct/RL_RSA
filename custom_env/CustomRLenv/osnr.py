from math import pi

from math import log, exp, asinh, log10, ceil
from env import constant

import numpy as np

import typing

from custom_env.optical_rl_gym.envs.rmsa_env import RMSAEnv, Service


def calculate_osnr(env: RMSAEnv, current_service: Service):
    if not current_service.accepted and current_service not in env.topology.graph["running_services"]:
        return None, None, None

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
                # phi = asinh(pi ** 2 * abs(beta_2) * l_eff_a * service.bandwidth * \
                #             (d_frequency + (service.bandwidth / 2))) - \
                #         asinh(pi ** 2 * abs(beta_2) * l_eff_a * service.bandwidth * \
                #             (d_frequency - (service.bandwidth / 2))
                #         ) 
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


def compute_ase_nli(env: RMSAEnv, current_service: Service, update_old_service=True):
    if not current_service.accepted and current_service not in env.topology.graph["running_services"]:
        return None, None, None
    
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

    power_nli = sum([current_service.nli_inf_from[sid] for sid in current_service.nli_inf_from.keys()])
    nli = power_nli / current_service.launch_power
    ase = current_service.ase_inf / current_service.launch_power
    osnr = nli + ase

    osnr = 10 * np.log10(1 / osnr)
    ase = 10 * np.log10(1 / ase)
    nli = 10 * np.log10(1 / nli)

    return osnr, ase, nli