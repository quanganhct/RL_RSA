from math import pi

from math import log, exp, asinh, log10, ceil
from env import constant

import numpy as np

import typing

from custom_env.optical_rl_gym.envs.rmsa_env import RMSAEnv, Service


def calculate_osnr(env: RMSAEnv, current_service: Service):
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

    # print("#"*30)
    # print("Service:", current_service)
    # acc_gsnr = 0

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
                phi = np.log(abs(d_frequency + service.bandwidth/2) / abs(d_frequency - service.bandwidth/2))
                # - \
                #     (phi_modulation_format[service.path.current_modulation.spectral_efficiency - 1] * \
                #         (service.bandwidth / abs(service.center_frequency - current_service.center_frequency)) * \
                #         5 / 3 * (l_eff / (constant.fiber_span * 1e3)))
                
                sum_phi += phi

        power_nli_span = nb_span * (current_service.launch_power / (current_service.bandwidth)) ** 3 * \
            (8 / (27 * pi * abs(beta_2))) * gamma ** 2 * l_eff * sum_phi * current_service.bandwidth
        power_ase = nb_span * h_plank * current_service.center_frequency * \
            (exp(2 * attenuation_normalized * constant.fiber_span * 1e3) - 1) * noise_figure_normalized

        acc_gsnr = acc_gsnr + 1 / (current_service.launch_power / (power_ase + power_nli_span))
        acc_ase = acc_ase + 1 / (env.current_service.launch_power / power_ase)
        acc_nli = acc_nli + 1 / (env.current_service.launch_power / power_nli_span)

    # print("acc_gsnr", acc_gsnr)
    # print("acc_nli", acc_nli)
    gsnr = 10 * np.log10(1 / acc_gsnr)
    ase = 10 * np.log10(1 / acc_ase)
    nli = 10 * np.log10(1 / acc_nli)
    return gsnr, ase, nli

    # for link in current_service.path.links:
    #     # print("\tLink:", link)
    #     # node1 = env.current_service.path.node_list[index]
    #     # node2 = env.current_service.path.node_list[index + 1]
    #     # gsnr_link = 0
    #     for span in env.topology[link.node1][link.node2]["link"].spans:
    #         # print("\t\tSpan:", span)
    #         l_eff_a = 1 / (2 * attenuation_normalized)
    #         # print(f"\t\t\t{l_eff_a=}")
    #         l_eff = (
    #             1 - np.exp(-2 * attenuation_normalized * span.length * 1e3)
    #         ) / (
    #             2 * attenuation_normalized
    #         )
    #         # print(f"\t\t\t{l_eff=}")

    #         # calculate SCI
    #         sum_phi = asinh(
    #             pi ** 2 * \
    #             abs(beta_2) * \
    #             (current_service.bandwidth) ** 2 / \
    #             (4 * attenuation_normalized)
    #         )

    #         # print(f"\t\t\t{sum_phi=}")

    #         for service in env.topology[link.node1][link.node2]["running_services"]:
    #             # if service.center_frequency - current_service.center_frequency == 0:
    #             #     print(service)
    #             #     print(current_service)
    #             if service.service_id != current_service.service_id:
    #                 # print(f"\t\t\t\t{service=}")
    #                 phi = (
    #                     asinh(
    #                         pi ** 2 * \
    #                         abs(beta_2) * \
    #                         l_eff_a * \
    #                         service.bandwidth * \
    #                         (  # TODO: double-check this part below
    #                             service.center_frequency - \
    #                             current_service.center_frequency + \
    #                             (service.bandwidth / 2)
    #                         )
    #                     ) - \
    #                     asinh(
    #                         pi ** 2 * \
    #                         abs(beta_2) * \
    #                         l_eff_a * \
    #                         service.bandwidth * \
    #                         (  # TODO: double-check this part below
    #                             service.center_frequency - \
    #                             current_service.center_frequency - \
    #                             (service.bandwidth / 2)
    #                         )
    #                     )
    #                 ) - \
    #                 (
    #                     phi_modulation_format[service.current_modulation.spectral_efficiency - 1] * \
    #                     (
    #                         service.bandwidth / \
    #                         abs(service.center_frequency - current_service.center_frequency)
    #                     ) * \
    #                     5 / 3 * \
    #                     (l_eff / (span.length * 1e3))
    #                 )
    #                 # print(f"\t\t\t\t{phi=}")
    #             sum_phi += phi
    #             # print(f"\t\t\t\t{sum_phi=}")

    #         power_nli_span = (current_service.launch_power / (current_service.bandwidth)) ** 3 * \
    #         (8 / (27 * pi * abs(beta_2))) * gamma ** 2 * l_eff * sum_phi * current_service.bandwidth
    #         power_ase = current_service.bandwidth * h_plank * current_service.center_frequency * \
    #             (exp(2 * attenuation_normalized * span.length * 1e3) - 1) * span.noise_figure_normalized

    #         acc_gsnr = acc_gsnr + 1 / (current_service.launch_power / (power_ase + power_nli_span))
    #         acc_ase = acc_ase + 1 / (env.current_service.launch_power / power_ase)
    #         acc_nli = acc_nli + 1 / (env.current_service.launch_power / power_nli_span)

    # gsnr = 10 * np.log10(1 / acc_gsnr)
    # ase = 10 * np.log10(1 / acc_ase)
    # nli = 10 * np.log10(1 / acc_nli)
    # return gsnr, ase, nli