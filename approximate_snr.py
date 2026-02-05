# -*- coding: utf-8 -*-

import math
import numpy as np

space = 12.5 * 10**12
ref_bw = 12.5e+9    # Mohammad, 12.5 Hz
#alpha = 0.0507                  # unit (km^-1)
alpha = 0.046/2
#h = 6.62607004 * 10**(-34)      # unit (m^2 kg s^-1)
h = 6.62607004e-31  # Mohammad, Planck's constant in milli-Joules-seconds to be consistent with Power in milli-Watts

v = 193.55 * 10**12             # unit (Hz)
nsp = 1.58                      # unitless

#v = 191 * 10**12
#v = 193.55 * 10**3             # unit (GHz)
hvn_sp = h * v * nsp * 10**9      # unit (W GHz^-1)

fiber_span = 80                # unit (km)

#gamma = 1.3                     # unit (W^-1 km^-1)
gamma = 1.3e-3#Mohammad, in 1/(mW Km)
beta2 = -21.3 * 10**-24         # unit (s^2 km^-1)
#beta2 = -8.87 * 10**-24         # unit (s^2 km^-1)
mu = 3 * gamma**2 / (2 * math.pi * alpha * np.abs(beta2)) * 10**-18
rho = math.pi**2 * np.abs(beta2) / (2 * alpha) * 10**18

#mu = 7.47 * 10**5               # unit (W^-2 GHz^2)
#rho = 2.073 * 10**-3            # unit (GHz^-2)

#space = 12.5 * 10**9           # unit (Hz)
space = 12.5                    # unit (GHz)


def approximate_XCI(nb_channel, bandwidth, used_bandwidth, nb_span, G):
    L_eff = (1 - np.exp(-2 * alpha * fiber_span)) / (2 * alpha)
    print("L_eff", L_eff)
    L_eff_a = 1/(2 * alpha)
    print("L_eff_a", L_eff)
    d = np.pi * L_eff_a * abs(beta2)# added by mohammad, denominator: Pi*|beta2|*Leff_a
    print("d", d)
    coeff=nb_span * 8/27 * (gamma**2 * G**3 * L_eff**2)/d
    print("coeff", coeff)
    Nch_factor = nb_channel ** (2 * (bandwidth/used_bandwidth))
    print("Nch_factor", Nch_factor)
    #arcsinhCoeff=np.arcsinh(.5*math.pi**2 * np.abs(beta2) * L_eff_a * bandwidth**2 * math.pow(nb_channel, 2 * bandwidth/used_bandwidth))
    arcsinhCoeff=np.arcsinh(.5*math.pi**2 * np.abs(beta2) * L_eff_a * bandwidth**2 * Nch_factor)

    print("arcsinhCoeff", arcsinhCoeff)
    G_NLI = (coeff)* np.arcsinh(.5*math.pi**2 * np.abs(beta2) * L_eff_a * bandwidth**2 * math.pow(nb_channel, 2 * bandwidth/used_bandwidth))
    den=G_NLI*coeff
    print("den", den)
    print("G_NLI", G_NLI)
    return G_NLI
#%%
    
#G = 4E-05 # W/GHz
G = 4 * 1e-11    #Mohammad, in mW/(Hz)
GOSNR_list=[]
for nb_channel in range(1,89):
    #nb_channel = 1
    print(nb_channel)
    #bandwidth = 35 #GHz
    bandwidth =  35e+9  # Mohammad, channel bandwidth in GHz

    used_bandwidth = 50e+9 #Mohammad, in Hz
    nb_span = 10
    G_NLI = approximate_XCI(nb_channel, bandwidth, used_bandwidth, nb_span, G)
    gnli_power =G_NLI*ref_bw#Mohammad, in mW
    print("G_NLI", G_NLI)
    print("gnli_power", gnli_power)
    print("SNR without ASE", G/G_NLI)

    #The followuing is written by Mohammad

    # ASE
    EDFA_NF = 5.01  # Amp Noise Figure (NOT in dB)
    fcns = 193.41  # center freq of span in THz
    fcut = 193.71895e+12     # channel under investigation in THz

    EDFA_Gain = np.e**(2 * alpha * fiber_span)
    ASE = nb_span * EDFA_NF * EDFA_Gain * h * fcut * ref_bw
    print("ASE", ASE)

    OSNR = (G * ref_bw) / ASE
    OSNR_dB = 10 * np.log10(OSNR)
    print("OSNR", OSNR)
    print("OSNR_dB", OSNR_dB)
    #%%
    GOSNR = (G * ref_bw) / (ASE + gnli_power)
    GOSNR_dB = 10 * np.log10(GOSNR)
    print("GOSNR_dB", GOSNR_dB)
    # Optimal Power
    p_opt = np.power(ASE/(2*gnli_power), 1/3)
    GOSNR_optimal = p_opt/(ASE + gnli_power)
    GOSNR_optimal_dB = 10 * np.log10(GOSNR_optimal)
    GOSNR_list.append(GOSNR_optimal_dB)

    print("GOSNR_optimal_dB", GOSNR_optimal_dB)
print("GOSNR_optimal_dB", GOSNR_list)
