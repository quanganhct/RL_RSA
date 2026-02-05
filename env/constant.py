
import numpy as np

fiber_span = 80         #km
gamma = 1.3e-3          #1/(mW km)
h = 6.62607004e-31      #Planck's constant in milli-Joules-seconds
n_sp = 1.58             #spontaneous noise factor
alpha = 0.046/2         #power attenuation neper/km
f0_c = 193.7E12         #lowest frequency of C-band in Hz
ref_bw = 12.5E9         #bandwidth of 1 slot in Hz
beta2 = 21.3E-24        #absolute value of group velocity dispersion in s^2/km, actual value is -21.3E-24
gamma = 1.3e-3          #Nonlinear coefficient in (W km)^-1
roll_off_factor = 0.1   #0: rectangular, 1: triangular

L_eff = (1 - np.exp(-2 * alpha * fiber_span)) / (2 * alpha)
