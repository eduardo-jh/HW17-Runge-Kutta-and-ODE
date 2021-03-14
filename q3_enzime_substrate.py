#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BE523 Biosystems Analysis & Design
HW17 - Question 3. Enzime and substrate reaction
       Example from Dunn, Biomedical Engineering textbook

Created on Sun Mar 14 00:01:51 2021
@author: eduardo
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def enzime(C, t, k1, k_1, k2):
    """ Enzyme E reacts with substrate S, forms ES which then forms the
    product P and E, or goes back to S and E. 
    C: list, initial state variables
    t: list, time steps
    ki: coefficients
    """
    S, E, ES, P = C
    d_S = -k1*S*E + k_1*ES  # S:substrate
    d_E = -k1*S*E + k_1*ES + k2*ES  # E: enzime
    d_ES = k1*S*E - k_1*ES - k2*ES  # ES: intermediate complex
    d_P = k2*ES  # P: product
    return [d_S, d_E, d_ES, d_P]

dt = 1  # time step
steps = 1000  # number of time steps
init_state = [0.1, 0.1, 0, 0]  # initial state variables
t = np.linspace(0, steps, int(steps/dt)+1)
k1 = 0.1
k2 = 0.3
k_1 = 0.1

# Get a solution to the differential equations using 'odeint'
C = odeint(enzime, init_state, t, args = (k1,k_1,k2))

plt.figure(0)
plt.plot(t, C[:,0], 'g', label = 'S')
plt.plot(t, C[:,1], 'm', label = 'E')
plt.plot(t, C[:,2], 'r', label = 'ES')
plt.plot(t, C[:,3], 'b', label = 'P' )
plt.legend(loc='best')
plt.xlabel('Time (seconds)')
plt.ylabel('Concentration (moles/L)')
plt.savefig('q3_enzime_substrate.png', dpi=300, bbox_inches='tight')