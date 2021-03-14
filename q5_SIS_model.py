#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BE523 Biosystems Analysis & Design
HW17 - Question 5. SIS model with born and death rates

Created on Sun Mar 14 00:41:52 2021
@author: eduardo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def sismodel(P0, t, coeff):
    """ SIS model with born and death rates
    P0: list, initial population
    t: list, time vector
    coeff: list, coefficients
    """
    S, I = P0
    alpha, gamma, epsilon = coeff
    dI = I * (alpha * S - gamma)
    dS = S * (-alpha * I + epsilon)
    return [dS, dI]

alpha = 0.005  #rate of infection 1/person-hr
gamma = 0.02 # hrs for recovery
epsilon = 0.05 # initial number of infectives
coefficients = (alpha, gamma, epsilon)

steps = 1000  # time of simulation
dt = 0.01  # time step, hours
N = 18  # total population
S0 = 10
I0 = N - S0
P0 = [S0, I0]  # initial population

t = np.linspace(0, steps, int(steps/dt)+1)

# Get a solution to the differential equations using 'odeint'
P = odeint(sismodel, P0, t, args=(coefficients,))

plt.figure(0)
plt.plot(t, P[:,0], label='S')
plt.plot(t, P[:,1], label='I')
plt.legend(loc='best')
plt.xlabel('Time (hours)')
plt.ylabel('Population')
plt.savefig('q5_SIS_model.png', dpi=300, bbox_inches='tight')
plt.show()
