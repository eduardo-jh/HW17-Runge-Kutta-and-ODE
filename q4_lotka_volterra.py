#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BE523 Biosystems Analysis & Design
HW17 - Question 4. Lotka-Volterra, without resource limitation

Created on Sun Mar 14 00:18:39 2021
@author: eduardo
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def lotkavolterra(P0, t, coeff):
    """ Lotka-Volterra function for voles and owls
    P0: list, initial population of voles and owls
    t: list, time vector
    coeff: list, coefficients alpha, gamma, delta, epsilon
    """
    V, O = P0
    alpha, gamma, delta, epsilon = coeff
    dV = V * (alpha - gamma*O)
    dO = O * (-delta + epsilon*V)
    return [dV, dO]

dt = 1
steps = 500

delta = 0.02
alpha = 0.06
gamma = 1e-3
epsilon = 2e-4
coefficients = (alpha, gamma, delta, epsilon)
P0 = [40, 40]  # initial population of voles and owls

t = np.linspace(0, steps, int(steps/dt)+1)

# Get a solution to the differential equations using 'odeint'
P = odeint(lotkavolterra, P0, t, args=(coefficients,))

plt.figure(0)
plt.plot(t, P[:,0], 'b-', label='V')
plt.plot(t, P[:,1], 'r-', label='O')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend(loc='best')
plt.savefig('q4_lotka_volterra.png', dpi=300, bbox_inches='tight')
plt.show()