#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BE523 Biosystems Analysis & Design
HW17 - Question 1. Binary fission for bacteria growth

Created on Sat Mar 13 17:38:16 2021
@author: eduardo
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def bacteria(P0, time, r):
    """ Bacteria population with binary fission
    P0: initial population
    time: time vector to solve the differential equation dP/dt
    r: growth factor for the differential equation
    """
    P = P0
    dP = r * P # Differential eq.: dP/dt = r*P
    return dP

B0 = 0.1  # initial concentration
dt = 1  # time step
N_t = 10  # number of steps
mu = 0.5  # growth rate

t = np.linspace(0, N_t, int(N_t/dt)+1)
BEuler = np.zeros(len(t))
BCN = np.zeros(len(t))
BEuler[0] = B0
BCN[0] = B0

for n in range(1, len(t)):
    # Numerical solution Euler
    BEuler[n] = BEuler[n-1] + mu*BEuler[n-1]*dt
    # Numerical solution Crank-Nicholson
    dB1 = mu * dt * BCN[n-1]
    B_future = BCN[n-1] + dB1  # B_future is estimate for BCN[n+1]
    dB2 = mu * dt * B_future
    dBavg = (dB1 + dB2)/2
    BCN[n] = BCN[n-1] + dBavg

# Prediction with exponential eq. (analytic solution)
Bexp = B0 * np.exp(mu*t)

# Numerical solution using 'odeint' to solve the differential equation
B = odeint(bacteria, B0, t, args=(mu,))

plt.plot(t, BEuler, 'bo', label='Numerical (Euler)')
plt.plot(t, BCN, 'c+', label='Numerical (C-N)')
plt.plot(t, B, 'kx', label='Numerical (odeint)')
plt.plot(t, Bexp, 'r-', label=r'Analytic Sol. $\mu=$%.2f' % mu)
plt.legend(loc='best')
plt.xlabel('Time (days)'); plt.ylabel('Bacteria population')
plt.savefig('q1_binary_fission.png', dpi=300, bbox_inches='tight')
