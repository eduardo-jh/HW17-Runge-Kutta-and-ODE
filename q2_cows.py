#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BE523 Biosystems Analysis & Design
HW17 - Question 2. Logistic model of cows

Created on Sat Mar 13 19:01:08 2021
@author: eduardo
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def cows(P0, t, coeff):
    """ Cows model differential equation
    P0: int, initial cow population
    t: list, time vector
    coeff: list, coefficients r and K
    """
    P = P0
    r, K = coeff
    dP = r * (K - P) * P  # differential equation dP/dt
    return dP

dt = 1  # time step in years
P0 = 2
steps = 100
sell = 52  # one cow per week, in one year=52
r = 0.001  # growth rate, per year
K = 500  # carrying capacity

t = np.array(range(0, steps+1, dt))
# t = np.linspace(0, steps, int(steps/dt)+1)  # when dt is not integer
P = np.zeros(len(t))
Psell = np.zeros(len(t))
P[0], Psell[0] = P0, P0

dPselldt = np.zeros(len(t))
dPdt = np.zeros(len(t))
dPdt2 = np.zeros(K+1)

# Numerical solutions
for i in range(1, len(t)):
    ################### No sell ##################
    # Numerical solution, Euler
    dPdt[i] = r*(K - P[i-1])*P[i-1]
    P[i] = P[i-1] + dPdt[i]*dt
    ################### Selling cows #############
    dPselldt[i] = r*(K - Psell[i-1])*Psell[i-1]
    # Psell[i] = Psell[i-1] + dPselldt[i]*dt - sell
    if Psell[i-1] > sell:
        Psell[i] = Psell[i-1] + dPselldt[i]*dt - sell
    else:
        Psell[i] = Psell[i-1] + dPselldt[i]*dt

# Get a solution to the differential equations using 'odeint'
coefficients = (r, K)
P_ode = odeint(cows, P0, t, args=(coefficients,))

# Make predictions with the exponential eq. (analytic solution)
# dt = 0.1
r=1
# time = np.linspace(0, steps, int(steps/dt)+1)
# Pana = (K * P0 * np.exp(r*time)) / (K + P0 * (np.exp(r*time) - 1))
Pana = (K * P0 * np.exp(r*t)) / (K + P0 * (np.exp(r*t) - 1))

# # dP/dt2
# for i in range(1, K):
#     dPdt2[i] = r*(K-i)*i - sell

plt.figure(0)
plt.plot(t, P, 'b-', label='Numerical (Euler)')
plt.plot(t, P_ode[:,0], 'g:', label='Numerical (odeint)')
plt.plot(t, Psell, 'r--', label='Cows selling %d/yr' % sell)
plt.plot(t, Pana, 'c--', label='Analytic sol.')
plt.legend(loc='best')
plt.xlabel('Time (years)')
plt.ylabel('Cows')
plt.grid()
plt.savefig('q2_cows.png', dpi=300, bbox_inches='tight')

# plt.figure(1)
# plt.plot(range(K+1), dPdt2, 'b-')
# plt.xlabel('K')
# plt.ylabel('dP/dt2')
# plt.grid()
# # plt.savefig('q10_cows_dPdt2.png', dpi=300, bbox_inches='tight')