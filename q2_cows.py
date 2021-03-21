#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BE523 Biosystems Analysis & Design
HW17 - Question 2. Logistic model of cows
       Haley's solution

Created on Wed Mar 17 14:28:44 2021
@author: eduardo
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

dt = 1
N_t = 12
t = np.linspace(0, N_t*dt, N_t+1)
P_CN = np.zeros(len(t))
P_3RK = np.zeros(len(t))
P_4RK = np.zeros(len(t))
r = 1
K = 500
P0 = 2

P_Euler = np.zeros(len(t))
P_CN[0] = P_Euler[0] = P_3RK[0] = P_4RK[0] = P0

def cows(P0, t, coeff):
    """ Cows model differential equation
    P0: int, initial cow population
    t: list, time vector
    coeff: list, coefficients r and K
    """
    P = P0
    r, K = coeff
    dP = r * (1 - P/K) * P  # differential equation dP/dt
    return dP

for i in range(1, len(t)):
    #**************************** Euler ***************************************
    P_Euler[i] = P_Euler[i-1] + r*P_Euler[i-1]*(1-(P_Euler[i-1]/K))*dt
    #************************ Crank-Nicholson *********************************
    dP1 = r * dt * P_CN[i-1] * (1-P_CN[i-1]/K)
    P_future = P_CN[i-1] * dP1
    dP2 = r * dt * P_future * (1-P_future/K)
    dPavg = (dP1+dP2)/2
    P_CN[i] = P_CN[i-1] + dPavg
    # ********************** 3rd order Runge-Kutta ****************************
    k1 = r * dt * P_3RK[i-1] * (1 - P_3RK[i-1]/K)
    P_half = P_3RK[i-1] + k1 / 2
    k2 = P_half * r * (1 - P_half/K) * dt
    P_future_3RK = P_3RK[i-1] + 2 * k2 - k1
    k3 = r * (1-P_future_3RK/K) * dt * P_future_3RK
    k_avg = (k1 + 4*k2 + k3) / 6
    P_3RK[i] = P_3RK[i-1] + k_avg
    # ********************** 4th order Runge-Kutta ****************************
    k1_4RK = r * P_4RK[i-1] * (1-P_4RK[i-1]/K) * dt
    P_half_4RK =P_4RK[i-1] + k1_4RK/2
    k2_4RK = r * P_half_4RK * (1 - P_half_4RK/K) * dt
    P_half2 = P_4RK[i-1] + k2_4RK/2
    k3_4RK = r * P_half2 * (1 - P_half2/K) * dt
    P_future_4RK = P_4RK[i-1] + k3_4RK
    k4_4RK = r * P_future_4RK * (1 - P_future_4RK/K) * dt
    k_avg_4RK = (k1_4RK + 2*k2_4RK + 2*k3_4RK + k4_4RK)/6
    P_4RK[i] = P_4RK[i-1] + k_avg_4RK

# Get a solution to the differential equations using 'odeint'
coefficients = (r, K)
P_ode = odeint(cows, P0, t, args=(coefficients,))

# Make predictions with the exponential eq. (analytic solution)
Pana = (K * P0 * np.exp(r*t)) / (K + P0 * (np.exp(r*t) - 1))

plt.figure(0)
plt.plot(t, Pana, 'b-', label='Analytic solution')
plt.plot(t, P_Euler, 'c--', label='Numerical (Euler)')
plt.plot(t, P_ode[:,0], 'go', label='Numerical (odeint)')
plt.plot(t, P_3RK, 'k+', label='Numerical (3rd RK)')
plt.plot(t, P_4RK, 'rx', label='Numerical (4th RK)')
plt.legend(loc='best')
plt.xlabel('Time (years)')
plt.ylabel('Cows')
plt.grid()
plt.savefig('q2_cows.png', dpi=300, bbox_inches='tight')
