##### Goal of this code: Plot the result of BG variables with different K, N_0 and sigma

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import optimize
from astropy import units as u
from astropy.constants import c

# Planck constant
t_p = 1.0 # second
l_p = 1.0 # km
m_p = 1.0 # reduced planck mass = \sqrt(c*hbar/8/pi/G) ??

# constants
V0 = m_p**4
K = [-1,-1,-1,-1,0,1,1,1,1,1,1,1,1,1]
Num_K_minus1 = 4
Num_K_1 = len(K) - Num_K_minus1 -1
sigma = [1.1375366390496873e-05, 1.1438332734639645e-05, 1.1427262120574145e-05, 1.1424624667273704e-05, 1.1438151843821638e-05, 1.59741599e-05, 1.1500316486097475e-05, 1.1469857859311461e-05, 1.1440139183478413e-05, 1.1442215985427509e-05, 1.1443582335390184e-05, 1.1445445450489322e-05, 1.1462396048179523e-05, 1.1460193110670108e-05]
phi_0 = [5.406168755538037, 5.4933025115388725, 5.6448673222896995, 5.815354804140279, 5.906947898026466, 6.08481575, 9.988837324387065, 7.250335673539723, 6.96436430183166, 6.6577897135531, 6.514081191532225, 6.422452274776457, 6.2020686549600486, 6.106810584463498]
N_0_as = [9, 10, 11, 12, 12, 11.713, 11.72, 11.73, 11.74, 11.76, 11.78, 11.8, 11.9, 12.0]
N_0_bs = [N_0_as[i] + np.log(sigma[i]) for i in range(len(sigma))] 

# Data storage
t_tot_array = []
sol_tot_array = []
w_phi_array = []


for k in range(len(phi_0)):

    # Define functions

    def potential(phi):
        return V0 * (1.0 - np.exp(- math.sqrt(2.0/3.0)* phi/m_p ) )**2

    def potential_prime(phi):
        return 2.0* math.sqrt(2.0/3.0) * V0/m_p *(1.0- np.exp(- math.sqrt(2.0/3.0)* phi/m_p)) *np.exp(- math.sqrt(2.0/3.0)* phi/m_p)

    def phi_dot(phi): # when start of inflation, V = phi_prime**2
        return - math.sqrt(V0)*( 1.0- np.exp(- math.sqrt(2.0/3.0)* phi/m_p))

    def N_dot(phi, N): # when start of inflation
        return math.sqrt( 1.0/(2.0*m_p**2) * phi_dot(phi)**2 - K[k]* np.exp(-2.0*N) )

    def P_R(phi_dot, N_dot): # N_dot = H
        return ( N_dot**2/2.0/np.pi/phi_dot )**2

    # define ODEs
    def odes(t, y):
        phi = y[0]
        N = y[1]
        dphidt = y[2]
        dNdt = y[3]

        # define each ODE
        d2phidt = - 3.0*dNdt*dphidt - potential_prime(phi)
        d2Ndt = -1.0/(2.0*m_p**2) * dphidt**2 + K[k]* np.exp(-2*N)

        return [dphidt, dNdt, d2phidt, d2Ndt]

    # inflating event
    def inflating(t,y):
        phi = y[0]
        dphidt = y[2]
        return dphidt**2 - potential(phi)

    inflating.terminal = True
    inflating.direction = 1

    # solve ODEs to get solution of BG variables 
    y0 = [phi_0[k], N_0_bs[k], phi_dot(phi_0[k]), N_dot(phi_0[k], N_0_bs[k])]
    sol_inf = solve_ivp(odes, [0, 1.e8], y0, method='RK45', events=inflating)

    # find the time when Big Bang started
    def BBstart(t, y):
        N = y[1]
        a = np.exp(N)
        return a - 1.e-6

    BBstart.terminal = True
    BBstart.direction = -1

    # Solve ODEs to get solution during kinetic dominance
    y0 = [phi_0[k], N_0_bs[k], phi_dot(phi_0[k]), N_dot(phi_0[k], N_0_bs[k])]
    sol_KD = solve_ivp(odes, [0, -1.e5], y0, method='RK45', events=BBstart)

    # Combine solutions of KD and inflation together
    t_tot = np.concatenate((np.flipud(sol_KD.t), sol_inf.t), axis=0)
    sol_tot = np.concatenate((np.fliplr(sol_KD.y), sol_inf.y), axis=1)

    # Shift t such that the universe start from t=0
    for i in range(len(t_tot)):
        t_tot[i] = t_tot[i] - sol_KD.t[-1]

    # calculate equation of state (w = p/rho)
    w_phi = []
    for i in range(len(sol_tot[0])):
        p_phi = 0.5* sol_tot[2][i]**2 - potential(sol_tot[0][i]) 
        rho_phi = 0.5* sol_tot[2][i]**2 + potential(sol_tot[0][i])
        w_phi.append(p_phi/rho_phi)

    # Scaling all variables
    for i in range(len(t_tot)):
        t_tot[i] = t_tot[i] / sigma[k]
        sol_tot[1][i] = sol_tot[1][i] - np.log(sigma[k])
        sol_tot[2][i] = sol_tot[2][i] * sigma[k]
        sol_tot[3][i] = sol_tot[3][i] * sigma[k]

    # Store data
    t_tot_array.append(t_tot)
    sol_tot_array.append(sol_tot)
    w_phi_array.append(w_phi)

# plot solution of BG variables

color_K_1 = 0.0
color_K_minus1 = 0.0
for k in range(len(phi_0)):
    if k< Num_K_minus1:
        plt.plot(sol_tot_array[k][1], sol_tot_array[k][2], c='g', alpha=1.-(1./Num_K_minus1)*color_K_minus1,label='K,Ni='+str(K[k])+','+str(N_0_as[k]))
        color_K_minus1 += 1
    elif k==Num_K_minus1:
        plt.plot(sol_tot_array[k][1], sol_tot_array[k][2], c='r', label='K,Ni='+str(K[k])+','+str(N_0_as[k]))
    else:
        plt.plot(sol_tot_array[k][1], sol_tot_array[k][2], c='b', alpha=1.-(1./Num_K_1)*color_K_1, label='K,Ni='+str(K[k])+','+str(N_0_as[k]))
        color_K_1 += 1

plt.yscale('symlog', linthresh=1.e-5)
#plt.yscale('log')
#plt.xscale('log')
#plt.xlim([8.e0, 2.e7])
plt.xlim([-2, 75])
plt.ylim([-1.e-1, 2.e-5])
plt.xlabel('N')
plt.ylabel('phi_dot')
plt.title('N - phi_dot')
plt.legend()
plt.show()
