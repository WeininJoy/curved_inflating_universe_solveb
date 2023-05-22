##### Goal of this code: Plot the result of Hubble Horizon evolution with different K and N_0

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
ks_cons = 0.05/u.Mpc # (comoving wavevector)
H0 = 70 * u.km/u.s/u.Mpc
a0 = c * 10.0/H0

# Data
"""
# Close (K=1)
K = 1 
sigma = [1.1500316486097475e-05, 1.1469857859311461e-05, 1.1440139183478413e-05, 1.1442215985427509e-05, 1.1443582335390184e-05, 1.1445445450489322e-05, 1.1462396048179523e-05, 1.1460193110670108e-05, 1.14345473e-05, 1.14278051e-05]
phi_0 = [9.988840220583233, 7.250475911021399, 6.964558830849157, 6.6579949028031455, 6.5142855187809445, 6.422653227810694, 6.202229802575044, 6.1069710832541535, 5.90369084, 5.86751537]
N_0_as = [11.72, 11.73, 11.74, 11.76, 11.78, 11.8, 11.9, 12.0, 13.0, 14.0]
N_0 = [N_0_as[i] + np.log(sigma[i]) for i in range(len(sigma))]
"""

# Open (K=-1)
K = -1
sigma = [1.03002908e-05, 9.97935022e-06, 1.1375366390496873e-05, 1.1438332734639645e-05, 1.1427262120574145e-05, 1.1424624667273704e-05, 1.14303575e-05, 1.14276805e-05]
phi_0 = [3.338373783463738, 5.00201368, 5.406349850473778, 5.493478795487219, 5.645057104655267, 5.815553924592122, 5.87036075, 5.86300853]
N_0_as = [7, 8, 9, 10, 11, 12, 13, 14]
N_0 = [N_0_as[i] + np.log(sigma[i]) for i in range(len(sigma))]

# Data storage
t_tot_array = []
sol_tot_array = []
Hubble_Horizon_array = []

# Define functions

def potential(phi):
    return V0 * (1.0 - np.exp(- math.sqrt(2.0/3.0)* phi/m_p ) )**2

def potential_prime(phi):
    return 2.0* math.sqrt(2.0/3.0) * V0/m_p *(1.0- np.exp(- math.sqrt(2.0/3.0)* phi/m_p)) *np.exp(- math.sqrt(2.0/3.0)* phi/m_p)

def phi_dot(phi): # when start of inflation, V = phi_prime**2
    return - math.sqrt(V0)*( 1.0- np.exp(- math.sqrt(2.0/3.0)* phi/m_p))

def N_dot(phi, N): # when start of inflation
    return math.sqrt( 1.0/(2.0*m_p**2) * phi_dot(phi)**2 - K* np.exp(-2.0*N) )

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
    d2Ndt = -1.0/(2.0*m_p**2) * dphidt**2 + K* np.exp(-2*N)

    return [dphidt, dNdt, d2phidt, d2Ndt]

# inflating event
def inflating(t,y):
    phi = y[0]
    dphidt = y[2]
    return dphidt**2 - potential(phi)

inflating.terminal = True
inflating.direction = 1

for k in range(len(N_0)):

    # solve ODEs to get solution of BG variables 
    y0 = [phi_0[k], N_0[k], phi_dot(phi_0[k]), N_dot(phi_0[k], N_0[k])]
    sol_inf = solve_ivp(odes, [0, 1.e5], y0, method='RK45', events=inflating)


    # find the time when Big Bang started
    def BBstart(t, y):
        N = y[1]
        a = np.exp(N)
        return a - 1.e-6

    BBstart.terminal = True
    BBstart.direction = -1

    # Solve ODEs to get solution during kinetic dominance
    y0 = [phi_0[k], N_0[k], phi_dot(phi_0[k]), N_dot(phi_0[k], N_0[k])]
    sol_KD = solve_ivp(odes, [0, -1.e5], y0, method='RK45', events=BBstart)

    # Combine solutions of KD and inflation together
    t_tot = np.concatenate((np.flipud(sol_KD.t), sol_inf.t), axis=0)
    sol_tot = np.concatenate((np.fliplr(sol_KD.y), sol_inf.y), axis=1)

    # Shift t such that the universe start from t=0
    for i in range(len(t_tot)):
        t_tot[i] = t_tot[i] - sol_KD.t[-1]

    # calculate 1/(aH)
    Hubble_Horizon = []
    for i in range(len(sol_tot[0])):
        Hubble_Horizon.append( a0.to(u.Mpc).value / (sol_tot[3][i]*np.exp(sol_tot[1][i])) )
    
    # Scaling all variables
    for i in range(len(t_tot)):
        t_tot[i] = t_tot[i] / sigma[k]
        sol_tot[1][i] = sol_tot[1][i] - np.log(sigma[k])
        sol_tot[2][i] = sol_tot[2][i] * sigma[k]
        sol_tot[3][i] = sol_tot[3][i] * sigma[k]

    # Store data
    t_tot_array.append(t_tot)
    Hubble_Horizon_array.append(Hubble_Horizon)
    sol_tot_array.append(sol_tot)

# Plot all lines
for k in range(len(phi_0)):
    plt.semilogy(sol_tot_array[k][1], Hubble_Horizon_array[k], c='g', alpha=1.-(1./len(N_0))*k,label='Ni='+str(N_0_as[k]))
    
plt.semilogy(sol_tot_array[0][1], [(1./ks_cons).to(u.Mpc).value for i in range(len(sol_tot_array[0][1]))] )
plt.xlim([-2, 75])
plt.ylim([1.e-22, 1.e9])
#plt.xlim([-1, 16])
#plt.ylim([3.e2, 1.e8])
plt.xlabel('N')
plt.ylabel('a0*(aH)^-1 / Mpc')
plt.title('N - a0*(aH)^-1 / Mpc')
plt.legend()
plt.show()
