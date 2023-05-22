##### Goal of this code: Solve and plot Primordial power spectrum (PPS)

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
sigma = 1.1424624667273704e-05
phi_0 = 5.815354804140279
V0 = m_p**4
N_0 = 12.0 + np.log(sigma)  # N_paper = ln(a/lp) = 10.0

# Cross horizon (k=0.05 Mpc^-1)
H0 = 70 * u.km/u.s/u.Mpc
K = -1.0
if K>0: Omega_K = - 0.01
else: Omega_K = 0.01
a0 = c * np.sqrt(-K/Omega_K)/H0
ks_cons = 0.05/u.Mpc # (comoving wavevector)
As_cons = 2.e-9


### Define functions

# functions for solving BG variables

def potential(phi):
    return V0 * (1.0 - np.exp(- math.sqrt(2.0/3.0)* phi/m_p ) )**2

def potential_prime(phi):
    return 2.0* math.sqrt(2.0/3.0) * V0/m_p *(1.0- np.exp(- math.sqrt(2.0/3.0)* phi/m_p)) *np.exp(- math.sqrt(2.0/3.0)* phi/m_p)

def phi_dot_initial(phi): # when start of inflation, V = phi_prime**2
    return - math.sqrt(V0)*( 1.0- np.exp(- math.sqrt(2.0/3.0)* phi/m_p))

def N_dot_initial(phi, N): # when start of inflation
    return math.sqrt( 1.0/(2.0*m_p**2) * phi_dot_initial(phi)**2 - K* np.exp(-2.0*N) )

def P_R(phi_dot, N_dot): # N_dot = H
    return ( N_dot**2/2.0/np.pi/phi_dot )**2

# functions for solving R

def epsilon_R(phi_dot, N_dot):
    return phi_dot**2 / (2.* N_dot**2)

def xi_R(phi, phi_dot, N, N_dot):
    return 2.* kappa_R_2/ (kappa_R_2 + K*epsilon_R(phi_dot, N_dot)) * ( epsilon_R(phi_dot, N_dot) -3.0 - potential_prime(phi)/N_dot/phi_dot - K/(np.exp(N)*N_dot)**2 )

def gamma_R(phi, phi_dot, N, N_dot):
    return 0.5* (3. + xi_R(phi, phi_dot, N, N_dot)) * N_dot

def omega_R_2(phi, phi_dot, N, N_dot):
    return kappa_R_2/ np.exp(N)**2 - K/np.exp(N)**2 *(1.+ xi_R(phi, phi_dot, N, N_dot))

def z_R(N, phi_dot, N_dot):
    return np.exp(N)*phi_dot/ N_dot * np.sqrt(D_2/ (D_2- K*epsilon_R(phi_dot, N_dot) ) )

def R_initial(N, phi_dot, N_dot):
    return 1./ ( z_R(N, phi_dot, N_dot) * np.sqrt(2.*k_R) )

def R_dot_initial(N, phi_dot, N_dot):
    return - 1.j * k_R/ np.exp(N) * R_initial(N, phi_dot, N_dot)

def h_initial(N):
    return 1./np.exp(N)* np.sqrt(2./k_R)

def h_dot_initial(N):
    return - 1.j * k_R/ np.exp(N) * h_initial(N)


# define ODEs
def odes(t, y):
    phi = y[0]
    N = y[1]
    dphidt = y[2]
    dNdt = y[3]
    R = y[4]
    dRdt = y[5]
    h = y[6]
    dhdt = y[7]

    # define each ODE
    d2phidt = - 3.0*dNdt*dphidt - potential_prime(phi)
    d2Ndt = -1.0/(2.0*m_p**2) * dphidt**2 + K* np.exp(-2*N)
    d2Rdt = -2.* gamma_R(phi, dphidt, N, dNdt) * dRdt - omega_R_2(phi, dphidt, N, dNdt) * R
    d2hdt = -3.*dNdt*dhdt - ( kappa_R_2 + 5.*K )/np.exp(N)**2 * h

    return [dphidt, dNdt, d2phidt, d2Ndt, dRdt, d2Rdt, dhdt, d2hdt]

# inflating event
def inflating(t,y):
    phi = y[0]
    dphidt = y[2]
    return dphidt**2 - potential(phi)

inflating.terminal = True
inflating.direction = 1

# find the time when Big Bang started
def BBstart(t, y):
    N = y[1]
    a = np.exp(N)
    return a - 1.e-6

BBstart.terminal = True
BBstart.direction = -1

P_R_list = []  # scalar PPS
P_t_list = []  # tensor PPS
#k_array = np.logspace(-5, -1, base=10, num=500)
k_array = np.logspace(-2, -1, base=10, num=10)
#k_array = [2.e-4]

for k in k_array:

    # solve R
    k_R_physical = k/u.Mpc
    k_R = (k_R_physical*a0).si.value
    kappa_R_2 = k_R**2 + k_R*K * (K+1) - 3*K 
    D_2 = -k_R**2 + 3*K

    # solve ODEs to get solution of BG variables 
    y0 = [phi_0, N_0, phi_dot_initial(phi_0), N_dot_initial(phi_0, N_0), R_initial(N_0, phi_dot_initial(phi_0), N_dot_initial(phi_0, N_0)),  R_dot_initial(N_0, phi_dot_initial(phi_0), N_dot_initial(phi_0, N_0)), h_initial(N_0), h_dot_initial(N_0)]
    sol_inf = solve_ivp(odes, [0, 1.e8], y0, method='RK45', events=inflating)


    # Solve ODEs to get solution during kinetic dominance
    y0 = [phi_0, N_0, phi_dot_initial(phi_0), N_dot_initial(phi_0, N_0), R_initial(N_0, phi_dot_initial(phi_0), N_dot_initial(phi_0, N_0)),  R_dot_initial(N_0, phi_dot_initial(phi_0), N_dot_initial(phi_0, N_0)), h_initial(N_0), h_dot_initial(N_0)]
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

    # calculate Hubble Horizon
    Hubble_Horizon = []
    for i in range(len(sol_tot[0])):
        Hubble_Horizon.append( a0.to(u.Mpc).value / (sol_tot[3][i]*np.exp(sol_tot[1][i])) )

    # Scaling all variables
    for i in range(len(t_tot)):
        t_tot[i] = t_tot[i] / sigma
        sol_tot[1][i] = sol_tot[1][i] - np.log(sigma)  # e-folding (N)
        sol_tot[2][i] = sol_tot[2][i] * sigma          # phi_dot
        sol_tot[3][i] = sol_tot[3][i] * sigma          # N_dot
        sol_tot[4][i] = sol_tot[4][i] * sigma          # R
        sol_tot[5][i] = sol_tot[5][i] * sigma**2       # R_dot
        sol_tot[6][i] = sol_tot[6][i] * sigma          # h
        sol_tot[7][i] = sol_tot[7][i] * sigma**2       # h_dot

    # seperate real and imaginary part of R
    R_real = []
    R_imagine = []
    for i in range(len(sol_tot[4])):
        R_real.append(k_R**3/ (2.*np.pi**2) * abs(sol_tot[4][i])**2)
        R_imagine.append(sol_tot[4][i].imag)
    
    # calculate P_R:
    P_R = k_R**3/ (2.*np.pi**2) * abs(sol_tot[4][-1])**2
    P_h = k_R**3/ (2.*np.pi**2) * abs(sol_tot[6][-1])**2
    P_R_list.append(P_R)
    print('k, P_R='+str(k)+','+str(P_R))
    print('t_end='+str(t_tot[-1]))
    P_t_list.append(2.*P_h)
    
    
plt.loglog(k_array, P_R_list)
plt.xlabel('k(Mpc^-1)')
plt.ylabel('P_R(k)')
plt.title('k - P_R(k)')
plt.show()