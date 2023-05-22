##### Goal of this code: Find phi_0 and sigma by one root-finder

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
N_0_origin = 12.5 # N_paper = ln(a/lp) = 10.0
K = 1.0
N_end_cons = 70.0

# Cross horizon (k=0.05 Mpc^-1)
H0 = 70 * u.km/u.s/u.Mpc
a0 = c * 10.0/H0
ks_cons = 0.05/u.Mpc # (comoving wavevector)
As_cons = 2.e-9

# Define functions

def potential(phi):
    #return V0* phi**(4./3.)
    return V0 * (1.0 - np.exp(- math.sqrt(2.0/3.0)* phi/m_p ) )**2

def potential_prime(phi):
    #return V0* 4./3. * phi**(1./3.)
    return 2.0* math.sqrt(2.0/3.0) * V0/m_p *(1.0- np.exp(- math.sqrt(2.0/3.0)* phi/m_p)) *np.exp(- math.sqrt(2.0/3.0)* phi/m_p)

def phi_dot(phi): # when start of inflation, V = phi_prime**2
    return - math.sqrt(potential(phi))

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

def Horizon_crossing(t,y):
    N = y[1]
    N_dot = y[3]
    log_ks = np.log(N_dot) + N
    return log_ks - np.log((ks_cons*a0).si.value)

Horizon_crossing.terminal = False

# Integrate(phi_0) return N_tot
def Integrate(x):
    
    # define root
    phi_0 = x[0]
    sigma = x[1]
    # initial condition
    N_0 = N_0_origin + np.log(sigma) 
    y0 = [phi_0, N_0, phi_dot(phi_0), N_dot(phi_0, N_0)]
    # scipy.integrate.solve_ivp(fun, t_span, y0, method='RK45', t_eval=None, dense_output=False, events=None, vectorized=False, args=None, **options)
    sol_inf = solve_ivp(odes, [0, 1.e5], y0, method='RK45', events=[inflating, Horizon_crossing])
 
    N_dot_star = sol_inf.y_events[1][0][3]
    phi_dot_star = sol_inf.y_events[1][0][2]
    P_R_star = P_R(phi_dot_star, N_dot_star)
    print('P_R_star='+str(P_R_star * sigma**2))
    
    # N_end 
    N_end = sol_inf.y_events[0][0][1] - np.log(sigma)
    print('N_end='+str(N_end))
    return [ N_end - N_end_cons, P_R_star * sigma**2 - As_cons ]  # [phi_0, sigma]

# get phi_0 by root finder
res = optimize.root(Integrate, [6.0, 1.2e-5], method='hybr')
print(res.x)