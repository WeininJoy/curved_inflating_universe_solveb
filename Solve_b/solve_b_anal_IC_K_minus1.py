##### Goal of this code: get approximated constants of analytic solutions of 'b' in KD and deep inflation (K=-1)

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import optimize
from scipy.optimize import curve_fit
import math


# Planck constant
t_p = 1.0 # second
l_p = 1.0 # km
m_p = 1.0 # reduced planck mass = \sqrt(c*hbar/8/pi/G) ??

# constants
sigma = 1.1424624667273704e-05
phi_0 = 5.815354804140279
V0 = m_p**4
N_0 = 12 + np.log(sigma)  # N_paper = ln(a/lp) = 10.0
K = -1

#### Define functions for ODE

def potential(phi):
    return V0 * (1.0 - np.exp(- np.sqrt(2.0/3.0)* phi/m_p ) )**2

def potential_prime(phi):
    return 2.0* np.sqrt(2.0/3.0) * V0/m_p *(1.0- np.exp(- np.sqrt(2.0/3.0)* phi/m_p)) *np.exp(- np.sqrt(2.0/3.0)* phi/m_p)

def phi_dot(phi): # when start of inflation, V = phi_prime**2
    return - np.sqrt(V0)*( 1.0- np.exp(- np.sqrt(2.0/3.0)* phi/m_p))

def b_dot(N, N_dot, X, X_dot):
    return N_dot* np.exp(N)* X + np.exp(N) * X_dot

def N_dot(phi, N): # when start of inflation
    return np.sqrt( 1.0/(2.0*m_p**2) * phi_dot(phi)**2 - K* np.exp(-2.0*N) )

def P_R(phi_dot, N_dot): # N_dot = H
    return ( N_dot**2/2.0/np.pi/phi_dot )**2

# define ODEs
def odes(t, y):
    phi = y[0]
    N = y[1]
    dphidt = y[2]
    dNdt = y[3]
    tau = y[4]
    dtaudt = 1./np.exp(N)

    # define each ODE
    d2phidt = - 3.0*dNdt*dphidt - potential_prime(phi)
    d2Ndt = -1.0/(2.0*m_p**2) * dphidt**2 + K* np.exp(-2*N)
    
    return [dphidt, dNdt, d2phidt, d2Ndt, dtaudt]

#### Inflation 

## Define events

def inflating(t,y):
    phi = y[0]
    dphidt = y[2]
    return dphidt**2 - potential(phi)

inflating.terminal = True
inflating.direction = 1

def deep_inflation(t,y):
    phi = y[0]
    dphidt = y[2]
    p_phi = 0.5* dphidt**2 - potential(phi) 
    rho_phi = 0.5* dphidt**2 + potential(phi)
    return p_phi/rho_phi- (-0.999)

deep_inflation.terminal = False

## solve ODEs to get BG variables in deep inflation
y0 = [phi_0, N_0, phi_dot(phi_0), N_dot(phi_0, N_0), 0.0]
sol_inf = solve_ivp(odes, [0, 1.e8], y0, method='Radau', events=[inflating, deep_inflation])
Dinf_start = sol_inf.t_events[1][0]
Dinf_end = sol_inf.t_events[1][1]
Dinf_index = 0

for i in range(len(sol_inf.t)):
    if sol_inf.t[i] < (8./9.)*Dinf_start + (1./9.)*Dinf_end < sol_inf.t[i+1]:
        Dinf_index = i

print('t_Dinf, t_end='+str(sol_inf.t[Dinf_index])+','+str(sol_inf.t[-1]))

# Find tau_SR
tau_SR = []
t_SR = []
N_SR = []
for i in range(len(sol_inf.t)):
    if Dinf_start < sol_inf.t[i] < 0.8*Dinf_start+0.2*Dinf_end:
        tau_SR.append(sol_inf.y[4][i])
        t_SR.append(sol_inf.t[i])
        N_SR.append(sol_inf.y[1][i])

#### Fit analytic solution
print('tau_inf_end='+str(tau_SR[-1]))
# analytic solution of variable b in SR
def anal_b_SR(tau, C):
    return np.log(C/(tau_SR[-1] - tau))

def Find_C(C):
    # define root
    Diff = 0.0
    for i in range(len(tau_SR)-1):
        Diff += anal_b_SR(tau_SR[i], C) - N_SR[i]
    print(Diff)
    return Diff

sol_C = optimize.root_scalar(Find_C, bracket=[1., 10.], method='brentq')
C = sol_C.root
print('C='+str(C))


#### Kinetic dominance

## Define events

# find the time when Big Bang started
def BBstart(t, y):
    N = y[1]
    a = np.exp(N)
    return a - 1.e-6

BBstart.terminal = True
BBstart.direction = -1

# find the deep kinetic dominance era
def deep_KD(t,y):
    phi = y[0]
    dphidt = y[2]
    return dphidt**2 / abs(potential(phi)) -100

deep_KD.terminal = False

# Solve ODEs to get solution during kinetic dominance
y0 = [phi_0, N_0, phi_dot(phi_0), N_dot(phi_0, N_0), 0.0]
sol_KD = solve_ivp(odes, [0.0, -1.e5], y0, method='Radau', events=[BBstart, deep_KD])
DKD_start = sol_KD.t_events[1][0]
BB_start = sol_KD.t[-1]
print('DKD_start, BB_start='+str(DKD_start)+','+str(BB_start))

#### Fit analytic solution

### Kinetic dominance 
# Find tau_KD
tau_KD = []
N_KD = []
for i in range(len(sol_KD.t)):
    if 0.9999*BB_start+0.0001*DKD_start <= sol_KD.t[i] <= DKD_start:
        tau_KD.append(sol_KD.y[4][i])
        N_KD.append(sol_KD.y[1][i])

# analytic solution of variable b in KD
A = 1.
tau_BBstart = sol_KD.y[4][-1]* (1.+1.e-15)
def anal_b_KD(tau, A, B):
    result = (2.+2.j)*np.pi**2*np.sqrt((tau-tau_BBstart))/(B*math.gamma(-1./4.)**2) \
    - ( 8./3.*(1.+1.j)*np.pi**2*(tau-tau_BBstart)**2.5* (6.*(-1.)**0.25*A*np.pi**2 + B* math.gamma(3./4.)**2 \
    *(-1. + (3.-3j)*np.pi+ 12.* np.log(2) - 6.* np.log((tau-tau_BBstart))) )) / (B**2*math.gamma(-0.25)**4)
    return result.real

def Find_B(B):
    Diff = 0.0
    for i in range(len(tau_KD)):
        Diff += np.log(anal_b_KD(tau_KD[i], A, B)) - N_KD[i]
    print(Diff)
    return Diff

sol_B = optimize.root_scalar(Find_B, bracket=[1.e-3, 0.5], method='brentq')
B = sol_B.root
print('B='+str(B))

print('tau_SR_end, tau_BB_start='+str(tau_SR[-1])+','+str(tau_BBstart))

# plot solution of BG variables
plt.plot(sol_inf.y[4], sol_inf.y[1])
plt.plot(sol_KD.y[4], sol_KD.y[1])
plt.plot(tau_SR, [anal_b_SR(tau, C) for tau in tau_SR])
plt.plot(tau_KD, [np.log(anal_b_KD(tau, A, B)) for tau in tau_KD])
plt.xlabel('eta')
plt.ylabel('log(a) & log(b)')
plt.title('eta - log(a)&log(b)')
plt.legend()
plt.show()