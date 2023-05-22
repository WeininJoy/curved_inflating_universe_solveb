##### Goal of this code: Find exact constant of analytic solutions of b numerically by connection solutions starting from KD and deep inflation (K=-1).

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import optimize
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

#### Define functions for ODEs

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

def X_dot(N, N_dot, b, b_prime):
    return b_prime/np.exp(N)**2 - N_dot*b/np.exp(N)

# define ODEs (only BG variables)
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

# define ODEs (include X=b/a)
def odes2(t, y):
    phi = y[0]
    N = y[1]
    dphidt = y[2]
    dNdt = y[3]
    tau = y[4]
    dtaudt = 1./np.exp(N)
    Nb = y[5]
    dNbdt = y[6]

    # define each ODE
    d2phidt = - 3.0*dNdt*dphidt - potential_prime(phi)
    d2Ndt = -1.0/(2.0*m_p**2) * dphidt**2 + K* np.exp(-2*N)
    d2Nbdt = dNbdt**2 + d2Ndt - dNdt**2 - K*np.exp(-2*N)
    #d2Xdt = ( -2.0*dNdt**2 -K* np.exp(-2.0*N) + 2.0*(dNdt + dXdt/X)**2 ) *X - 2.0*dNdt*dXdt

    return [dphidt, dNdt, d2phidt, d2Ndt, dtaudt, dNbdt, d2Nbdt]

### Define events 

# inflating event
def inflating(t,y):
    phi = y[0]
    dphidt = y[2]
    return dphidt**2 - potential(phi)

inflating.terminal = True
inflating.direction = 1

# deep inflation
def deep_inflation(t,y):
    phi = y[0]
    dphidt = y[2]
    p_phi = 0.5* dphidt**2 - potential(phi) 
    rho_phi = 0.5* dphidt**2 + potential(phi)
    return p_phi/rho_phi- (-0.999)

deep_inflation.terminal = False

# find the time when Big Bang started
def BBstart(t, y):
    N = y[1]
    a = np.exp(N)
    return a - 1.e-6

BBstart.terminal = True
BBstart.direction = -1

# deep kinetic dominance
def deep_KD(t,y):
    phi = y[0]
    dphidt = y[2]
    return dphidt**2 / abs(potential(phi)) -100

deep_KD.terminal = False


#### Set Initial condition of BG variables in deep SR and KD

## Deep Inflation
y0 = [phi_0, N_0, phi_dot(phi_0), N_dot(phi_0, N_0), 0.0]
sol_inf = solve_ivp(odes, [0, 1.e8], y0, method='Radau', events=[inflating, deep_inflation])
Dinf_start = sol_inf.t_events[1][0]
Dinf_end = sol_inf.t_events[1][1]
Dinf_index = 0
tau_end = sol_inf.y[4][-1]
for i in range(len(sol_inf.t)):
    if sol_inf.t[i] < (9./10.)*Dinf_start + (1./10.)*Dinf_end < sol_inf.t[i+1]:
        Dinf_index = i


print('Dinf_start, Dinf_end, t_Dinf, t_end='+str(Dinf_start)+','+str(Dinf_end)+','+str(sol_inf.t[Dinf_index])+','+str(sol_inf.t[-1]))

t_Dinf = sol_inf.t[Dinf_index]
N_Dinf = sol_inf.y[1][Dinf_index]
N_dot_Dinf = sol_inf.y[3][Dinf_index]
#y0 = [sol.y[0][Dinf_index], N_Dinf, sol.y[2][Dinf_index], N_dot_Dinf, 1.0*np.exp(N_Dinf), b_dot(N_Dinf, N_dot_Dinf, 1.0, 0.0), sol.y[4][Dinf_index]]


## Deep Kinetic dominance
y0 = [phi_0, N_0, phi_dot(phi_0), N_dot(phi_0, N_0), 0.0]
sol_KD = solve_ivp(odes, [0, -1.e5], y0, method='Radau', events=[BBstart, deep_KD])
DKD_start = sol_KD.t_events[1][0]
BB_start = sol_KD.t[-1]
tau_BBstart = sol_KD.y[4][-1]* (1.+1.e-15)

# Get index of deep KD
DKD_index = 0
for i in range(len(sol_KD.t)):
    if sol_KD.t[i] < (7./8.)*DKD_start + (1./8.)*BB_start < sol_KD.t[i-1]:
        DKD_index = i

print('DKD_start,t_DKD, t_end='+str(DKD_start)+','+str(sol_KD.t[DKD_index])+','+str(sol_KD.t[-1]))

t_DKD = sol_KD.t[DKD_index]
N_DKD = sol_KD.y[1][DKD_index]
N_dot_DKD = sol_KD.y[3][DKD_index]


#### Analytic solution of variable b in SR and KD

## Analytic solution of variable b in SR
def anal_b_SR(tau, C):
    return C/ (tau_end - tau)

def anal_b_SR_prime(tau, C):
    return C/ (tau_end - tau)**2

# Find BG variables in deep inflation
tau_SR = []
t_SR = []
N_SR = []
for i in range(len(sol_inf.t)):
    if Dinf_start < sol_inf.t[i] < 0.8*Dinf_start + 0.2*Dinf_end:
        tau_SR.append(sol_inf.y[4][i])
        t_SR.append(sol_inf.t[i])
        N_SR.append(sol_inf.y[1][i])

# Find parameter C
def Find_C(C):
    # define root
    Diff = 0.0
    for i in range(len(tau_SR)-1):
        Diff += np.log(anal_b_SR(tau_SR[i], C)) - N_SR[i]
    return Diff

sol_C = optimize.root_scalar(Find_C, bracket=[1., 10.], method='brentq')
C = sol_C.root
print('C='+str(C))

## Analytic solution of variable b in KD
def anal_b_KD(tau, A, B):
    result = (2.+2.j)*np.pi**2*np.sqrt((tau-tau_BBstart))/(B*math.gamma(-1./4.)**2) \
    - ( 8./3.*(1.+1.j)*np.pi**2*(tau-tau_BBstart)**2.5* (6.*(-1.)**0.25*A*np.pi**2 + B* math.gamma(3./4.)**2 \
    *(-1. + (3.-3j)*np.pi+ 12.* np.log(2) - 6.* np.log((tau-tau_BBstart))) )) / (B**2*math.gamma(-0.25)**4)
    return result.real

def anal_b_KD_prime(tau, A, B):
    result = (1.+1.j)*np.pi**2/(B*math.gamma(-0.25)**4) * ( math.gamma(-0.25)**2/np.sqrt((tau-tau_BBstart)) \
        + 16.*(tau-tau_BBstart)**(3./2.)*math.gamma(3./4.)**2 - 20./3./B*(tau-tau_BBstart)**(3./2.) * ( 6*(-1)**0.25*A*np.pi**2 \
        + B*math.gamma(3./4.)**2 * ( -1+(3.-3.j)*np.pi+12.*np.log(2)-6.*np.log((tau-tau_BBstart)) ) ) ) 
    return result.real


#### Solve A by root finding (make the b variables from SR and KD consistant with each other)
def Find_AB(x):
    
    A = x[0]
    B = x[1]
    # solve ODEs to get solution during deep inflation to the start of inflation
    y0_Dinf = [sol_inf.y[0][Dinf_index], N_Dinf, sol_inf.y[2][Dinf_index], N_dot_Dinf, sol_inf.y[4][Dinf_index], np.log(anal_b_SR(sol_inf.y[4][Dinf_index], C)), anal_b_SR_prime(sol_inf.y[4][Dinf_index], C)/ (anal_b_SR(sol_inf.y[4][Dinf_index], C)*np.exp(N_Dinf))]
    #y0_Dinf = [sol_inf.y[0][Dinf_index], N_Dinf, sol_inf.y[2][Dinf_index], N_dot_Dinf, sol_inf.y[4][Dinf_index], 1.0, 0.0]
    sol_inf_end = solve_ivp(odes2, [t_Dinf, -1.e5], y0_Dinf, method='Radau', events=BBstart)
    delta_t = sol_inf_end.t[-1] - sol_KD.t[-1]
    sol_inf2 = solve_ivp(odes2, [t_Dinf, DKD_start + delta_t], y0_Dinf, method='Radau')

    # Solve ODEs to get solution during kinetic dominance to the start of inflation
    y0_DKD = [sol_KD.y[0][DKD_index], N_DKD, sol_KD.y[2][DKD_index], N_dot_DKD, sol_KD.y[4][DKD_index], np.log(anal_b_KD(sol_KD.y[4][DKD_index],A,B)), anal_b_KD_prime(sol_KD.y[4][DKD_index],A,B)/ (anal_b_KD(sol_KD.y[4][DKD_index],A,B)*np.exp(N_DKD))]
    sol_KD2 = solve_ivp(odes2, [t_DKD, DKD_start], y0_DKD, method='Radau')
    
    # Determine whether the b from SR and KD consistant with each other
    Diff = abs(sol_inf2.y[5][-1]-sol_KD2.y[5][-1])+ abs(sol_inf2.y[6][-1]-sol_KD2.y[6][-1])
    print('Diff='+str(Diff))
    return [sol_inf2.y[5][-1]-sol_KD2.y[5][-1], sol_inf2.y[6][-1]-sol_KD2.y[6][-1]]

# get phi_0 by root finder
res = optimize.root(Find_AB, [1., 0.359738724564026], method='hybr')
A = res.x[0]
B = res.x[1]
print('A,B='+str(A)+','+str(B))

#### Solve ODEs to get all BG variables 

# solve ODEs to get solution during deep inflation to the start of inflation
y0_Dinf = [sol_inf.y[0][Dinf_index], N_Dinf, sol_inf.y[2][Dinf_index], N_dot_Dinf, sol_inf.y[4][Dinf_index], np.log(anal_b_SR(sol_inf.y[4][Dinf_index], C)), anal_b_SR_prime(sol_inf.y[4][Dinf_index], C)/ (anal_b_SR(sol_inf.y[4][Dinf_index], C)*np.exp(N_Dinf))]
#y0_Dinf = [sol_inf.y[0][Dinf_index], N_Dinf, sol_inf.y[2][Dinf_index], N_dot_Dinf, sol_inf.y[4][Dinf_index], N_Dinf, N_dot_Dinf]
sol_inf2 = solve_ivp(odes2, [t_Dinf, -1.e5], y0_Dinf, method='Radau', events=BBstart)
sol_inf3 = solve_ivp(odes2, [t_Dinf, 1.e8], y0_Dinf, method='Radau', events=inflating)

# Solve ODEs to get solution during kinetic dominance to the start of inflation
y0_DKD = [sol_KD.y[0][DKD_index], N_DKD, sol_KD.y[2][DKD_index], N_dot_DKD, sol_KD.y[4][DKD_index], np.log(anal_b_KD(sol_KD.y[4][DKD_index],A,B)), anal_b_KD_prime(sol_KD.y[4][DKD_index],A,B)/ (anal_b_KD(sol_KD.y[4][DKD_index],A,B)*np.exp(N_DKD))]
#y0_DKD = [sol_KD.y[0][DKD_index], N_DKD, sol_KD.y[2][DKD_index], N_dot_DKD, sol_KD.y[4][DKD_index], N_DKD, N_dot_DKD]
sol_KD2 = solve_ivp(odes2, [t_DKD, -1.e5], y0_DKD, method='Radau', events=BBstart)
sol_KD3 = solve_ivp(odes2, [t_DKD, t_Dinf], y0_DKD, method='Radau', events=inflating)

print('b, b_dot='+str(sol_inf2.y[5][-1])+','+str(sol_inf2.y[6][-1]))
"""
# Combine solutions of KD and inflation together
t_tot = np.concatenate((np.flipud(sol_KD.t), sol_inf.t), axis=0)
sol_tot = np.concatenate((np.fliplr(sol_KD.y), sol_inf.y), axis=1)

# Shift t such that the universe start from t=0
for i in range(len(t_tot)):
    t_tot[i] = t_tot[i] - sol_KD.t[-1]
    sol_tot[4][i] = sol_tot[4][i] - sol_KD.y[4][-1]

# calculate equation of state (w = p/rho) 
w_phi = []
for i in range(len(sol_tot[0])):
    p_phi = 0.5* sol_tot[2][i]**2 - potential(sol_tot[0][i]) 
    rho_phi = 0.5* sol_tot[2][i]**2 + potential(sol_tot[0][i])
    w_phi.append(p_phi/rho_phi)
    
log_b = []
# Scaling all variables
for i in range(len(t_tot)):
    t_tot[i] = t_tot[i] / sigma
    sol_tot[1][i] = sol_tot[1][i] - np.log(sigma)
    sol_tot[2][i] = sol_tot[2][i] * sigma
    sol_tot[3][i] = sol_tot[3][i] * sigma
    #log_b.append(np.log(sol_tot[4][i]))
    log_b.append(np.log(sol_tot[5][i])+ sol_tot[1][i])
"""
# plot solution of BG variables
plt.plot(sol_inf.t, sol_inf.y[1], label='a_for')
plt.plot(sol_KD.t, sol_KD.y[1], label='a_back')

#plt.plot(t_SR, [np.log(anal_b_SR(tau, C)) for tau in tau_SR], label='b_inf_anal')
plt.plot(sol_inf2.t, sol_inf2.y[5], label='b_inf_back')
plt.plot(sol_inf3.t, sol_inf3.y[5], label='b_inf_for')
#plt.plot(sol_inf2.t, sol_inf2.y[1], label='a_inf_back')
#plt.plot(sol_inf3.t, sol_inf3.y[1], label='a_inf_for')

plt.plot(sol_KD2.t, sol_KD2.y[5], label='b_KD_back')
plt.plot(sol_KD3.t, sol_KD3.y[5], label='b_KD_for')
#plt.plot(sol_KD2.t, sol_KD2.y[1], label='a_KD_back')
#plt.plot(sol_KD3.t, sol_KD3.y[1], label='a_KD_for')
#plt.yscale('symlog', linthresh=1.e-5)
#plt.yscale('log')
#plt.xscale('log')
#plt.xlim([1.e-12, 3.e-7])
#plt.xlim([8.e0, 2.e7])
plt.xlim([-1., 2])
plt.ylim([-1, 2])
plt.xlabel('t')
plt.ylabel('log(a)&log(b)')
plt.title('t - log(a)&log(b)')
plt.legend()
plt.show()
