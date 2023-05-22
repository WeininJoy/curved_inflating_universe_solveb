######## Goal: generate PPS and CMB by (1) BG variables depends N_end,
########       (2) apply actual numerical b solution to set quantum IC
import sys
sys.path
sys.path.append('/home/weinin/miniconda3/lib/python3.8/site-packages/class_public')
import classy
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar, root
from math import gamma
from scipy.constants import c, hbar, G
Mpc = 3.086e+22

# Planck constant
t_p = 1.0 # second
l_p = 1.0 # km
m_p = 1.0 # reduced planck mass = \sqrt(c*hbar/8/pi/G) ??

# constants
V0 = m_p**4
N_i_origin_list = [11.713, 11.72, 11.73] # N_paper = ln(a/lp) = 10.0
phi_i_list = [10.44997084519851, 9.988837324387065, 7.250335673539723]
sigma_list = [1.1580032025900702e-05, 1.1500316486097475e-05, 1.1469857859311461e-05]
N_end_cons = 70.0

# Cross horizon (k=0.05 Mpc^-1)
Omega_K = -0.0092
H0 = 64.03
ks_cons = 0.05 # (comoving wavevector) Mpc^-1
As_cons = 2.e-9

V = lambda phi: V0 * (1.0 - np.exp(- math.sqrt(2.0/3.0)* phi/m_p ) )**2
V.d = lambda phi: 2.0* math.sqrt(2.0/3.0) * V0/m_p *(1.0- np.exp(- np.sqrt(2.0/3.0)* phi/m_p)) *np.exp(- np.sqrt(2.0/3.0)* phi/m_p)
V.dd = lambda phi: 4.0/3.0* V0/m_p**2 *(2.0*np.exp(- np.sqrt(2.0/3.0)* phi/m_p) -1.0) *np.exp(- np.sqrt(2.0/3.0)* phi/m_p)

class PrimordialSolver(object):
    def __init__(self, n, V, K):
        self.n = n
        self.V = V
        self.K = K

    def calcH(self, t, y):
        N, phi, dphi, eta = y
        H2 = (dphi**2/2 + self.V(phi))/(3*m_p**2) - self.K*np.exp(-2*N)
        return np.sqrt(H2)

    def P_R(self, t, y): # N_dot = H
        N, phi, dphi, eta = y
        H = self.calcH(t, y)
        return ( H**2/2.0/np.pi/dphi )**2

    def f(self, t, y):
        N, phi, dphi, eta = y
        H = self.calcH(t, y)
        ddphi = -3*H*dphi - self.V.d(phi)
        deta = np.exp(-N)
        return [H, dphi, ddphi, deta]
    
    def solve(self, N_i, phi_i, d=+1, **kwargs):
        dphi_i = -np.sign(phi_i)*np.sqrt(self.V(phi_i))
        y0 = [N_i, phi_i, dphi_i, 0]
        return solve_ivp(self.f, [0, d*np.inf], y0, rtol=1e-10, atol=1e-10, **kwargs)

    def find_ns_As(self, N_i, phi_i, sigma, logaH):

        sol_inf = self.solve(N_i, phi_i, d=+1, events=[Inflating(self), Until_aH(self, logaH)])
        N, phi, dphi, eta = sol_inf.y_events[1][0]
        H = self.calcH(sol_inf.t_events[1][0], sol_inf.y_events[1][0])
        ddphi = (-3*H*dphi - self.V.d(phi)) 
        dH =  -1.0/(2.0*m_p**2) * dphi**2 + self.K* np.exp(-2*N) 
        ddH =  -1./m_p**2 * dphi*ddphi - 2.*self.K*np.exp(-2*N)* H

        n_eps = - dH/H**2
        n_eta = ddH/(H*dH) - 2.*dH/H**2
        ns = 1 - 2.*n_eps - n_eta
        As = H**4/dphi**2/(2*np.pi)**2 * sigma**2
        
        return ns, As

    def find_phi0_sigma(self, N_i_origin, logaH):
        def Integrate(x):
    
            # define root
            phi_i = x[0]
            sigma = x[1]
            # initial condition
            N_i = N_i_origin + np.log(sigma) 
            sol_inf = self.solve(N_i, phi_i, d=+1, events=[Inflating(self), Until_aH(self, logaH)])

            P_R_star = self.P_R(sol_inf.t_events[1][0], sol_inf.y_events[1][0])
            print('P_R_star='+str(P_R_star * sigma**2))
            
            # N_end 
            N_end = sol_inf.y[0][-1] - np.log(sigma)
            print('N_end='+str(N_end))
            return [ N_end - N_end_cons, P_R_star * sigma**2 - As_cons ]  # [phi_0, sigma]

        # get phi_0 by root finder
        res = root(Integrate, [phi_i_list[self.n], sigma_list[self.n]], method='hybr')
        phi_i, sigma = res.x
        return [phi_i, sigma]

class IC_b_Solver(object):
    def __init__(self, n, V, K, phi_i, N_i):
        self.n = n
        self.V = V
        self.K = K
        self.phi_i = phi_i
        self.N_i = N_i

    @property
    def PrimordialSolver(self):
        return PrimordialSolver(self.n, self.V, self.K)

    def f_b(self, t, y):
        N, dN, Nb, dNb, phi, dphi, eta = y
        ddN = -1.0/(2.0*m_p**2) * dphi**2 + self.K* np.exp(-2*N)
        ddNb = dNb**2 + ddN - dN**2 - self.K*np.exp(-2*N)
        ddphi = -3*dN*dphi - self.V.d(phi)
        deta = np.exp(-N)
        return [dN, ddN, dNb, ddNb, dphi, ddphi, deta]

    def SR_t_eta(self):
        sol_inflating = self.PrimordialSolver.solve(self.N_i, self.phi_i, method='Radau', events=[Inflating(self), SlowRow(self)])
        SR_start = sol_inflating.t_events[1][0]
        SR_end = sol_inflating.t_events[1][1]

        # Find BG variables in deep inflation
        eta_SR_end = 0.0
        for i in range(len(sol_inflating.t)):
            if sol_inflating.t[i] <= SR_end < sol_inflating.t[i+1]:
                eta_SR_end = sol_inflating.y[3][i]       
        return [SR_start, SR_end, eta_SR_end]

    def KD_t_eta(self):
        sol_KD = self.PrimordialSolver.solve(self.N_i, self.phi_i, d=-1, events=Deep_KD(self))
        DKD_start = sol_KD.t_events[0][0]
        BB_start = sol_KD.t[-1]
        eta_BBstart = 0.0
        for i in range(len(sol_KD.t)):
            if sol_KD.t[i] <= BB_start < sol_KD.t[i-1]:
                eta_BBstart = sol_KD.y[3][i]* (1.+1.e-10)
        return [DKD_start, BB_start, eta_BBstart]
   
    ## Analytic solution of variable b in SR
    def anal_b_SR(self, eta, eta_Dinf_end, C):
        return C/ (eta_Dinf_end - eta)

    def anal_b_SR_prime(self, eta, eta_Dinf_end, C):
        return C/ (eta_Dinf_end - eta)**2

    ## Analytic solution of variable b in KD

    def anal_b_KD(self, eta, eta_BBstart, A, B):
        if self.K==1:
            return  B*((2.* (eta-eta_BBstart))**0.5 + (-1.-48.*A+3.*np.pi+12*np.log(2)-6.*np.log((eta-eta_BBstart)))*(eta-eta_BBstart)**2.5 / (6.*(2)**0.5) )
        elif self.K==-1:
            result = (2.+2.j)*np.pi**2*np.sqrt((eta-eta_BBstart))/(B*gamma(-1./4.)**2) \
                - ( 8./3.*(1.+1.j)*np.pi**2*(eta-eta_BBstart)**2.5* (6.*(-1.)**0.25*A*np.pi**2 + B* gamma(3./4.)**2 \
                *(-1. + (3.-3j)*np.pi+ 12.* np.log(2) - 6.* np.log((eta-eta_BBstart))) )) / (B**2*gamma(-0.25)**4)
            return result.real
        
    def anal_b_KD_prime(self, eta, eta_BBstart, A, B):
        if self.K==1:
            return  B* (1./(2.*(eta-eta_BBstart))**0.5 - (eta-eta_BBstart)**(3./2.)/2.**0.5 + 5.*(eta-eta_BBstart)**(3./2.) * (-1.-48.*A+3*np.pi+12.*np.log(2)-6.*np.log(eta-eta_BBstart)) / (12.*2**0.5) )
        elif self.K==-1:
            result = (1.+1.j)*np.pi**2/(B*gamma(-0.25)**4) * ( gamma(-0.25)**2/np.sqrt((eta-eta_BBstart)) \
                + 16.*(eta-eta_BBstart)**(3./2.)*gamma(3./4.)**2 - 20./3./B*(eta-eta_BBstart)**(3./2.) * ( 6*(-1)**0.25*A*np.pi**2 \
                + B*gamma(3./4.)**2 * ( -1+(3.-3.j)*np.pi+12.*np.log(2)-6.*np.log((eta-eta_BBstart)) ) ) ) 
            return result.real


    def solve_C(self):

        SR_start, SR_end, eta_SR_end = self.SR_t_eta()
        sol_inflating = self.PrimordialSolver.solve(self.N_i, self.phi_i, method='Radau', events=Inflating(self))
        # Find BG variables in deep inflation
        eta_SR = []
        N_SR = []
        for i in range(len(sol_inflating.t)):
            if SR_start < sol_inflating.t[i] < 0.2*SR_start + 0.8*SR_end:
                eta_SR.append(sol_inflating.y[3][i])
                N_SR.append(sol_inflating.y[0][i]) 

        def Find_C(C):
            # define root
            Diff = 0.0
            for i in range(len(eta_SR)-1):
                Diff += np.log(self.anal_b_SR(eta_SR[i], eta_SR_end, C)) - N_SR[i]
            return Diff

        sol_C = root_scalar(Find_C, bracket=[1., 10.], method='brentq')
        C = sol_C.root
        return C

    def IC_SR(self):
        SR_start, SR_end, eta_SR_end = self.SR_t_eta()
        sol_inflating = self.PrimordialSolver.solve(self.N_i, self.phi_i, method='Radau', events=Inflating(self))
        SR_index = 0
        for i in range(len(sol_inflating.t)):
            if sol_inflating.t[i] < (15./15.)*SR_start + (0./15.)*SR_end < sol_inflating.t[i+1]:
                SR_index = i
       
        t_SR = sol_inflating.t[SR_index]
        N_SR = sol_inflating.y[0][SR_index]
        dN_SR = self.PrimordialSolver.calcH(t_SR, sol_inflating.y[:,SR_index])

        # anal_b_SR_IC = self.anal_b_SR(sol_inflating.y[3][SR_index], eta_SR_end, self.solve_C())
        # anal_b_SR_prime_IC = self.anal_b_SR_prime(sol_inflating.y[3][SR_index], eta_SR_end, self.solve_C())
        # y0_SR = [N_SR, dN_SR, np.log(anal_b_SR_IC), anal_b_SR_prime_IC/ (anal_b_SR_IC*np.exp(N_SR)), sol_inflating.y[1][SR_index], sol_inflating.y[2][SR_index], sol_inflating.y[3][SR_index]]
        y0_SR = [N_SR, dN_SR, N_SR, dN_SR, sol_inflating.y[1][SR_index], sol_inflating.y[2][SR_index], sol_inflating.y[3][SR_index]]
        return [t_SR, y0_SR]

    def IC_KD(self, A, B):

        DKD_start, BB_start, eta_BBstart = self.KD_t_eta()
        ## Deep Kinetic dominance
        sol_KD = self.PrimordialSolver.solve(self.N_i, self.phi_i, d=-1)

        # Get index of deep KD
        DKD_index = 0
        for i in range(len(sol_KD.t)):
            if sol_KD.t[i] < (7./8.)*DKD_start + (1./8.)*BB_start < sol_KD.t[i-1]:
                DKD_index = i

        #print('t_DKD, t_end='+str(sol_KD.t[DKD_index])+','+str(sol_KD.t[-1]))
        
        t_DKD = sol_KD.t[DKD_index]
        N_DKD = sol_KD.y[0][DKD_index]
        dN_DKD = self.PrimordialSolver.calcH(t_DKD, sol_KD.y[:,DKD_index])

        # anal_b_KD_IC = self.anal_b_KD(sol_KD.y[3][DKD_index], eta_BBstart, A, B)
        # anal_b_KD_prime_IC = self.anal_b_KD_prime(sol_KD.y[3][DKD_index], eta_BBstart, A, B)
        # y0_KD = [N_DKD, dN_DKD, np.log(anal_b_KD_IC), anal_b_KD_prime_IC/ (anal_b_KD_IC*np.exp(N_DKD)), sol_KD.y[1][DKD_index], sol_KD.y[2][DKD_index], sol_KD.y[3][DKD_index]]
        y0_KD = [N_DKD, dN_DKD, N_DKD, dN_DKD, sol_KD.y[1][DKD_index], sol_KD.y[2][DKD_index], sol_KD.y[3][DKD_index]]
        return [t_DKD, y0_KD]

    def approx_B(self):
        DKD_start, BB_start, eta_BBstart = self.KD_t_eta()
        sol_KD = self.PrimordialSolver.solve(self.N_i, self.phi_i, d=-1)
        # Find BG variables in deep inflation
        eta_KD = []
        N_KD = []
        dN_KD = []
        for i in range(len(sol_KD.t)):
            if BB_start <= sol_KD.t[i] <= DKD_start:
                eta_KD.append(sol_KD.y[3][i])
                N_KD.append(sol_KD.y[0][i])
                dN_KD.append(self.PrimordialSolver.calcH(sol_KD.t[i], sol_KD.y[:,i]))
        A = 0.45
        def Find_B(B):
            Diff = 0.0
            for i in range(len(eta_KD)):
                Nb = np.log(self.anal_b_KD(eta_KD[i], eta_BBstart, A, B))
                Diff += Nb - N_KD[i]
            return Diff 

        sol_B = root_scalar(Find_B, bracket=[0.1, 10.], method='brentq')
        B = sol_B.root
        
        print('approximate A,B='+str(A)+','+str(B))
        return [A, B]

    def solve_AB(self):
        
        SR_start = self.SR_t_eta()[0]
        DKD_start = self.KD_t_eta()[0]

        def Find_AB(x):
    
            A = x[0]
            B = x[1]
            if self.K ==1:
                sol_inf = solve_ivp(self.f_b, [self.IC_SR()[0], SR_start], self.IC_SR()[1], method='Radau', d=-1)
                sol_KD = solve_ivp(self.f_b, [self.IC_KD(A, B)[0], SR_start], self.IC_KD(A, B)[1], method='Radau')
            elif self.K == -1:
                sol_inf = solve_ivp(self.f_b, [self.IC_SR()[0], DKD_start], self.IC_SR()[1], method='Radau', d=-1)
                sol_KD = solve_ivp(self.f_b, [self.IC_KD(A, B)[0], DKD_start], self.IC_KD(A, B)[1], method='Radau')
            # Determine whether the b from SR and KD consistant with each other
            return [sol_inf.y[2][-1]-sol_KD.y[2][-1], sol_inf.y[3][-1]-sol_KD.y[3][-1]]

        res = root(Find_AB, self.approx_B(), method='hybr')
        A = res.x[0]
        B = res.x[1]
        print('A,B='+str(A)+','+str(B))
        return [A, B]

    def get_b_IC(self):
        ## n=(KD, t0, SR)
        # sol_KD = self.PrimordialSolver.solve(self.N_i, self.phi_i, d=-1)

        # if i == 0:
        #     A, B = 0.5, 1.7
        #     sol_KD_for = solve_ivp(self.f_b, [self.IC_KD(A, B)[0], 0.0], self.IC_KD(A, B)[1], method='Radau')
        #     return [sol_KD_for.y[2][-1], sol_KD_for.y[3][-1]]

        # elif i==1:
        #     dN_i = self.PrimordialSolver.calcH(sol_KD.t[0], sol_KD.y[:,0])
        #     return [self.N_i, dN_i]

        # elif i==2:
        #     sol_inf_back = solve_ivp(self.f_b, [self.IC_SR()[0], 0.0], self.IC_SR()[1], method='Radau', d=-1)
        #     return [sol_inf_back.y[2][-1], sol_inf_back.y[3][-1]]
        # else: 
        #     print('n should be 0, 1, or 2')

        # sol_inf = solve_ivp(self.f_b, [self.IC_SR()[0], 0.0], self.IC_SR()[1], method='Radau', d=-1)
        # return [sol_inf.y[2][-1], sol_inf.y[3][-1]]
        sol_KD = self.PrimordialSolver.solve(self.N_i, self.phi_i, d=-1)
        dN_i = self.PrimordialSolver.calcH(sol_KD.t[0], sol_KD.y[:,0])
        return [self.N_i, dN_i]
        

class Inflating(object):
    terminal=True
    direction=1
    def __init__(self, solver):
        self.solver = solver

    def __call__(self, t, y):
        N, phi, dphi, eta = y
        return dphi**2 - self.solver.V(phi)

class BBstart(object):
    terminal=True
    direction=-1
    def __init__(self, solver):
        self.solver = solver

    def __call__(self, t, y):
        N = y[0]
        a = np.exp(N)
        return a - 1.e-6


class Until_aH(object):
    terminal=False
    def __init__(self, solver, logaH):
        self.solver = solver
        self.logaH = logaH

    def __call__(self, t, y):
        N, phi, dphi, eta = y
        H = self.solver.calcH(t, y)
        return N + np.log(H) - self.logaH

class SlowRow(object):
    terminal=False
    def __init__(self, solver):
        self.solver = solver

    def __call__(self, t, y):
        N, phi, dphi, eta = y
        p_phi = 0.5* dphi**2 - self.solver.V(phi)
        rho_phi = 0.5* dphi**2 + self.solver.V(phi)
        return p_phi/rho_phi- (-0.999)

class Deep_KD(object):
    terminal=False
    def __init__(self, solver):
        self.solver = solver

    def __call__(self, t, y):
        N, phi, dphi, eta = y
        return dphi**2 / abs(self.solver.V(phi)) -100


class R_func(object):
    def __init__(self, V, cs=1):
        self.cs = cs
        self.V = V
    
    def keppaD_2(self, k, K):
        if K==0 or K==-1:
            return k**2 - 3*K
        if K==1:
            return k*(k+2) - 3*K  ## k > 2, k\in Z
        else:
            raise ValueError("K should be 0, 1 or -1.")

    def epsilon(self, a, da, dphi):
        return 0.5* (a* dphi/da)**2

    def phi_dot_IC(self, phi): # when start of inflation, V = phi_prime**2
        return - np.sqrt(V(phi))

    def N_dot_IC(self, K, phi, N): # when start of inflation
        return np.sqrt( 1.0/2.0 * self.phi_dot_IC(phi)**2 - K* np.exp(-2.0*N) )

    def z_g(self, dphi, b, db):
        return dphi * b**2 / self.cs / db

    def z_g_dot(self, dphi, ddphi, b, db, ddb):
        return (ddphi*b**2 + 2.*b*db*dphi)/(self.cs*db) - (self.cs*ddb*dphi*b**2)/(self.cs*db)**2

    def g(self, a, da, b, db):
        return (a/b)**2 * (db/da)

    def dg(self, a, a_dot, a_2dot, b, b_dot, b_2dot):
        return 2.*(b_dot/a_dot) * (a*a_dot*b**2-b*b_dot*a**2)/ b**4 + (a/b)**2 * (b_2dot*a_dot-b_dot*a_2dot)/a_dot**2

    def f(self, K, a, da, dda, b, db, ddb):
        return a* da* self.dg(a, da, dda, b, db, ddb) / K + self.g(a, da, b, db)

    def zeta_IC(self, dphi, b, db, k, K):
        return 1./ ( 2*self.cs*self.z_g(dphi, b, db)**2 * self.keppaD_2(k, K)**0.5 )**0.5

    def zeta_dot_IC(self, dphi, ddphi, a, da, b, db, ddb, k, K):
        return  (- 1.j*self.keppaD_2(k,K)**0.5/a + da/a - self.z_g_dot(dphi, ddphi, b, db, ddb)/self.z_g(dphi, b, db) ) * self.zeta_IC(dphi, b, db, k, K)

    def R_PPS(self, k, K, zeta, dzeta, a, da, dda, b, db, ddb):
        return zeta/ self.g(a, da, b, db) - K* (a/da)* self.f(K, a, da, dda, b, db, ddb)*dzeta/ ( self.g(a, da, b, db)**2 *self.cs**2 *self.keppaD_2(k,K) )

    def R_dot_PPS(self, k, K, zeta, dzeta, dphi, a, da, dda, b, db, ddb):
        return 1./ ( self.g(a, da, b, db)*self.keppaD_2(k, K)) *( -K**2* self.f(K, a, da, dda, b, db, ddb)/(self.g(a, da, b, db)*self.cs**2*da**2) + (self.keppaD_2(k,K)+K*self.epsilon(a, da, dphi)) ) * dzeta + K*zeta/(a*da*self.g(a, da, b, db))
    
    def get_R_IC(self, k, K, phi, N, Nb, dNb):
        
        b = np.exp(Nb)
        db = b* dNb
        # solve ODEs to get solution of BG variables
        ddN = -1.0/2.0 * self.phi_dot_IC(phi)**2 + K* np.exp(-2*N)
        ddNb = dNb**2 + ddN - self.N_dot_IC(K, phi, N)**2 - K*np.exp(-2*N)
        ddb = b * (ddNb + (db/b)**2)
        dda = np.exp(N) * (ddN + self.N_dot_IC(K, phi, N)**2)

        zeta_IC = self.zeta_IC(self.phi_dot_IC(phi), b, db, k, K)
        zeta_dot_IC = self.zeta_dot_IC(self.phi_dot_IC(phi), - 3.0*self.N_dot_IC(K, phi, N)*self.phi_dot_IC(phi) - self.V.d(phi), np.exp(N), np.exp(N)*self.N_dot_IC(K, phi, N), b, db, ddb, k, K)
        R_IC = self.R_PPS(k, K, zeta_IC, zeta_dot_IC, np.exp(N), np.exp(N)*self.N_dot_IC(K, phi, N), dda, b, db, ddb)
        R_dot_IC =  self.R_dot_PPS(k, K, zeta_IC, zeta_dot_IC, self.phi_dot_IC(phi), np.exp(N), np.exp(N)*self.N_dot_IC(K, phi, N), dda, b, db, ddb)
        
        return [R_IC, R_dot_IC]


class Solver(object):
    def __init__(self, n, N_i_origin, V, H0=H0, omegabh2=0.022509, omegach2=0.11839, tau=0.0515, Omega_K=Omega_K, N_end_cons=N_end_cons, As_cons=As_cons, ks_cons=ks_cons):
        # Initialise late-time parameters
        self.n = n
        self.H0 = H0
        self.omegabh2 = omegabh2
        self.omegach2 = omegach2
        self.tau = tau
        self.Omega_K = Omega_K
        if self.K == 0:
            self.a0 = 1
        else:
            self.a0 = np.sqrt(-self.K/self.Omega_K)*(c*1e-3)/self.H0
        self.N_end_cons = N_end_cons
        self.As_cons = As_cons
        self.ks_cons = ks_cons
        
        self.N_i_origin = N_i_origin
        self.V = V
        self.phi_i, self.sigma = self.PrimordialSolver.find_phi0_sigma(self.N_i_origin, np.log((ks_cons*self.a0)))
        self.N_i = self.N_i_origin  + np.log(self.sigma)
        self.Nb_i, self.dNb_i = self.IC_b_Solver.get_b_IC()
        self.ns, self.As = self.PrimordialSolver.find_ns_As(self.N_i, self.phi_i, self.sigma, np.log((ks_cons*self.a0)))

    @property
    def PrimordialSolver(self):
        return PrimordialSolver(self.n, self.V, self.K)
    
    @property
    def IC_b_Solver(self):
        return IC_b_Solver(self.n, self.V, self.K, self.phi_i, self.N_i)
    
    @property
    def R_func(self):
        return R_func(self.V)

    @property
    def K(self):
        return -np.sign(self.Omega_K)

    def inflation(self, solution=0):
        sol = self.PrimordialSolver.solve(self.N_i, self.phi_i,
                                          events=Inflating(self.PrimordialSolver))
        H = self.PrimordialSolver.calcH(sol.t, sol.y)
        if solution:
            t_eval = np.linspace(sol.t[0], sol.t[-1], solution)
            sol = self.PrimordialSolver.solve(self.N_i, self.phi_i, t_eval=t_eval,
                                              events=Inflating(self.PrimordialSolver))
            K = self.PrimordialSolver.K
            N, phi, dphi, _ = sol.y
            H, _, ddphi, deta = self.PrimordialSolver.f(sol.t, sol.y)
            dH = -dphi**2/2 + K*np.exp(-2*N)
            eps = dphi**2/H**2/2 # the original code without /2, but I still don't know why
            z = np.exp(N)*dphi/H
            dlogz = H + ddphi/dphi - dH/H
            return sol.t, eps, dlogz, N, H, z, K

        N = self.N_to_Mpc(sol.y[0])
        Omega = -2*(np.log(H)+sol.y[0])
        eta = sol.y[3,-1]
        return N, Omega, eta

    def approx_PR(self):
        sol = self.PrimordialSolver.solve(self.N_i, self.phi_i, events=Inflating(self.PrimordialSolver))
        H = self.PrimordialSolver.calcH(sol.t, sol.y)
        N, phi, dphi, eta = sol.y
        PR = ( H**2/2.0/np.pi/dphi )**2 *self.sigma**2
        k = np.exp(N)*H/self.a0
        return k, PR
    
    def get_R_i(self, k): 
        R_i, dR_i = self.R_func.get_R_IC(k, self.PrimordialSolver.K, self.phi_i, self.N_i, self.Nb_i, self.dNb_i)
        return [R_i, dR_i]

    def test_b_IC(self):
        
        SR_start, SR_end, eta_SR_end = self.IC_b_Solver.SR_t_eta()
        DKD_start, BB_start, eta_BBstart = self.IC_b_Solver.KD_t_eta()
        print('phi_i, sigma='+str(self.phi_i)+', '+str(self.sigma))
        print('SR_start, SR_end='+str(SR_start)+','+str(SR_end))
        # A, B = self.IC_b_Solver.solve_AB()
        A, B = 0.5, 1.7
        print('C='+str(self.IC_b_Solver.solve_C()))
        print('Nb, dNb='+str(self.Nb_i)+', '+str(self.dNb_i))
        
        sol_inflating = self.PrimordialSolver.solve(self.N_i, self.phi_i, method='Radau', events=Inflating(self))
        sol_KD = self.PrimordialSolver.solve(self.N_i, self.phi_i, method='Radau', d=-1, events = BBstart(self))
        
        dN_i = self.PrimordialSolver.calcH(sol_KD.t[0], sol_KD.y[:,0])
        y_t0 = [self.N_i, dN_i, self.N_i, dN_i, sol_KD.y[1][0], sol_KD.y[2][0], sol_KD.y[3][0]]
        # sol_t0_for = solve_ivp(self.IC_b_Solver.f_b, [0.0, SR_end], y_t0, method='Radau')
        # sol_t0_back = solve_ivp(self.IC_b_Solver.f_b, [0.0, BB_start], y_t0, method='Radau', d=-1, events = BBstart(self))
        # sol_inf_for = solve_ivp(self.IC_b_Solver.f_b, [self.IC_b_Solver.IC_SR()[0], SR_end], self.IC_b_Solver.IC_SR()[1], method='Radau')
        # sol_inf_back = solve_ivp(self.IC_b_Solver.f_b, [self.IC_b_Solver.IC_SR()[0], BB_start], self.IC_b_Solver.IC_SR()[1], method='Radau', d=-1, events = BBstart(self))
        sol_KD_for = solve_ivp(self.IC_b_Solver.f_b, [self.IC_b_Solver.IC_KD(A, B)[0], SR_end], self.IC_b_Solver.IC_KD(A, B)[1], method='Radau')
        sol_KD_back = solve_ivp(self.IC_b_Solver.f_b, [self.IC_b_Solver.IC_KD(A, B)[0], BB_start], self.IC_b_Solver.IC_KD(A, B)[1], method='Radau', d=-1, events = BBstart(self))

        plt.plot(sol_inflating.t, sol_inflating.y[0],label='a,Ni='+labellist[self.n], linestyle='dashed', color=colorlist[self.n])
        plt.plot(sol_KD.t, sol_KD.y[0], linestyle='dashed', color=colorlist[self.n])

        # plt.plot(sol_t0_for.t, sol_t0_for.y[0], label='a_t0_for')
        # plt.plot(sol_t0_back.t, sol_t0_back.y[0], label='a_t0_back')
        # plt.plot(sol_t0_for.t, sol_t0_for.y[2], label='b,Ni='+labellist[self.n], color=colorlist[self.n])
        # plt.plot(sol_t0_back.t, sol_t0_back.y[2], color=colorlist[self.n])

        # plt.plot(sol_inf_for.t, sol_inf_for.y[0], label='a_inf_for')
        # plt.plot(sol_inf_back.t, sol_inf_back.y[0], label='a_inf_back')
        # plt.plot(sol_inf_for.t, sol_inf_for.y[2], label='b,Ni='+labellist[self.n], color=colorlist[self.n])
        # plt.plot(sol_inf_back.t, sol_inf_back.y[2], color=colorlist[self.n])
        
        # plt.plot(sol_KD_for.t, sol_KD_for.y[0], label='a_KD_for')
        # plt.plot(sol_KD_back.t, sol_KD_back.y[0], label='a_KD_back')
        plt.plot(sol_KD_for.t, sol_KD_for.y[2], label='b,Ni='+labellist[self.n], color=colorlist[self.n])
        plt.plot(sol_KD_back.t, sol_KD_back.y[2], color=colorlist[self.n])
        
        return 0

# universes = [Solver(n, N_i_origin_list[n], V) for n in range(len(N_i_origin_list))]

# for n in range(len(N_i_origin_list)):
#     universes[n].test_b_IC()
# plt.legend()
# plt.xlim([-3, 5])
# plt.ylim([-5, 5])
# plt.xlabel('t')
# plt.ylabel('log(a)&log(b)')
# plt.show()

import pyoscode
import tqdm

# Maximum N_i -- as flat as it can be whilst solving horizon problem
# print(eta_frac(-1))
# N_i = root_scalar(eta_frac, bracket=[-1, 1]).root
# N_i_max = N_i
# N_i_min = -1.5
# N_i_med = (N_i_max + N_i_min)/2

# N_i_min, N_i_med, N_i_max= -1.5, -0.4676751301973933, 0.5646497396052134
# print('N_i_min, N_i_med, N_i_max= '+str(N_i_min)+', '+str(N_i_med)+', '+str(N_i_max))

# universes =[Solver(N, V) for N in [N_i_min, N_i_med, N_i_max]]
# labellist = ["max","med","min"]
# colorlist = ['darkblue', 'steelblue', 'lightsteelblue']

universes = [Solver(n, N_i_origin_list[n], V) for n in range(len(N_i_origin_list))]
labellist = ["max", "med", "min"]
colorlist = ['darkblue', 'steelblue', 'lightsteelblue']


for figname in ['PPS-diffNi-ab_t0-small_Ni']:
    ks = np.arange(3,25000)

    def PR_analytic(k, s):
        return s.As * (k / s.ks_cons)**(s.ns-1)

    fig, ax = plt.subplots(1)
    
    i = 0
    for universe in universes:
        t, eps, dlogz, N, H, z, K = universe.inflation(10000)

        R = []
        for k in tqdm.tqdm(ks):
            # D squared operator in fourier space = Laplacian + 3K
            D2 = -k*(k+2)+3*K

            # Differential Equation for R variable in coordinate time (t) for PyOscode Package
            gamma = ((H+2*dlogz)*D2 - 3*K*H*eps)/(D2-K*eps)/2
            w2 = (-D2**2 + K*(1+eps-2/H*dlogz)*D2+K**2*eps)*np.exp(-2*N)/(D2-K*eps)
            w = np.sqrt(w2)
            
            # RST for R and zetapf
            R_i, dR_i = universe.get_R_i(k)
            sol = pyoscode.solve(t, np.log(w), gamma, t[0], t[-1], R_i, dR_i, logw=True)
            R.append(sol['sol'][-1])
            # print(sol['sol'][-1])

        R = np.array(R)
        P = ks**3/2/np.pi**2*abs(R)**2 * universe.sigma**2

        P = P /P[-1]*PR_analytic(ks[-1]/universe.a0, universe)

        labelval = labellist[i]
        ax.plot(ks/universe.a0, np.log(1e10*P), zorder=3, label="{} primordial $\Omega_K$".format(labelval),color=colorlist[i])
        # ax.plot(ks/universe.a0, np.log(1e10*P), zorder=3, label="$a&b$ match at {} ".format(labelval),color=colorlist[i])
        i = i+1

        # ----
        # This uses the CLASS package to do power spectrum fitting to give error bars figure for low values of wavenumber k

        # cosmo = classy.Class()
        # params = {
        #         'output': 'tCl lCl',
        #         'l_max_scalars': 2000,
        #         'lensing': 'yes',
        #         'A_s': universe.As,
        #         'n_s': universe.ns,
        #         'tau_reio': universe.tau,
        #         'h': universe.H0/100,
        #         'omega_b': universe.omegabh2,
        #         'Omega_k': universe.Omega_K,
        #         'omega_cdm': universe.omegach2}
        
        # cosmo.set(params)
        # cosmo.compute()
        # cls = cosmo.lensed_cl(2000)
        # l = cls['ell'][2:]
        # cosmo_tt = l*(l+1)*cls['tt'][2:] * (1e6 * 2.7255)**2 / (2*np.pi)
        
        # np.savetxt('nov28-5pm-correctedR.dat', np.array([
        #     np.concatenate([[1e-8, 1e-7], ks/universe.a0, [1e5]]),
        #     np.concatenate([[P[0], P[0]], P, [PR_analytic(1e5, universe)]])
        #     ]).T)
        # curved = classy.Class()
        # params = {
        #         'output': 'tCl lCl',
        #         'l_max_scalars': 2000,
        #         'lensing': 'yes',
        #         'P_k_ini type': 'external_Pk',
        #         'command': 'cat nov28-5pm-correctedR.dat',
        #         'tau_reio': universe.tau,
        #         'h': universe.H0/100,
        #         'omega_b': universe.omegabh2,
        #         'Omega_k': universe.Omega_K,
        #         'omega_cdm': universe.omegach2}
        # curved.set(params)
        # curved.compute()
        # cls = curved.lensed_cl(2000)
        # l = cls['ell'][2:]
        # tt0 = l*(l+1)*cls['tt'][2:] * (1e6 * 2.7255)**2 / (2*np.pi)
        
        # Cl = np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt')
        # tt_dat = Cl[:len(cosmo_tt),1]
        # tt_err_minus = Cl[:len(cosmo_tt),2]
        # tt_err_plus = Cl[:len(cosmo_tt),3]
        
        # def chi2(tt, lmax=-1):
        #     return ((tt_dat - tt)/np.where(tt_dat<tt,tt_err_plus,tt_err_minus))**2
        
        # chi0 = (chi2(tt0)-chi2(cosmo_tt))[:30].sum()
   
        # ax.plot(l,tt0, zorder=3, label='$\Delta\chi^2=%.2f$' % chi0, color=colorlist[i])
        
        # if i == 2:
        #     ax.plot(l,cosmo_tt, zorder=4,label="$\Lambda$CDM", linestyle='dashed',color="red")
        # i = i+1
    
    
    universe = universes[1]
    ax.plot(ks/universe.a0, np.log(1e10*PR_analytic(ks/universe.a0,universe)), zorder=4, linestyle='dashed', label="$\Lambda$CDM", color="red")
    ax.set_xscale('log')
    ax.set_xlim(1e-4,10**(-0.3))
    ax.set_ylim(2, 4)
    ax.set_xlabel('$k\:[\mathrm{Mpc}^{-1}]$')
    ax.set_ylabel(r'$\log\left( 10^{10}\mathcal{P}_\mathcal{R}\right)$')
    ax.set_yticks([2,2.5,3,3.5,4])
    ax.legend(loc = 'lower right')

    # Plots the right hand side of the figures in paper
    
    # ax.set_xscale('log')
    # ax.errorbar(l, tt_dat, yerr=(tt_err_minus, tt_err_plus), fmt='.', color='r', ecolor='k',marker='o',linestyle='None',capsize=0, markersize=4,zorder=2000,elinewidth=1, markeredgecolor='k')
    # ax.set_xlim(1.9,30.5)
    # ax.set_ylim(0,2500)
    # ax.set_ylabel('$\mathcal{D}_\ell^{TT}\: [\mu \mathrm{K}^2]$')
    # ax.set_xlabel('$\ell$')
    # ax.set_xticks([2,10,30])
    # ax.set_xticklabels([2,10,30])
    ax.legend()
    fig.set_size_inches(3.5,3.3)
    fig.tight_layout()
    fig.savefig(figname + '.pdf')


"""
### Make the PPS figure with different R_IC

colorlist = ['darkblue', 'steelblue', 'lightsteelblue', 'g']
R_IC_name_list = ['BD', 'FlatRST', 'MaryRST', 'bRST']
universe =Solver(2, N_i_origin_list[2], V) 

for figname in ['CMB-diffIC-Ni_max']:
    ks = np.arange(3,25000)

    def PR_analytic(k, s):
        ks = 0.05
        return s.As * (k / ks)**(s.ns-1)

    fig, ax = plt.subplots(1)
    i = 0
    for R_IC_name in R_IC_name_list:
        t, eps, dlogz, N, H, z, K = universe.inflation(10000)
        m = len(t)//5  # why choose this?
        eps = eps[:m]
        dlogz = dlogz[:m]
        N = N[:m]
        H = H[:m]
        z = z[:m]
        t = t[:m]

        R = []
        for k in tqdm.tqdm(ks):
            # D squared operator in fourier space = Laplacian + 3K
            D2 = -k*(k+2)+3*K

            # Differential Equation for R variable in coordinate time (t) for PyOscode Package
            gamma = ((H+2*dlogz)*D2 - 3*K*H*eps)/(D2-K*eps)/2
            w2 = (-D2**2 + K*(1+eps-2/H*dlogz)*D2+K**2*eps)*np.exp(-2*N)/(D2-K*eps)
            w = np.sqrt(w2)

            # Stuff
            a0 = np.exp(N[0])
            z0 = z[0]
            H0 = H[0]
            dphi0 = H0*z0/a0
            dH0 = (-1/2)*(dphi0**2) + K/(a0**2)
            #ddphi0 = (1/a0)*(dH0*z0 + H0*z0*dlogz[0] - z0*H0**2)#(dlogz[0] + dH0/H0 - H0)*dphi0
            #ddH0 = H0*(a0**2)*(-dH0*z0**2 - H0*dlogz[0]*z0**2 - (H0**2)*(z0**2) + 2*K)
            ddphi0 = -3*H0*dphi0 - universe.V.d(universe.phi_i)
            dddphi0 = -3*dH0*dphi0 - 3*H0*ddphi0 - dphi0*universe.V.dd(universe.phi_i)

            ddH0 = -dphi0*ddphi0 - 2*K*H0/(a0**2)
            ddz0 = z0*(dlogz[0])**2 + z0*(dH0 + dddphi0/dphi0 - (ddphi0/dphi0)**2 - ddH0/H0 + (dH0/H0)**2)

            # Sound speed
            ca2 = (1/(3*H0))*(3*H0 + 2*dH0/H0 - 2*dlogz[0])
            dlogca = ((-1/3)*(dH0/(H0**2))*(3*H0 + 2*dH0/H0 - 2*dlogz[0]) + (1/(3*H0))*(3*dH0 + 2*ddH0/H0 - 2*(dH0/H0)**2 - 2*ddz0/z0 + 2*(dlogz[0])**2))/(2*ca2)
            
            def R_IC(R_IC_name, k):

                if R_IC_name =='BD': # Bunch-Davies
                    dlogz0 =  H0 + ddphi0/dphi0 - dH0/H0
                    R0 = 1./ (z0*np.sqrt(2*k))
                    dR0 = ( -1.j*k/a0 - dlogz0 )* R0
                
                elif R_IC_name =='FlatRST':
                    R0 = 1./ (z0*np.sqrt(2*k))
                    dR0 = - 1.j*k*R0/ a0

                elif R_IC_name =='MaryRST':
                    # RST  for R and zetapf (Mary's)
                    R0 = 1/(z[0]*np.sqrt(2*(np.sqrt(-D2))))
                    dR0 = R0*(K/(H0*(a0**2)) + (1 - K*(z[0]**2)/(2*D2*(a0**2)))*((-1j * np.sqrt(-D2)/a0) + H0 - dlogz[0] - K/(H0*(a0**2))))
                
                elif R_IC_name =='bRST':
                    R0, dR0 = universe.get_R_i(k)

                else:print('R_IC should be BD, FlatRST, MaryRST, or bRST')

                return [R0, dR0]

            R0, dR0 = R_IC(R_IC_name, k)
            # Ignore this
            # gamma = dlogz - dlogca + H/2
            # w2 = -(ca2)*(D2-3*K)*np.exp(-2*N)
            # w = np.sqrt(w2)


            # zeta perfect fluid - Ignore this
            # dR0 = R0*(1 + (1/D2)*(K/H0)*(-3*H0 - 2*dH0/H0 + 2*dlogz[0]))*((-1j * np.sqrt(-D2)/a0) + H0 - dlogz[0])
            sol = pyoscode.solve(t, np.log(w), gamma, t[0], t[-1], R0, dR0, logw=True)
            R.append(sol['sol'][-1])
            # print(sol['sol'][-1])

        R = np.array(R)
        P = ks**3/2/np.pi**2*abs(R)**2 * universe.sigma**2

        P = P /P[-1]*PR_analytic(ks[-1]/universe.a0, universe)

        # labelval = labellist[i]
        # ax.plot(ks/universe.a0, np.log(1e10*P), zorder=3, label="{} primordial $\Omega_K$".format(labelval),color=colorlist[i])
        # i = i+1 

        # ----
        # This uses the CLASS package to do power spectrum fitting to give error bars figure for low values of wavenumber k

        cosmo = classy.Class()
        params = {
                'output': 'tCl lCl',
                'l_max_scalars': 2000,
                'lensing': 'yes',
                'A_s': universe.As,
                'n_s': universe.ns,
                'tau_reio': universe.tau,
                'h': universe.H0/100,
                'omega_b': universe.omegabh2,
                'Omega_k': universe.Omega_K,
                'omega_cdm': universe.omegach2}
        
        cosmo.set(params)
        cosmo.compute()
        cls = cosmo.lensed_cl(2000)
        l = cls['ell'][2:]
        cosmo_tt = l*(l+1)*cls['tt'][2:] * (1e6 * 2.7255)**2 / (2*np.pi)
        
        np.savetxt('nov28-5pm-correctedR.dat', np.array([
            np.concatenate([[1e-8, 1e-7], ks/universe.a0, [1e5]]),
            np.concatenate([[P[0], P[0]], P, [PR_analytic(1e5, universe)]])
            ]).T)
        curved = classy.Class()
        params = {
                'output': 'tCl lCl',
                'l_max_scalars': 2000,
                'lensing': 'yes',
                'P_k_ini type': 'external_Pk',
                'command': 'cat nov28-5pm-correctedR.dat',
                'tau_reio': universe.tau,
                'h': universe.H0/100,
                'omega_b': universe.omegabh2,
                'Omega_k': universe.Omega_K,
                'omega_cdm': universe.omegach2}
        curved.set(params)
        curved.compute()
        cls = curved.lensed_cl(2000)
        l = cls['ell'][2:]
        tt0 = l*(l+1)*cls['tt'][2:] * (1e6 * 2.7255)**2 / (2*np.pi)
        
        Cl = np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt')
        tt_dat = Cl[:len(cosmo_tt),1]
        tt_err_minus = Cl[:len(cosmo_tt),2]
        tt_err_plus = Cl[:len(cosmo_tt),3]
        
        def chi2(tt, lmax=-1):
            return ((tt_dat - tt)/np.where(tt_dat<tt,tt_err_plus,tt_err_minus))**2
        
        chi0 = (chi2(tt0)-chi2(cosmo_tt))[:30].sum()
   
        ax.plot(l,tt0, zorder=3, label='$\Delta\chi^2=%.2f$' % chi0, color=colorlist[i])
        
        if i == 2:
            ax.plot(l,cosmo_tt, zorder=4,label="$\Lambda$CDM", color="red")
        i = i+1

    # universe = universes[2]
    # ax.plot(ks/universe.a0, np.log(1e10*PR_analytic(ks/universe.a0,universe)), zorder=4, label="$\Lambda$CDM", color="red")
    # ax.set_xscale('log')
    # ax.set_xlim(1e-4,10**(-0.3))
    # ax.set_ylim(2, 4)
    # ax.set_xlabel('$k\:[\mathrm{Mpc}^{-1}]$')
    # ax.set_ylabel(r'$\log\left( 10^{10}\mathcal{P}_\mathcal{R}\right)$')
    # ax.set_yticks([2,2.5,3,3.5,4])
    # ax.legend()

    # Plots the right hand side of the figures in paper
    
    ax.set_xscale('log')
    ax.errorbar(l, tt_dat, yerr=(tt_err_minus, tt_err_plus), fmt='.', color='r', ecolor='k',marker='o',linestyle='None',capsize=0, markersize=4,zorder=2000,elinewidth=1, markeredgecolor='k')
    ax.set_xlim(1.9,30.5)
    ax.set_ylim(0,2500)
    ax.set_ylabel('$\mathcal{D}_\ell^{TT}\: [\mu \mathrm{K}^2]$')
    ax.set_xlabel('$\ell$')
    ax.set_xticks([2,10,30])
    ax.set_xticklabels([2,10,30])
    ax.legend()
    fig.set_size_inches(3.5,3.3)
    fig.tight_layout()
    fig.savefig(figname + '.pdf')

"""