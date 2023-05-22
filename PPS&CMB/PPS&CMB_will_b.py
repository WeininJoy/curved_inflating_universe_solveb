######## Goal: generate PPS and CMB by (1) BG variables depends on ns (rather than N_end or N_star),
########       (2) apply actual numerical b solution to set quantum IC
######## Based on PPS&CMB_willworks.py
import sys
sys.path.append('/home/weinin/miniconda3/lib/python3.8/site-packages/class_public')
import classy
print(classy.__file__)
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import patches
from numpy import exp
import numpy
from scipy.optimize import root_scalar, root
from scipy.integrate import solve_ivp
from scipy.special import logsumexp

from scipy.constants import c, hbar, G
import scipy.interpolate
from math import gamma

lp = numpy.sqrt(hbar*G/c**3)
mp = numpy.sqrt(hbar*c/G)
tp = numpy.sqrt(hbar*G/c**5)
Mpc = 3.086e+22
ly = c*60*60*24*365.24

# Plot formatting
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['text.usetex'] = True
myfont = matplotlib.font_manager.FontProperties(
    fname=r'/home/weinin/miniconda3/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf/jsMath-cmbx10.ttf')
#plt.rcParams['font.family'] = "serif"
#plt.rcParams['font.serif'] = "cm"


# n = 2
# V = lambda phi: phi**2/2
# V.d = lambda phi: phi
# V.dd = lambda phi: 1

n = 2
V = lambda phi: (1-numpy.exp(-numpy.sqrt(2./3)*phi))**2
V.d = lambda phi: 2*(1-numpy.exp(-numpy.sqrt(2./3)*phi))* numpy.sqrt(2./3)*numpy.exp(-numpy.sqrt(2./3)*phi)
V.dd = lambda phi: 8./3*numpy.exp(-2*numpy.sqrt(2./3)*phi) - 4./3*numpy.exp(-numpy.sqrt(2./3)*phi)

# n = 4./3
# V = lambda phi: numpy.abs(phi)**n
# V.d = lambda phi: n * numpy.sign(phi)*numpy.abs(phi)**(n-1)
# V.dd = lambda phi: n * (n-1) *numpy.abs(phi)**(n-2)

V.w = (n-2)/(n+2)


class PrimordialSolver(object):
    def __init__(self, V, K):
        self.V = V
        self.K = K

    def calcH(self, t, y):
        N, phi, dphi, eta = y
        H2 = (dphi**2/2 + self.V(phi))/3 - self.K*numpy.exp(-2*N)
        return numpy.sqrt(H2)

    def f(self, t, y):
        N, phi, dphi, eta = y
        H = self.calcH(t, y)
        ddphi = -3*H*dphi - self.V.d(phi)
        deta = numpy.exp(-N)
        return [H, dphi, ddphi, deta]

    def solve(self, N_i, phi_i, d=+1, **kwargs):
        dphi_i = -numpy.sign(phi_i)*numpy.sqrt(self.V(phi_i))
        y0 = [N_i, phi_i, dphi_i, 0]
        return solve_ivp(self.f, [0, d*numpy.inf], y0, rtol=1e-10, atol=1e-10, **kwargs)

    def solve_post(self, N_i, phi_i, depth):
        sol_inflating = self.solve(N_i, phi_i, events=Inflating(self))
        H_end = self.calcH(sol_inflating.t[-1], sol_inflating.y[:,-1])
        sol_inflating.y[3,-1] = 0
        return solve_ivp(self.f, [0, numpy.inf], sol_inflating.y[:,-1], rtol=1e-10, atol=1e-10, events=Until_H(self, depth*H_end))

    def find_ns_As_r(self, N_i, phi_i, logaH):
        #print(N_i, phi_i)
        sol_pivot = self.solve(N_i, phi_i, d=+1, events=[Until_aH(self, logaH), Inflating(self)])
        if not sol_pivot.t_events[0]:
            raise ValueError

        N, phi, dphi, eta = sol_pivot.y[:,-1]
        H = self.calcH(sol_pivot.t[-1], sol_pivot.y[:,-1])
        ddphi = -3*H*dphi - self.V.d(phi)
        dH = -dphi**2/2 + self.K*numpy.exp(-2*N)
        ns = 1+ (4*dH/H - 2*ddphi/dphi)/(H+dH/H)
        r = 8 * dphi**2/H**2
        As = H**4/dphi**2/(2*numpy.pi)**2

        return ns, As, r

    def find_phi_i(self, Ni, logaH, ns):
        def calcns(phi):
            try:
                return self.find_ns_As_r(Ni, phi, logaH)[0] -ns
            except ValueError:
                return -1
        return root_scalar(calcns, bracket=[self.phimin(Ni)+1e-4, 100]).root

    def phimin(self, Ni):
        if self.K > 0:
            return root_scalar(lambda phi: self.V(phi) - 2*self.K*numpy.exp(-2*Ni), bracket=[0,1e11]).root
        else:
            return 0.

class IC_b_Solver(object):
    def __init__(self, V, K, phi_i, N_i):
        self.V = V
        self.K = K
        self.phi_i = phi_i
        self.N_i = N_i

    @property
    def PrimordialSolver(self):
        return PrimordialSolver(self.V, self.K)

    def f_b(self, t, y):
        N, dN, Nb, dNb, phi, dphi, eta = y
        ddN = -1.0/2.0 * dphi**2 + self.K* numpy.exp(-2*N)
        ddNb = dNb**2 + ddN - dN**2 - self.K*numpy.exp(-2*N)
        ddphi = -3*dN*dphi - self.V.d(phi)
        deta = numpy.exp(-N)
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
            return  B*((2.* (eta-eta_BBstart))**0.5 + (-1.-48.*A+3.*numpy.pi+12*numpy.log(2)-6.*numpy.log((eta-eta_BBstart)))*(eta-eta_BBstart)**2.5 / (6.*(2)**0.5) )
        elif self.K==-1:
            result = (2.+2.j)*numpy.pi**2*numpy.sqrt((eta-eta_BBstart))/(B*gamma(-1./4.)**2) \
                - ( 8./3.*(1.+1.j)*numpy.pi**2*(eta-eta_BBstart)**2.5* (6.*(-1.)**0.25*A*numpy.pi**2 + B* gamma(3./4.)**2 \
                *(-1. + (3.-3j)*numpy.pi+ 12.* numpy.log(2) - 6.* numpy.log((eta-eta_BBstart))) )) / (B**2*gamma(-0.25)**4)
            return result.real
        
    def anal_b_KD_prime(self, eta, eta_BBstart, A, B):
        if self.K==1:
            return  B* (1./(2.*(eta-eta_BBstart))**0.5 - (eta-eta_BBstart)**(3./2.)/2.**0.5 + 5.*(eta-eta_BBstart)**(3./2.) * (-1.-48.*A+3*numpy.pi+12.*numpy.log(2)-6.*numpy.log(eta-eta_BBstart)) / (12.*2**0.5) )
        elif self.K==-1:
            result = (1.+1.j)*numpy.pi**2/(B*gamma(-0.25)**4) * ( gamma(-0.25)**2/numpy.sqrt((eta-eta_BBstart)) \
                + 16.*(eta-eta_BBstart)**(3./2.)*gamma(3./4.)**2 - 20./3./B*(eta-eta_BBstart)**(3./2.) * ( 6*(-1)**0.25*A*numpy.pi**2 \
                + B*gamma(3./4.)**2 * ( -1+(3.-3.j)*numpy.pi+12.*numpy.log(2)-6.*numpy.log((eta-eta_BBstart)) ) ) ) 
            return result.real


    def solve_C(self):

        SR_start, SR_end, eta_SR_end = self.SR_t_eta()
        sol_inflating = self.PrimordialSolver.solve(self.N_i, self.phi_i, method='Radau', events=[Inflating(self), SlowRow(self)])
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
                Diff += numpy.log(self.anal_b_SR(eta_SR[i], eta_SR_end, C)) - N_SR[i]
            return Diff

        sol_C = root_scalar(Find_C, bracket=[0., 1.], method='brentq')
        C = sol_C.root
        return C

    def IC_SR(self, n):
        SR_start, SR_end, eta_SR_end = self.SR_t_eta()
        sol_inflating = self.PrimordialSolver.solve(self.N_i, self.phi_i, method='Radau', events=Inflating(self))
        SR_index = 0
        for i in range(len(sol_inflating.t)):
            if sol_inflating.t[i] < n* (1./5.)*SR_start < sol_inflating.t[i+1]:
                SR_index = i
       
        t_SR = sol_inflating.t[SR_index]
        N_SR = sol_inflating.y[0][SR_index]
        dN_SR = self.PrimordialSolver.calcH(t_SR, sol_inflating.y[:,SR_index])

        anal_b_SR_IC = self.anal_b_SR(sol_inflating.y[3][SR_index], eta_SR_end, self.solve_C())
        anal_b_SR_prime_IC = self.anal_b_SR_prime(sol_inflating.y[3][SR_index], eta_SR_end, self.solve_C())
        # y0_SR = [N_SR, dN_SR, numpy.log(anal_b_SR_IC), anal_b_SR_prime_IC/ (anal_b_SR_IC*numpy.exp(N_SR)), sol_inflating.y[1][SR_index], sol_inflating.y[2][SR_index], sol_inflating.y[3][SR_index]]
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

        anal_b_KD_IC = self.anal_b_KD(sol_KD.y[3][DKD_index], eta_BBstart, A, B)
        anal_b_KD_prime_IC = self.anal_b_KD_prime(sol_KD.y[3][DKD_index], eta_BBstart, A, B)
        y0_KD = [N_DKD, dN_DKD, numpy.log(anal_b_KD_IC), anal_b_KD_prime_IC/ (anal_b_KD_IC*numpy.exp(N_DKD)), sol_KD.y[1][DKD_index], sol_KD.y[2][DKD_index], sol_KD.y[3][DKD_index]]
        return [t_DKD, y0_KD]

    def approx_B(self):
        DKD_start, BB_start, eta_BBstart = self.KD_t_eta()
        sol_KD = self.PrimordialSolver.solve(self.N_i, self.phi_i, d=-1)
        # Find BG variables in deep inflation
        eta_KD = []
        N_KD = []
        dN_KD = []
        for i in range(len(sol_KD.t)):
            if BB_start <= sol_KD.t[i] <= 0.1*BB_start + 0.9*DKD_start:
                eta_KD.append(sol_KD.y[3][i])
                N_KD.append(sol_KD.y[0][i])
                dN_KD.append(self.PrimordialSolver.calcH(sol_KD.t[i], sol_KD.y[:,i]))
        A = 0.4
        def Find_B(B):
            Diff = 0.0
            for i in range(len(eta_KD)):
                Nb = numpy.log(self.anal_b_KD(eta_KD[i], eta_BBstart, A, B))
                Diff += Nb - N_KD[i]
            print(Diff)
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
            sol_inf = solve_ivp(self.f_b, [self.IC_SR()[0], 0.0], self.IC_SR()[1], method='Radau', d=-1)
            sol_KD = solve_ivp(self.f_b, [self.IC_KD(A, B)[0], 0.0], self.IC_KD(A, B)[1], method='Radau')
            # Determine whether the b from SR and KD consistant with each other
            print([sol_inf.y[2][-1]-sol_KD.y[2][-1], sol_inf.y[3][-1]-sol_KD.y[3][-1]])
            return [sol_inf.y[2][-1]-sol_KD.y[2][-1], sol_inf.y[3][-1]-sol_KD.y[3][-1]]

        res = root(Find_AB, self.approx_B(), method='hybr')
        A = res.x[0]
        B = res.x[1]
        print('A,B='+str(A)+','+str(B))
        return [A, B]

    def get_b_IC(self, n):
        sol_inf = solve_ivp(self.f_b, [self.IC_SR(n)[0], 0.0], self.IC_SR(n)[1], method='Radau', d=-1)
        return [sol_inf.y[2][-1], sol_inf.y[3][-1]]



class Inflating(object):
    terminal=True
    direction=1
    def __init__(self, solver):
        self.solver = solver

    def __call__(self, t, y):
        N, phi, dphi, eta = y
        return dphi**2 - self.solver.V(phi)

class SlowRow(object):
    terminal=False
    def __init__(self, solver):
        self.solver = solver

    def __call__(self, t, y):
        N, phi, dphi, eta = y
        p_phi = 0.5* dphi**2 - self.solver.V(phi)
        rho_phi = 0.5* dphi**2 + self.solver.V(phi)
        return p_phi/rho_phi- (-0.99)

class Deep_KD(object):
    terminal=False
    def __init__(self, solver):
        self.solver = solver

    def __call__(self, t, y):
        N, phi, dphi, eta = y
        return dphi**2 / abs(self.solver.V(phi)) -100


class Until_aH(object):
    terminal=True
    def __init__(self, solver, logaH):
        self.solver = solver
        self.logaH = logaH

    def __call__(self, t, y):
        N, phi, dphi, eta = y
        H = self.solver.calcH(t, y)
        return N + numpy.log(H) - self.logaH

class Until_H(object):
    terminal=True
    def __init__(self, solver, H):
        self.solver = solver
        self.H = H

    def __call__(self, t, y):
        return self.solver.calcH(t, y)/self.H - 1


def solve(x0, x, y):
    f = scipy.interpolate.interp1d(x,y)
    return float(f(x0))

def create_figure(K):
    # fig = plt.figure(tight_layout=True, fontproperties=myfont)
    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(2, 3, height_ratios=[2,1])
    ax0 = fig.add_subplot(gs[0, :])
    ax0.set_xticks([])

    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[1, 2])
    ax4 = ax0.twiny()


    if K == 0:
        minor_xticks = major_xticks = numpy.log(numpy.logspace(-90,0,31))
        minor_xticklabels = major_xticklabels = ['$10^{%i}$' % i for i in range(-90,1,3)]
    else:
        major_xticks = numpy.concatenate([
            numpy.log(numpy.logspace(0,12,5)/(Mpc/lp)),
            numpy.log(numpy.logspace(-12,9,8)/(Mpc)),
            numpy.log(numpy.logspace(0,18,7)/(Mpc/ly))
            ])

        major_xticklabels  = numpy.concatenate([
            ['%s$\ell_\mathrm{p}$' % i for i in ['', 'k', 'M', 'G', 'T']],
            ['%sm' % i for i in ['p', 'n', '$\mu$', 'm', '', 'k','M','G']],
            ['%sly' % i for i in ['', 'k', 'M', 'G', 'T', 'P', 'Y']],
            ])

        minor_xticks = numpy.concatenate([
            numpy.log(numpy.logspace(0,12,13)/(Mpc/lp)),
            numpy.log(numpy.logspace(-12,9,22)/(Mpc)),
            numpy.log(numpy.logspace(0,18,19)/(Mpc/ly))
            ])

        minor_xticklabels  = numpy.concatenate([
            ['%s%s$\ell_\mathrm{p}$' % (j,i) for i in ['', 'k', 'M', 'G', 'T'] for j in [1, 10, 100]][:-2],
            ['%s%sm' % (j,i) for i in ['p', 'n', '$\mu$', 'm', '', 'k','M','G'] for j in [1, 10, 100]][:-2],
            ['%s%sly' % (j,i) for i in ['', 'k', 'M', 'G', 'T', 'P', 'Y'] for j in [1, 10, 100]][:-2],
            ])


    i = [i for i in range(-60,61,3)]
    major_yticks = numpy.log(10.**(numpy.array(i)))
    major_yticklabels = ['$10^{%i}$' % j for j in i]


    ax0.set_yticks(major_yticks)
    ax0.set_yticklabels(major_yticklabels)
    ax4.set_xticks(major_xticks)
    ax4.set_xticklabels(major_xticklabels, rotation=90)

    for ax in [ax1, ax2, ax3]:
        ax.set_yticks(major_yticks)
        ax.set_yticklabels(major_yticklabels)
        ax.set_xticks(minor_xticks)
        ax.set_xticklabels(minor_xticklabels,rotation=90)


    if K==0:
        ax4.set_xlabel('Relative size of the universe, $a$')
        ax0.set_ylabel('Curvature, $|\Omega_K| = (aH)^{-2}$ $[\mathrm{Mpc}^{-2}]$')
    else:
        ax4.set_xlabel('Size of the universe, $a$')
        ax0.set_ylabel('Curvature, $|\Omega_K| = (aH)^{-2}$')
    return fig, ax0, ax1, ax2, ax3, ax4

class Solver_b(object):
    def __init__(self, N_i, V, H0=64.03, Omega_m=0.3453, omegabh2=0.022509, omegach2=0.11839, tau=0.0515, Omega_K=-0.0092, logA=3.0336, ns=0.96535, z=1089.61): # ns=0.9699
        # Initialise late-time parameters
        self.H0 = H0
        self.omegach2 = omegach2
        self.omegabh2 = omegabh2
        self.tau = tau
        self.Omega_m = Omega_m
        self.Omega_K = Omega_K
        self.z = z

        h = H0/100
        self.Omega_r = 4.18343e-5*h**-2

        if self.K == 0:
            self.a0 = 1
        else:
            self.a0 = numpy.sqrt(-self.K/self.Omega_K)*(c*1e-3)/self.H0
        self.logaH = self.logk_in_natural_units(0.05)

        # Initialise primordial by tuning phi_i to ns
        self.As = 1e-10*numpy.exp(logA)
        self.ns = ns
        self.N_i = N_i
        self.V = V
        self.phi_i = self.PrimordialSolver.find_phi_i(self.N_i, self.logaH, self.ns)
    
    def logk_in_natural_units(self, k_in_invMpc):
        #if self.K==0:
        #    return numpy.log(k_in_invMpc*(c*1e-3)/self.H0)
        #else:
        return numpy.log(k_in_invMpc*self.a0)

    @property
    def PrimordialSolver(self):
        return PrimordialSolver(self.V, self.K)

    @property
    def IC_b_Solver(self):
        return IC_b_Solver(self.V, self.K, self.phi_i, self.N_i)
    
    @property
    def K(self):
        return -numpy.sign(self.Omega_K)

    def test_b_IC(self):
        
        SR_start, SR_end, eta_SR_end = self.IC_b_Solver.SR_t_eta()
        print('SR_start, SR_end='+str(SR_start)+','+str(SR_end))
        # A, B = self.IC_b_Solver.solve_AB()

        sol_inflating = self.PrimordialSolver.solve(self.N_i, self.phi_i, method='Radau', events=Inflating(self))
        sol_KD = self.PrimordialSolver.solve(self.N_i, self.phi_i, method='Radau', d=-1)

        sol_inf_for = solve_ivp(self.IC_b_Solver.f_b, [self.IC_b_Solver.IC_SR()[0], SR_end], self.IC_b_Solver.IC_SR()[1], method='Radau')
        sol_inf_back = solve_ivp(self.IC_b_Solver.f_b, [self.IC_b_Solver.IC_SR()[0], -numpy.inf], self.IC_b_Solver.IC_SR()[1], method='Radau', d=-1)
        # sol_KD_for = solve_ivp(self.IC_b_Solver.f_b, [self.IC_b_Solver.IC_KD(A, B)[0], SR_end], self.IC_b_Solver.IC_KD(A, B)[1], method='Radau')
        # sol_KD_back = solve_ivp(self.IC_b_Solver.f_b, [self.IC_b_Solver.IC_KD(A, B)[0], -numpy.inf], self.IC_b_Solver.IC_KD(A, B)[1], method='Radau', d=-1)

        # plt.plot(sol_inflating.t, sol_inflating.y[0], label='a_for')
        # plt.plot(sol_KD.t, sol_KD.y[0], label='a_back')

        plt.plot(sol_inf_for.t, sol_inf_for.y[0], label='a_inf_for')
        plt.plot(sol_inf_back.t, sol_inf_back.y[0], label='a_inf_back')
        plt.plot(sol_inf_for.t, sol_inf_for.y[2], label='b_inf_for')
        plt.plot(sol_inf_back.t, sol_inf_back.y[2], label='b_inf_back')
        
        # plt.plot(sol_KD_for.t, sol_KD_for.y[0], label='a_KD_for')
        # plt.plot(sol_KD_back.t, sol_KD_back.y[0], label='a_KD_back')
        # plt.plot(sol_KD_for.t, sol_KD_for.y[2], label='b_KD_for')
        # plt.plot(sol_KD_back.t, sol_KD_back.y[2], label='b_KD_back')
        plt.legend()
        plt.xlim([-1, 3.5])
        plt.ylim([-6, 3.5])
        plt.show()

        return 0

# N_i_min, N_i_med, N_i_max= -1.5, -0.4676751301973933, 0.5646497396052134
# universe = Solver_b(N_i_min, V)
# universe.test_b_IC()

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
        return - numpy.sqrt(V(phi))

    def N_dot_IC(self, K, phi, N): # when start of inflation
        return numpy.sqrt( 1.0/2.0 * self.phi_dot_IC(phi)**2 - K* numpy.exp(-2.0*N) )

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
        
        b = numpy.exp(Nb)
        db = b* dNb
        # solve ODEs to get solution of BG variables
        ddN = -1.0/2.0 * self.phi_dot_IC(phi)**2 + K* numpy.exp(-2*N)
        ddNb = dNb**2 + ddN - self.N_dot_IC(K, phi, N)**2 - K*numpy.exp(-2*N)
        ddb = b * (ddNb + (db/b)**2)
        dda = numpy.exp(N) * (ddN + self.N_dot_IC(K, phi, N)**2)

        zeta_IC = self.zeta_IC(self.phi_dot_IC(phi), b, db, k, K)
        zeta_dot_IC = self.zeta_dot_IC(self.phi_dot_IC(phi), - 3.0*self.N_dot_IC(K, phi, N)*self.phi_dot_IC(phi) - self.V.d(phi), numpy.exp(N), numpy.exp(N)*self.N_dot_IC(K, phi, N), b, db, ddb, k, K)
        R_IC = self.R_PPS(k, K, zeta_IC, zeta_dot_IC, numpy.exp(N), numpy.exp(N)*self.N_dot_IC(K, phi, N), dda, b, db, ddb)
        R_dot_IC =  self.R_dot_PPS(k, K, zeta_IC, zeta_dot_IC, self.phi_dot_IC(phi), numpy.exp(N), numpy.exp(N)*self.N_dot_IC(K, phi, N), dda, b, db, ddb)
        
        return [R_IC, R_dot_IC]


class Solver(object):
    def __init__(self, n, N_i, V, H0=64.03 , omegabh2=0.022509, omegach2=0.11839, Omega_K=-0.0092, Omega_m=0.3453, tau=0.0515, logA=3.0336, ns=0.96535, z=1089.61): # ns=0.9699
        # Initialise late-time parameters
        self.n = n
        self.H0 = H0
        self.omegach2 = omegach2
        self.omegabh2 = omegabh2
        self.tau = tau
        self.Omega_m = Omega_m
        self.Omega_K = Omega_K
        self.z = z

        h = H0/100
        self.Omega_r = 4.18343e-5*h**-2

        if self.K == 0:
            self.a0 = 1
        else:
            self.a0 = numpy.sqrt(-self.K/self.Omega_K)*(c*1e-3)/self.H0
        self.logaH = self.logk_in_natural_units(0.05)

        # Initialise primordial by tuning phi_i to ns
        self.As = 1e-10*numpy.exp(logA)
        self.ns = ns
        self.N_i = N_i
        self.V = V
        self.phi_i = self.PrimordialSolver.find_phi_i(self.N_i, self.logaH, self.ns)
        self.Nb_i, self.dNb_i = self.IC_b_Solver.get_b_IC(self.n)

        # Determine the mass
        _, As_, r = self.PrimordialSolver.find_ns_As_r(self.N_i, self.phi_i, self.logaH)
        self.m = numpy.sqrt(self.As / As_)

    def logk_in_natural_units(self, k_in_invMpc):
        #if self.K==0:
        #    return numpy.log(k_in_invMpc*(c*1e-3)/self.H0)
        #else:
        return numpy.log(k_in_invMpc*self.a0)

    def N_to_Mpc(self, N):
        return N -numpy.log(self.m) + numpy.log(lp/Mpc)

    @property
    def PrimordialSolver(self):
        return PrimordialSolver(self.V, self.K)

    @property
    def Omega_l(self):
        return 1 - self.Omega_m - self.Omega_r - self.Omega_K
    
    @property
    def IC_b_Solver(self):
        return IC_b_Solver(self.V, self.K, self.phi_i, self.N_i)

    @property
    def R_func(self):
        return R_func(self.V)

    @property
    def K(self):
        return -numpy.sign(self.Omega_K)

    def pre_inflation(self):
        sol = self.PrimordialSolver.solve(self.N_i, self.phi_i, d=-1)
        H = self.PrimordialSolver.calcH(sol.t, sol.y)
        N = numpy.flip(self.N_to_Mpc(sol.y[0]))
        Omega = numpy.flip(-2*(numpy.log(H)+sol.y[0]))
        eta = -sol.y[3,-1]
        return N, Omega, eta

    def inflation(self, solution=0):
        sol = self.PrimordialSolver.solve(self.N_i, self.phi_i,
                                          events=Inflating(self.PrimordialSolver))
        H = self.PrimordialSolver.calcH(sol.t, sol.y)
        if solution:
            t_eval = numpy.linspace(sol.t[0], sol.t[-1], solution)
            sol = self.PrimordialSolver.solve(self.N_i, self.phi_i, t_eval=t_eval,
                                              events=Inflating(self.PrimordialSolver))
            K = self.PrimordialSolver.K
            N, phi, dphi, _ = sol.y
            H, _, ddphi, deta = self.PrimordialSolver.f(sol.t, sol.y)
            dH = -dphi**2/2 + K*numpy.exp(-2*N)
            eps = dphi**2/H**2/2 # the original code without /2, but I still don't know why
            z = numpy.exp(N)*dphi/H
            dlogz = H + ddphi/dphi - dH/H
            return sol.t, eps, dlogz, N, H, z, K

        N = self.N_to_Mpc(sol.y[0])
        Omega = -2*(numpy.log(H)+sol.y[0])
        eta = sol.y[3,-1]
        return N, Omega, eta

    def reheating(self):
        sol = self.PrimordialSolver.solve_post(self.N_i, self.phi_i, 0.1)
        H = self.PrimordialSolver.calcH(sol.t, sol.y)
        N = self.N_to_Mpc(sol.y[0])
        Omega = -2*(numpy.log(H)+sol.y[0])
        eta = sol.y[3,-1]

        # N1 = root_scalar(lambda N_: self.Omega_late_time(N_)[0] - (Omega[-1] + (1+3*V.w)*(N_-N[-1])), bracket=[-120,numpy.log(self.a0)]).root
        # Omega1 = self.Omega_late_time(N1)[0]

        # x = numpy.linspace(0,1,1000)
        # N = numpy.concatenate([N, N1*x + N[-1]*(1-x)])
        # Omega = numpy.concatenate([Omega, Omega1*x + Omega[-1]*(1-x)])

        return N, Omega, eta

    def Omega_late_time(self, N):
        N = numpy.atleast_1d(N)
        N0 = numpy.log(self.a0)
        logH = numpy.log(self.H0) + logsumexp(numpy.transpose([4*(N0-N),3*(N0-N),2*(N0-N),numpy.zeros_like(N)]), b=[[self.Omega_r,self.Omega_m,self.Omega_K,self.Omega_l]], axis=1)/2
        return -2*(N+logH)+2*numpy.log(c*1e-3)

    def eta_frac(self):
        eta_pre = self.pre_inflation()[2]
        # print(eta_pre)
        eta_inflating = self.inflation()[2]
        eta_reheating = self.reheating()[2]
        eta_pre_cmb = scipy.integrate.quad(lambda N: numpy.exp(self.Omega_late_time(N)[0]*0.5), -numpy.inf, numpy.log(self.a0/(1+self.z)))[0]
        eta_post_cmb = scipy.integrate.quad(lambda N: numpy.exp(self.Omega_late_time(N)[0]*0.5), numpy.log(self.a0/(1+self.z)),numpy.log(self.a0))[0]
        return (eta_pre+eta_inflating + eta_reheating + eta_pre_cmb)/eta_post_cmb

    def approx_PR(self):
        sol = self.PrimordialSolver.solve(self.N_i, self.phi_i,
                                          events=Inflating(self.PrimordialSolver))
        H = self.PrimordialSolver.calcH(sol.t, sol.y)
        N, phi, dphi, eta = sol.y
        PR = H**4/dphi**2/(2*numpy.pi)**2 *self.m**2
        k = numpy.exp(N)*H/self.a0
        return k, PR
    
    def get_R_i(self, k): 
        R_i, dR_i = self.R_func.get_R_IC(k, self.PrimordialSolver.K, self.phi_i, self.N_i, self.Nb_i, self.dNb_i)
        return [R_i, dR_i]
    
    def plot_aH(self):
        sol = self.PrimordialSolver.solve_post(self.N_i, self.phi_i, 0.1)
        N_late_time = numpy.linspace(self.reheating()[0][-1], numpy.log(self.a0/(1+self.z)), 1000)
        Omega_late_time = self.Omega_late_time(N_late_time)
        
        # plt.plot(self.pre_inflation()[0], self.pre_inflation()[1],label="{} primordial $\Omega_K$".format(labellist[self.n]),color=colorlist[self.n])
        plt.plot(self.pre_inflation()[0], self.pre_inflation()[1],label="ns={}".format(self.ns),color=colorlist[self.n])
        plt.plot(self.inflation()[0], self.inflation()[1], color=colorlist[self.n])
        plt.plot(self.reheating()[0], self.reheating()[1], color=colorlist[self.n])
        plt.plot(N_late_time, Omega_late_time, color=colorlist[self.n])
        return 0
        # return Omega_late_time[0] - self.reheating()[1][-1]
    


def eta_frac(N_i):
    print('N_i='+str(N_i))
    try:
        return Solver(0, N_i, V).eta_frac() - 1.0
    except ValueError:
        return -1


import pyoscode
import tqdm

## find ns
# def find_ns(ns):
#     #print(N_i)
#     try:
#         return Solver(0, N_i, V, ns).plot_aH() 
#     except ValueError:
#         return -1

# ns = root_scalar(find_ns, bracket=[0.965, 0.966]).root
# print('ns='+str(ns))


# Maximum N_i -- as flat as it can be whilst solving horizon problem
# print(eta_frac(-1))
# N_i = root_scalar(eta_frac, bracket=[1, 3]).root
# N_i_max = N_i
# N_i_min = 0.35
# N_i_med = (N_i_max + N_i_min)/2

# N_i_min, N_i_med, N_i_max= -1.5, -0.4676751301973933, 0.5646497396052134  # V=phi^4/3, ns=0.9699
# N_i_min, N_i_med, N_i_max= 0.35, 1.315024545227802, 2.280049090455604 # V=Starobinsky, ns=0.96535
# print('N_i_min, N_i_med, N_i_max= '+str(N_i_min)+', '+str(N_i_med)+', '+str(N_i_max))



# # labellist = ["max","med","min"]
# # colorlist = ['darkblue', 'steelblue', 'lightsteelblue']

# for universe in universes:
#     universe.plot_aH()
# plt.legend()
# plt.show()



## total PPS by A,B
# N_i_list = [0.5646497396052134, 0.3, 0.0, -0.4676751301973933, -0.8, -1, -1.2]
# A_list = [0.015434555824034746, 0.2663686756381103, 0.4045194401051108, 0.4757350283585328, 0.47150000231818257, 0.4570910789851049, 0.4339596232958878]
# B_list = [3.490187594681386,2.337205365309828,1.4758890120102401, 0.7015989176569104, 0.3986209881911241, 0.26958840528786576, 0.16357678838050468]
# universes =[Solver(N_i_list[i], V, A_list[i], B_list[i]) for i in range(len(N_i_list))]
# labellist = ['0.565', '0.3', '0.0', '-0.468', '-0.8', '-1', '-1.2']
# colorlist = ['red', 'sandybrown', 'gold', 'darkkhaki', 'chartreuse', 'deepskyblue', 'blue']

N_i = 0.5
universes = [Solver(0, N_i, V)]
colorlist = ['darkblue']
labellist = ['med']

for figname in ['jan22-8am-CMB-diffNi-actual-b-will']:

    fig, ax = plt.subplots(1)

    def PR_analytic(k, s):
        ks = 0.05
        return s.As * (k / ks)**(s.ns-1)

    i = 0
    ks = numpy.arange(3,25000)

    for universe in universes:

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
            w2 = (-D2**2 + K*(1+eps-2/H*dlogz)*D2+K**2*eps)*numpy.exp(-2*N)/(D2-K*eps)
            w = numpy.sqrt(w2)

            # RST for R and zetapf
            R_i, dR_i = universe.get_R_i(k)
            sol = pyoscode.solve(t, numpy.log(w), gamma, t[0], t[-1], R_i, dR_i, logw=True)
            R.append(sol['sol'][-1])
            # print(sol['sol'][-1])

        R = numpy.array(R)
        P = ks**3/2/numpy.pi**2*abs(R)**2 * universe.m**2

        P = P /P[-1]*PR_analytic(ks[-1]/universe.a0, universe)

        labelval = labellist[i]
        # ax.plot(ks/universe.a0, numpy.log(1e10*P), zorder=3, label="{} primordial $\Omega_K$".format(labelval),color=colorlist[i])
        # ax.plot(ks/universe.a0, numpy.log(1e10*P), zorder=3, label="$N_i=$ {} ".format(labelval),color=colorlist[i])

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
        cosmo_tt = l*(l+1)*cls['tt'][2:] * (1e6 * 2.7255)**2 / (2*numpy.pi)
        
        numpy.savetxt('nov28-5pm-correctedR.dat', numpy.array([
            numpy.concatenate([[1e-8, 1e-7], ks/universe.a0, [1e5]]),
            numpy.concatenate([[P[0], P[0]], P, [PR_analytic(1e5, universe)]])
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
        tt0 = l*(l+1)*cls['tt'][2:] * (1e6 * 2.7255)**2 / (2*numpy.pi)
        
        Cl = numpy.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt')
        tt_dat = Cl[:len(cosmo_tt),1]
        tt_err_minus = Cl[:len(cosmo_tt),2]
        tt_err_plus = Cl[:len(cosmo_tt),3]
        
        def chi2(tt, lmax=-1):
            return ((tt_dat - tt)/numpy.where(tt_dat<tt,tt_err_plus,tt_err_minus))**2
        
        chi0 = (chi2(tt0)-chi2(cosmo_tt))[:30].sum()
        print(chi0)
        ax.plot(l,tt0, zorder=3, label='$\Delta\chi^2=%.2f$' % chi0, color=colorlist[i])
        
        if i == 2:
            ax.plot(l,cosmo_tt, zorder=4,label="$\Lambda$CDM", color="red")
        i = i+1


    # universe = universes[1]
    # ax.plot(ks/universe.a0, numpy.log(1e10*PR_analytic(ks/universe.a0,universe)), zorder=4, label="$\Lambda$CDM", color="red")
    # ax.set_xscale('log')
    # ax.set_xlim(1e-4,10**(-0.3))
    # ax.set_ylim(2, 4)
    # ax.set_xlabel('$k\:[\mathrm{Mpc}^{-1}]$')
    # ax.set_ylabel(r'$\log\left( 10^{10}\mathcal{P}_\mathcal{R}\right)$')
    # ax.set_yticks([2,2.5,3,3.5,4])
    # ax.legend(loc = 'lower right')


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
### Make the PPS figure with different R_IC

labellist = ["max","med","min"]
colorlist = ['darkblue', 'steelblue', 'lightsteelblue', 'g']
R_IC_name_list = ['BD', 'FlatRST', 'MaryRST', 'bRST']
universe =Solver(N_i_med, V) 

for figname in ['jan15-9am-PPS-diffRIC-Nmed']:
    ks = numpy.arange(3,25000)

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
            w2 = (-D2**2 + K*(1+eps-2/H*dlogz)*D2+K**2*eps)*numpy.exp(-2*N)/(D2-K*eps)
            w = numpy.sqrt(w2)

            # Stuff
            a0 = numpy.exp(N[0])
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
                    R0 = 1./ (z0*numpy.sqrt(2*k))
                    dR0 = ( -1.j*k/a0 - dlogz0 )* R0
                
                elif R_IC_name =='FlatRST':
                    R0 = 1./ (z0*numpy.sqrt(2*k))
                    dR0 = - 1.j*k*R0/ a0

                elif R_IC_name =='MaryRST':
                    # RST  for R and zetapf (Mary's)
                    R0 = 1/(z[0]*numpy.sqrt(2*(numpy.sqrt(-D2))))
                    dR0 = R0*(K/(H0*(a0**2)) + (1 - K*(z[0]**2)/(2*D2*(a0**2)))*((-1j * numpy.sqrt(-D2)/a0) + H0 - dlogz[0] - K/(H0*(a0**2))))
                
                elif R_IC_name =='bRST':
                    R0, dR0 = universe.get_R_i(k)

                else:print('R_IC should be BD, FlatRST, MaryRST, or bRST')

                return [R0, dR0]

            R0, dR0 = R_IC(R_IC_name, k)
            # Ignore this
            # gamma = dlogz - dlogca + H/2
            # w2 = -(ca2)*(D2-3*K)*numpy.exp(-2*N)
            # w = numpy.sqrt(w2)


            # zeta perfect fluid - Ignore this
            # dR0 = R0*(1 + (1/D2)*(K/H0)*(-3*H0 - 2*dH0/H0 + 2*dlogz[0]))*((-1j * numpy.sqrt(-D2)/a0) + H0 - dlogz[0])
            sol = pyoscode.solve(t, numpy.log(w), gamma, t[0], t[-1], R0, dR0, logw=True)
            R.append(sol['sol'][-1])
            # print(sol['sol'][-1])

        R = numpy.array(R)
        P = ks**3/2/numpy.pi**2*abs(R)**2 * universe.m**2

        P = P /P[-1]*PR_analytic(ks[-1]/universe.a0, universe)

        # labelval = labellist[i]
        # ax.plot(ks/universe.a0, numpy.log(1e10*P), zorder=3, label="{} primordial $\Omega_K$".format(labelval),color=colorlist[i])
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
        cosmo_tt = l*(l+1)*cls['tt'][2:] * (1e6 * 2.7255)**2 / (2*numpy.pi)
        
        numpy.savetxt('nov28-5pm-correctedR.dat', numpy.array([
            numpy.concatenate([[1e-8, 1e-7], ks/universe.a0, [1e5]]),
            numpy.concatenate([[P[0], P[0]], P, [PR_analytic(1e5, universe)]])
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
        tt0 = l*(l+1)*cls['tt'][2:] * (1e6 * 2.7255)**2 / (2*numpy.pi)
        
        Cl = numpy.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt')
        tt_dat = Cl[:len(cosmo_tt),1]
        tt_err_minus = Cl[:len(cosmo_tt),2]
        tt_err_plus = Cl[:len(cosmo_tt),3]
        
        def chi2(tt, lmax=-1):
            return ((tt_dat - tt)/numpy.where(tt_dat<tt,tt_err_plus,tt_err_minus))**2
        
        chi0 = (chi2(tt0)-chi2(cosmo_tt))[:30].sum()
   
        ax.plot(l,tt0, zorder=3, label='$\Delta\chi^2=%.2f$' % chi0, color=colorlist[i])
        
        if i == 2:
            ax.plot(l,cosmo_tt, zorder=4,label="$\Lambda$CDM", color="red")
        i = i+1

    # universe = universes[2]
    # ax.plot(ks/universe.a0, numpy.log(1e10*PR_analytic(ks/universe.a0,universe)), zorder=4, label="$\Lambda$CDM", color="red")
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
"""
#######################
# Define the universes
N_i = 0.5
universes = []
for n in range(6):
    universes.append(Solver(n, N_i, V))

########################
# chi of PPS (with a&b match at different t): 
chi0_list = [-2.6170938267356494, -2.619444576678246, -2.622022846498224, -2.6338828810735553, -2.618434736003726, -2.6320884016769655]

########################
# output PPS 
k_iMpc = numpy.concatenate((
        numpy.logspace(numpy.log10(5e-7), numpy.log10(5e-5), 2 * 100 + 1),
        numpy.logspace(numpy.log10(5e-5), numpy.log10(5e-1), 4 * 200 + 1)[1:],
        numpy.logspace(numpy.log10(5e-1), numpy.log10(5e0), 1 * 50 + 1)[1:],
        numpy.logspace(numpy.log10(5e0), numpy.log10(5e1), 1 * 10 + 1)[1:]
    ))

def PR_analytic(k, s):
    ks = 0.05
    return s.As * (k / ks)**(s.ns-1)

n=0
for universe in universes:

    t, eps, dlogz, N, H, z, K = universe.inflation(10000)
    R = []

    ks = k_iMpc * universe.a0
    if K == +1:
        ks = ks[ks >= 1]

    for k in tqdm.tqdm(ks):
        # D squared operator in fourier space = Laplacian + 3K
        D2 = -k*(k+2)+3*K

        # Differential Equation for R variable in coordinate time (t) for PyOscode Package
        gamma = ((H+2*dlogz)*D2 - 3*K*H*eps)/(D2-K*eps)/2
        w2 = (-D2**2 + K*(1+eps-2/H*dlogz)*D2+K**2*eps)*numpy.exp(-2*N)/(D2-K*eps)
        w = numpy.sqrt(w2)

        # RST for R and zetapf
        R_i, dR_i = universe.get_R_i(k)
        sol = pyoscode.solve(t, numpy.log(w), gamma, t[0], t[-1], R_i, dR_i, logw=True)
        R.append(sol['sol'][-1])
        # print(sol['sol'][-1])

    R = numpy.array(R)
    P = ks**3/2/numpy.pi**2*abs(R)**2 * universe.m**2

    P = P /P[-1]*PR_analytic(ks[-1]/universe.a0, universe)
    
    plt.plot(ks/universe.a0, P, label='n='+str(n))
    numpy.savetxt('PPS_will_b-n='+str(n)+'.dat', numpy.array([ks/universe.a0, P]).T)
    n += 1

plt.show()
plt.legend()
"""