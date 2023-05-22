import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import patches
from numpy import exp
import numpy
from scipy.optimize import root_scalar
from scipy.integrate import solve_ivp
from scipy.special import logsumexp
import classy

from scipy.constants import c, hbar, G
import scipy.interpolate

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
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "cm"


n = 2
V = lambda phi: phi**2/2
V.d = lambda phi: phi
V.dd = lambda phi: 1

#n = 2
#V = lambda phi: (1-numpy.exp(-numpy.sqrt(2./3)*phi))**2
#V.d = lambda phi: 2*(1-numpy.exp(-numpy.sqrt(2./3)*phi))* numpy.sqrt(2./3)*numpy.exp(-numpy.sqrt(2./3)*phi)
#V.dd = lambda phi: 8./3*numpy.exp(-2*numpy.sqrt(2./3)*phi) - 4./3*numpy.exp(-numpy.sqrt(2./3)*phi)

n = 4./3
V = lambda phi: numpy.abs(phi)**n
V.d = lambda phi: n * numpy.sign(phi)*numpy.abs(phi)**(n-1)
V.dd = lambda phi: n * (n-1) *numpy.abs(phi)**(n-2)

V.w = (n-2)/(n+2)


class PrimordialSolver(object):
    def __init__(self, V, K):
        self.V = V
        self.K = K

    def calcH(self, t, y):
        N, phi, dphi, eta = y
        H2 = (dphi**2/2 + self.V(phi))/3 - self.K*exp(-2*N)
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

class Inflating(object):
    terminal=True
    direction=1
    def __init__(self, solver):
        self.solver = solver

    def __call__(self, t, y):
        N, phi, dphi, eta = y
        return dphi**2 - self.solver.V(phi)

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


class Solver(object):
    def __init__(self, N_i, V, H0=64.03, Omega_m=0.3453, omegabh2=0.022509, omegach2=0.11839, tau=0.0515, Omega_K=-0.0092, logA=3.0336, ns=0.9699, z=1089.61):
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
            eps = dphi**2/H**2
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

        N1 = root_scalar(lambda N_: self.Omega_late_time(N_)[0] - (Omega[-1] + (1+3*V.w)*(N_-N[-1])), bracket=[-120,numpy.log(self.a0)]).root
        Omega1 = self.Omega_late_time(N1)[0]

        x = numpy.linspace(0,1,1000)
        N = numpy.concatenate([N, N1*x + N[-1]*(1-x)])
        Omega = numpy.concatenate([Omega, Omega1*x + Omega[-1]*(1-x)])

        return N, Omega, eta

    def Omega_late_time(self, N):
        N = numpy.atleast_1d(N)
        N0 = numpy.log(self.a0)
        logH = numpy.log(self.H0) + logsumexp(numpy.transpose([4*(N0-N),3*(N0-N),2*(N0-N),numpy.zeros_like(N)]), b=[[self.Omega_r,self.Omega_m,self.Omega_K,self.Omega_l]], axis=1)/2
        return -2*(N+logH)+2*numpy.log(c*1e-3)

    def eta_frac(self):
        eta_pre = self.pre_inflation()[2]
        print(eta_pre)
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


def eta_frac(N_i):
    #print(N_i)
    try:
        return Solver(N_i, V).eta_frac() - 1
    except ValueError:
        return -1



# Maximum N_i -- as flat as it can be whilst solving horizon problem
print(eta_frac(-1))
N_i = root_scalar(eta_frac, bracket=[-1, 1]).root
N_i_max = N_i
N_i_min = -1.5
N_i_med = (N_i_max + N_i_min)/2

universes =[Solver(N, V) for N in [N_i_min, N_i_med, N_i_max]]


import pyoscode
import tqdm

labellist = ["max","med","min"]
colorlist = ['darkblue', 'steelblue', 'lightsteelblue']

for figname in ['nov28-10pm-zeta-min']:
    ks = numpy.arange(3,25000)

    def PR_analytic(k, s):
        ks = 0.05
        return s.As * (k / ks)**(s.ns-1)

    fig, ax = plt.subplots(1)
    i = 0
    for universe in universes:
        t, eps, dlogz, N, H, z, K = universe.inflation(10000)
        m = len(t)//5
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

            # Ignore this
            # gamma = dlogz - dlogca + H/2
            # w2 = -(ca2)*(D2-3*K)*numpy.exp(-2*N)
            # w = numpy.sqrt(w2)

            # RST  for R and zetapf
            R0 = 1/(z[0]*numpy.sqrt(2*(numpy.sqrt(-D2))))
            dR0 = R0*(K/(H0*(a0**2)) + (1 - K*(z[0]**2)/(2*D2*(a0**2)))*((-1j * numpy.sqrt(-D2)/a0) + H0 - dlogz[0] - K/(H0*(a0**2))))

            # zeta perfect fluid - Ignore this
            # dR0 = R0*(1 + (1/D2)*(K/H0)*(-3*H0 - 2*dH0/H0 + 2*dlogz[0]))*((-1j * numpy.sqrt(-D2)/a0) + H0 - dlogz[0])
            sol = pyoscode.solve(t, numpy.log(w), gamma, t[0], t[-1], R0, dR0, logw=True)
            R.append(sol['sol'][-1])
            print(sol['sol'][-1])

        R = numpy.array(R)
        P = ks**3/2/numpy.pi**2*abs(R)**2 * universe.m**2

        P = P /P[-1]*PR_analytic(ks[-1]/universe.a0, universe)

        labelval = labellist[i]
        ax.plot(ks/universe.a0, numpy.log(1e10*P), zorder=3, label="{} primordial $\Omega_K$".format(labelval),color=colorlist[i])
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
        # cosmo_tt = l*(l+1)*cls['tt'][2:] * (1e6 * 2.7255)**2 / (2*numpy.pi)
        
        # numpy.savetxt('nov28-5pm-correctedR.dat', numpy.array([
        #     numpy.concatenate([[1e-8, 1e-7], ks/universe.a0, [1e5]]),
        #     numpy.concatenate([[P[0], P[0]], P, [PR_analytic(1e5, universe)]])
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
        # tt0 = l*(l+1)*cls['tt'][2:] * (1e6 * 2.7255)**2 / (2*numpy.pi)
        
        # Cl = numpy.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt')
        # tt_dat = Cl[:len(cosmo_tt),1]
        # tt_err_minus = Cl[:len(cosmo_tt),2]
        # tt_err_plus = Cl[:len(cosmo_tt),3]
        
        # def chi2(tt, lmax=-1):
        #     return ((tt_dat - tt)/numpy.where(tt_dat<tt,tt_err_plus,tt_err_minus))**2
        
        # chi0 = (chi2(tt0)-chi2(cosmo_tt))[:30].sum()
        
        # ax = axes[1]
        # ax.plot(l,tt0, zorder=3, label='$\Delta\chi^2=%.2f$' % chi0, color=colorlist[i])
        
        # if i == 2:
        #     ax.plot(l,cosmo_tt, zorder=4,label="$\Lambda$CDM", color="red")
        # i = i+1

    universe = universes[2]
    ax.plot(ks/universe.a0, numpy.log(1e10*PR_analytic(ks/universe.a0,universe)), zorder=4, label="$\Lambda$CDM", color="red")
    ax.set_xscale('log')
    ax.set_xlim(1e-4,10**(-0.3))
    ax.set_ylim(2, 4)
    ax.set_xlabel('$k\:[\mathrm{Mpc}^{-1}]$')
    ax.set_ylabel(r'$\log\left( 10^{10}\mathcal{P}_\mathcal{R}\right)$')
    ax.set_yticks([2,2.5,3,3.5,4])
    ax.legend()

    # Plots the right hand side of the figures in paper
    
    # ax = axes[1]
    # ax.set_xscale('log')
    # ax.errorbar(l, tt_dat, yerr=(tt_err_minus, tt_err_plus), fmt='.', color='r', ecolor='k',marker='o',linestyle='None',capsize=0, markersize=4,zorder=2000,elinewidth=1, markeredgecolor='k')
    # ax.set_xlim(1.9,30.5)
    # ax.set_ylim(0,2500)
    # ax.set_ylabel('$\mathcal{D}_\ell^{TT}\: [\mu \mathrm{K}^2]$')
    # ax.set_xlabel('$\ell$')
    # ax.set_xticks([2,10,30])
    # ax.set_xticklabels([2,10,30])
    # ax.legend()
    fig.set_size_inches(3.5,3.3)
    fig.tight_layout()
    fig.savefig(figname + '.pdf')


#lmax = 30
#
#axes[1].plot(l, numpy.zeros_like(l))
#axes[1].plot(l, chi2(cosmo_tt) - chi2(tt0))
#axes[1].plot(l, chi2(cosmo_tt) - chi2(tt1))
#axes[1].set_yscale('log')
#chi2(cosmo_tt)
#
#chi2(cosmo_tt)[:50].sum()
#chi2(tt0)[:50].sum()
#chi2(tt1)[:50].sum()

# universe = universes[-1]
# fig, ax0, ax1, ax2, ax3, ax4 = create_figure(universe.K)
# axes = [ax4, ax1, ax2, ax3]
#
# for ax in axes:
#     color = next(ax._get_lines.prop_cycler)['color']
#
# # Pre-inflation
# # During-inflation
# for i, (label, universe) in enumerate(zip(['min', 'med', 'max'], universes)):
#     N_pre, Omega_pre, eta_pre = universe.pre_inflation()
#     numpy.exp(N_pre[-1])*(Mpc/lp)/1000
#     N_pre
#     N_inflating, Omega_inflating, eta_inflating = universe.inflation()
#     label = '%s primordial $\Omega_K$' % label
#     for ax in axes:
#         ax.plot(numpy.concatenate([N_pre, N_inflating]), numpy.concatenate([Omega_pre, Omega_inflating]), label=label, zorder=-i)
#
# # Post-inflation
# N_reheating, Omega_reheating, eta_reheating = universe.reheating()
# for ax in axes:
#     ax.plot(N_reheating, Omega_reheating, label='Reheating', color=color)
#
# # Late-time universe
# N_late = numpy.linspace(N_reheating[-1],numpy.log(universe.a0),10000)
# Omega_late = universe.Omega_late_time(N_late)
# for ax in axes:
#     ax.plot(N_late, Omega_late, label='Late time evolution')
#
#
# Omega_star = -2*universe.logk_in_natural_units(0.05)
# Omega_min = -2*universe.logk_in_natural_units(1e-4)
# Omega_max = -2*universe.logk_in_natural_units(1)
#
# y = numpy.linspace(*ax4.get_ylim(),1000)
# x = numpy.log(universe.a0/(1+universe.z)) + 0.2*numpy.sin(0.2*2*numpy.pi*y)
#
# for ax in axes:
#     ax.axhline(Omega_min, color='k', linestyle=':', linewidth=0.5, label=r'$k_\mathrm{min}=10^{-4} \mathrm{Mpc}^{-1}$')
#     ax.axhline(Omega_star, color='k', linestyle='-', linewidth=0.5, label='$k_*=0.05 \mathrm{Mpc}^{-1}$')
#     ax.axhline(Omega_max, color='k', linestyle='--', linewidth=0.5, label='$k_\mathrm{max}=1 \mathrm{Mpc}^{-1}$')
#     ax.plot(x, y, label='CMB')
#
# eta_pre_cmb = scipy.integrate.quad(lambda N: numpy.exp(universe.Omega_late_time(N)[0]*0.5), -numpy.inf, numpy.log(universe.a0/(1+universe.z)))[0]
# eta_post_cmb = scipy.integrate.quad(lambda N: numpy.exp(universe.Omega_late_time(N)[0]*0.5), numpy.log(universe.a0/(1+universe.z)),numpy.log(universe.a0))[0]
#
# ((eta_pre+eta_inflating) + eta_pre_cmb)/eta_post_cmb
#
#
# #for N_i in numpy.linspace(N_i_min, N_i_max, 10):
# #    universe = Solver(N_i, V)
# #    # Pre-inflation
# #    N_pre, Omega_pre, eta_pre = universe.pre_inflation()
# #    for ax in [ax1]:
# #        ax.plot(N_pre, Omega_pre, 'k-', zorder=-1000, linewidth=0.25)
# #
# #    # During-inflation
# #    N_inflating, Omega_inflating, eta_inflating = universe.inflation()
# #    for ax in [ax1]:
# #        ax.plot(N_inflating, Omega_inflating, 'k-', zorder=-1000, linewidth=0.25)
# #
#
# pad = 2
# xmin = min(N_reheating) - pad
# xmax = max(N_reheating) + pad
# ymin = min(Omega_reheating) - pad
# ymax = max(Omega_reheating) + pad
#
# rect2 = patches.Rectangle([xmin, ymin], xmax-xmin, ymax-ymin,fill=False)
# ax4.add_patch(rect2)
# (xmin, ymin), (xmax, ymax) = rect2.get_bbox().get_points()
# ax2.set_xlim(xmin, xmax)
# ax2.set_ylim(ymin, ymax)
#
#
# pad = 2
# ymax = Omega_pre[-1] + pad
# ymin = max(Omega_max - pad, Omega_pre[0])
# xmin = solve(ymin, Omega_pre, N_pre)
# xmax = solve(ymin, Omega_inflating, N_inflating)
#
# rect1 = patches.Rectangle([xmin, ymin], xmax-xmin, ymax-ymin,fill=False)
#
# ax4.add_patch(rect1)
# (xmin, ymin), (xmax, ymax) = rect1.get_bbox().get_points()
# ax1.set_xlim(xmin, xmax)
# ax1.set_ylim(ymin, ymax)
#
#
# pad = 2
# ymin = Omega_max - pad
# ymax = Omega_min + pad
# xmax = N_late[-1]
# xmin = solve(ymin, Omega_late, N_late)
#
# rect3 = patches.Rectangle([xmin, ymin], xmax-xmin, ymax-ymin,fill=False)
# ax4.add_patch(rect3)
# (xmin, ymin), (xmax, ymax) = rect3.get_bbox().get_points()
# ax3.set_xlim(xmin, xmax)
# ax3.set_ylim(ymin, ymax)
#
#
# # Adjust overall axes
# right_pad = 2
# left_pad = 2
# upper_pad = 2
# lower_pad = 2
#
# ax4.set_ylim(rect2.get_bbox().get_points()[0,1]-lower_pad, rect1.get_bbox().get_points()[1,1]+upper_pad)
# ax4.set_xlim(rect1.get_bbox().get_points()[0,0]-left_pad, rect3.get_bbox().get_points()[1,0]+right_pad)
#
#
#
# ax4.legend(loc='lower left')
#
# fig.set_size_inches(7,6)
# fig.tight_layout()
# fig.savefig('phi4o3.pdf')
#
# primordial = PrimordialSolver(V, +1)
#
# #phi_is = numpy.linspace(4,8,20)
# #N_is = numpy.linspace(-3,5,20)
# #
# #nss = []
# #for phi_i in phi_is:
# #    nss.append([])
# #    for N_i in N_is:
# #        if phi_i < primordial.phimin(N_i):
# #            ns = numpy.nan
# #        else:
# #            try:
# #                ns = primordial.find_ns_As_r(N_i, phi_i, universe.logaH)[0]
# #            except:
# #                print("error")
# #                ns = numpy.nan
# #
# #        nss[-1].append(ns)
# #
# #nss = numpy.array(nss)
# #fig, ax = plt.subplots()
# #ax.contourf(phi_is, N_is, nss.T)
# #nss
