##### Goal: generate quantum IC of R (comoving curvature perturbation) by solving b

import numpy as np
import pyoscode
import primpy.potentials as pp
import matplotlib.pyplot as plt
from primpy.events import UntilNEvent, InflationEvent, SlowRowEvent, KineticDominanceEvent
from primpy.initialconditions import InflationStartIC, ISIC_NsOk
from primpy.time.inflation import InflationEquationsT as InflationEquations
from primpy.solver import solve


class SlowRowIC(object): # set IC in slow row
    """Deep slow row initial conditions given background."""
    def __init__(self, background, equations):
        self.background = background
        self.equations = equations

    def __call__(self, y0, **ivp_kwargs):
    
        # #########################################################################
        # Set background equations of inflation for `N`, `phi` and `dphi`.
        # #########################################################################
        SR_key = 'SlowRow_dir1_term0'
        self.x_ini = self.background.t_events[SR_key][0]
        self.x_end = self.background.t[0]
        

        SR_phi, SR_dphidt, SR_N, SR_eta = self.background.y_events[SR_key][0]
        SR_V = self.background.potential.V(SR_phi)
        SR_dNdt = np.sqrt((SR_dphidt**2 /2 + SR_V) /3 - self.background.K * np.exp(-2*SR_N))
        y0[self.equations.idx['phi']] = SR_phi
        y0[self.equations.idx['dphidt']] = SR_dphidt
        y0[self.equations.idx['N']] = SR_N
        # y0[self.equations.idx['Nb']] = SR_N
        # y0[self.equations.idx['dNbdt']] = SR_dNdt


class KD_IC(object): # set IC in Kinetic dominance
    """Deep kinetic dominance initial conditions given background."""
    def __init__(self, background, equations):
        self.background = background
        self.equations = equations

    def __call__(self, y0, **ivp_kwargs):
    
        # #########################################################################
        # Set background equations of inflation for `N`, `phi` and `dphi`.
        # #########################################################################
        KD_key = 'KineticDominance_dir1_term1'
        self.x_ini = self.background.t_events[KD_key][0]
        self.x_end = self.background.t[0]

        KD_phi, KD_dphidt, KD_N, KD_eta = self.background.y_events[KD_key][0]
        KD_V = self.background.potential.V(KD_phi)
        KD_dNdt = np.sqrt((KD_dphidt**2 /2 + KD_V) /3 - self.background.K * np.exp(-2*KD_N))
        y0[self.equations.idx['phi']] = KD_phi
        y0[self.equations.idx['dphidt']] = KD_dphidt
        y0[self.equations.idx['N']] = KD_N
        # y0[self.equations.idx['Nb']] = KD_N
        # y0[self.equations.idx['dNbdt']] = KD_dNdt


class IC_RST_diff_t(object):
    def __init__(self, background, w, k, cs=1):
        self.background = background  
        self.V = self.background.potential.V
        self.dV = self.background.potential.dV
        self.K = self.background.K
        self.cs = cs
        self.w = w
        self.k = k

        if -1./3.*(1.+1.e-2) < self.w < -1./3.*(1.-1.e-2): # set at the start of inflation
            self.newbackground = self.background
           
        elif -1. < self.w < -1./3.*(1.+1.e-2): # quantum IC is set in inflation era
            equations = InflationEquations(K=self.K, potential=self.background.potential, track_eta=False, track_b=False)
            ic_SR = SlowRowIC(self.background, equations)
            ev = [InflationEvent(equations, -1, terminal=True)]   # end at inflation start
            self.newbackground = solve(ic=ic_SR, events=ev)
            if self.newbackground.t[-1] < self.background.t[0]* (1.+1.e-2): # make sure get R_IC at the start of inflation
                pass
            else: 
                print("Cannot get R_IC at the start of inflation.")



        elif -1./3.*(1.-1.e-2) < self.w <= 1.: # quantum IC is set in kinetic dominance era
            # define backward background
            eq = InflationEquations(K=self.K, potential=self.background.potential, track_eta=True, verbose=False)
            ic = InflationStartIC(equations=eq, phi_i=self.background.phi[0], N_i=self.background.N[0], t_i=self.background.t[0], eta_i=0, x_end=-1.e10)
            ev = [KineticDominanceEvent(eq, +1, terminal=True, value=self.w)]
            background = solve(ic=ic, events=ev, dense_output=True)
            # integrate forward to start of inflation
            equations = InflationEquations(K=self.K, potential=background.potential, track_eta=False, track_b=False)
            ic_KD = KD_IC(background, equations)
            ev = [InflationEvent(equations, +1, terminal=True)]   # end at inflation start
            self.newbackground = solve(ic=ic_KD, events=ev)
            if self.newbackground.t[-1] > self.background.t[0]* (1.-1.e-2): # make sure get R_IC at the start of inflation
                pass
            else: 
                print("Cannot get R_IC at the start of inflation.")
                
    
    
    def get_vacuum_ic_RST(self):
        """Get initial conditions for scalar modes for RST vacuum w.r.t. cosmic time `t`."""
        a_i = np.exp(self.newbackground.N[0])
        dphidt_i = self.newbackground.dphidt[0]
        H_i = self.newbackground.H[0]
        z_i = a_i * dphidt_i / H_i
        Rk_i = 1 / np.sqrt(2 * self.k) / z_i
        dRk_i = -1j * self.k / a_i * Rk_i
        return Rk_i, dRk_i    
        

    def mukhanov_sasaki_frequency_damping(self):
        """Frequency and damping term of the Mukhanov-Sasaki equations for scalar modes.

        Frequency and damping term of the Mukhanov-Sasaki equations for the
        comoving curvature perturbations `R` w.r.t. time `t`, where the e.o.m. is
        written as `ddR + 2 * damping * dR + frequency**2 R = 0`.

        """
        K = self.newbackground.K
        N = self.newbackground.N[:]
        dphidt = self.newbackground.dphidt[:]
        H = self.newbackground.H[:]
        dV = self.newbackground.potential.dV(self.newbackground.phi[:])

        kappa2 = self.k**2 + self.k * K * (K + 1) - 3 * K
        shared = 2 * kappa2 / (kappa2 + K * dphidt**2 / (2 * H**2))
        terms = dphidt**2 / (2 * H**2) - 3 - dV / (H * dphidt) - K * np.exp(-2 * N) / H**2

        frequency2 = kappa2 * np.exp(-2 * N) - K * np.exp(-2 * N) * (1 + shared * terms)
        damping = (3 * H + shared * terms * H) / 2
        if np.all(frequency2 > 0):
            return [np.sqrt(frequency2), damping]
        else:
            return [np.sqrt(frequency2 + 0j), damping]


    def get_R_IC(self):

        if -1./3.*(1.+1.e-2) < self.w < -1./3.*(1.-1.e-2):# set at the start of inflation
            R_IC, R_dot_IC = self.get_vacuum_ic_RST()

        else:
            y0 = self.get_vacuum_ic_RST()
            rtol = 5e-5
            b = self.newbackground
            frequency, damping = self.mukhanov_sasaki_frequency_damping()
            oscode_sol = pyoscode.solve(ts=b.x[:], ti=b.x[0], tf=b.x[-1],
                                            ws=np.log(frequency), logw=True,
                                            gs=damping, logg=False,
                                            x0=y0[0],
                                            dx0=y0[1],
                                            rtol=rtol, even_grid=False)
            R_IC, R_dot_IC = oscode_sol['sol'][-1], oscode_sol['dsol'][-1]

        return [R_IC, R_dot_IC]
    
    def plot_R(self):
        y0 = self.get_vacuum_ic_RST()
        rtol = 5e-5
        b = self.newbackground
        frequency, damping = self.mukhanov_sasaki_frequency_damping()
        oscode_sol = pyoscode.solve(ts=b.x, ti=b.x[0], tf=b.x[-1],
                                        ws=np.log(frequency), logw=True,
                                        gs=damping, logg=False,
                                        x0=y0[0],
                                        dx0=y0[1],
                                        rtol=rtol, even_grid=False)
        t, R, R_dot = np.asarray(oscode_sol['t']), np.asarray(oscode_sol['sol']), np.asarray(oscode_sol['dsol'])
        plt.plot(t, R, label='R(t)')
        # plt.plot(t, R_dot, label = 'R_dot(t)')
        plt.legend()
        plt.show()