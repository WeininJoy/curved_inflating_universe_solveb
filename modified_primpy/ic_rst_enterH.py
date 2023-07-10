##### Goal: generate quantum IC of R (comoving curvature perturbation) by solving b

import numpy as np
import pyoscode
import primpy.potentials as pp
import matplotlib.pyplot as plt
from primpy.events import InflationEvent, TouchHorizonEvent
from primpy.initialconditions import InflationStartIC, ISIC_NsOk
from primpy.time.inflation import InflationEquationsT as InflationEquations
from primpy.solver import solve


class Horizon_IC(object): # set IC when the scale enter Hubble Horizon
    """Deep kinetic dominance initial conditions given background."""
    def __init__(self, background, equations):
        self.background = background
        self.equations = equations

    def __call__(self, y0, **ivp_kwargs):
    
        # #########################################################################
        # Set background equations of inflation for `N`, `phi` and `dphi`.
        # #########################################################################
        try:
            H_key = 'TouchHorizon_dir1_term1'
            self.x_ini = self.background.t_events[H_key][0]
            self.x_end = self.background.t[0]

            H_phi, H_dphidt, H_N, H_eta = self.background.y_events[H_key][0]
            y0[self.equations.idx['phi']] = H_phi
            y0[self.equations.idx['dphidt']] = H_dphidt
            y0[self.equations.idx['N']] = H_N
        except: 
            # print('1/k does not touch the Hubble Horizon. Set IC at the start of inflation')
            self.x_ini = self.background.t[0]
            self.x_end = self.background.t[0]

            H_phi, H_dphidt, H_N = self.background.phi[0], self.background.dphidt[0], self.background.N[0]
            y0[self.equations.idx['phi']] = H_phi
            y0[self.equations.idx['dphidt']] = H_dphidt
            y0[self.equations.idx['N']] = H_N


class IC_RST_enterH(object):
    def __init__(self, background, k, cs=1):
        self.background = background  
        self.V = self.background.potential.V
        self.dV = self.background.potential.dV
        self.K = self.background.K
        self.cs = cs
        self.k = k
        
        # define backward background
        eq = InflationEquations(K=self.K, potential=self.background.potential, track_eta=True, verbose=False)
        ic = InflationStartIC(equations=eq, phi_i=self.background.phi[0], N_i=self.background.N[0], t_i=self.background.t[0], eta_i=self.background.eta[0], x_end=-1.e10)
        ev = [TouchHorizonEvent(eq, +1, terminal=True, value=1./self.k)]
        background = solve(ic=ic, events=ev)
        # integrate forward to start of inflation
        equations = InflationEquations(K=self.K, potential=background.potential, track_eta=False, track_b=False)
        ic_horizon = Horizon_IC(background, equations)
        ev = [InflationEvent(equations, +1, terminal=True)]   # end at inflation start
        self.newbackground = solve(ic=ic_horizon, events=ev)
        if self.newbackground.t[-1] > self.background.t[0]* (1.-1.e-2): # make sure get R_IC at the start of inflation
            pass
        else: 
            print("Cannot get R_IC at the start of inflation.")
    
    # #########################################################################            
    # Defining IC of R and dRdt for flat RSET
    # #########################################################################
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