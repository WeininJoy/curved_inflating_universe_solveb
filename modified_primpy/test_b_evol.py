###### Goal: code for testing the evolution of variable b

import numpy as np
import matplotlib.pyplot as plt

from primpy.parameters import K_STAR
import primpy.potentials as pp
from primpy.events import UntilNEvent, InflationEvent, SlowRowEvent
from primpy.initialconditions import InflationStartIC, ISIC_Nt
from primpy.time.inflation import InflationEquationsT as InflationEquations
from primpy.solver import solve
from primpy.oscode_solver import solve_oscode

t_eval = np.logspace(5, 8, 2000)
K = 0            # flat universe
N_star = 50      # number of e-folds of inflation after horizon crossing
N_tot = 60       # total number of e-folds of inflation
N_end = 70       # end time/size after inflation, arbitrary in flat universe
delta_N_reh = 2  # extra e-folds after end of inflation to see reheating oscillations
A_s = 2e-9       # amplitude of primordial power spectrum at pivot scale
Pot = pp.StarobinskyPotential
w_value = -0.99

class SlowRowIC(object):
    """Deep slow row initial conditions given backgound."""
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

        SR_phi, SR_dphidt, SR_N = self.background.y_events[SR_key][0]
        SR_V = self.background.potential.V(SR_phi)
        SR_dNdt = np.sqrt((SR_dphidt**2 /2 + SR_V) /3 - self.background.K * np.exp(-2*SR_N))
        y0[self.equations.idx['phi']] = SR_phi
        y0[self.equations.idx['dphidt']] = SR_dphidt
        y0[self.equations.idx['N']] = SR_N
        y0[self.equations.idx['Nb']] = SR_N
        y0[self.equations.idx['dNbdt']] = SR_dNdt


Lambda, phi_star, N_star = Pot.sr_As2Lambda(A_s=A_s, N_star=N_star, phi_star=None) # crude slow-roll estimate
pot = Pot(Lambda=Lambda)
eq = InflationEquations(K=K, potential=pot, track_eta=False)
ev = [UntilNEvent(eq, value=N_end+delta_N_reh),  # decides stopping criterion
      SlowRowEvent(eq, +1, terminal=False, value=w_value),
      InflationEvent(eq, +1, terminal=False),    # records inflation start
      InflationEvent(eq, -1, terminal=False)]    # records inflation end

# from inflation start forwards in time, optimising to get `N_tot` e-folds of inflation
ic_fore = ISIC_Nt(equations=eq, N_tot=N_tot, N_i=N_end-N_tot, phi_i_bracket=[phi_star-3, phi_star+3], t_i=t_eval[0])
forewards = solve(ic=ic_fore, events=ev, t_eval=t_eval)
equations = InflationEquations(K=K, potential=pot, track_eta=False, track_b=True)
"""get initial condition of Nb and dNbdt"""
ic_SR = SlowRowIC(forewards, equations)
ev = [UntilNEvent(eq, value=0),  # decides stopping criterion
      InflationEvent(eq, +1, terminal=False),    # records inflation start
      InflationEvent(eq, -1, terminal=False)]    # records inflation end
backwards = solve(ic=ic_SR, events=ev)

# need to shift time, since we initially did not know the precise starting time of inflation
# backwards_t = (backwards.t - backwards.t.min())
# forewards_t = (forewards.t - backwards.t.min())
inf_start_key = 'Inflation_dir-1_term0'
inf_start_t = backwards.t_events[inf_start_key][0]
print('inf_start_t='+str(inf_start_t))


plt.semilogx(backwards.t, backwards.Nb, label='b_back',linestyle='dashed')
plt.semilogx(backwards.t, backwards.N, label='a_back')

plt.legend()
plt.show()