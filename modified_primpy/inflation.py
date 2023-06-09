###### Goal: modify primpy/time/inflation.py by adding new variable b

#!/usr/bin/env python
""":mod:`primpy.time.inflation`: differential equations for inflation w.r.t. time `t`."""
import numpy as np
from primpy.inflation import InflationEquations


class InflationEquationsT(InflationEquations):
    """Background equations during inflation w.r.t. time `t`.

    Solves background variables in cosmic time for curved and flat universes
    using the Klein-Gordon and Friedmann equations.

    Independent variable:
        t: cosmic time

    Dependent variables:
        N: number of e-folds
        phi: inflaton field
        dphidt: `d(phi)/dt`

    """

    def __init__(self, K, potential, track_eta=False, track_b=False, verbose=False):
        super(InflationEquationsT, self).__init__(K=K, potential=potential, verbose=verbose)
        self._set_independent_variable('t')
        self.add_variable('phi', 'dphidt', 'N')
        self.track_eta = track_eta
        self.track_b = track_b
        if track_eta:
            self.add_variable('eta')
        if track_b:
            self.add_variable('Nb', 'dNbdt')

    def __call__(self, x, y):
        """System of coupled ODEs for underlying variables."""
        N = self.N(x, y)
        H = self.H(x, y)
        dphidt = self.dphidt(x, y)
        dVdphi = self.dVdphi(x, y)

        dy = np.zeros_like(y)
        dy[self.idx['phi']] = dphidt
        dy[self.idx['dphidt']] = -3 * H * dphidt - dVdphi
        dy[self.idx['N']] = H
        
        if self.track_eta:
            dy[self.idx['eta']] = np.exp(-N)
        if self.track_b:
            d2Ndt = -1.0/2.0 * dphidt**2 + self.K* np.exp(-2*N)
            dNbdt= self.dNbdt(x, y)
            dy[self.idx['Nb']] = dNbdt
            dy[self.idx['dNbdt']] = dNbdt**2 + d2Ndt - H**2 - self.K*np.exp(-2*N)
        return dy

    def H2(self, x, y):
        """Compute the square of the Hubble parameter using the Friedmann equation."""
        N = self.N(x, y)
        V = self.V(x, y)
        dphidt = self.dphidt(x, y)
        return (dphidt**2 / 2 + V) / 3 - self.K * np.exp(-2 * N)
    
    def Horizon(self, x, y):
        """Compute the Hubble Horizon."""
        H = self.H(x, y)
        N = self.N(x, y)
        return 1./(np.exp(N)*H)

    def w(self, x, y):
        """Compute the equation of state parameter."""
        V = self.V(x, y)
        dphidt = self.dphidt(x, y)
        p = dphidt**2 / 2 - V
        rho = dphidt**2 / 2 + V
        return p / rho

    def inflating(self, x, y):
        """Inflation diagnostic for event tracking."""
        return self.V(x, y) - self.dphidt(x, y)**2

    def sol(self, sol, **kwargs):
        """Post-processing of `solve_ivp` solution."""
        sol = super(InflationEquationsT, self).sol(sol, **kwargs)
        return sol
