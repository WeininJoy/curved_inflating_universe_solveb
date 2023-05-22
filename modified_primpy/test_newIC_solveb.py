###### Goal: code for testing the new quantum IC based on b

import numpy as np
import matplotlib.pyplot as plt
from primpy.events import InflationEvent, SlowRowEvent, CollapseEvent, UntilNEvent
from primpy.initialconditions import InflationStartIC, ISIC_NsOk
from primpy.potentials import StarobinskyPotential as Pot
from primpy.exceptionhandling import PrimpyError, StepSizeError
from primpy.time.inflation import InflationEquationsT
import primpy.bigbang as bb
from primpy.solver import solve
from primpy.time.ic_rst_b import IC_RST_b


logA_SR   = np.log(20)
N_star    = 55
log10f_i  = 5
omega_k   = -0.005
H0        = 70


h = H0 / 100
Omega_K0 = omega_k / h**2
A_s = np.exp(logA_SR) * 1e-10
f_i = 10**log10f_i
Omega_Ki = f_i * Omega_K0
K = -np.sign(Omega_K0)

msg = ("\n\t Failed to compute PPS for parameters: "
        "\n\t A_s=%.18e, N_star=%.18e, f_i=%.18e, Omega_K0=%.18e, h=%.18e"
        "\n\t reason: " % (A_s, N_star, f_i, Omega_K0, h))

t_i = 7e4
N_BBN = bb.get_N_BBN(h=h, Omega_K0=Omega_K0)
N_max = N_BBN
print('N_max='+str(N_max))

# #########################################################################
# Potential
# #########################################################################
Lambda, phi_star, _ = Pot.sr_As2Lambda(A_s=A_s, phi_star=None, N_star=N_star)
pot = Pot(Lambda=Lambda)


# w_value_list = np.linspace(-0.4, -0.99, num=5)
w_value_list = [-0.99]

for w_value in w_value_list:

    # #########################################################################
    # Background setup
    # #########################################################################
    eq = InflationEquationsT(K=K, potential=pot, track_eta=True, verbose=False)
    ev = [InflationEvent(eq, +1, terminal=False),
            InflationEvent(eq, -1, terminal=True), 
            SlowRowEvent(eq, +1, terminal=False, value=w_value),
            UntilNEvent(eq, N_max),
            CollapseEvent(eq)]

    # #########################################################################
    # Background integration
    # #########################################################################
    phi_add = 1
    for i in range(100):
        ic = ISIC_NsOk(eq, Omega_Ki=Omega_Ki, N_star=N_star, Omega_K0=Omega_K0, h=h, phi_i_bracket=[phi_star+i*1e-12, 9+phi_add], t_i=t_i, eta_i=0)
        try:
            b = solve(ic=ic, events=ev, dense_output=True)
            break
        except StepSizeError:
            if i > 10:
                raise StepSizeError(msg)
        except ValueError as verr:
            if verr.args[0] == 'f(a) and f(b) must have different signs' and phi_add < 20:
                phi_add *= 2
            else:
                print(msg)
                raise ValueError(verr)
        except Exception as err:
            print(msg)
            raise Exception(err)
    if b.N_events['Collapse'].size > 0 or b.N_tot == 0 or b.N[-1] - ic.N_i < 5:
        raise PrimpyError(msg + "collapsed, N_tot=%g, N[-1]=%g \n" % (b.N_tot, b.N[-1]))
    elif b.N_tot < 20:
        raise PrimpyError(msg + "insufficient inflation, N_tot=%g \n" % b.N_tot)
    elif b.N[-1] >= N_max or b.N_end >= N_max or b.N_events['UntilN'].size > 0:
        raise PrimpyError(msg + "excess inflation, N_tot=%g, N[-1]=%g \n" % (b.N_tot, b.N[-1]))
    elif not np.isfinite(b.N_tot):
        raise PrimpyError(msg + "unknown problem, N_tot=%g, N[-1]=%g \n" % (b.N_tot, b.N[-1]))

    try:
        b.derive_comoving_hubble_horizon(Omega_K0=Omega_K0, h=h)
        b.derive_approx_power(Omega_K0=Omega_K0, h=h)
    except:
        raise PrimpyError(msg + "unknown problem, N_tot=%g, N[-1]=%g \n" % (b.N_tot, b.N[-1]))


    IC_b = IC_RST_b(b, w_value=w_value)
    IC_b.test_b_IC()

plt.legend()
plt.show()