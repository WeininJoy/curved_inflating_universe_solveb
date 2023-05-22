######## Goal: generate PPS by primpy
######## Origin: this code is written by Lukas Hergt, which can be downloaded from Finite_inflation_in_curved_space: https://zenodo.org/record/6547872#.Y-UpNBzP3HA
########         In cb3-0_pc1-20_cl2-9/stb_omegak/pcs3d500_cl_hf_p18_TTTEEElite_lowl_lowE_BK15_stb_omegak_AsfoH_perm

import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'
os.environ['VECLIB_MAXIMUM_THREADS'] = '2'
import sys
from warnings import catch_warnings
from warnings import filterwarnings
from scipy.integrate import IntegrationWarning
import numpy as np
from primpy.exceptionhandling import PrimpyWarning
from primpy.exceptionhandling import PrimpyError, StepSizeError
from primpy.oscode_solver import solve_oscode
from primpy.solver import solve
from primpy.potentials import StarobinskyPotential as Pot
from primpy.events import InflationEvent, SlowRowEvent, KineticDominanceEvent, CollapseEvent, UntilNEvent
from primpy.initialconditions import InflationStartIC, ISIC_NsOk
from primpy.time.inflation import InflationEquationsT
import primpy.bigbang as bb
import matplotlib.pyplot as plt

# N_star_list = np.linspace(50, 60, num=5)
# w_value_list = np.linspace(-0.4, -0.99, num=5)
w_value_list = np.linspace(-0.3, 0.99, num=5)
# logA_SR_list = np.linspace(2.5, 3.7, num=5)
# H0_list = np.linspace(67, 74, num=5)
# omega_k_list = np.linspace(-0.04, -0.001, num=5)
# log10f_i_list = np.linspace(-1, 5, num=5)

for j in range(len(w_value_list)):
    try:
        w_value = w_value_list[j]
        logA_SR   = np.log(20)
        N_star    = 55
        log10f_i  = 5
        omega_k   = -0.005
        H0        = 70 
        # logA_SR   = float(sys.argv[1])
        # N_star    = float(sys.argv[2])
        # log10f_i  = float(sys.argv[3])
        # omega_k   = float(sys.argv[4])
        # H0        = float(sys.argv[11])
    except IndexError:
        raise IndexError("It seems you are calling this script with too few arguments.")
    except ValueError:
        raise ValueError("It seems some of the arguments are not correctly formatted. " +
                        "Remember that they must be floating point numbers.")


    with catch_warnings():
        filterwarnings('ignore', category=RuntimeWarning)
        filterwarnings('ignore', category=IntegrationWarning)
        filterwarnings('ignore', category=PrimpyWarning)

        h = H0 / 100
        Omega_K0 = omega_k / h**2
        A_s = np.exp(logA_SR) * 1e-10
        f_i = 10**log10f_i
        Omega_Ki = f_i * Omega_K0
        K = -np.sign(Omega_K0)
        if Omega_Ki >= 1 or Omega_K0 == 0:
            raise PrimpyError("Primordial curvature for open universes has to be Omega_Ki < 1, "
                            "but Omega_Ki = %g was requested." % Omega_Ki, geometry="open")

        msg = ("\n\t Failed to compute PPS for parameters: "
            "\n\t A_s=%.18e, N_star=%.18e, f_i=%.18e, Omega_K0=%.18e, h=%.18e"
            "\n\t reason: " % (A_s, N_star, f_i, Omega_K0, h))

        t_i = 7e4
        N_BBN = bb.get_N_BBN(h=h, Omega_K0=Omega_K0)
        N_max = N_BBN

        # #########################################################################
        # Potential
        # #########################################################################
        Lambda, phi_star, _ = Pot.sr_As2Lambda(A_s=A_s, phi_star=None, N_star=N_star)
        pot = Pot(Lambda=Lambda)

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


        # #########################################################################
        # Primordial Power Spectrum
        # #########################################################################
        k_iMpc = np.concatenate((
            np.logspace(np.log10(5e-7), np.log10(5e-5), 2 * 100 + 1),
            np.logspace(np.log10(5e-5), np.log10(5e-1), 4 * 200 + 1)[1:],
            np.logspace(np.log10(5e-1), np.log10(5e0), 1 * 50 + 1)[1:],
            np.logspace(np.log10(5e0), np.log10(5e1), 1 * 10 + 1)[1:]
        ))
        k = k_iMpc * b.a0_Mpc
        if K == +1:
            k = k[k >= 1]

        pps = solve_oscode(background=b, k=k, w=w_value)
        


        # 3. Limits for k and precision:
        #    Check that the boundaries are correct for your case.
        #    It is safer to set k_per_decade primordial slightly bigger than that of Class.

        # k_min  = 5e-7
        # k_max  = 5e1
        # k_per_decade_primordial = 200
        # ks = np.logspace(np.log10(k_min), np.log10(k_max), (7 + 1) * k_per_decade_primordial + 1)

        #
        # And nothing should need to be edited from here on.
        #

        """
        # Filling the array of Pk's
        for i, k in enumerate(pps.k_iMpc):
            if np.isfinite(pps.P_s_RST[i]) and np.isfinite(pps.P_t_RST[i]):
                print("%.18e %.18e %.18e" % (k, pps.P_s_RST[i], pps.P_t_RST[i]))
        """
        # plt.loglog(pps.k_iMpc, pps.P_s_RST, label='w='+str(w_value_list[j]))
        # np.savetxt('./PPS_primpy/PPS_test_w='+str(j)+'.dat', np.array([pps.k_iMpc, pps.P_s_RST, pps.P_t_RST]).T)

# plt.xlim([2.e-5, 5.])
# plt.xlabel('k/Mpc^-1')
# plt.ylabel('PPS')
plt.legend()
plt.show()