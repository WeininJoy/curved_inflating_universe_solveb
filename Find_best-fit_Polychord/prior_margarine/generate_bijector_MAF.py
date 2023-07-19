import numpy as np

#######################################
# load data from chains
#######################################

from anesthetic import read_chains

samples = read_chains("../Lukas_Likelihood_prior/chains/inflation")
sample_prior = samples.prior()
names = ['logA_SR', 'N_star', 'log10f_i', 'omega_k', 'H0']


#######################################
# Prior constraints
#######################################

from primpy.exceptionhandling import PrimpyError, StepSizeError
from primpy.solver import solve
from primpy.potentials import StarobinskyPotential as Pot
from primpy.events import InflationEvent, CollapseEvent, UntilNEvent
from primpy.initialconditions import ISIC_NsOk, InflationStartIC
from primpy.time.inflation import InflationEquationsT
from primpy.efolds.inflation import InflationEquationsN
import primpy.bigbang as bb

def prior_constraint(params):
    

    logA_SR   = params[0] 
    N_star    = params[1]
    log10f_i  = params[2]
    omega_k   = params[3]
    H0        = params[4]

    logzero = -np.inf
    logone = 0

    h = H0 / 100
    Omega_K0 = omega_k / h**2
    omega_b, omega_cdm = 2.2632e-02, 1.1792e-01
    Omega_m0 = omega_b / h**2 + omega_cdm / h**2
    A_s = np.exp(logA_SR) * 1e-10
    f_i = 10**log10f_i
    Omega_Ki = f_i * Omega_K0
    K = -np.sign(Omega_K0)
    if Omega_Ki >= 1 or Omega_K0 == 0:
        return logzero

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
                return logzero
        except ValueError as verr:
            if verr.args[0] == 'f(a) and f(b) must have different signs' and phi_add < 20:
                phi_add *= 2
            else:
                print(msg)
                raise ValueError(verr)
        except Exception as err:
            print("A_s=%.18e, N_star=%.18g, f_i=%.18e, Omega_K0=%.18g, h=%.18g" % (A_s, N_star, f_i, Omega_K0, h))
            raise Exception(err)
    if b.N_events['Collapse'].size > 0 or b.N_tot == 0 or b.N[-1] - ic.N_i < 5:
        return logzero  # universe collapsed
    elif b.N_tot < 20:
        return logzero  # insufficient inflation
    elif b.N[-1] >= N_max or b.N_end >= N_max or b.N_events['UntilN'].size > 0:
        return logzero  # excess inflation
    elif not np.isfinite(b.N_tot):
        return logzero  # unknown problem

    try:
        b.derive_comoving_hubble_horizon(Omega_K0=Omega_K0, h=h)
        b.derive_approx_power(Omega_K0=Omega_K0, h=h)
    except:
        return logzero  # unknown problem

    # #########################################################################
    # Reheating
    # #########################################################################
    cHH_BBN_lp = bb.comoving_Hubble_horizon(N=N_BBN, Omega_m0=Omega_m0, Omega_K0=Omega_K0, h=h,
                                            units='planck')
    w_reh_BBN = bb.get_w_reh(N1=b.N_end, N2=N_BBN,
                             log_cHH1=b.log_cHH_end_lp, log_cHH2=np.log(cHH_BBN_lp))
    if not (-1/3 <= w_reh_BBN <= 1):
        return logzero
        
    # #########################################################################
    # Horizon Problem
    # #########################################################################
    eqn = InflationEquationsN(K=K, potential=pot, track_eta=True, verbose=False)
    icbn = InflationStartIC(equations=eqn, phi_i=ic.phi_i, Omega_Ki=ic.Omega_Ki, x_end=0, eta_i=0)
    bisn = solve(ic=icbn)
    bisn.derive_comoving_hubble_horizon(Omega_K0=Omega_K0, h=h)
    if not np.allclose(bisn.eta[-5:], bisn.eta[-1], rtol=1e-2):
        raise PrimpyError("conformal time problem: eta[-5:]=%s" % bisn.eta[-5:])
    ratio = bb.conformal_time_ratio(Omega_m0=Omega_m0, Omega_K0=Omega_K0, h=h,
                                    b_forward=b, b_backward=bisn)
    if not ratio >= 1:
        return logzero
    
    return logone


import random

sample_prior = sample_prior.head(10)
prior_constraint_list = [] 
for i in range(sample_prior.shape[0]):
    params = [sample_prior[name][i] for name in names]
    prior_constraint_list.append(prior_constraint(params))

sample_prior['prior_constraint'] = prior_constraint_list
sample_prior_constraint = sample_prior[sample_prior.prior_constraint == 0]

#######################################
# Masked Autoregressive Flows
#######################################

from margarine.maf import MAF

weights = sample_prior_constraint.get_weights()
theta = sample_prior_constraint[names].values
bij = MAF(theta, weights)
bij.train(10, early_stop=True)
filename = 'MAF_bijector.pkl'
bij.save(filename)

bij_load = MAF.load(filename)
x = bij_load.sample(10)
# x = bij_load(np.random.uniform(0, 1, size=(len(theta), theta.shape[-1])))
print(x)