import os
# os.environ['OMP_NUM_THREADS'] = '8'
# os.environ['MKL_NUM_THREADS'] = '8'
# os.environ['OPENBLAS_NUM_THREADS'] = '8'
# os.environ['NUMEXPR_NUM_THREADS'] = '8'
# os.environ['VECLIB_MAXIMUM_THREADS'] = '8'
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
from primpy.events import InflationEvent, CollapseEvent, UntilNEvent
from primpy.initialconditions import ISIC_NsOk, InflationStartIC
from primpy.time.inflation import InflationEquationsT
import primpy.bigbang as bb
from scipy.optimize import minimize
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/wnd22/rds/hpc-work/env_/lib/python3.8/site-packages/class_public')
sys.path.insert(0, '/home/wnd22/rds/hpc-work/env_/lib/python3.8/site-packages')
import classy
from classy import CosmoComputationError
print(classy.__file__)

plc = os.path.join(os.getcwd(),'/home/wnd22/rds/hpc-work/clik_installs/code/planck/code/plc_3.0/plc-3.1')
data = os.path.join(os.getcwd(),'/home/wnd22/rds/hpc-work/clik_installs/data/planck_2018/baseline/plc_3.0') 

#sys.path.insert(plc + 'include/')
sys.path.append(os.path.join(plc, 'lib/python/site-packages')) 
sys.path.append(os.path.join(plc, 'lib/python/site-packages')) 
print(sys.path)
import clik

from numpy import pi, log
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior
try:
    from mpi4py import MPI
except ImportError:
    pass


c = 2.99792458 * 10 ** 5  # Speed of light in km/s
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if not os.path.exists('PPS'):
    os.makedirs('PPS')
dat_name = './PPS/PPS_rank='+str(rank)+'.dat' # rank of the MPI process 
############


class PlanckLikelihood(object):
    """Baseline Planck Likelihood"""
    
    def __init__(self): # don't consider lensing
        self.plik = clik.clik(os.path.join(data, "hi_l/plik_lite/plik_lite_v22_TTTEEE.clik"))
        self.lowl = clik.clik(os.path.join(data, "low_l/commander/commander_dx12_v3_2_29.clik"))
        self.lowE = clik.clik(os.path.join(data, "low_l/simall/simall_100x143_offlike5_EE_Aplanck_B.clik"))
        
    
    def __call__(self, cls, nuis):
        lkl = []
        for like in [self.plik, self.lowl, self.lowE]:
            #        for like in [self.plik]:
            lmaxes = like.get_lmax()
            dat = []
            order = ['tt','ee','bb','te','tb','eb']
            
            # print(order,len(lmaxes),len(order))
            for spec, lmax in zip(order, lmaxes):
                if lmax>-1:
                    if spec == 'pp':
                        dat += list(cls[spec][:lmax+1])
                    else:
                        dat += list(cls[spec][:lmax+1]* (1e6 * 2.7255)**2 )
        
            for param in like.get_extra_parameter_names():
                dat.append(nuis[param])
    
            lkl.append(like(dat))
    
        return np.array(lkl).flatten()

lkl = PlanckLikelihood()

# calculate and output PPS.dat file with parameters = [logA_SR, N_star, log10f_i, omega_k, H0]
def PPS_dat(params):
    try:
        logA_SR   = params[0] 
        N_star    = params[1]
        log10f_i  = params[2]
        omega_k   = params[3]
        H0        = params[4]

    except IndexError:
        raise IndexError("It seems you are calling this script with too few arguments.")
    except ValueError:
        raise ValueError("It seems some of the arguments are not correctly formatted. " +
                        "Remember that they must be floating point numbers.")


    with catch_warnings():
        #filterwarnings('ignore', category=RuntimeWarning)
        #filterwarnings('ignore', category=IntegrationWarning)
        #filterwarnings('ignore', category=PrimpyWarning)
        
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

        pps = solve_oscode(background=b, k=k)
        np.savetxt(dat_name, np.array([pps.k_iMpc, pps.P_s_RST]).T)


# Runs the CLASS code
def run_class(params):
    
    # calulate PPS and make dat file
    PPS_dat(params)
   
    # parameters
    omega_k   = params[3]
    H0        = params[4]
    h = H0/100
    
    ####### solve_b cls

    params_pstb = {
        # 'omega_b': 2.25521925e-02,
        'omega_b': 2.2632e-02,
        # 'omega_cdm': 1.18636571e-01,
        'omega_cdm': 1.1792e-01,
        'h': h,
        # 'tau_reio': 5.09446854e-02,
        'tau_reio': 4.95e-02,
        # 'Omega_k': -1.15169968e-02 / h**2,
        'Omega_k':  omega_k/ h**2,
        # # 'custom1': 3.05219166,
        # 'custom1': 3.0522,
        # # 'custom2': 5.75507947e+01,
        # 'custom2': 5.755e+01,
        # # 'custom3': 1.28491857e-01,
        # 'custom3': 1.28e-01,
        # # 'custom4': -1.15169968e-02,
        # 'custom4': -1.15e-02,
        'N_ncdm': 1,
        'N_ur': 2.0328,
        'lensing': 'yes',
        'non linear': 'halofit',
        # 'modes': 's t',
        'output': 'tCl,pCl,lCl,mPk',
        'P_k_ini type': 'external_Pk',
        'P_k_max_1/Mpc':3.0,
        'l_max_scalars':2508,
        'command': 'cat ' + dat_name
    }

    pstb = classy.Class()
    pstb.set(params_pstb)
    pstb.compute()
    cls = pstb.lensed_cl(2508)
    pstb.struct_cleanup()
    pstb.empty()

    return cls


# Checks the input values are within the priors
def check_prior(params):

    prior = np.array([[2.5, 3.7],      #logA_SR
                      [20, 90],          #N_star
                      [-1, 5],        #log10f_i
                      [-0.035, -1.e-5],  #omega_k
                      [20, 100]])         #H0

    diff_below = params - prior[:, 0]
    diff_above = prior[:, 1] - params
      
    if min(diff_below) < 0:
        result = False
    elif min(diff_above) < 0:
        result = False
    else:
        result = True
                      
    return result

#################
# set constraints on prior
#################

# conformal time before/after inflation end should be larger than 1 -> enough inflation
def conformal_time_ratio(params):
    try:
        logA_SR   = params[0] 
        N_star    = params[1]
        log10f_i  = params[2]
        omega_k   = params[3]
        H0        = params[4]

    except IndexError:
        raise IndexError("It seems you are calling this script with too few arguments.")
    except ValueError:
        raise ValueError("It seems some of the arguments are not correctly formatted. " +
                        "Remember that they must be floating point numbers.")

    with catch_warnings():
        #filterwarnings('ignore', category=RuntimeWarning)
        #filterwarnings('ignore', category=IntegrationWarning)
        #filterwarnings('ignore', category=PrimpyWarning)
        
        Omega_m0 = 0.3166  # for calculating conformal_time_after
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
        eta_i =  0.0

        # #########################################################################
        # Potential
        # #########################################################################
        Lambda, phi_star, _ = Pot.sr_As2Lambda(A_s=A_s, phi_star=None, N_star=N_star)
        pot = Pot(Lambda=Lambda)

    assert np.sign(Omega_Ki) == np.sign(Omega_K0)
    eq = InflationEquationsT(K=K, potential=pot, track_eta=True, verbose=False)
    phi_add = 1
    for i in range(100):
        icf = ISIC_NsOk(equations=eq, Omega_Ki=Omega_Ki, N_star=N_star, Omega_K0=Omega_K0, h=h, phi_i_bracket=[phi_star+i*1e-12, 9+phi_add], t_i=t_i, eta_i=eta_i)
        evf = [InflationEvent(eq, +1, terminal=False),
                InflationEvent(eq, -1, terminal=True),
                CollapseEvent(eq)]
        icb = InflationStartIC(equations=eq, Omega_Ki=icf.Omega_Ki, phi_i=icf.phi_i, t_i=t_i, eta_i=eta_i, x_end=1)
        evb = [UntilNEvent(eq, value=0, terminal=True)]
        try: 
            bistf = solve(ic=icf, events=evf)
            bistb = solve(ic=icb, events=evb)
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
    try:
        bistf.derive_comoving_hubble_horizon(Omega_K0=Omega_K0, h=h)
        conformal_time_before = bistf.eta[-1] - bistb.eta[-1]
        conformal_time_after = bb.conformal_time(N_start=bistf.N[-1], N=np.log(bistf.a0_lp), Omega_m0=Omega_m0, Omega_K0=Omega_K0, h=h)[0]
    except:
        raise PrimpyError(msg)

    return  conformal_time_before / conformal_time_after

# Global variable for best chi^2
best_chisq = 2e+30 

# Computes TT spectrum and returns chi^2 for given set of parameters params in form  [logA_SR, N_star, log10f_i, omega_k, H0] using linear quantisation
def run_TT(params):
    
    if check_prior(params) == True:
        
        try:
            # Find corresponding spectra
            cls = run_class(params)
            
            # calculate conformal_time_ratio
            try:
                eta_ratio = conformal_time_ratio(params)
                print('eta_ratio='+str(eta_ratio))
            except ValueError as err:
                eta_ratio = np.nan
                print(err)

            if np.isnan([val for val in cls.values()]).any(): # if there is any Nan in cls -> raise CosmoComputationError
                raise CosmoComputationError
            
            elif eta_ratio < 1: # make sure there is enough inflation
                raise CosmoComputationError

            elif params[1] > 70: # N_star should be < 70
                raise CosmoComputationError

            else:
                nuisance_params = [1.00044, 46.1, 0.66, 7.08, 248.2, 50.7, 53.3, 121.9, 0., 8.80, 11.01, 20.16, 95.5, 0.1138, 0.1346, 0.479, 0.225, 0.665, 2.082, 0.99974, 0.99819]

                nuis={
                    'ycal':nuisance_params[0],
                    'A_cib_217':nuisance_params[1],
                    'xi_sz_cib':nuisance_params[2],
                    'A_sz':nuisance_params[3],
                    'ps_A_100_100':nuisance_params[4],
                    'ps_A_143_143':nuisance_params[5],
                    'ps_A_143_217':nuisance_params[6],
                    'ps_A_217_217':nuisance_params[7],
                    'ksz_norm':nuisance_params[8],
                    'gal545_A_100':nuisance_params[9],
                    'gal545_A_143':nuisance_params[10],
                    'gal545_A_143_217':nuisance_params[11],
                    'gal545_A_217':nuisance_params[12],
                    'galf_TE_A_100':nuisance_params[13],
                    'galf_TE_A_100_143':nuisance_params[14],
                    'galf_TE_A_100_217':nuisance_params[15],
                    'galf_TE_A_143':nuisance_params[16],
                    'galf_TE_A_143_217':nuisance_params[17],
                    'galf_TE_A_217':nuisance_params[18],
                    'calib_100T':nuisance_params[19],
                    'calib_217T':nuisance_params[20],
                    'cib_index':-1.3, #no range given in table so assume fixed
                    #-------------------------------------------------------------------
                    # These are all set to 1, so assume that these are fixed -----------
                    'A_cnoise_e2e_100_100_EE':1.,
                    'A_cnoise_e2e_143_143_EE':1.,
                    'A_cnoise_e2e_217_217_EE':1.,
                    'A_sbpx_100_100_TT':1.,
                    'A_sbpx_143_143_TT':1.,
                    'A_sbpx_143_217_TT':1.,
                    'A_sbpx_217_217_TT':1.,
                    'A_sbpx_100_100_EE':1.,
                    'A_sbpx_100_143_EE':1.,
                    'A_sbpx_100_217_EE':1.,
                    'A_sbpx_143_143_EE':1.,
                    'A_sbpx_143_217_EE':1.,
                    'A_sbpx_217_217_EE':1.,
                    'A_pol':1,
                    'A_planck':1.,
                    #-------------------------------------------------------------------
                    # These are fixed from Planck 2018 Likelihood Paper, Table 16 ------
                    'galf_EE_A_100':0.055,
                    'galf_EE_A_100_143':0.040,
                    'galf_EE_A_100_217':0.094,
                    'galf_EE_A_143':0.086,
                    'galf_EE_A_143_217':0.21,
                    'galf_EE_A_217':0.70,
                    'calib_100P':1.021,
                    'calib_143P':0.966,
                    'calib_217P':1.04,
                    #-------------------------------------------------------------------
                    # These are fixed from Planck 2018 Likelihood Paper, pg 39 ---------
                    'galf_EE_index':-2.4,
                    'galf_TE_index':-2.4,
                #-------------------------------------------------------------------
                }

                plik, lowl, lowE = -2 * lkl(cls, nuis)
                chi_eff_sq = plik + lowl + lowE 

        except CosmoComputationError:
            print('CosmoComputationError')
            plik = 2e+30
            lowl = 2e+30
            lowE = 2e+30
            chi_eff_sq = 2e+30
        except clik.lkl.CError:
            print('CError')
            plik = 2e+30
            lowl = 2e+30
            lowE = 2e+30
            chi_eff_sq = 2e+30
    else:
        plik = 2e+30
        lowl = 2e+30
        lowE = 2e+30
        chi_eff_sq = 2e+30
    
    global best_chisq

    if chi_eff_sq < best_chisq:
        best_chisq = chi_eff_sq

    return plik, lowl, lowE, chi_eff_sq


####################
# run Pypolychord
####################

#| Initialise the settings
nDims = 5
nDerived = 0
settings = PolyChordSettings(nDims, nDerived)
settings.file_root = 'inflation'
settings.nlive = 224
settings.do_clustering = True
settings.read_resume = True

#| Define loglikelihood
def to_optimise(params):
    plik, lowl, lowE, chi_eff_sq = run_TT(params)
    return -chi_eff_sq


def likelihood(theta): # theta=params
    """ get Likelihood by to_optimise """
    try: 
        logl = to_optimise(theta)
    except CosmoComputationError:
        logl = settings.logzero
    if np.isnan(logl):
        logl = settings.logzero
    print('logl='+str(logl))
    return logl, []

#| Define a box uniform prior from min to max

def prior(hypercube): # parameters = [logA_SR, N_star, log10f_i, omega_k, H0]
    """ Uniform prior from [min,max]^D. """
    # return [UniformPrior(2.5, 3.7)(hypercube[0]), UniformPrior(45, 70)(hypercube[1]), UniformPrior(-1, 5)(hypercube[2]), UniformPrior(-0.04, -0.0001)(hypercube[3]), UniformPrior(50, 75)(hypercube[4])]
    return [UniformPrior(2.5, 3.7)(hypercube[0]), UniformPrior(20, 90)(hypercube[1]), UniformPrior(-1, 5)(hypercube[2]), UniformPrior(-0.035, -1.e-5)(hypercube[3]), UniformPrior(20, 100)(hypercube[4])]
                      
#| Optional dumper function giving run-time read access to
#| the live points, dead points, weights and evidences

def dumper(live, dead, logweights, logZ, logZerr):
    print("Last dead point:", dead[-1])


#| Run PolyChord

output = pypolychord.run_polychord(likelihood, nDims, nDerived, settings, prior, dumper)

#| Create a paramnames file

parameters = ['logA_SR', 'N_star', 'log10f_i', 'omega_k', 'H0']
param_latex = [r'$\ln(10^{10} A_\mathrm{SR})$', r'$N_\ast$', r'$\log_{10} f_\mathrm{i}$', r'$\Omega_{K,0} h^2$', r'$H_0$']
paramnames = [(parameters[i], param_latex[i]) for i in range(nDims)]
output.make_paramnames_files(paramnames)

#| Make an anesthetic plot (could also use getdist)
try:
    from matplotlib import pyplot as plt
    from anesthetic import read_chains
    samples = read_chains(settings.base_dir + '/' + settings.file_root)
    samples.plot_2d(['logA_SR', 'N_star', 'log10f_i', 'omega_k', 'H0'])
    plt.savefig('posterior.pdf')

except ImportError:
    try:
        import getdist.plots
        posterior = output.posterior
        g = getdist.plots.getSubplotPlotter()
        g.triangle_plot(posterior, filled=True)
        g.export('posterior.pdf')
    except ImportError:
        print("Install matplotlib and getdist for plotting examples")

    print("Install anesthetic or getdist  for for plotting examples")
