######### Goal: Find best-fit parameter set by minimizing chi_eff_sq -> most simplier to observation
######### Method: use scipy.optimize.minimize() function

import os
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
os.environ['VECLIB_MAXIMUM_THREADS'] = '8'
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
from primpy.events import InflationEvent, SlowRowEvent, CollapseEvent, UntilNEvent
from primpy.initialconditions import ISIC_NsOk
from primpy.time.inflation import InflationEquationsT
import primpy.bigbang as bb
from scipy.optimize import minimize
import matplotlib.pyplot as plt


sys.path.insert(0, '/home/wnd22/rds/hpc-work/env_/lib/python3.8/site-packages/class_public')
sys.path.insert(0, '/home/wnd22/rds/hpc-work/env_/lib/python3.8/site-packages')
import classy
from classy import CosmoComputationError
print(classy.__file__)

plc = os.path.join(os.getcwd(),'packages/code/planck/code/plc_3.0/plc-3.1')
data = os.path.join(os.getcwd(),'packages/data/planck_2018/baseline/plc_3.0') 

#sys.path.insert(plc + 'include/')
sys.path.append(os.path.join(plc, 'lib/python3.8/site-packages')) 
sys.path.append(os.path.join(plc, 'lib/python3.8/site-packages')) 
import clik


c = 2.99792458 * 10 ** 5  # Speed of light in km/s
# w_values = np.linspace(-0.999, 0.999, num=100)
# w_values = [-0.5953636363636363, -0.5751818181818182, -0.5549999999999999, -0.5348181818181819, \
#             -0.5146363636363636, -0.39354545454545453, -0.37336363636363634, -0.35318181818181815,\
#             -0.2926363636363636, -0.2522727272727272, -0.21190909090909082, -0.19172727272727264, \
#             0.01009090909090904, 0.03027272727272734, 0.05045454545454542, 0.07063636363636372, \
#             0.0908181818181818, 0.1110000000000001, 0.13118181818181818, 0.15136363636363648, \
#             0.17154545454545456, 0.19172727272727286, 0.21190909090909094, 0.23209090909090924, \
#             0.2724545454545456, 0.3128181818181818, 0.3330000000000001, 0.35318181818181815, \
#             0.39354545454545453, 0.41372727272727283, 0.4339090909090909, 0.4540909090909092, \
#             0.4742727272727273, 0.4944545454545456, 0.5146363636363637, 0.534818181818182, 0.555, \
#             0.5751818181818183, 0.5953636363636364, 0.6155454545454547, 0.6357272727272728, \
#             0.6559090909090909, 0.6760909090909092, 0.6962727272727273, 0.7164545454545456, \
#             0.7366363636363636, 0.756818181818182, 0.777, 0.8173636363636364, 0.8375454545454547,\
#             0.8577272727272728, 0.8779090909090911, 0.8980909090909092, 0.9182727272727275, 0.9384545454545455]
w_values = [0.999]
w_value = w_values[int(sys.argv[1])] # equation of state when matching a&b
dat_name = 'PPS_simplex_w='+str(w_value)+'.dat'
file_root = 'find_bestfit_simplex_w='+str(w_value)+'.txt'
############
# read start params from files 
if os.path.isfile(file_root):
    print('Use start_params from '+ file_root)
    with open(file_root,'rb') as f:
        # [plik, lowl, lowE, lensing, chi_eff_sq, logA_SR, N_star, log10f_i, omega_k, H0]
        # [  0 ,   1 ,   2 ,    3   ,     4     ,    5   ,    6  ,    7    ,   8    ,  9]
        data_ini = f.read().split()
        start_params = [float(ele) for ele in data_ini[5:10]]
else: 
    print('Set start_params as default ')
    start_params = [3.045, 60.0, 4.94, -0.005, 63] # [np.log(20), 55, 5, -0.005, 70]
############


class PlanckLikelihood(object):
    """Baseline Planck Likelihood"""
    
    def __init__(self):
        self.plik = clik.clik(os.path.join(data, "hi_l/plik/plik_rd12_HM_v22b_TTTEEE.clik"))
        self.lowl = clik.clik(os.path.join(data, "low_l/commander/commander_dx12_v3_2_29.clik"))
        self.lowE = clik.clik(os.path.join(data, "low_l/simall/simall_100x143_offlike5_EE_Aplanck_B.clik"))
        self.lensing = clik.clik_lensing(os.path.join(data, "lensing/smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8.clik_lensing"))
    
    def __call__(self, cls, nuis):
        lkl = []
        for like in [self.plik, self.lowl, self.lowE, self.lensing]:
            #        for like in [self.plik]:
            lmaxes = like.get_lmax()
            dat = []
            order = ['tt','ee','bb','te','tb','eb']
            if like is self.lensing:
                order = ['pp'] + order
            
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
        'output': 'tCl pCl lCl mPk',
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
                      [45, 70],        #N_star
                      [-1, 5],         #log10f_i
                      [-0.04, -0.0001], #omega_k
                      [50, 75]])       #H0

    diff_below = params - prior[:, 0]
    diff_above = prior[:, 1] - params
      
    if min(diff_below) < 0:
        result = False
    elif min(diff_above) < 0:
        result = False
    else:
        result = True
                      
    return result

# Global variable for best chi^2
best_chisq = 2e+30 

# Computes TT spectrum and returns chi^2 for given set of parameters params in form  [logA_SR, N_star, log10f_i, omega_k, H0] using linear quantisation
def run_TT(params, filename):
    
    if check_prior(params) == True:
        
        try:
            # Find corresponding spectra
            cls = run_class(params)
            0.02238280, 0.1201075, 6.451439, 0.6732117, 2.100549, 0.9660499, 0.05430842, 1.00044, 46.1, 0.66, 7.08, 248.2, 50.7, 53.3, 121.9, 0., 8.80, 11.01, 20.16, 95.5, 0.1138, 0.1346, 0.479, 0.225, 0.665, 2.082, 0.99974, 0.99819
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

            plik, lowl, lowE, lensing = -2 * lkl(cls, nuis)
            chi_eff_sq = plik + lowl + lowE + lensing
            print('chi_eff_sq='+str(chi_eff_sq))

        except CosmoComputationError:
            print('CosmoComputationError')
            plik = 2e+30
            lowl = 2e+30
            lowE = 2e+30
            lensing = 2e+30
            chi_eff_sq = 2e+30
        except clik.lkl.CError:
            print('CError')
            plik = 2e+30
            lowl = 2e+30
            lowE = 2e+30
            lensing = 2e+30
            chi_eff_sq = 2e+30
    else:
        plik = 2e+30
        lowl = 2e+30
        lowE = 2e+30
        lensing = 2e+30
        chi_eff_sq = 2e+30
    
    global best_chisq

    if chi_eff_sq < best_chisq:
        best_chisq = chi_eff_sq
        with open(filename, 'w') as f:
            print(plik, lowl, lowE, lensing, chi_eff_sq, *params, 'False', file=f)
    
    print(chi_eff_sq)

    return plik, lowl, lowE, lensing, chi_eff_sq


def get_data(file_root):
    
    sim = np.transpose(np.loadtxt('simplex.txt', unpack=True))
    print(sim)
    def to_optimise(params):
        plik, lowl, lowE, lensing, chi_eff_sq = run_TT(params, file_root)
        return chi_eff_sq

    # test simulation
    
    for i in range(sim.shape[0]):
        to_optimise(sim[i, :])

    if best_chisq < 2e+30:
        res = minimize(to_optimise, start_params, method='Nelder-Mead', options={'initial_simplex': sim, 'fatol': 0.03, 'xatol':0.001})
        plik, lowl, lowE, lensing, chi_eff_sq = run_TT(res.x, file_root)

        print(plik, lowl, lowE, lensing, chi_eff_sq, res.success)
        print(res.x)

        with open(file_root, 'w') as f:
            print(plik, lowl, lowE, lensing, chi_eff_sq, *res.x, res.success, file=f)
    
    else:
        print('Invalid')
        with open(file_root, 'w') as f:
            print('Invalid', file=f)


if __name__ == '__main__':
    get_data(file_root)

