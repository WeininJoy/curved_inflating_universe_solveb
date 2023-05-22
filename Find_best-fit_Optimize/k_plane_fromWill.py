import os
import sys
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import generate_k_point as genk

sys.path.insert(1, './class_parallel/scripts')
from classy import Class
from classy import CosmoComputationError

sys.path.insert(1, './plc_3.0/plc-3.0/include')
import clik

c = 2.99792458 * 10 ** 5  # Speed of light in km/s


class PlanckLikelihood(object):
    """Baseline Planck Likelihood"""
    
    def __init__(self):
        clikpath = os.environ["CLIK_PATH"]
        self.plik = clik.clik(os.path.join(clikpath, "../hi_l/plik/plik_rd12_HM_v22b_TTTEEE.clik"))
        self.lowl = clik.clik(os.path.join(clikpath, "../low_l/commander/commander_dx12_v3_2_29.clik"))
        self.lowE = clik.clik(os.path.join(clikpath, "../low_l/simall/simall_100x143_offlike5_EE_Aplanck_B.clik"))
        self.lensing = clik.clik_lensing(os.path.join(clikpath, "../lensing/smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8.clik_lensing"))
    
    def __call__(self, cls, nuis):
        lkl = []
        for like in [self.plik, self.lowl, self.lowE, self.lensing]:
            #        for like in [self.plik]:
            lmaxes = like.get_lmax()
            dat = []
            order = ['tt','ee','bb','te','tb','eb']
            if like is self.lensing:
                order = ['pp'] + order
            
            # print(order,len(lmaxes),len(order))
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
#        return np.array(lkl).flatten().sum()

lkl = PlanckLikelihood()


# Runs the CLASS code with given parameters params in form [omega_b, omega_cdm, 10^4 * omega_ncdm, h, 10^9 * A_s, n_s, tau_reio]
def run_class(params, Neff, N_ncdm, k0, delta_k):
    
    # create instance of the class " Class "
    LambdaCDM = Class()
    
    # pass input parameters
    LambdaCDM.set({'omega_b':params[0], 'omega_cdm':params[1], 'omega_ncdm':params[2]*1e-4, 'h':params[3], 'A_s':params[4]*1e-9, 'n_s':params[5], 'tau_reio':params[6], 'N_ur':Neff, 'N_ncdm':N_ncdm, 'k_min':k0, 'delta_k':delta_k})
    LambdaCDM.set({'output':'tCl,pCl,lCl,mPk','lensing':'yes','P_k_max_1/Mpc':3.0, 'l_max_scalars':2508})
    
    # run class
    LambdaCDM.compute()
    
    # get all C_l output
    cls = LambdaCDM.lensed_cl(2508)
    
    LambdaCDM.struct_cleanup()
    LambdaCDM.empty()
    
    return cls


# Checks the input values are within the priors
def check_prior(params):
    
    prior = np.array([[0.005, 0.1],
                      [0.001, 0.99],
                      [0, 10], # My choice
                      [0.2, 1.0],
                      [1.48, 54.6],
                      [0.9, 1.1],
                      [0.01, 0.8],
                      [0.9, 1.1],
                      [0, 200],
                      [0, 1],
                      [0, 10],
                      [0, 400],
                      [0, 400],
                      [0, 400],
                      [0, 400],
                      [0, 10],
                      [0, 50],
                      [0, 50],
                      [0, 100],
                      [0, 400],
                      [0, 10],
                      [0, 10],
                      [0, 10],
                      [0, 10],
                      [0, 10],
                      [0, 10],
                      [0, 3],
                      [0, 3]])

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


# Computes TT spectrum and returns chi^2 for given set of parameters params in form [omega_b, omega_cdm, omega_ncdm, h, 10^9 * A_s, n_s, tau_reio] using linear quantisation
def run_TT(params, k0, delta_k, filename):
    
    if check_prior(params) == True:
        
        lcdm_params = params[:7]
        
        Neff = 2.0328
        N_ncdm = 1
        T0 = 2.7255
        
        try:
            
            # Find corresponding spectra
            cls = run_class(lcdm_params, Neff, N_ncdm, k0, delta_k)
            
            nuis={
                'ycal':params[7],
                'A_cib_217':params[8],
                'xi_sz_cib':params[9],
                'A_sz':params[10],
                'ps_A_100_100':params[11],
                'ps_A_143_143':params[12],
                'ps_A_143_217':params[13],
                'ps_A_217_217':params[14],
                'ksz_norm':params[15],
                'gal545_A_100':params[16],
                'gal545_A_143':params[17],
                'gal545_A_143_217':params[18],
                'gal545_A_217':params[19],
                'galf_TE_A_100':params[20],
                'galf_TE_A_100_143':params[21],
                'galf_TE_A_100_217':params[22],
                'galf_TE_A_143':params[23],
                'galf_TE_A_143_217':params[24],
                'galf_TE_A_217':params[25],
                'calib_100T':params[26],
                'calib_217T':params[27],
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
        except CosmoComputationError:
            plik = 2e+30
            lowl = 2e+30
            lowE = 2e+30
            lensing = 2e+30
            chi_eff_sq = 2e+30
        except clik.lkl.CError:
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
            print(k0, delta_k, plik, lowl, lowE, lensing, chi_eff_sq, *params, 'False', file=f)
    
    print(chi_eff_sq)

    return plik, lowl, lowE, lensing, chi_eff_sq


def get_data(file_root):
    
    start_params = [0.02238280, 0.1201075, 6.451439, 0.6732117, 2.100549, 0.9660499, 0.05430842, 1.00044, 46.1, 0.66, 7.08, 248.2, 50.7, 53.3, 121.9, 0., 8.80, 11.01, 20.16, 95.5, 0.1138, 0.1346, 0.479, 0.225, 0.665, 2.082, 0.99974, 0.99819]
    
    sim = np.transpose(np.loadtxt('simplex.txt', unpack=True))
    
    k0, delta_k = genk.generate_k0_deltak_pair()
    
    print('k0 = ', k0 * 1e3, 'x 10^{-3} Mpc^{-1}')
    print('delta_k = ', delta_k * 1e3, 'x 10^{-3} Mpc^{-1}')
    
    def to_optimise(params):
        plik, lowl, lowE, lensing, chi_eff_sq = run_TT(params, k0, delta_k, file_root)
        return chi_eff_sq

    # test simulation
    for i in range(sim.shape[0]):
        to_optimise(sim[i, :])
    
    if best_chisq < 2e+30:
    
        res = minimize(to_optimise, start_params, method='Nelder-Mead', options={'initial_simplex': sim, 'fatol': 0.03, 'xatol':0.001})
        plik, lowl, lowE, lensing, chi_eff_sq = run_TT(res.x, k0, delta_k, file_root)

        print(k0, delta_k, plik, lowl, lowE, lensing, chi_eff_sq, res.success)
        print(res.x)

        with open(file_root, 'w') as f:
            print(k0, delta_k, plik, lowl, lowE, lensing, chi_eff_sq, *res.x, res.success, file=f)
    else:
        print('Invalid')
        with open(file_root, 'w') as f:
            print(k0, delta_k, 'Invalid', file=f)

if __name__ == '__main__':
    get_data(*sys.argv[1:])
