########### Goal: calculate chi_eff by known PPS (generated based on different parameters)
########### Method: Refer to k_plane_fromWill.py

import os
import sys
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from classy import Class
from classy import CosmoComputationError

plc = os.path.join(os.getcwd(),'packages/code/planck/code/plc_3.0/plc-3.1')
data = os.path.join(os.getcwd(),'packages/data/planck_2018/baseline/plc_3.0') 

#sys.path.insert(plc + 'include/')
sys.path.append(os.path.join(plc, 'lib/python/site-packages')) 
sys.path.append(os.path.join(plc, 'lib/python/site-packages')) 
print(sys.path)
import clik

c = 2.99792458 * 10 ** 5  # Speed of light in km/s


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

#######
# CDM cls
#######

# LambdaCDM = Class()
# LambdaCDM.set() # Adjust cosmological parameters/primordial power spectrum as appropriate here
# LambdaCDM.set({'output':'tCl,pCl,lCl,mPk','lensing':'yes','P_k_max_1/Mpc':3.0, 'l_max_scalars':2508})
# LambdaCDM.compute()
# cls = LambdaCDM.lensed_cl(2508)



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


###################
# calculate chi of PPS from primpy
###################

f = open('chi_eff_sq_diff_w_H0_74.txt', 'a')
# N_star_list = np.linspace(50, 60, num=5)
w_value_list = np.linspace(-0.4, -0.99, num=5)
# logA_SR_list = np.linspace(2.5, 3.7, num=5)
# H0_list = np.linspace(67, 74, num=5)
# omega_k_list = np.linspace(-0.04, -0.001, num=5)
# log10f_i_list = np.linspace(-1, 5, num=5)

chi_eff_sq_list = []
for n in range(5):
 
    ##### primpy constants
    logA_SR   = np.log(20)
    N_star    = 55
    log10f_i  = 5
    omega_k   = -0.005
    H0        = 74
    
    ####### solve_b cls

    h = H0/100
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
        # 'custom1': 3.05219166,
        'custom1': 3.0522,
        # 'custom2': 5.75507947e+01,
        'custom2': 5.755e+01,
        # 'custom3': 1.28491857e-01,
        'custom3': 1.28e-01,
        # 'custom4': -1.15169968e-02,
        'custom4': -1.15e-02,
        'N_ncdm': 1,
        'N_ur': 2.0328,
        'lensing': 'yes',
        'non linear': 'halofit',
        'modes': 's t',
        'output': 'tCl pCl lCl mPk',
        'P_k_ini type': 'external_Pk',
        'P_k_max_1/Mpc':3.0,
        'l_max_scalars':2508,
        'command': 'cat ./PPS_primpy/PPS_H0_74_w='+str(n)+'.dat'
    }

    pstb = Class()
    pstb.set(params_pstb)
    # LambdaCDM.set({'output':'tCl,pCl,lCl,mPk','lensing':'yes', 'l_max_scalars':2508})
    pstb.compute()
    cls = pstb.lensed_cl(2508)
    

    plik, lowl, lowE, lensing = -2 * lkl(cls, nuis)
    chi_eff_sq = plik + lowl + lowE + lensing

    chi_eff_sq_list.append(chi_eff_sq)
    print(str(chi_eff_sq)+' ')
    f.write(str(chi_eff_sq)+' ')

f.close()
plt.plot(w_value_list, chi_eff_sq_list)
plt.xlabel('w (when a&b match)')
plt.ylabel('chi_eff_sq')
#plt.legend()
plt.show()

"""
###################
# calculate chi of PPS from PPS_will_b.py
###################

f = open('chi_eff_sq_will_b.txt', 'a')

for n in range(6):

    curved = Class()
    params = {
            'output': 'tCl pCl lCl mPk',
            'l_max_scalars': 2508,
            'lensing': 'yes',
            'P_k_ini type': 'external_Pk',
            'command': 'cat ./PPS_will_b/PPS_will_b-n='+str(n)+'.dat',
            'tau_reio': 0.0515,
            'h': 64.03/100,
            'omega_b': 0.022509,
            'Omega_k': -0.0092,
            'omega_cdm': 0.11839}
    curved.set(params)
    curved.compute()
    cls = curved.lensed_cl(2508)
    

    plik, lowl, lowE, lensing = -2 * lkl(cls, nuis)
    chi_eff_sq = plik + lowl + lowE + lensing

    print(str(chi_eff_sq)+' ')
    f.write(str(chi_eff_sq)+' ')

f.close()

"""
