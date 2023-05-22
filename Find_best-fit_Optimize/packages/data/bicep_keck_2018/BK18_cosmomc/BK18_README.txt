BICEP2/Keck Array June 2021 Data Products
BICEP/Keck XIII: Constraints on Primordial Gravitational Waves using Planck, WMAP, and BICEP/Keck Observations through the 2018 Observing Season
http://bicepkeck.org/

File: BK18_README.txt
Date: 2021-06-07

This README file gives an overview of the BK18 data release products for CosmoMC.
Additional information can be found in the header comments of each file.

Contents of this tarball (file locations as normally organized in CosmoMC):
1.   As BICEP3 maps have larger sky covergae than BICEP2/Keck's, in our studies there are 3 different ways to apodize 
     BICEP/Keck and WMAP/Planck maps: 1) maps are apodized to the largest possible map size (i.e. BICEP2/Keck maps and 
     BICEP3 maps are used as they are, and WMAP/Planck maps are apodized to the BICEP3 map size); 2) maps are apodized 
     to the smallest field size (i.e. the BICEP2/Keck map size); and 3) maps are apodized to the BICEP3 map size. (i.e. 
     BICEP2/Keck maps are dropped). In this release only data apodized by the first method is included, and they are 
     marked as "BK18lf". 
2.   data/BK18lf_dust/BK18lf*: These files contain the B-modes only data (bandpowers, bandpower covariance, and ancillary 
     data) needed to use the BK18lf dataset in CosmoMC (including WMAP and Planck polarization data in the BICEP field).
     BK18lf_dust.dataset is the main file. Furthur information can be found in comments therein.
3.   data/BK18lf_dust/windows/: This directory contains BK18lf bandpower window functions.
4.   data/BK18lf_dust/bandpass*: These files contain instrument frequency response.
5.   data/BK18lf_dust_incEE/: This is the corresponding data directory contains BB, EE (and EB) data.
6.   batch3/BK18lf.ini: This file sets the data selection and foreground nuisance parameters used in BK18lf of BK-XIII.
7.   batch3/BK18lfonly.ini: For CosmoMC runs where you are using *only* the BK18lf data set, you should include it via
     this file, which sets scalar cosmological parameters to nominal values. These parameters are otherwise not
     well constrained by BK18lf data. If you are running chains using BK18lf alongside CMB TT data or similar, then it
     is not necessary to fix these parameters.
8.   batch3/BK18/BK18_04_BK18lf_freebdust.ini, BK18_04_BK18lf_freebdust_dist.ini: These files run CosmoMC and getdist 
     to recompute the results of the BK18 baseline analysis, as seen in Figure 4 of BK-XIII.
9.   batch3/BK18/BK18_07_BK18lf_freebdust_dclin.ini, BK18_08_BK18lf_freebdust_nop353.ini, ...: These files run CosmoMC 
     and getdist to recompute results for some of the alternate likelihoods seen in Figures 5, 17-21 of BK-XIII.
10.   chains/BK18_04_BK18lf_freebdust/ and BK18_17_BK18lf_freebdust_incP2018_BAO/: These directories contain CosmoMC 
     chains of likelihoods studied in the main body (figure 4 and figure 5) of BK-XIII. 
11.  planck_covmats/BK18.covmat, ...: These files are included by the various ini files found in batch3/BK18/. Using 
     them should speed up the convergence of your chains.
12.  source/CMB_BK_Planck.f90: This file encodes the foreground model used in BK-XIII. It is intended to be compiled 
     and run as part of CosmoMC, but may also be a useful reference for technical details of the model. It was 
     originally distributed with the BICEP2/Keck/Planck joint result and data release. The addition of foreground
     decorrelation parameters in BK-X breaks backwards compatibility. If you want to run older datasets (BKP, BK14)
     with the new model, then you should get an up-to-date copy of CosmoMC via Antony Lewis' github: 
     https://github.com/cmbant/CosmoMC
