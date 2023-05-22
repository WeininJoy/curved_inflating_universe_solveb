##### Goal of this code: some cosmological constants used in these codes

from astropy import units as u
from astropy.constants import c
import numpy as np

ks = 0.05/u.Mpc
K = 1
Omega_K = -0.01
H0 = 70 * u.km/u.s/u.Mpc
h = 0.7
As = 2e-9
ns = 0.972535
tau = 0.0495
Omega_b = 0.022632/h**2
Omega_c = 0.11792/h**2
fi = 5
Ns = 55

a0 = c * np.sqrt(-K/Omega_K)/H0
print("a0 =", a0.si)
print("a0 =", a0.to(u.Mpc))
print("k a0 =", (ks*a0))
aH = (ks*a0).si.value
print("aH =", aH)
print("Omega_K*=(a*H*)^-2 =", 1/(aH)**2)
print(np.exp(10)/np.sqrt(2))
print(np.log(1.1427262120574145e-05))
print( 1./2.*(1.-np.exp(-np.sqrt(2./3.)*5.5))**2 - np.exp(-2*(11.713+np.log(1.18e-05))) )
N0 = 11.713
sigma= 1.158e-5
print(1.- np.sqrt( 2.*np.exp(-2.*(N0 + np.log(sigma)))))
print( -np.sqrt(3./2.)* np.log(1.- np.sqrt( 2.*np.exp(-2.*(N0 + np.log(sigma))) ) )) 
print( 3.e8/9.8* np.arctanh(9.46e16/3.e8))