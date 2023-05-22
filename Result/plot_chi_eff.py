import matplotlib.pyplot as plt
import numpy as np


###############
# plot chi_eff from primpy
###############
with open('chi_eff/chi_eff_sq_diff_w_logA_SR_2_5.txt','rb') as f:
    data = f.read().split()

chi_eff_sq = [float(num) for num in data]

w_value_list = np.linspace(-0.4, -0.99, num=5)
# N_star_list = np.linspace(50, 60, num=5)
# logA_SR_list = np.linspace(2.5, 3.7, num=5)
# H0_list = np.linspace(67, 74, num=5)
# omega_k_list = np.linspace(-0.04, -0.001, num=5)
# log10f_i_list = np.linspace(-1, 5, num=5)


plt.plot(w_value_list, chi_eff_sq)
plt.xlabel('w (when a&b match)')
plt.ylabel('chi_eff_sq')
#plt.legend()
plt.show()

"""
###############
# plot chi_eff from PPS_will_b.py
###############

with open('chi_eff/chi_eff_sq_will_b.txt','rb') as f:
    data = f.read().split()

chi_eff_sq_will_b = [float(num) for num in data]

match_t_list = np.linspace(0.0, 2.0, num=6)

plt.plot(match_t_list, chi_eff_sq_will_b)
plt.xlabel('t (when a&b match)')
plt.ylabel('chi_eff_sq')
plt.legend()
plt.show()
"""
