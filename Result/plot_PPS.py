import matplotlib.pyplot as plt
import numpy as np


###############
# plot PPS from primpy
###############

# N_star_list = np.linspace(50, 60, num=5)
w_value_list = np.linspace(-0.4, -0.99, num=5)
# logA_SR_list = np.linspace(2.5, 3.7, num=5)
# H0_list = np.linspace(67, 74, num=5)
# omega_k_list = np.linspace(-0.04, -0.001, num=5)
# log10f_i_list = np.linspace(-1, 5, num=5)

for n in range(len(w_value_list)):
    # read flash.dat to a list of lists
    datContent = [i.strip().split() for i in open('./PPS_primpy/PPS_Nstar60_w='+str(n)+'.dat').readlines()]
    k_list = []
    PPS_list = []
    for i in range(len(datContent)):
        k_list.append(float(datContent[i][0]))
        PPS_list.append(float(datContent[i][1]))
    plt.loglog(k_list, PPS_list, label='w='+str(n))

plt.xlim([2.e-5, 5.])
plt.xlabel('k/Mpc^-1')
plt.ylabel('PPS')
plt.legend()
plt.show()