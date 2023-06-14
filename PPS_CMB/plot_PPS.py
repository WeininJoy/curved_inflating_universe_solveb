import matplotlib.pyplot as plt
import numpy as np


###############
# plot PPS from primpy
###############

# w_value_list = [-0.33, 0.0, 0.33, 0.66, 0.99]
# w_value_list = np.linspace(-0.99, 0.99, num=5)
w_value_list = [-0.3333, -0.3, -0.2, -0.1, 0.0]

for n in range(len(w_value_list)):
    # read flash.dat to a list of lists
    datContent = [i.strip().split() for i in open('./PPS_b_difft/PPS_test_flat_w='+str(w_value_list[n])+'.dat').readlines()]
    k_list = []
    PPS_list = []
    for i in range(len(datContent)):
        k_list.append(float(datContent[i][0]))
        PPS_list.append(float(datContent[i][1]))
    plt.loglog(k_list, PPS_list, label='w='+str(w_value_list[n]))

plt.xlim([1.e-5, 5.])
plt.xlabel('$k/Mpc^-1$')
plt.ylabel('$P$')
plt.legend()
plt.show()
