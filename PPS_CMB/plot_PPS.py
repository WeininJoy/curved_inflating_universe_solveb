import matplotlib.pyplot as plt
import numpy as np


###############
# plot PPS from primpy
###############

cm = 1./2.54  # centimeters in inches
plt.figure(figsize=(10*cm, 7.8*cm))

# w_value_list = [-0.33, 0.0, 0.33, 0.66, 0.99]
w_value_list = np.linspace(-0.99, 0.99, num=5)
# dat_name = ['8','9','10','11', '11.8']

for n in range(len(w_value_list)):
    # read flash.dat to a list of lists
    # datContent = [i.strip().split() for i in open('./PPS_difft/PPS_deepKD_w='+str(w_value_list[n])+'.dat').readlines()]
    w_label = round(w_value_list[n], 3)
    datContent = [i.strip().split() for i in open('./PPS_diffw/PPS_w='+str(w_label)+'.dat').readlines()]
    k_list = []
    PPS_list = []
    for i in range(len(datContent)):
        k_list.append(float(datContent[i][0]))
        PPS_list.append(float(datContent[i][1]))
    plt.loglog(k_list, PPS_list, label='$w=$'+str(w_label))

# datContent = [i.strip().split() for i in open('./PPS/PPS_enterH.dat').readlines()]
# k_list = []
# PPS_list = []
# for i in range(len(datContent)):
#     k_list.append(float(datContent[i][0]))
#     PPS_list.append(float(datContent[i][1]))
# plt.loglog(k_list, PPS_list, 'k', label='N = enter Horizon')

plt.xlim([1.e-5, 1.e-1])
plt.ylim([5.e-12, 4.e-8])
plt.xlabel('$k/Mpc^{-1}$', fontsize=10)
plt.ylabel('$PPS$', fontsize=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.legend(fontsize=9, loc='best')
plt.savefig('PPS_diffw.pdf', format='pdf', bbox_inches="tight")

