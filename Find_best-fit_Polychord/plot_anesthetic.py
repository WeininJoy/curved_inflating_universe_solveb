from anesthetic import read_chains, make_2d_axes
from matplotlib import pyplot as plt

samples = read_chains("chains/inflation")
prior = samples.prior()
params = ['logA_SR', 'N_star', 'log10f_i', 'omega_k', 'H0']
fig, axes = make_2d_axes(params, figsize=(6, 6), facecolor='w')
prior.plot_2d(axes, alpha=0.9, label="prior")
samples.plot_2d(axes, alpha=0.9, label="posterior")
axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes)/2, len(axes)), loc='lower center', ncols=2)
plt.savefig('posterior.pdf')
