from anesthetic import read_chains, make_2d_axes
from matplotlib import pyplot as plt


#| Make an anesthetic plot (could also use getdist)
try:

    # set params and figure
    params = ['logA_SR', 'N_star', 'log10f_i', 'omega_k', 'H0']
    cm = 1./2.54  # centimeters in inches
    fig, axes = make_2d_axes(params, figsize=(7.5, 7.5), facecolor='w')
     
    # plot prior and posterior of Lukas' result
    samples_Lukas = read_chains("/home/weinin/Documents/Research/Will_solve_b/finite_inflation_curved_space_data/cb3-0_pc1-20_cl2-9/stb_omegak/pcs3d500_cl_hf_p18_TTTEEElite_lowl_lowE_BK15_stb_omegak_AsfoH_perm/pcs3d500_TTTEEElite_lowl_lowE_BK15_stb_omegak_AsfoH_perm_polychord_raw/pcs3d500_TTTEEElite_lowl_lowE_BK15_stb_omegak_AsfoH_perm")
    prior_Lukas = samples_Lukas.prior()
    prior_Lukas.plot_2d(axes, alpha=0.9, label="prior_Lukas")
    # samples_Lukas.plot_2d(axes, alpha=0.9, label="posterior_Lukas")
    
    # plot prior and posterior of Weinin's result
    samples_weinin = read_chains("chains/inflation")
    prior_weinin = samples_weinin.prior()
    prior_weinin.plot_2d(axes, alpha=0.9, label="prior_weinin")
    # samples_weinin.plot_2d(axes, alpha=0.9, label="posterior_weinin")

    # figure setting and savefigure
    axes.set_xlabel(fontsize=8)
    axes.set_ylabel(fontsize=8)
    for tick in axes.xaxis.get_major_ticks():
        tick.label.set_fontsize(7) 
    for tick in self.ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(7) 
    axes.iloc[-1, 0].legend(fontsize=8, bbox_to_anchor=(len(axes)/2, len(axes)), loc='lower center', ncols=2)
    fig.savefig('prior_Lukas_weinin.png')

except ImportError:
    try:
        import getdist.plots
        posterior = output.posterior
        g = getdist.plots.getSubplotPlotter()
        g.triangle_plot(posterior, filled=True)
        g.export('posterior.png')
    except ImportError:
        print("Install matplotlib and getdist for plotting examples")

    print("Install anesthetic or getdist  for for plotting examples")
