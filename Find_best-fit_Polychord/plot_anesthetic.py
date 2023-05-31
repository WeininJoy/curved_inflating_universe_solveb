from anesthetic import read_chains, make_2d_axes
from matplotlib import pyplot as plt


#| Make an anesthetic plot (could also use getdist)
try:
    from matplotlib import pyplot as plt
    from anesthetic import read_chains
    samples = read_chains("chains/inflation")
    samples.plot_2d(['logA_SR', 'N_star', 'log10f_i', 'omega_k', 'H0'])
    plt.savefig('posterior.pdf')

except ImportError:
    try:
        import getdist.plots
        posterior = output.posterior
        g = getdist.plots.getSubplotPlotter()
        g.triangle_plot(posterior, filled=True)
        g.export('posterior.pdf')
    except ImportError:
        print("Install matplotlib and getdist for plotting examples")

    print("Install anesthetic or getdist  for for plotting examples")
