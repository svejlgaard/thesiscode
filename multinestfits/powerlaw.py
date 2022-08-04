import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, contextlib, sys
from yellowbrick.style import set_palette
from seaborn import pairplot
import seaborn as sns
import pandas as pd
from pymultinest.solve import solve
from astropy.stats import sigma_clip

# selfmade packages
from functions.DataReader import reader
from functions.LineExclusion import excl

L_init = 1.90
L_sigma = 0.15

zabs = 6.312

# Sets the directory to the current directory
os.chdir(sys.path[0])
np.random.seed(27)

# The smoothing function
def smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

set_palette('flatui')   
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Charter"
plt.rcParams['font.size'] =  14
plt.rcParams['legend.fontsize'] = 22
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 5.0
plt.rcParams['xtick.minor.size'] = 3.0
plt.rcParams['ytick.major.size'] = 5.0
plt.rcParams['ytick.minor.size'] = 3.0
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['xtick.minor.visible'] = True
colorlist = plt.rcParams['axes.prop_cycle'].by_key()['color']


# LOADING DATA
data = reader('GRB210905A', 'binned_50')
data['WAVE_BIN'] = 10 * data['WAVE_BIN']
wave = data['WAVE_BIN'].to_numpy()
flux = data['FLUX_BIN'].to_numpy()
flux_err = data['ERR_FLUX_BIN'].to_numpy()


fit_wave = wave.copy()
fit_flux = flux.copy()
fit_flux_err = flux_err.copy()

# EXCLUDING TELLURIC AND ABS LINES
for z in [6.312, 6.318, 5.7390, 2.8296]:
    fit_wave, fit_flux, fit_flux_err = excl(fit_wave, fit_flux, fit_flux_err, z, 0.8)

incl = ((fit_wave > 14500) & (fit_wave < 18000) & (fit_flux > 0) & (fit_flux_err < fit_flux) & (fit_flux < 1.5e-17) & (fit_flux > 0.75e-17))

fit_flux = sigma_clip(fit_flux, 3)

# Defining the fit model
def power(theta):
    f0, L = theta
    return fit_wave[incl]**(L-3.5) * f0 * 1e-10 

def power_init(x, f0, L):
    return x**(L-3.5) * f0 * 1e-10

# Defining the prior
def prior(utheta):
    uf0, uL = utheta
    f0 = uf0
    L = L_sigma*uL + L_init
    return f0, L

#Defining the likelihood
def lnlike(theta):
    model = power(theta)
    f0, L = theta
    sigma2 =  fit_flux_err[incl] ** 2
    LnLike = -0.5 * np.sum((fit_flux[incl] - model) ** 2 / sigma2)
    return LnLike


parameters = [r'$F_0$',r'$\Gamma$']
n_params = len(parameters)

result = solve(LogLikelihood=lnlike, Prior=prior, 
	n_dims=n_params, verbose=True, n_live_points = 500, sampling_efficiency=0.8)

res_array = np.zeros((n_params,2))

print()
print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
print()
print('parameter values:')
for name, col in zip(parameters, result['samples'].transpose()):
	print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))

for c, col in enumerate(result['samples'].transpose()):
    res_array[c] = np.array([col.mean(), col.std()])


samples = result['samples'].transpose()
best_fit_model = power_init(wave, *res_array[:,0])

# PLOTTING A ZOOM ON RESULTS
fig, ax = plt.subplots(figsize=(16,9))
ax.plot(wave,smooth(flux,5))
ax.plot(wave,best_fit_model)
ax.plot(fit_wave[incl], smooth(fit_flux[incl],35))
ax.set(ylim=(0, 3e-17), xlabel='Wavelength [Å]', ylabel='Flux [erg/s/cm/Å]', xlim=(15000, 18000))
#plt.savefig('../figures/powerlaw_zoom.pdf')

# PLOTTING THE QUANTILES IN CORNER PLOT
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

df = pd.DataFrame(data=samples.T, columns=parameters)
grid = pairplot(df,diag_kind='hist', corner=True, grid_kws=dict(despine=False), plot_kws=dict(marker=".", linewidth=1))
grid.map_lower(sns.kdeplot, levels=5, color="k")
for i in [0,1]:
    grid.figure.axes[i*2].axvline(x=res_array[i,0], color='k', linestyle='dashed')
    grid.figure.axes[i*2].axvline(x=res_array[i,0]-res_array[i,1], color='k', linestyle='dashed')
    grid.figure.axes[i*2].axvline(x=res_array[i,0]+res_array[i,1], color='k', linestyle='dashed')
grid.figure.axes[0].set_title(parameters[0]+' = '+f'({res_array[0,0]:.3f}'+r'$\pm$'+f'{res_array[0,1]:.3f})'+r'$\times$10$^{-10}$')
grid.figure.axes[2].set_title(parameters[1]+' = '+f'({res_array[1,0]:.3f}'+r'$\pm$'+f'{res_array[1,1]:.3f})')
plt.tight_layout()
#plt.savefig('../figures/powerlaw_corner.pdf')