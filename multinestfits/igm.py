import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, contextlib, sys
from yellowbrick.style import set_palette
from seaborn import pairplot
import seaborn as sns
import pandas as pd
from pymultinest.solve import solve

# selfmade packages
from functions.DataReader import reader
from functions.LineExclusion import excl
from functions.VoigtHjerting import H, addAbs
from functions.Selsing import aLambda
from functions.MiraldaEscude import addIGM

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
wave = data['WAVE_BIN']
flux = data['FLUX_BIN']
flux_err = data['ERR_FLUX_BIN']

fit_wave = wave.copy()
fit_flux = flux.copy()
fit_flux_err = flux_err.copy()

# EXCLUDING TELLURIC AND ABS LINES
for z in [6.312, 6.318, 5.7390, 2.8296]:
    fit_wave, fit_flux, fit_flux_err = excl(fit_wave, fit_flux, fit_flux_err, z, 0.9)

exclusion = ((fit_wave > 8970) & (fit_wave < 8979)) |\
    ((fit_wave > 8989) & (fit_wave < 8994)) |\
    ((fit_wave > 9060) & (fit_wave < 9065)) |\
    ((fit_wave > 9084) & (fit_wave < 9094)) |\
    ((fit_wave > 9116) & (fit_wave < 9130)) |\
    ((fit_wave > 9205) & (fit_wave < 9225)) |\
    ((fit_wave > 9516) & (fit_wave < 9546)) |\
    ((fit_wave > 8948) & (fit_wave < 8960)) |\
    (fit_wave < 8900) |\
    (fit_flux < 0.01e-19)

incl = ~exclusion


# Defining the fit model
def power(theta):
    f0, L, n1, av, hi = theta
    return fit_wave[incl]**(L-3.5) * f0 * 1e-10 * addAbs(fit_wave[incl],10**(n1),zabs) * 10**(-0.4*aLambda(fit_wave[incl],av)) * addIGM(fit_wave[incl], hi)

def power_init(x, f0, L, n1, av, hi):
    return x**(L-3.5) * f0 * 1e-10 * addAbs(x,10**(n1),zabs) * 10**(-0.4*aLambda(x,av)) * addIGM(x, hi)


# Defining the prior
def prior(utheta):
    uf0, uL, un1, uav, uhi = utheta
    f0 = uf0
    L = L_sigma*uL + L_init
    n1 = 3*un1 + 19
    av = uav
    hi = uhi
    return f0, L, n1, av, hi

#Defining the likelihood
def lnlike(theta):
    model = power(theta)
    f0, L, n1, av, hi = theta
    sigma2 =  fit_flux_err[incl] ** 2
    LnLike = np.sum(-0.5 * ((fit_flux[incl] - model) ** 2 / sigma2))
    return LnLike

parameters = [r'$F_0$',r'$\Gamma$',r'log(N$_{DLA}$/cm$^{-2})$', r' a$_V$/mag', r'x$_{HI}$']
n_params = len(parameters)

result = solve(LogLikelihood=lnlike, Prior=prior, 
	n_dims=n_params, verbose=True, n_live_points = 500, sampling_efficiency=0.3)

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
ax.set(ylim=(0, 3e-17), xlabel='Wavelength [Å]', ylabel='Flux [erg/s/cm/Å]', xlim=(8850, 9300))
#plt.savefig('../figures/igm_zoom.pdf')

# PLOTTING THE QUANTILES IN CORNER PLOT
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
samples = samples[2:5]
labels = [r'N$_{DLA}$/cm$^{-2}$', r' a$_V$/mag', r'$x_{HI}$']
df = pd.DataFrame(data=samples.T, columns=labels)
grid = pairplot(df,diag_kind='hist', corner=True, grid_kws=dict(despine=False), plot_kws=dict(marker=".", linewidth=1))
grid.map_lower(sns.kdeplot, levels=5, color="k")
for i in [0,1]:
    grid.figure.axes[i*2].axvline(x=res_array[i+2,0], color='k', linestyle='dashed')
    grid.figure.axes[i*2].axvline(x=res_array[i+2,0]-res_array[i+2,1], color='k', linestyle='dashed')
    grid.figure.axes[i*2].axvline(x=res_array[i+2,0]+res_array[i+2,1], color='k', linestyle='dashed')

grid.figure.axes[8].axvline(x=res_array[4,0], color='k', linestyle='dashed')
grid.figure.axes[8].axvline(x=res_array[4,0]-res_array[4,1], color='k', linestyle='dashed')
grid.figure.axes[8].axvline(x=res_array[4,0]+res_array[4,1], color='k', linestyle='dashed')
grid.figure.axes[0].set_title(labels[0]+' = '+f'{res_array[2,0]:.4f}'+r'$\pm$'+f'{res_array[2,1]:.4f}')
grid.figure.axes[2].set_title(labels[1]+' = '+f'({res_array[3,0]*1e4:.2f}'+r'$\pm$'+f'{res_array[3,1]*1e4:.2f})'+r'$\times$10$^{-4}$')
grid.figure.axes[8].set_title(labels[2]+' = '+f'{res_array[4,0]:.3f}'+r'$\pm$'+f'{res_array[4,1]:.3f}')
plt.tight_layout()
#plt.savefig('../figures/igm_corner.pdf')