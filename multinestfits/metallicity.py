import numpy as np
import matplotlib.pyplot as plt
import os, sys
from uncertainties.core import ufloat
from yellowbrick.style import set_palette
from uncertainties import ufloat
from seaborn import pairplot
import seaborn as sns
import pandas as pd
from pymultinest.solve import solve
from functions.Reader import reader

# SETS THE DIRECTORY TO THE CURRENT DIRECTORY
os.chdir(sys.path[0])

# CHOOSING SEEDS FOR LESS STOCHASTIC RE-RUNS
np.random.seed(27)

# PRETTY PLOTTING  
plt.style.use('seaborn-paper')
set_palette('flatui')   
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Charter"
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['xtick.minor.visible'] = False
colorlist = plt.rcParams['axes.prop_cycle'].by_key()['color']

# INITIAL SETTINGS
obs_logNH = ufloat(21.17, 0.03)
elements = ['SII','SiII','FeII']
plotelements = ['SII','SiII','MgII','FeII', 'OI','AlII', 'CII']
lowlims = [0,0,1,0,1,0,1]


# LOADING DATA
def get_xy(elem):
    obs, obs_abun, par, solar_abundance, dtm_new, fe_zn = reader('../voigtfit/ObsAbundance', elem, obs_logNH)
    print(obs, obs_abun)
    xvals = [ufloat(b, berr) for b,berr in zip(par['B'].to_numpy(dtype=float), par['Berr'].to_numpy(dtype=float))]
    xvals = pd.DataFrame(data=xvals, index=par.index.to_list(), columns=['B'])
    a = [ufloat(b, berr) for b,berr in zip(par['A'].to_numpy(dtype=float), par['Aerr'].to_numpy(dtype=float))]
    a = pd.DataFrame(data=a, index=obs.index.to_list(), columns=['A'])
    logN = obs_abun.to_numpy() 
    solar = solar_abundance.to_numpy()[:,0]
    a = a.to_numpy()[:,0]
    yvals = logN - solar + 12 - a - obs_logNH
    y = np.array([v.n for v in yvals])
    y_err = np.array([v.s for v in yvals])
    x = np.array([v.n for v in xvals.to_numpy()[:,0]])
    x_err = np.array([v.s for v in xvals.to_numpy()[:,0]])
    return x,y,x_err, y_err, dtm_new, fe_zn


x,y,x_err,y_err,dtm_new,fe_zn = get_xy(elements)

xplot,yplot,xplot_err,yplot_err,_,_ = get_xy(plotelements)

# DEFINING FUNCTIONS TO FIT
def dust_init(x, theta):
    m, z = theta
    return m + z*x

def dust(theta):
    m, z = theta
    return m + z*x

# DEFINING THE PRIORS
def prior(utheta):
    um, uz = utheta
    m = 4.0*um - 3.0
    z = 4.0*uz - 2.0
    return m, z

evidence_list = list()
   
def lnlike(theta):
    model = dust(theta)
    sigma2 =  y_err ** 2
    LnLike = -0.5 * np.sum((y - model) ** 2 / sigma2)
    return LnLike



parameters = [r'$[M/H]_{tot}$', r'$[Zn/Fe]_{fit}$']
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


zn_fe = ufloat(res_array[1,0],res_array[1,1])
fe_zn = -zn_fe
dtm_2013 = (1 - 10**(fe_zn*(-0.95)/(-0.95+0.11))) / 0.98
print(dtm_2013)

samples = result['samples']

plt.clf()
plt.style.use('seaborn-paper')
set_palette('flatui')   
plt.rcParams["font.family"] = "serif"
plt.errorbar(xplot, yplot, yerr=yplot_err,xerr=xplot_err, fmt='o', label='Observed abundance', color=colorlist[2], lolims=lowlims)
plt.plot(np.arange(-2, 2, step=1), dust_init(np.arange(-2, 2, step=1),(res_array[0,0], res_array[1,0])), label=fr'Depletion pattern with $[M/H]$ = {res_array[0,0]:.2f} +- {res_array[0,1]:.2f} and $[Zn/Fe]$ = {res_array[1,0]:.2f} +- {res_array[1,1]:.2f}', color=colorlist[0])
for i, txt in enumerate(plotelements):
    if txt == 'SiII':
        plt.annotate(txt.replace('I',''), (xplot[i]-0.07, yplot[i]+0.02))
    elif txt == 'MgII':
        plt.annotate(txt.replace('I',''), (xplot[i]+0.02, yplot[i]-0.06))
    elif txt == 'SII':
        plt.annotate(txt.replace('I',''), (xplot[i]-0.05, yplot[i]+0.02))
    elif txt == 'FeII':
        plt.annotate(txt.replace('I',''), (xplot[i]+0.02, yplot[i]-0.06))
    else:
        plt.annotate(txt.replace('I',''), (xplot[i]+0.03, yplot[i]+0.02))
plt.legend()
plt.xlim([-1.75, 0.1])
plt.ylim([-2.25, -1])
plt.xlabel('x')
plt.ylabel('y')
#plt.savefig(f'figures/DynestyFittingResults_relative.pdf')



plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plotsamples = samples
df = pd.DataFrame(data=plotsamples, columns=parameters)
grid = pairplot(df,diag_kind='hist', corner=True, grid_kws=dict(despine=False), plot_kws=dict(marker=".", linewidth=1))
grid.map_lower(sns.kdeplot, levels=5, color="k")
for i in [0,1]:
    grid.figure.axes[i*2].axvline(x=res_array[i,0], color='k', linestyle='dashed')
    grid.figure.axes[i*2].axvline(x=res_array[i,0]-res_array[i,1], color='k', linestyle='dashed')
    grid.figure.axes[i*2].axvline(x=res_array[i,0]+res_array[i,1], color='k', linestyle='dashed')
grid.figure.axes[0].set_title(parameters[0]+' = '+f'{res_array[0,0]:.2f}'+r'$\pm$'+f'{res_array[0,1]:.2f}')
grid.figure.axes[2].set_title(parameters[1]+' = '+f'{res_array[1,0]:.2f}'+r'$\pm$'+f'{res_array[1,1]:.2f}')
plt.tight_layout()
plt.savefig('../../Text/figures/mn_corner_relative.pdf')

plt.clf()
set_palette('flatui')   
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Charter"
plt.rcParams['font.size'] =  22
plt.rcParams['legend.fontsize'] = 22
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 5.0
plt.rcParams['xtick.minor.size'] = 3.0
plt.rcParams['ytick.major.size'] = 5.0
plt.rcParams['ytick.minor.size'] = 3.0
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['xtick.minor.visible'] = False
colorlist = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig,ax = plt.subplots(figsize=(6,9))
ax.errorbar(xplot, yplot, yerr=yplot_err,xerr=xplot_err, fmt='o', label='Observed abundance', color=colorlist[2], linewidth=2, markersize=10, lolims=lowlims, capthick=4)
ax.plot(np.arange(-2, 2, step=1), dust_init(np.arange(-2, 2, step=1),(res_array[0,0], res_array[1,0])), label=fr'Depletion pattern with $[M/H]$ = {res_array[0,0]:.2f} +- {res_array[0,1]:.2f} and $[Zn/Fe]$ = {res_array[1,0]:.2f} +- {res_array[1,1]:.2f}', color=colorlist[0])
for i, txt in enumerate(plotelements):
    if txt == 'SiII':
        ax.annotate(txt.replace('I',''), (xplot[i]-0.15, yplot[i]+0.02))
    elif txt == 'MgII':
        ax.annotate(txt.replace('I',''), (xplot[i]-0.21, yplot[i]+0.06))
    elif txt == 'SII':
        ax.annotate(txt.replace('I',''), (xplot[i]-0.09, yplot[i]+0.02))
    elif txt == 'FeII':
        ax.annotate(txt.replace('I',''), (xplot[i]+0.02, yplot[i]-0.06))
    else:
        ax.annotate(txt.replace('I',''), (xplot[i]+0.03, yplot[i]+0.02))
ax.set(xlim=(-1.75, 0.1), ylim=(-2.25,-1.30), xlabel='x', ylabel='y')
plt.tight_layout()
plt.savefig(f'../../Text/figures/DepletionFittingResults_relative.pdf')