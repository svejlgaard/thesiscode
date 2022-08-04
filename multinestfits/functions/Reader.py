import pandas as pd
import os, sys
import numpy as np
from uncertainties import ufloat


# Sets the directory to the current directory
os.chdir(sys.path[0])

def reader(filename, elements, obs_logNH):

    # Loading the fitted line column densities
    observations = np.loadtxt(f'{filename}.dat', delimiter='=', dtype=str).T
    observations = pd.DataFrame(data=observations[1:].T, index = [i.replace('logN(', '').replace(')', '').replace(' ', '') for i in observations[0]], columns=['logN', 'err_logN'])
    observations['logN'] = [ufloat(i,j) for i,j in zip(observations['logN'].to_numpy(dtype=float), observations['err_logN'].to_numpy(dtype=float))]
    observations = observations.drop(columns='err_logN')
    not_elements = [i for i in observations.index.to_numpy() if i not in elements]
    observations = observations.drop(index=not_elements)
    observations = observations.sort_index()


    # Loading the Lodders 2009 table values
    datafile = 'functions/Asplund2009.dat'

    dt = [('element', 'U2'), ('N', 'f4'), ('N_err', 'f4'), ('N_m', 'f4'), ('N_m_err', 'f4')]
    data = np.loadtxt(datafile, dtype=dt)

    fname = 'functions/Lodders2009.dat'
    Lodders2009 = np.loadtxt(fname, usecols=(1, 2), dtype=str)

    photosphere = dict()
    meteorite = dict()
    solar = dict()

    for element, N_phot, N_phot_err, N_met, N_met_err in data:
        photosphere[element] = [N_phot, N_phot_err]
        meteorite[element] = [N_met, N_met_err]
        idx = (Lodders2009 == element).nonzero()[0][0]
        typeN = Lodders2009[idx][1]
        if typeN == 's':
            solar[element] = [N_phot, N_phot_err]
        elif typeN == 'm':
            solar[element] = [N_met, N_met_err]
        elif typeN == 'a':
            # Calculate the weighted average
            this_N = np.array([N_phot, N_met])
            this_e = np.array([N_phot_err, N_met_err])
            w = 1./this_e**2
            N_avg = np.sum(w*this_N) / np.sum(w)
            N_err = np.round(1./np.sqrt(np.sum(w)), 3)
            solar[element] = [N_avg, N_err]

    sol_abundance = list()

    # Calculating relative abundances
    obs_metal = observations['logN']
    obs_abun = observations['logN'].copy()

    for n, name in enumerate(observations.index.to_numpy(dtype=str)):
        element = name.replace('I', '').replace('a','')
        N_solar, N_solar_err = solar[element]
        solar_abundance = ufloat(N_solar, N_solar_err)
        metal_array = obs_metal[n] - obs_logNH - (solar_abundance - 12.)
        obs_metal[n] = ufloat(metal_array.n, metal_array.s)
        sol_abundance.append(ufloat(solar_abundance.n, solar_abundance.s))

    # Saving the results
    sol_abundance = pd.DataFrame(data=sol_abundance, index=observations.index.to_numpy(dtype=str), columns=['(X/H)_o'])
    observations['[N/H]'] = obs_metal
    observations = observations.sort_index()

    # DeCia DTM method
    try:
        fe_zn = obs_metal['FeII'] - obs_metal['ZnII'] 
        dtm = 1 - 10**(fe_zn * (-0.95) / (-0.95 + 0.11))
    except:
        fe_zn = obs_metal['FeII'] - obs_metal['SiII'] 
        dtm = 1 - 10**(fe_zn * (-0.95) / (-0.95 + 0.26))
    dtm = dtm/0.89

    # Loading the depletion parameters
    dep_params = np.loadtxt('functions/LinearDepletionParameters.dat',dtype='str')
    dep_params = pd.DataFrame(data=dep_params[1:], columns=dep_params[0])
    dep_params.index = dep_params['X'].to_numpy(dtype=str)
    dep_params = dep_params.drop('X', axis=1)
    dep_params = dep_params.loc[[i.replace('I','').replace('a','') for i in elements]]
    dep_params = dep_params.sort_index()

    
    observations = observations.reindex(index = elements)
    obs_abun = obs_abun.reindex(index = elements)
    dep_params = dep_params.reindex(index=[i.replace('I','').replace('a','') for i in elements])
    sol_abundance = sol_abundance.reindex(index = elements)

    return observations, obs_abun, dep_params, sol_abundance, dtm, fe_zn