import pandas as pd
import numpy as np
import os, sys


# Sets the directory to the current directory
os.chdir(sys.path[0])

def reader(filename, ending):
    """
    A function that loads the data into python from the dat-format:
        filename: str, name of the GRB
        ending: str, either binned or combined
    Returns:
        data: pandas DataFrame, array-like format of the data, shaped n_datapoints x n_filters
    
    """
    data_list = list()
    for filt in ['UVB', 'VIS', 'NIR']:
        data = pd.read_csv(f'../../{filename}/data/'+filename+f'_{filt}_{ending}.dat', sep=' ')
        if filt == 'NIR':
            dat = data['#WAVE_BIN'].to_numpy()
            dat_filt = dat>1020
            data = pd.DataFrame(data=np.array([dat[dat_filt], data['FLUX_BIN'].to_numpy()[dat_filt], data['ERR_FLUX_BIN'].to_numpy()[dat_filt]]).T, columns=['#WAVE_BIN', 'FLUX_BIN', 'ERR_FLUX_BIN'])
        data_list.append(data)
    data = pd.concat(data_list)
    data.columns = [col.replace('#', '') for col in data.columns]
    return data