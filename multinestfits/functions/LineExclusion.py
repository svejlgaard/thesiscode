import pandas as pd
import numpy as np
import os, sys

# Sets the directory to the current directory
os.chdir(sys.path[0])

def excl(wave, flux, flux_err,z,sig):
    lines = pd.read_csv('functions/linelist.dat')
    abs_waves = lines['Value'].to_numpy(dtype=float)
    names = lines['Name'].to_numpy(dtype=str)

    wavetops = (abs_waves+sig*lines['EW'].to_numpy(dtype=float)+sig*lines['EW_err'].to_numpy(dtype=float))*(z+1) 
    wavebots = (abs_waves-sig*lines['EW'].to_numpy(dtype=float)-sig*lines['EW_err'].to_numpy(dtype=float))*(z+1) 
    filt = np.ones_like(wave)
    for w, top in enumerate(wavetops):
        for wav in wave[wave<top]:
                if wav > wavebots[w]:
                        index = np.where(wave == wav)[0][0]
                        filt[index] = 0


    filt = np.array(filt, dtype=bool)
    return wave[filt], flux[filt], flux_err[filt]