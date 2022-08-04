import pandas as pd
import os, sys
from astropy.stats import sigma_clip


# Sets the directory to the current directory
os.chdir(sys.path[0])

def s_clip(flux, s):
    
    clipped_flux = sigma_clip(flux, sigma=s)
    return clipped_flux 