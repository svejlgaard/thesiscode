import numpy as np
import os, sys


# Sets the directory to the current directory
os.chdir(sys.path[0])

# Constants
c1 = -4.959
c2 = 2.264
c3 = 0.389
c4 = 0.461
c5 = 5.9
gam = 1.
rv = 2.74
x0 = 4.6
m_e = 9.1095e-28
e = 4.8032e-10
c = 2.998e10
lamb = 1215.67
f = 0.416
gamma = 6.265e8
broad = 1
ra = 2.02*1e-8
zabs1 = 6.3168
zabs2 = 6.312
zgrb = np.max([zabs1, zabs2])
zu = 6.31028
zl = 6


def I(x):
    return x**(9/2)/(1-x) + 9/7 * x**(7/2) + 9/5 * x**(5/2) + 3*x**(3/2) + 9*x**(1/2) - 9/2 * np.log((1+x**(1/2))/(1-x**(1/2)))

def addIGM(wl_mod, hi):
    """
    A function that calculates the absorption from foreground source:
        wl_mod: list, wavelength values in units of Å
        hi: float, the neutral hydrogen fraction in the IGM
    Returns:
        exp(-tau): float, absorption fraction
    
    """
    tau_GP = 3.88 * 10**5 * ((1+zgrb)/7)**(3/2)
    # Optical depth
    tau = np.array([(hi * ra * tau_GP)/np.pi * ((wl_mod/lamb)/(1+zgrb))**(3/2) * ( I((1+zu)/(wl_mod/lamb)) - I((1+zl)/(wl_mod/lamb)) ) ], dtype=np.float128)
    return np.exp(-tau)[0]
