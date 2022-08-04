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
zabs1 = 6.312
zabs2 = 6.318


def drude(x, x0, gam):
    return (x**2.)/(((x**2. - x0**2.)**2.) + (x**2.)*(gam**2))

def aLambda(l,av):
    """
    A function that calculates the dust-reddening from foreground source:
        l: list, wavelength values in units of Å
        av: float, the total dust extinction in magnitudes
    Returns:
        final: float, extinction fraction
    
    """
    x = 1 / (l * 0.1 * 0.001)
    k = np.zeros_like(x)
    D = drude(x, x0, gam)
    mask = (x <= c5)
    F = 0.539 * (x-5.9)**2 + 0.056*(x-5.9)**3
    k[mask] = c1 + c2*x[mask] + c3*D[mask] + c4 * F
    k[~mask] = c1 + c2*x[~mask] + c3*D[~mask]
    final = av/rv * (k + 1)
    return final
