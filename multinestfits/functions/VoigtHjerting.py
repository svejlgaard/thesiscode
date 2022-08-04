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

# The Voigt-Hjerting profile based on the numerical approximation by Garcia
def H(a,x):
    P = x**2
    H0 = np.exp(-x**2)
    Q = 1.5/x**2
    return H0 - a / np.sqrt(np.pi) /\
    P * ( H0 ** 2 * (4. * P**2 + 7. * P + 4. + Q) - Q - 1.0 )

def addAbs(wl_mod, t, zabs):
    """
    A function that calculates the absorption from foreground source:
        wl_mod: list, wavelength values in units of Å
        t: float, hydrogen column density in units of cm^{-2}
        zabs: float, redshift of absorption source
    Returns:
        exp(-tau): float, absorption fraction
    
    """
    C_a = np.sqrt(np.pi) * e**2 * f * lamb * 1E-8 / m_e / c / broad
    a = lamb * 1.E-8 * gamma / (4.*np.pi * broad)
    dl_D = broad/c * lamb
    x = (wl_mod/(zabs+1.0) - lamb)/dl_D+0.01

    # Optical depth
    tau = np.array([C_a * t * H(a,x)], dtype=np.float128)
    return np.exp(-tau)[0]