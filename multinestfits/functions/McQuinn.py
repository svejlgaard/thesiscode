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
h0 = 71 * 100 * 1000
omega_m = 0.27
omega_l = 0.73
omega_k = 9.24e-5
zabs1 = 6.312
zabs2 = 6.3173
zgrb = np.max([zabs1, zabs2])
zgrb = 6.31028
zu = 6.31028
zl = 6


def addIGM(wl_mod, hi, rb):
    """
    A function that calculates the absorption from foreground source:
        wl_mod: list, wavelength values in units of Å
        hi: float, the neutral hydrogen fraction in the IGM
        rb: float, the comoving size of an isolated bubble in units of Mpc
    Returns:
        res: float, absorption fraction
    
    """
    f_mod = c / (wl_mod * 10**(-10) * 100)
    f_z = f_mod * (1+zgrb)
    f_a = c / (lamb * 10**(-10) * 100)
    tau_GP = 9 * 10**7 * hi * ((1+zgrb)/8)**(3/2) # cm/s
    hz = h0 * np.sqrt(omega_m * (1+zgrb)**3 + omega_k * (1+zgrb)**2 + omega_l) # cm/s/Mpc
    tau_rb = ((hz*rb / (1+zgrb)) - c*((f_z-f_a)/f_a))**(-1)
    # Optical depth
    tau = np.array([tau_GP*tau_rb], dtype=np.float128)[0]
    res = np.zeros(len(wl_mod))
    res[wl_mod > 8900] = np.exp(-tau[wl_mod > 8900])
    return res
