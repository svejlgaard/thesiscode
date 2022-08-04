import os, sys
from uncertainties import ufloat
from astropy import constants as const

# SETS THE DIRECTORY TO THE CURRENT DIRECTORY
os.chdir(sys.path[0])

# DEFINING THE ZERO VELOCITY REDSHIFT
z_0 = 6.3168

def zconv(z, z_err):
    """
    A function that transforms points in redshift space into points in velocity space
    
    ~args~
    z: float, the value of the point in redshift space
    z_err: float, the uncertainty of the point in redshift space

    ~returns~
    v: ufloat, the value and uncertainty of the point in velocity space with v(z_0) = 0 km/s

    """

    diff = (z - z_0) / (z_0 + 1)
    c = const.c.to('km/s')
    c = c.value
    v = ufloat(c*diff, c*z_err / (z_0 + 1))
    return v

# USING THE FUNCTION TO TRANSLATE THE VOIGTFIT OUTPUT INTO VELOCITY SPACE
# FOR THE LOW IONISATION LINES
print(zconv(6.31102, 0.00003))
print(zconv(6.31236, 0.00004))
print(zconv(6.31415, 0.00007))
print(zconv(6.31682, 0.00007))
print(zconv(6.31862, 0.00011))
print(zconv(6.31931, 0.00004))

# FOR THE HIGH IONISATION LINES
print('\n')
print(zconv(6.3120, 0.0004))
print(zconv(6.31513, 0.00019))
print(zconv(6.3173, 0.0005))
print(zconv(6.31853, 0.00012))
