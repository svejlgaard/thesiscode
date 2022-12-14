import numpy as np
import os, sys
from uncertainties.core import ufloat


# SETS THE DIRECTORY TO THE CURRENT DIRECTORY
os.chdir(sys.path[0])


def logsum(xlist):
    """
    A function that calculates sum of a list, where each list element x is reported as log10(x) +/- err
    
    ~args~
    x: list, a list containing the elements to be summed

    ~returns~
    s: ufloat, the value and uncertainty of logarithmic sum

    """

    s = np.log10(np.sum([10**x.n for x in xlist]))
    err = np.sqrt(np.sum([x.s**2 for x in xlist])) /(len(xlist)-1)
    s = ufloat(s, err)
    return s

# SUMMING THE VOIGTFIT COMPONENTS
al = logsum([ufloat(13.21, 0.15), ufloat(13.35, 0.03), ufloat(12.24, 0.19), ufloat(12.80, 0.14)])
c = logsum([ufloat(15.68, 0.00), ufloat(15.05, 0.0), ufloat(14.08, 0.0), ufloat(14.22, 0.0), ufloat(14.52, 0.0), ufloat(15.07, 0.0),
            ufloat(13.24, 0.14), ufloat(13.48, 0.11)])
fe = logsum([ufloat(14.07, 0.12), ufloat(13.98, 0.06), ufloat(12.51, 0.13), ufloat(12.68, 0.2), ufloat(13.27, 0.07), ufloat(13.64, 0.08)])
mg = logsum([ufloat(14.31, 0.00), ufloat(15.02, 0.00), ufloat(13.42, 0.0), ufloat(13.19, 0.0), ufloat(13.43, 0.0), ufloat(12.79, 0.1)])
o = logsum([ufloat(15.23, 0.00), ufloat(15.32, 0.00), ufloat(14.04, 0.05), ufloat(13.93, 0.06), ufloat(13.91, 0.00), ufloat(14.95, 0.00), 
            ufloat(13.63, 0.13), ufloat(13.5, 0.3), ufloat(13.23, 0.22)] )
s = logsum([ufloat(14.15, 0.11), ufloat(14.24, 0.11), ufloat(13.67, 0.19)])
si = logsum([ufloat(14.37, 0.09), ufloat(14.50, 0.04), ufloat(13.39, 0.08), ufloat(13.09, 0.04), ufloat(13.72, 0.08), ufloat(14.22, 0.12),
            ufloat(12.76,0.06), ufloat(12.52,0.10), ufloat(11.96,0.07), ufloat(12.46,0.11)])

print(al, c, fe, mg, o, s, si)
print('\n')

# SUMMING THE COMPONENTS OF ONE OF THE TWO MAJOR SYSTEMS OF LINES
c1 = logsum([ufloat(15.68, 0.00), ufloat(15.05, 0.0), ufloat(14.08, 0.0)])
c2 = logsum([ufloat(14.22, 0.0), ufloat(14.52, 0.0), ufloat(15.07, 0.0), ufloat(13.24, 0.14), ufloat(13.48, 0.11)])
fe1 = logsum([ufloat(14.17, 0.12), ufloat(13.98, 0.06), ufloat(12.51, 0.13)])
fe2 = logsum([ufloat(12.68, 0.2), ufloat(13.27, 0.07), ufloat(13.64, 0.08)])
si1 = logsum([ufloat(14.37, 0.09), ufloat(14.50, 0.04), ufloat(13.39, 0.06), ufloat(12.76,0.06), ufloat(12.52,0.10)])
si2 = logsum([ufloat(13.09, 0.04), ufloat(13.72, 0.08), ufloat(14.22, 0.12), ufloat(11.96,0.07), ufloat(12.46,0.11)])

# COMPARING THE TWO SYSTEMS
print(si1/fe1)
print(si2/fe2)
print('\n')
print(c1/fe1)
print(c2/fe2)