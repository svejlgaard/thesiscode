import numpy as np
import os, sys


# Sets the directory to the current directory
os.chdir(sys.path[0])

# Define the likelihood
def lnlike(theta, x, y, yerr, func):
    """
    A function that calculates the log-likelihood using input parameters:
        theta: list, parameter values given to the model function
        x: list, x-values used to calculate model y-values
        y: list, observed y-values
        yerr: list, observed error on y-values
        func: function, model function using theta and x to calculate model y-values
    Returns:
        LnLike: float, log-likelihood of given model and parameters
    
    """
    model = func(theta, x)
    sigma2 = yerr ** 2
    LnLike = -0.5 * np.sum((y - model) ** 2 / sigma2)
    return LnLike

# Define the probability function
def lnprob(theta, x, y, yerr, func, prior):
    """
    A function that calculates the log-likelihood using input parameters and prior distribution (see also 'lnlike'):
        theta: list, parameter values given to the model function
        x: list, x-values used to calculate model y-values
        y: list, observed y-values
        yerr: list, observed error on y-values
        func: function, model function using theta and x to calculate model y-values
        prior: function, prior function returning -np.inf if parameter values not accepted by prior, else returning 0
    Returns:
        lnlike: float, log-likelihood of given model and parameters
    
    """
    lp = prior(theta)
    if not lp == 0: #check if lp is infinite
        return -np.inf
    return lp + lnlike(theta, x, y, yerr, func)