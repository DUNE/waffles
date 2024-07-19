import numpy as np
from scipy.special import erfc

def dec_gauss(f : np.ndarray,
              fc: int):
    '''
    Function to compute the Gaussian for the deconvolution Filter
    
    Parameter
    ---------
    f: array of frequencies
    fc: cutoff frequency

    Returns:
    ----------
    Gaussian 
    '''
    return np.exp(-0.5*(f/fc)**2)

def dec_fit_FastSlow(t     : np.ndarray,
                     As    : float,
                     taus  : float,
                     Af    : float,
                     tauf  : float,
                     sigma : float,
                     t0    : float,
                     offset: float):
    '''
    Method to compute the deconvolution of a bi-exponential function with a gaussain

    Parameter
    ---------
    t : time window
    A_: amplitude of the slow and fast component
    tau_: tau of the slow and fast component
    sigma: standar deviation of the gaussian
    t0: time where the gaussian is centered
    offset: offset of the signal for the case that the baseline is not centered in 0
    

    Returns:
    ----------
    Pulse shape function
    '''
    
    return (As / np.sqrt(2)) * np.exp((sigma**2) / (2 * taus**2)) * erfc(((t0 - t) / sigma) + (sigma / taus)) * np.exp((t0 - t) / taus) + \
           (Af / np.sqrt(2)) * np.exp((sigma**2) / (2 * tauf**2)) * erfc(((t0 - t) / sigma) + (sigma / tauf)) * np.exp((t0 - t) / tauf) - offset

def dec_fit_FastSlowIntermediate(t     : np.ndarray,
                                 As    : float,
                                 taus  : float,
                                 Ai    : float,
                                 taui  : float,
                                 Af    : float,
                                 tauf  : float,
                                 sigma : float,
                                 t0    : float,
                                 offset: float):
    '''
    Method to compute the deconvolution of a tri-exponential function with a gaussain

    Parameter
    ---------
    t : time window
    A_: amplitude of the slow, intermadiary and fast component
    tau_: tau of the slow, intermadiary and fast component
    sigma: standar deviation of the gaussian
    t0: time where the gaussian is centered
    offset: offset of the signal for the case that the baseline is not centered in 0
    

    Returns:
    ----------
    Pulse shape function
    '''
        
    return (As / np.sqrt(2)) * np.exp((sigma**2) / (2 * taus**2)) * erfc(((t0 - t) / sigma) + (sigma / taus)) * np.exp((t0 - t) / taus) + \
           (Ai / np.sqrt(2)) * np.exp((sigma**2) / (2 * taui**2)) * erfc(((t0 - t) / sigma) + (sigma / taui)) * np.exp((t0 - t) / taui) + \
           (Af / np.sqrt(2)) * np.exp((sigma**2) / (2 * tauf**2)) * erfc(((t0 - t) / sigma) + (sigma / tauf)) * np.exp((t0 - t) / tauf) - offset


