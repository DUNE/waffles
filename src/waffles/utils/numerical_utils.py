import numba
import numpy as np

def gaussian(   x : float, 
                scale : float, 
                mean : float, 
                std : float) -> float:

    """
    Evaluates an scaled gaussian function
    in x. The function is defined as:
    
    f(x) = scale * exp( -1 * (( x - mean ) / ( 2 * std )) ** 2)

    This function supports numpy arrays as input.
    
    Parameters
    ----------
    x : float
        The point at which the function is evaluated.
    scale : float
        The scale factor of the gaussian function
    mean : float
        The mean value of the gaussian function
    std : float
        The standard deviation of the gaussian function

    Returns
    -------
    float
        The value of the function at x
    """

    return scale * np.exp( -1. * (np.power( ( x - mean ) / ( 2 * std ), 2)))

@numba.njit(nogil=True, parallel=False)                 
def histogram2d(samples : np.ndarray, 
                bins : np.ndarray,                      # ~ 20 times faster than numpy.histogram2d
                ranges : np.ndarray) -> np.ndarray:     # for a dataset with ~1.8e+8 points

    """
    This function returns a bidimensional integer numpy 
    array which is the 2D histogram of the given samples.

    Parameters
    ----------
    samples : np.ndarray
        A 2xN numpy array where samples[0, i] (resp.
        samples[1, i]) gives, for the i-th point in the
        samples set, the value for the coordinate which 
        varies along the first (resp. second) axis of 
        the returned bidimensional matrix.
    bins : np.ndarray
        A 2x1 numpy array where bins[0] (resp. bins[1])
        gives the number of bins to be considered along
        the coordinate which varies along the first 
        (resp. second) axis of the returned bidimensional 
        matrix.
    ranges : np.ndarray
        A 2x2 numpy array where (ranges[0,0], ranges[0,1])
        (resp. (ranges[1,0], ranges[1,1])) gives the 
        range for the coordinate which varies along the 
        first (resp. second) axis of the returned 
        bidimensional. Any sample which falls outside 
        these ranges is ignored.

    Returns
    ----------
    result : np.ndarray
        A bidimensional integer numpy array which is the
        2D histogram of the given samples.
    """

    result = np.zeros((bins[0], bins[1]), dtype = np.uint64)

    inverse_step = 1. / ((ranges[:, 1] - ranges[:, 0]) / bins)

    for t in range(samples.shape[1]):

        i = (samples[0, t] - ranges[0, 0]) * inverse_step[0]
        j = (samples[1, t] - ranges[1, 0]) * inverse_step[1]

        if 0 <= i < bins[0] and 0 <= j < bins[1]:       # Using this condition is slightly faster than               
            result[int(i), int(j)] += 1                 # using four nested if-conditions (one for each        
                                                        # one of the four conditions). For a dataset with             
    return result                                       # 178993152 points, the average time (for 30        
                                                        # calls to this function) gave ~1.06 s vs ~1.22 s