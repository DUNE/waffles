import numba
import numpy as np
from typing import List

from waffles.Exceptions import generate_exception_message

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

def reference_to_minimum(input : List[int]) -> List[int]:

    """
    This function returns a list of integers, say output,
    so that output[i] is equal to input[i] minus the
    minimum value within input.

    Parameters
    ----------
    input : list of int

    Returns
    ----------
    list of int
    """

    aux = np.array(input)

    return list( aux - aux.min() )

@numba.njit(nogil=True, parallel=False)
def __cluster_integers_by_contiguity(increasingly_sorted_integers : np.ndarray) -> List[List[int]]:

    """
    This function is not intended for user usage. It 
    must only be called by the 
    cluster_integers_by_contiguity() function, where 
    some well-formedness checks have been already 
    perforemd. This is the low-level numba-optimized 
    implementation of the numerical process which is 
    time consuming.

    Parameters
    ----------
    increasingly_sorted_integers : np.ndarray
        An increasingly sorted numpy array of integers
        whose length is at least 2.

    Returns
    ----------
    extremals : list of list of int
        output[i] is a list containing two integers,
        so that output[i][0] (resp. output[i][1]) is
        the inclusive (resp. exclusive) lower (resp. 
        upper) bound of the i-th cluster of contiguous
        integers in the input array.
    """

    extremals = []
    extremals.append([increasingly_sorted_integers[0]])
    
    for i in range(1, len(increasingly_sorted_integers)-1):  # The last integer has an exclusive treatment

        if increasingly_sorted_integers[i] - increasingly_sorted_integers[i-1] != 1:    # We have stepped into a new cluster

            extremals[-1].append(increasingly_sorted_integers[i-1]+1)   # Add one to get the exclusive upper bound
            extremals.append([increasingly_sorted_integers[i]])

    if increasingly_sorted_integers[-1] - increasingly_sorted_integers[-2] != 1:  # Taking care of the last element of the given list

        extremals[-1].append(increasingly_sorted_integers[-2]+1)                                    # Add one to get the 
        extremals.append([increasingly_sorted_integers[-1], increasingly_sorted_integers[-1]+1])    # exclusive upper bound

    else:

        extremals[-1].append(increasingly_sorted_integers[-1]+1)

    return extremals

def cluster_integers_by_contiguity(increasingly_sorted_integers : np.ndarray) -> List[List[int]]:

    """
    This function gets an unidimensional numpy array of 
    integers, increasingly_sorted_integers, which 
    
        -   must contain at least two elements and
        -   must be strictly increasingly ordered, i.e.
            increasingly_sorted_integers[i] < increasingly_sorted_integers[i+1]
            for all i.

    The first requirement will be checked by this function,
    but it is the caller's responsibility to make sure that
    the second one is met. P.e. the output of 
    np.where(boolean_1d_array)[0], where boolean_1d_array 
    is an unidimensional boolean array, always meets the 
    second requirement.

    This function clusters the integers in such array by 
    contiguity. P.e. if increasingly_sorted_integers is
    array([1,2,3,5,6,8,10,11,12,13,16]), then this function 
    will return the following list: 
    '[[1,4],[5,7],[8,9],[10,14],[16,17]]'.
    
    Parameters
    ----------
    increasingly_sorted_integers : np.ndarray
        An increasingly sorted numpy array of integers
        whose length is at least 2.

    Returns
    ----------
    extremals : list of list of int
        output[i] is a list containing two integers,
        so that output[i][0] (resp. output[i][1]) is
        the inclusive (resp. exclusive) lower (resp. 
        upper) bound of the i-th cluster of contiguous
        integers in the input array.
    """

    if increasingly_sorted_integers.ndim != 1:
        raise Exception(generate_exception_message( 1,
                                                    'cluster_integers_by_contiguity()',
                                                    'The given numpy array must be unidimensional.'))
    if len(increasingly_sorted_integers) < 2:
        raise Exception(generate_exception_message( 2,
                                                    'cluster_integers_by_contiguity()',
                                                    'The given numpy array must contain at least two elements.'))
    
    return __cluster_integers_by_contiguity(increasingly_sorted_integers)

def fraction_is_well_formed(lower_limit : float = 0.0,
                            upper_limit : float = 1.0) -> bool:
    
    """
    This function returns True if 

        0.0 <= lower_limit < upper_limit <= 1.0,

    and False if else.

    Parameters
    ----------
    lower_limit : float
    upper_limit : float

    Returns
    ----------
    bool
    """

    if lower_limit < 0.0:
        return False
    elif upper_limit <= lower_limit:
        return False
    elif upper_limit > 1.0:
        return False
    
    return True