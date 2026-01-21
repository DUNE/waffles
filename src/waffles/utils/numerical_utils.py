import numba
import numpy as np
from typing import List, Tuple
from tqdm import tqdm

from waffles.data_classes.WaveformSet import WaveformSet
from waffles.Exceptions import GenerateExceptionMessage
from waffles.np02_utils.AutoMap import dict_uniqch_to_module


def gaussian(
        x: float,
        scale: float,
        mean: float,
        std: float
) -> float:
    """Evaluates an scaled gaussian function
    in x. The function is defined as:

    f(x) = scale * exp(-0.5 * ((x - mean) / std) ** 2)

    This function supports numpy arrays as input.

    Parameters
    ----------
    x: float
        The point at which the function is evaluated.
    scale: float
        The scale factor of the gaussian function
    mean: float
        The mean value of the gaussian function
    std: float
        The standard deviation of the gaussian function

    Returns
    -------
    float
        The value of the function at x
    """

    return scale * np.exp(-0.5 * (np.power((x - mean) / std, 2)))

def multigaussfit(x, *params):
    """Multivariate Gaussian function for fitting.
    Parameters
    ----------
    x : array_like
        The input data points.
    params : tuple
    n_peaks : int
        The number of peaks to fit.

    Returns
    -------
    output : array_like
        The evaluated multivariate Gaussian function.
    """
    
    output = np.zeros_like(x)
    
    n_peaks = (len(params)-4)
    baseline_amplitude = abs(params[0])
    baseline_mean = params[1]
    baseline_std = params[2]
    output += gaussian(x, baseline_amplitude, baseline_mean, baseline_std)

    gain = params[3]
    std_prop = params[4]

    for i in range(1, n_peaks):
        gaus_amplitude = abs(params[i + 4])
        gaus_mean = i*gain + baseline_mean
        gaus_std = np.sqrt( baseline_std ** 2 + std_prop ** 2 * i )
        output += gaussian(x, gaus_amplitude, gaus_mean, gaus_std)

    return output

def multigaussplot(x, *params):
    """Multivariate Gaussian function for plotting.
    Parameters
    ----------
    x : array_like
        The input data points.
    params : array with values of scale, mean and std
        The parameters of the Gaussian function.
    Returns
    -------
    output : array_like
        The evaluated multivariate Gaussian function.
    """
    output = np.zeros_like(x)
    for i in range(len(params)//3):
        output += gaussian(
            x,
            params[i * 3],  # scale
            params[i * 3 + 1],  # mean
            params[i * 3 + 2]  # std
        )
    return output




def correlated_sum_of_gaussians(
    x: float,
    gaussians_num: int,
    scaling_factors: np.ndarray,
    mean_0: float,
    mean_increment: float,
    std_0: float,
    std_increment: float,
) -> float:
    """Evaluates a correlated sum of gaussians
    in x. The function is defined as:

    f(x) = sum_{i=0}^{gaussians_num - 1} \
        gaussian(
            x,
            scaling_factors[i],
            mean_0 + (i * mean_increment),
            ((std_0 ** 2) + (i * (std_increment ** 2))) ** 0.5
        )

    It is the caller's responsibility to make sure
    that the input parameters are well-formed. No
    checks are done here.

    Parameters
    ----------
    x: float
        The point at which the function is evaluated
    gaussians_num: int
        The number of gaussians to be summed
    scaling_factors: np.ndarray
        A 1D numpy array of floats, where
        scaling_factors[i] gives the scale factor
        of the i-th gaussian function in the sum.
    mean_0: float
        The mean value of the first gaussian function
        in the sum
    mean_increment: float
        The increment in the mean value of each
        gaussian function in the sum with respect
        to the previous one
    std_0: float
        The standard deviation of the first gaussian
        function in the sum
    std_increment: float
        The i-th gaussian function in the sum
        has a standard deviation equal to
        ((std_0 ** 2) + (i * (std_increment ** 2))) ** 0.5.

    Returns
    -------
    float
        The value of the function at x
    """

    result = 0.

    for i in range(gaussians_num):
        result += gaussian(
            x,
            scaling_factors[i],
            mean_0 + (i * mean_increment),
            ((std_0 ** 2) + (i * (std_increment ** 2))) ** 0.5
        )

    return result


def correlated_sum_of_gaussians(
    x: float,
    gaussians_num: int,
    scaling_factors: np.ndarray,
    mean_0: float,
    mean_increment: float,
    std_0: float,
    std_increment: float,
) -> float:
    """Evaluates a correlated sum of gaussians
    in x. The function is defined as:

    f(x) = sum_{i=0}^{gaussians_num - 1} \
        gaussian(
            x,
            scaling_factors[i],
            mean_0 + (i * mean_increment),
            ((std_0 ** 2) + (i * (std_increment ** 2))) ** 0.5
        )

    It is the caller's responsibility to make sure
    that the input parameters are well-formed. No
    checks are done here.

    Parameters
    ----------
    x: float
        The point at which the function is evaluated
    gaussians_num: int
        The number of gaussians to be summed
    scaling_factors: np.ndarray
        A 1D numpy array of floats, where
        scaling_factors[i] gives the scale factor
        of the i-th gaussian function in the sum.
    mean_0: float
        The mean value of the first gaussian function
        in the sum
    mean_increment: float
        The increment in the mean value of each
        gaussian function in the sum with respect
        to the previous one
    std_0: float
        The standard deviation of the first gaussian
        function in the sum
    std_increment: float
        The i-th gaussian function in the sum
        has a standard deviation equal to
        ((std_0 ** 2) + (i * (std_increment ** 2))) ** 0.5.

    Returns
    -------
    float
        The value of the function at x
    """

    result = 0.

    for i in range(gaussians_num):
        result += gaussian(
            x,
            scaling_factors[i],
            mean_0 + (i * mean_increment),
            ((std_0 ** 2) + (i * (std_increment ** 2))) ** 0.5
        )

    return result


@numba.njit(nogil=True, parallel=False)
def __histogram1d(
    samples: np.ndarray,
    bins: int,
    # Not calling it 'range' because
    # it is a reserved keyword in Python
    domain: np.ndarray,
    keep_track_of_idcs: bool = False
) -> Tuple[np.ndarray, List[List[int]]]:
    """This function is not intended for user usage. 
    It must only be called by the histogram1d() 
    function. This is the low-level optimized 
    numerical implementation of the histogramming 
    process.

    Parameters
    ----------
    samples: np.ndarray
    bins: int
    domain: np.ndarray
    keep_track_of_idcs: bool

    Returns
    ----------
    counts: np.ndarray
        An unidimensional integer numpy array which 
        is the 1D histogram of the given samples
    formatted_idcs: list of int
        A list of integers. If keep_track_of_idcs is
        False, then this list is empty. If it is True,
        then this list is 2*len(samples) long at most.
        This list is such that formatted_idcs[2*i]
        gives the index of the i-th sample from
        samples which actually fell into the 
        specified domain, and formatted_idcs[2*i + 1] 
        gives the index of the bin where such sample 
        falls into.
    """

    # Of course, building a list of lists with the indices of the
    # samples which fell into each bin would be more conceptually
    # stragihtforward, but appending to a list which is contained
    # within another list is apparently not allowed within a numba.njit
    # function. So, we are using this format, which only implies
    # appending to a flat list, and it will be latter de-formatted
    # into a list of lists in the upper level histogram1d() function,
    # which is not numba decorated and should not perform very
    # demanding operations.

    counts = np.zeros(bins, dtype=np.uint64)
    formatted_idcs = []

    inverse_step = 1. / ((domain[1] - domain[0]) / bins)

    if not keep_track_of_idcs:
        for t in range(samples.shape[0]):
            i = (samples[t] - domain[0]) * inverse_step

            if 0 <= i < bins:
                counts[int(i)] += 1
    else:
        for t in range(samples.shape[0]):
            i = (samples[t] - domain[0]) * inverse_step

            if 0 <= i < bins:
                aux = int(i)
                counts[aux] += 1

                formatted_idcs.append(t)
                formatted_idcs.append(aux)

    return counts, formatted_idcs


def histogram1d(
    samples: np.ndarray,
    bins: int,
    # Not calling it 'range' because
    # it is a reserved keyword in Python
    domain: np.ndarray,
    keep_track_of_idcs: bool = False
) -> Tuple[np.ndarray, List[List[int]]]:
    """This function returns a tuple with two elements. The
    first one is an unidimensional integer numpy 
    array, say counts, which is the 1D histogram of the 
    given samples. I.e. counts[i] gives the number of
    samples that fall into the i-th bin of the histogram,
    with i = 0, 1, ..., bins - 1. The second element
    of the returned tuple is a list containing bins 
    empty lists. If keep_track_of_idcs is True, then 
    the returned list of lists contains integers, so 
    that the i-th list contains the indices of the 
    samples which fall into the i-th bin of the 
    histogram. It is the caller's responsibility to 
    make sure that the given input parameters are 
    well-formed. No checks are performed here.

    Parameters
    ----------
    samples: np.ndarray
        An unidimensional numpy array where samples[i] 
        gives the i-th sample.
    bins: int
        The number of bins
    domain: np.ndarray
        A 2x1 numpy array where (domain[0], domain[1])
        gives the range to consider for the histogram.
        Any sample which falls outside this range is 
        ignored.
    keep_track_of_idcs: bool
        If True, then the second element of the returned
        tuple is not empty

    Returns
    ----------
    counts: np.ndarray
        An unidimensional integer numpy array which 
        is the 1D histogram of the given samples
    idcs: list of list of int
        A list containing bins empty lists. If 
        keep_track_of_idcs is True, then the i-th 
        list contains the indices of the samples,
        with respect to the input samples array,
        which fall into the i-th bin of the histogram.
    """

    counts, formatted_idcs = __histogram1d(
        samples,
        bins,
        domain,
        keep_track_of_idcs=keep_track_of_idcs)

    deformatted_idcs = [[] for _ in range(bins)]

    if keep_track_of_idcs:
        for i in range(0, len(formatted_idcs), 2):
            deformatted_idcs[formatted_idcs[i + 1]].append(formatted_idcs[i])

    return counts, deformatted_idcs


@numba.njit(nogil=True, parallel=False)
def histogram2d(
        samples: np.ndarray,
        bins: np.ndarray,
        ranges: np.ndarray
) -> np.ndarray:
    # ~ 20 times faster than numpy.histogram2d
    # # for a dataset with ~1.8e+8 points
    """This function returns a bidimensional integer numpy 
    array which is the 2D histogram of the given samples.

    Parameters
    ----------
    samples: np.ndarray
        A 2xN numpy array where samples[0, i] (resp.
        samples[1, i]) gives, for the i-th point in the
        samples set, the value for the coordinate which 
        varies along the first (resp. second) axis of 
        the returned bidimensional matrix.
    bins: np.ndarray
        A 2x1 numpy array where bins[0] (resp. bins[1])
        gives the number of bins to be considered along
        the coordinate which varies along the first 
        (resp. second) axis of the returned bidimensional 
        matrix.
    ranges: np.ndarray
        A 2x2 numpy array where (ranges[0,0], ranges[0,1])
        (resp. (ranges[1,0], ranges[1,1])) gives the 
        range for the coordinate which varies along the 
        first (resp. second) axis of the returned 
        bidimensional. Any sample which falls outside 
        these ranges is ignored.

    Returns
    ----------
    result: np.ndarray
        A bidimensional integer numpy array which is the
        2D histogram of the given samples.
    """

    result = np.zeros((bins[0], bins[1]), dtype=np.uint64)

    inverse_step = 1. / ((ranges[:, 1] - ranges[:, 0]) / bins)

    for t in range(samples.shape[1]):

        i = (samples[0, t] - ranges[0, 0]) * inverse_step[0]
        j = (samples[1, t] - ranges[1, 0]) * inverse_step[1]

        # Using this condition is slightly faster than
        # using four nested if-conditions. For a dataset 
        # with 178993152 points, the average time (for 30
        # calls to this function) gave ~1.06 s vs ~1.22 s
        if 0 <= i < bins[0] and 0 <= j < bins[1]:
            result[int(i), int(j)] += 1

    return result


def reference_to_minimum(input: List[int]) -> List[int]:
    """This function returns a list of integers, say output,
    so that output[i] is equal to input[i] minus the
    minimum value within input.

    Parameters
    ----------
    input: list of int

    Returns
    ----------
    list of int
    """

    aux = np.array(input)

    return list(aux - aux.min())


@numba.njit(nogil=True, parallel=False)
def __cluster_integers_by_contiguity(
        increasingly_sorted_integers: np.ndarray
) -> List[List[int]]:
    """This function is not intended for user usage.
    It must only be called by the
    cluster_integers_by_contiguity() function, where
    some well-formedness checks have been already
    perforemd. This is the low-level numba-optimized
    implementation of the numerical process which is
    time consuming.

    Parameters
    ----------
    increasingly_sorted_integers: np.ndarray
        An increasingly sorted numpy array of integers
        whose length is at least 2.

    Returns
    ----------
    extremals: list of list of int
        output[i] is a list containing two integers,
        so that output[i][0] (resp. output[i][1]) is
        the inclusive (resp. exclusive) lower (resp. 
        upper) bound of the i-th cluster of contiguous
        integers in the input array.
    """

    extremals = []
    extremals.append([increasingly_sorted_integers[0]])

    # The last integer has an exclusive treatment
    for i in range(1, len(increasingly_sorted_integers) - 1):

        # We have stepped into a new cluster
        if increasingly_sorted_integers[i] - increasingly_sorted_integers[i - 1] != 1:

            # Add one to get the exclusive upper bound
            extremals[-1].append(increasingly_sorted_integers[i - 1] + 1)
            extremals.append([increasingly_sorted_integers[i]])

    # Taking care of the last element of the given list
    if increasingly_sorted_integers[-1] - increasingly_sorted_integers[-2] != 1:

        # Add one to get the exclusive upper bound
        extremals[-1].append(increasingly_sorted_integers[-2] + 1)
        
        extremals.append(
            [increasingly_sorted_integers[-1],
             increasingly_sorted_integers[-1] + 1])

    else:

        extremals[-1].append(
            increasingly_sorted_integers[-1] + 1)

    return extremals


def cluster_integers_by_contiguity(
        increasingly_sorted_integers: np.ndarray
) -> List[List[int]]:
    """This function gets an unidimensional numpy array of 
    integers, increasingly_sorted_integers, which 

        -   must contain at least two elements and
        -   must be strictly increasingly ordered, i.e.
            increasingly_sorted_integers[i] < increasingly_sorted_integers[i + 1]
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
    increasingly_sorted_integers: np.ndarray
        An increasingly sorted numpy array of integers
        whose length is at least 2.

    Returns
    ----------
    extremals: list of list of int
        output[i] is a list containing two integers,
        so that output[i][0] (resp. output[i][1]) is
        the inclusive (resp. exclusive) lower (resp. 
        upper) bound of the i-th cluster of contiguous
        integers in the input array.
    """

    if increasingly_sorted_integers.ndim != 1:
        raise Exception(GenerateExceptionMessage(
            1,
            'cluster_integers_by_contiguity()',
            'The given numpy array must be unidimensional.'))
    if len(increasingly_sorted_integers) < 2:
        raise Exception(GenerateExceptionMessage(
            2,
            'cluster_integers_by_contiguity()',
            'The given numpy array must contain at least two elements.'))

    return __cluster_integers_by_contiguity(increasingly_sorted_integers)


def average_wf_ch(wfch: WaveformSet, analysis_label="std"):
    """
    Compute the average waveform for a single channel after baseline subtraction.

    This function loops over all waveforms stored in a `WaveformSet`, subtracts
    the baseline computed in a previous analysis step, and returns the sample-wise
    average waveform.

    Parameters
    ----------
    wfch : WaveformSet
        WaveformSet object containing the waveforms to be averaged.
    analysis_label : str, optional
        Label of the analysis from which the baseline value is retrieved
        (default is "std"). 

    Returns
    -------
    average_waveform : ndarray
        One-dimensional NumPy array containing the average waveform (in ADC units)
        after baseline subtraction.

    Example
    --------
    avg_wf = average_wf_ch(wfch, analysis_label="std")
    
    """
    
    arrs = []
    for run in wfch.runs: 
        available_endpoints_and_channels = wfch.available_channels[run]
        if len(list(available_endpoints_and_channels.keys())) > 1:
            raise Exception("WaveformSet must contain exactly one endpoint and one channel.")
        for ep, channels in available_endpoints_and_channels.items():
            if len(list(channels)) > 1:
                raise Exception("WaveformSet must contain exactly one endpoint and one channel.")

    for wf in tqdm(wfch.waveforms):
        adcs_float = np.asarray(wf.adcs).astype(float)          
        if analysis_label in wf.analyses:
            baseline = wf.analyses[analysis_label].result["baseline"]
            adcs_float = adcs_float - baseline                
            arrs.append(adcs_float)
            
    return np.mean(arrs, axis=0)


def compute_peaks_rise_fall_ch(g: "ChannelWsGrid"):

    """
    Compute peak characteristics and rise/fall times for each channel in a ChannelWsGrid.

    For each valid channel in the provided grid, this function:
      - Computes the average waveform.
      - Finds the peak value and its index.
      - Calculates the rise time (10% → 90% of peak) and fall time (90% → 10% of peak).
      - Stores intermediate times corresponding to 10% and 90% amplitudes for rise and fall.

    Parameters
    ----------
    g : ChannelWsGrid
        A grid object containing waveform sets for multiple channels. 
        Channels not present in `dict_uniqch_to_module` or missing waveforms are skipped.

    Returns
    -------
    peaks_all : dict
        Dictionary with keys `(endpoint, channel)` and values as another dict containing:
            - "peak_index": int, index of the peak in the averaged waveform
            - "peak_time": int, time tick of the peak
            - "peak_value": float, value of the peak
            - "rise_time": float, duration from 10% to 90% of peak
            - "fall_time": float, duration from 90% to 10% of peak
            - "t_low": int, time of 10% of peak during rise
            - "t_high": int, time of 90% of peak during rise
            - "t_high_fall": int, time of 90% of peak during fall
            - "t_low_fall": int, time of 10% of peak during fall
            - "time": np.ndarray, array of time indices
            - "avg": np.ndarray, averaged waveform

    """

    peaks_all = {}

    for (row, col), uch in np.ndenumerate(g.ch_map.data):
        if str(uch) not in dict_uniqch_to_module:
            continue
        if uch.channel not in g.ch_wf_sets[uch.endpoint]:
            continue

        wfch = g.ch_wf_sets[uch.endpoint][uch.channel]
        avg = average_wf_ch(wfch)
        time = np.arange(avg.size)

        peak_idx = np.argmax(avg)
        peak_value = avg[peak_idx]
        peak_time = time[peak_idx]

        amp_10 = 0.1 * peak_value
        amp_90 = 0.9 * peak_value

        t_low = None
        t_high = None
        for j in range(peak_idx + 1):
            if t_low is None and avg[j] >= amp_10:
                t_low = time[j]
            if t_high is None and avg[j] >= amp_90:
                t_high = time[j]
                break
        rise_time = t_high - t_low

        t_high_fall = None
        t_low_fall = None
        for j in range(peak_idx, len(avg)):
            if t_high_fall is None and avg[j] <= amp_90:
                t_high_fall = time[j]
            if t_low_fall is None and avg[j] <= amp_10:
                t_low_fall = time[j]
                break
        fall_time = t_low_fall - t_high_fall

        peaks_all[(uch.endpoint, uch.channel)] = {
            "peak_index": peak_idx,
            "peak_time": peak_time,
            "peak_value": peak_value,
            "rise_time": rise_time,
            "fall_time": fall_time,
            "t_low": t_low,
            "t_high": t_high,
            "t_high_fall": t_high_fall,
            "t_low_fall": t_low_fall,
            "time": time,   
            "avg": avg      
        }

    return peaks_all
