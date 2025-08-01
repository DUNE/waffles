import math

import numpy as np
from scipy import signal as spsi
from scipy import optimize as spopt
from typing import Tuple, Dict

from waffles.data_classes.CalibrationHistogram import CalibrationHistogram

import waffles.utils.numerical_utils as wun

from waffles.Exceptions import GenerateExceptionMessage

def trim_spsi_find_peaks_output_to_max_peaks(
    spsi_output: Tuple[np.ndarray, Dict[str, np.ndarray]],
    max_peaks: int
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """This function gets the output of a certain call to 
    scipy.signal.find_peaks() and returns the same output, 
    but truncated to the first max_peaks found peaks. 

    Parameters
    ----------
    spsi_output: tuple of ( np.ndarray,  dict, )
        The output of a call to scipy.signal.find_peaks().
        No checks are performed here regarding the
        well-formedness of this input.
    max_peaks: int
        The maximum number of peaks to keep. It must be a
        positive integer. This is not checked here. It is
        the caller's responsibility to ensure this. If the 
        given spsi_output has a number less or equal to 
        max_peaks peaks, then the output of this function 
        will be the same as the given spsi_output. If it
        has more than max_peaks peaks, then the output of
        this function will contain only the information
        regarding the first max_peaks peaks of the given 
        spsi_output.

    Returns
    -------
    output: tuple of ( np.ndarray, dict, )
        The first element is an unidimensional numpy array 
        which contains the first max_peaks elements of the 
        first element of the given spsi_output. The second 
        element is a dictionary which contains the same keys 
        as the second element of the given spsi_output, but 
        the values (which are unidimensional numpy arrays) 
        contain only the first max_peaks elements of the 
        values of the second element of the given spsi_output.
    """

    if len(spsi_output[0]) <= max_peaks:
        return spsi_output
    else:
        return (spsi_output[0][0:max_peaks], {key: value[0:max_peaks] for key, value in spsi_output[1].items()})


def __spot_first_peaks_in_CalibrationHistogram(
    calibration_histogram: CalibrationHistogram,
    max_peaks: int,
    prominence: float,
    initial_percentage: float = 0.1,
    percentage_step: float = 0.1,
    return_last_addition_if_fail: bool = False
) -> Tuple[bool, Tuple]:
    """This function is not intended for user usage. It 
    must be only called by fit_peaks_of_CalibrationHistogram(),
    where the well-formedness checks of the input
    parameters have been performed. This function tries 
    to find peaks over the signal which is computed as

        signal = (calibration_histogram.counts - np.min(calibration_histogram.counts))/np.max(calibration_histogram.counts)

    This function iteratively calls

        scipy.signal.find_peaks(signal[0:points], 
                                prominence = prominence)

    to spot, at most, max_peaks peaks. To do so, at the 
    first iteration, points is computed as 
    math.floor(initial_percentage * len(signal)). If the 
    number of spotted peaks is less than max_peaks, then 
    points is increased by 
    math.ceil(percentage_step * len(signal)) and the 
    scipy.signal.find_peaks() function is called again. 
    This process is repeated until the number of spotted peaks
    is equal to max_peaks, or until the number of points 
    reaches len(signal). If the number of points reaches 
    len(signal), then scipy.signal.find_peaks() is called 
    one last time as

        scipy.signal.find_peaks(signal, 
                                prominence = prominence)

    If the last call found a number of peaks smaller than
    max_peaks, then this function returns (False, peaks),
    where peaks is
    
        - the output of the last call to
        scipy.signal.find_peaks() if return_last_addition_if_fail
        is False

        - the output of the last call to
        scipy.signal.find_peaks() which added one peak
        with respect to the immediately previous
        call, if return_last_addition_if_fail is
        True.
        
    Otherwise, if the last call found a number of peaks
    greater than or equal to max_peaks, then the function
    returns (True, peaks), where peaks is the output of
    scipy.signal.find_peaks() but truncated to the first
    max_peaks found peaks.

    Parameters
    ----------
    calibration_histogram: CalibrationHistogram
        The CalibrationHistogram object to spot peaks on
    max_peaks: int
        The maximum number of peaks to spot. It must 
        be a positive integer. This is not checked here, 
        it is the caller's responsibility to ensure this.
    prominence: float
        The prominence parameter to pass to the
        scipy.signal.find_peaks() function. Since the
        signal is normalized, this prominence can be
        understood as a fraction of the total amplitude
        of the signal. P.e. setting prominence to 0.5,
        will prevent scipy.signal.find_peaks() from
        spotting peaks whose amplitude is less than
        half of the total amplitude of the signal.
    initial_percentage: float
        The initial percentage of the signal to
        consider. It must be greater than 0.0
        and smaller than 1.0.
    percentage_step: float
        The percentage step to increase the signal
        to consider in successive calls of 
        scipy.signal.find_peaks(). It must be greater 
        than 0.0 and smaller than 1.0.
    return_last_addition_if_fail: bool
        This parameter only makes a difference if the
        specified number of peaks (max_peaks) is not
        found not even in the last call to
        scipy.signal.find_peaks() which includes the
        whole calibration histogram. In that case, if
        this parameter is False, then the second
        output returned by this function is the output
        of the last call to scipy.signal.find_peaks().
        If this parameter is True, then the second
        output returned by this function is the output
        of the last call to scipy.signal.find_peaks()
        which added one peak with respect to the
        immediately previous call.

    Returns
    -------
    output: tuple of ( bool, tuple, )
        The first entry is a boolean which is True if 
        the number of peaks found is greater than or 
        equal to max_peaks, and False otherwise. If
        the first entry of the tuple is False, then
        the second entry is the output of a call to
        scipy.signal.find_peaks(), depending on the
        value given to return_last_addition_if_fail.
        If the first entry of the tuple is True, then
        the second entry is the output of
        scipy.signal.find_peaks() but truncated to
        the first max_peaks found peaks. For more
        information, check the
        scipy.signal.find_peaks() documentation.
    """

    signal = (
        calibration_histogram.counts - np.min(calibration_histogram.counts)
    ) / np.max(calibration_histogram.counts)

    # Some initializations for the loop
    fFoundMax = False
    fReachedEnd = False
    last_spotted_peaks = 0
    last_spsi_output_with_addition = None
    points = math.floor(initial_percentage * len(signal))

    while not fFoundMax and not fReachedEnd:

        points = min(points, len(signal))

        # Adding a minimal 0 width, which constraints nothing,
        # but which makes scipy.signal.find_peaks() return
        # information about each peak-width at half its height.

        spsi_output = spsi.find_peaks(
            signal[0:points],
            prominence=prominence,
            width=0,
            rel_height=0.5
        )

        if len(spsi_output[0]) > last_spotted_peaks:
            last_spsi_output_with_addition = spsi_output

        last_spotted_peaks = len(spsi_output[0])
        
        if last_spotted_peaks >= max_peaks:

            spsi_output = trim_spsi_find_peaks_output_to_max_peaks(
                spsi_output,
                max_peaks
            )
            fFoundMax = True

        if points == len(signal):
            fReachedEnd = True

        points += math.ceil(percentage_step * len(signal))

    if fFoundMax:
        return (True, spsi_output)
    else:
        if return_last_addition_if_fail and \
            last_spsi_output_with_addition is not None:
            return (False, last_spsi_output_with_addition)
        else:
            return (False, spsi_output)
    

def __fit_independent_gaussians_to_calibration_histogram(
    spsi_output: Tuple[np.ndarray, Dict[str, np.ndarray]], 
    calibration_histogram: CalibrationHistogram,
    half_points_to_fit: int
) -> bool:
    """This function gets the output of a certain call to 
    scipy.signal.find_peaks() and a CalibrationHistogram
    object, and tries to fit one independent gaussian to
    each one of the peaks spotted by scipy.signal.find_peaks()
    in the given calibration histogram. The number of
    points to fit around each peak is related to the
    half_points_to_fit parameter.

    Parameters
    ----------
    spsi_output: tuple of ( np.ndarray,  dict, )
        The output of a call to scipy.signal.find_peaks().
        No checks are performed here regarding the
        well-formedness of this input.
    calibration_histogram: CalibrationHistogram
        The CalibrationHistogram object to fit peaks on
    half_points_to_fit: int
        It must be a positive integer. This is not checked
        here. It is the caller's responsibility to check so.
        For each peak, it gives the number of points to
        consider on either side of the peak maximum, to
        fit each gaussian function. I.e. if i is the
        iterator value for calibration_histogram.counts of
        the i-th peak, then the histogram bins which will
        be considered for the fit are given by the slice
        calibration_histogram.counts[i - half_points_to_fit : i + half_points_to_fit + 1].

    Returns
    -------
    fFitAll: bool
        True if all of the given peaks could be fit. I.e.
        it may happen for some peaks that
        scipy.optimize.curve_fit() could not converge to
        a solution. If that is the case for at least one
        peak, then this function returns False.
    """

    # Number of peaks to try fit
    peaks_n_to_fit = len(spsi_output[0])

    # Whether all of the peaks could be fit
    fFitAll = True

    for i in range(peaks_n_to_fit):
            
        aux_idx  = spsi_output[0][i]

        aux_seeds = [
            # Scale seed for wun.gaussian()
            calibration_histogram.counts[aux_idx],
            # Mean seed for wun.gaussian()
            (calibration_histogram.edges[aux_idx] \
             + calibration_histogram.edges[aux_idx + 1]) / 2.,
            # Std seed for wun.gaussian(): Here we are
            # assuming that scipy.signal.find_peaks()
            # was given the width=0 and rel_height=0.5
            # keyword arguments. This means that
            # spsi_output should contain the widths of
            # the peaks, in number of samples, at half
            # of their height. 2.355 is approximately
            # the conversion factor between the standard
            # deviation and the FWHM. Also, note that here
            # we are assuming that the binning is uniform.
            spsi_output[1]['widths'][i] * calibration_histogram.mean_bin_width / 2.355
        ]

        # Restrict the fit lower limit to 0
        aux_lower_lim = max(
            0,
            aux_idx - half_points_to_fit
        )
        
        # The upper limit should be restricted to
        # len(calibration_histogram.counts). Making it
        # be further restricted to 
        # len(calibration_histogram.counts) - 1 so that
        # there is always available data to compute
        # the center of the bins, in the following line.
        aux_upper_lim = min(
            len(calibration_histogram.counts) - 1,
            aux_idx + half_points_to_fit + 1
        )
        
        aux_bin_centers = ( 
            calibration_histogram.edges[aux_lower_lim : aux_upper_lim] \
            + calibration_histogram.edges[
                aux_lower_lim + 1 : aux_upper_lim + 1] ) / 2.
        
        aux_counts = calibration_histogram.counts[
            aux_lower_lim : aux_upper_lim]

        try:
            aux_optimal_parameters, aux_covariance_matrix = spopt.curve_fit(
                wun.gaussian,
                aux_bin_centers,
                aux_counts,
                p0=aux_seeds
            )
            
        # Happens if scipy.optimize.curve_fit()
        # could not converge to a solution
        except RuntimeError:

            # In this case, we will skip this peak
            # so, in case fFitAll was True, it
            # should be set to False
            fFitAll = False
            continue    

        aux_errors = np.sqrt(np.diag(aux_covariance_matrix))

        calibration_histogram._CalibrationHistogram__add_gaussian_fit_parameters(   
            aux_optimal_parameters[0],
            aux_errors[0],
            aux_optimal_parameters[1],
            aux_errors[1],
            aux_optimal_parameters[2],
            aux_errors[2])

    return fFitAll


def __fit_correlated_gaussians_to_calibration_histogram(
    spsi_output: Tuple[np.ndarray, Dict[str, np.ndarray]], 
    calibration_histogram: CalibrationHistogram,
    # Data from runs (27905, 27906, 27907 and 27908
    # typically gave std_increment optimal values of
    # this order of magnitude
    std_increment_seed_fallback: float = 1e+2,
    ch_span_fraction_around_peaks: float = 0.05
) -> bool:
    """This function gets the output of a certain call to 
    scipy.signal.find_peaks() and a CalibrationHistogram
    object, and tries to fit
    wun.correlated_sum_of_gaussians() to the given
    calibration histogram. For two or more peaks to fit,
    the number of free parameters is four plus N, where
    N is the number of peaks to fit, which is given by
    the spsi_output parameter. The free parameters are
    the following:

        - N scaling factors,
        - the mean of the first gaussian,
        - the mean increment between adjacent gaussians,
        - the standard deviation of the first gaussian,
        - a parameter which correlates the standard
        deviations of adjacent gaussians, which we will
        refer to as the standard deviation increment.

    If trying to fit the given peaks fails, then
    the last peak is skipped and the fit is
    attempted again with the remaining peaks. This is
    repeated until a fit converges or until all of
    the peaks have been removed.

    Parameters
    ----------
    spsi_output: tuple of ( np.ndarray,  dict, )
        The output of a call to scipy.signal.find_peaks().
        This information is used to determine the number
        of peaks to fit and to provide a seed for the
        fit parameters. No checks are performed here
        regarding the well-formedness of this input. 
    calibration_histogram: CalibrationHistogram
        The CalibrationHistogram object to fit peaks on
    std_increment_seed_fallback: float
        The fallback value for the fit seed of the standard
        deviation increment. For cases when the number
        of peaks to fit is bigger than one: it is used
        when the peak finder (whose return value is used
        to calculate the seeds for each fitting parameter)
        predicts that the standard deviation of the second
        peak is less than the standard deviation of the
        first peak, which is incompatible with our fitting
        function. In this case, the standard deviation
        increment is set to this fallback value.
    ch_span_fraction_around_peaks: float
        It must belong to the (0.0, 1.0] interval. This
        parameter is used to determine the number of
        bins before (resp. after) the first (resp. last)
        considered peak to fit. P.e. setting it to 0.1
        means that the fit range will span from the
        i-th bin to the j-th bin, where i (resp. j)
        is computed as the index of the first (resp. last)
        considered peak minus (resp. plus) a number
        of bins which is equal to the 10% of the
        total of bins in the calibration histogram. 

    Returns
    -------
    fFitAll: bool
        True if all of the given peaks could be fit. I.e.
        if the fit was successful at the first attempt.
        It returns False otherwise.
    """

    if ch_span_fraction_around_peaks <= 0.0 or \
       ch_span_fraction_around_peaks > 1.0:
         
        raise Exception(
            GenerateExceptionMessage(
                1,
                '__fit_correlated_gaussians_to_calibration_histogram()',
                "The given ch_span_fraction_around_peaks "
                f"({ch_span_fraction_around_peaks}) must belong to the "
                "(0.0, 1.0] interval."
            )
        )
    
    points_to_fit_around_peaks = round(
        ch_span_fraction_around_peaks * len(calibration_histogram.counts)
    )

    # Number of peaks to try to fit on the first attempt
    max_peaks_n_to_fit = len(spsi_output[0])

    # Whether all of the peaks could be fit
    fFitAll = True

    fTryAgain = True
    peaks_n_to_fit = max_peaks_n_to_fit
    while fTryAgain:

        # first_peak_idx, mean_0_seed and std_0_seed are the same
        # for both cases (peaks_n_to_fit == 1 or peaks_n_to_fit >= 2)
        first_peak_idx = spsi_output[0][0]
        
        # mean_0 seed for fitting_function
        mean_0_seed = (calibration_histogram.edges[
            first_peak_idx
        ] + calibration_histogram.edges[
            first_peak_idx + 1
        ]) / 2.

        # std_0 seed for fitting_function: Here we are
        # assuming that scipy.signal.find_peaks()
        # was given the width=0 and rel_height=0.5
        # keyword arguments. This means that
        # spsi_output should contain the widths of
        # the peaks, in number of samples, at half
        # of their height. 2.355 is approximately
        # the conversion factor between the standard
        # deviation and the FWHM. Also, note that here
        # we are assuming that the binning is uniform.
        std_0_seed = spsi_output[1]['widths'][0] * \
            calibration_histogram.mean_bin_width / 2.355

        if peaks_n_to_fit >= 2:

            second_peak_idx = spsi_output[0][1]

            fitting_function = lambda \
                x, \
                mean_0, \
                mean_increment, \
                std_0, \
                std_increment, \
                *scaling_factors : wun.correlated_sum_of_gaussians(
                    x,
                    peaks_n_to_fit,
                    np.array(scaling_factors),
                    mean_0,
                    mean_increment,
                    std_0,
                    std_increment,
                )

            # Needed for the computation of the
            # mean_increment seed
            mean_1_seed = (calibration_histogram.edges[
                second_peak_idx
            ] + calibration_histogram.edges[
                second_peak_idx + 1
            ]) / 2.

            # mean_increment seed for fitting_function
            mean_increment_seed = mean_1_seed - mean_0_seed

            # Needed for the computation of the std_increment
            # seed
            std_1_seed = spsi_output[1]['widths'][1] * \
                calibration_histogram.mean_bin_width / 2.355

            # std_increment seed for fitting_function
            # Using std_i = ((std_0^2) + (i * (std_increment^2))) ** 0.5
            # for i=1
            if std_1_seed > std_0_seed:
                std_increment_seed = ((std_1_seed ** 2) - \
                    (std_0_seed ** 2)) ** 0.5
            else:
                std_increment_seed = std_increment_seed_fallback
            
            # scaling_factors seed for fitting_function
            scaling_factors_seed = [
                calibration_histogram.counts[idx] 
                for idx in spsi_output[0][0:peaks_n_to_fit]
            ]

            aux_seeds = [
                mean_0_seed,
                mean_increment_seed,
                std_0_seed,
                std_increment_seed,
                *scaling_factors_seed,
            ]
        
        else:

            fitting_function = lambda \
                x, \
                mean_0, \
                std_0, \
                scaling_0 : wun.correlated_sum_of_gaussians(
                    x,
                    1,
                    np.array((scaling_0,)),
                    mean_0,
                    0.,
                    std_0,
                    0.,
                )
            
            # scaling_factors seed for fitting_function
            scaling_0_seed = calibration_histogram.counts[
                first_peak_idx
            ]

            aux_seeds = [
                mean_0_seed,
                std_0_seed,
                scaling_0_seed,
            ]

        # Prevent the fit limits from going out of bounds
        # of the calibration_histogram arrays
        first_fitting_idx = round(
            max(
                0,
                spsi_output[0][0] - \
                    points_to_fit_around_peaks
            )
        )

        last_fitting_idx = round(
            min(
                len(calibration_histogram.edges) - 1,
                spsi_output[0][peaks_n_to_fit - 1] + \
                    points_to_fit_around_peaks
            )
        )

        fit_x = (calibration_histogram.edges[
            first_fitting_idx:last_fitting_idx
            ] + calibration_histogram.edges[
            first_fitting_idx + 1:last_fitting_idx + 1
            ]) / 2.

        fit_y = calibration_histogram.counts[
            first_fitting_idx:last_fitting_idx
        ]
        
        try:
            aux_optimal_parameters, aux_covariance_matrix = spopt.curve_fit(
                fitting_function,
                fit_x,
                fit_y,
                p0=aux_seeds
            )
            
        # Happens if scipy.optimize.curve_fit()
        # could not converge to a solution
        except RuntimeError:

            # In this case, we will skip the last
            # peak so, in case fFitAll was True,
            # it should be set to False
            fFitAll = False

            peaks_n_to_fit -= 1

            if peaks_n_to_fit == 0:
                fTryAgain = False
    
            continue

        # If no exception was raised, then the fit was successful,
        # so we can stop trying to fit the peaks
        fTryAgain = False

        aux_errors = np.sqrt(np.diag(aux_covariance_matrix))
        
        if peaks_n_to_fit >= 2:

            # Needed for later i-dependent computations
            optimal_std_0 = aux_optimal_parameters[2]

            error_std_0 = aux_errors[2]

            for i in range(peaks_n_to_fit):

                ith_optimal_scaling_factor = aux_optimal_parameters[4 + i]
                
                ith_scaling_factor_error = aux_errors[4 + i]
                
                ith_optimal_mean = aux_optimal_parameters[0] + \
                    (i * aux_optimal_parameters[1])

                # Error propagation of mean_i = mean_0 + (i * mean_increment)
                ith_mean_error = ((aux_errors[0] ** 2) + \
                    ((i * aux_errors[1]) ** 2)) ** 0.5 \
                
                ith_optimal_std = ((aux_optimal_parameters[2] ** 2) + \
                    (i * (aux_optimal_parameters[3] ** 2))) ** 0.5
                
                # Error propagation of std_i = ((std_0^2) + (i * (std_increment^2))) ** 0.5
                ith_std_error = (
                    (
                        ((optimal_std_0 * error_std_0) ** 2) + \
                        ((i*aux_optimal_parameters[3]*aux_errors[3]) ** 2)
                    ) ** 0.5) / ith_optimal_std

                calibration_histogram._CalibrationHistogram__add_gaussian_fit_parameters(   
                    ith_optimal_scaling_factor,
                    ith_scaling_factor_error,
                    ith_optimal_mean,
                    ith_mean_error,
                    ith_optimal_std,
                    ith_std_error
                )

        else:

            # Needed for later i-dependent computations
            optimal_std_0 = aux_optimal_parameters[1]

            error_std_0 = aux_errors[1]

            for i in range(peaks_n_to_fit):

                ith_optimal_scaling_factor = aux_optimal_parameters[2]
                
                ith_scaling_factor_error = aux_errors[2]
                
                ith_optimal_mean = aux_optimal_parameters[0]

                ith_mean_error = aux_errors[0]
                
                ith_optimal_std = aux_optimal_parameters[1]
                
                ith_std_error = aux_errors[1]

                calibration_histogram._CalibrationHistogram__add_gaussian_fit_parameters(   
                    ith_optimal_scaling_factor,
                    ith_scaling_factor_error,
                    ith_optimal_mean,
                    ith_mean_error,
                    ith_optimal_std,
                    ith_std_error
                )

    return fFitAll