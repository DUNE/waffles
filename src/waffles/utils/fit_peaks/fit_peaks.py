import numpy as np
from scipy import optimize as spopt

from waffles.data_classes.CalibrationHistogram import CalibrationHistogram
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid

import waffles.utils.numerical_utils as wun
import waffles.utils.fit_peaks.fit_peaks_utils as wuff

from waffles.Exceptions import GenerateExceptionMessage

def fit_peaks_of_CalibrationHistogram(
    calibration_histogram: CalibrationHistogram,
    max_peaks: int,
    prominence: float,
    half_points_to_fit: int,
    initial_percentage = 0.1,
    percentage_step = 0.1
) -> bool:
    """For the given CalibrationHistogram object, 
    calibration_histogram, this function
        
        -   tries to find the first max_peaks whose 
            prominence is greater than the given prominence 
            parameter, using the scipy.signal.find_peaks() 
            function iteratively. This function delegates 
            this task to the
            wuff.__spot_first_peaks_in_CalibrationHistogram()
            function.

        -   Then, it fits a gaussian function to each one
            of the found peaks using the output of 
            the last call to scipy.signal.find_peaks()
            (which is returned by 
            wuff.__spot_first_peaks_in_CalibrationHistogram())
            as a seed for the fit.

        -   Finally, it stores the fit parameters in the
            gaussian_fits_parameters attribute of the given
            CalibrationHistogram object, according to its 
            structure, which can be found in the CalibrationHistogram
            class documentation.

    This function returns True if the number of found peaks
    matches the given max_peaks parameter, and False
    if it is smaller than max_peaks.
    
    Parameters
    ----------
    calibration_histogram: CalibrationHistogram
        The CalibrationHistogram object to fit peaks on
    max_peaks: int
        It must be a positive integer. It gives the
        maximum number of peaks that could be possibly
        fit. This parameter is passed to the 'max_peaks'
        parameter of the 
        wuff.__spot_first_peaks_in_CalibrationHistogram()
        function.
    prominence: float
        It must be greater than 0.0 and smaller than 1.0.
        It gives the minimal prominence of the peaks to 
        spot. This parameter is passed to the 'prominence' 
        parameter of the 
        wuff.__spot_first_peaks_in_CalibrationHistogram()
        function, where it is interpreted as the fraction 
        of the total amplitude of the histogram which is 
        required for a peak to be spotted as such. P.e. 
        setting prominence to 0.5, will prevent 
        scipy.signal.find_peaks() from spotting peaks 
        whose amplitude is less than half of the total 
        amplitude of the histogram.
    half_points_to_fit: int
        It must be a positive integer. For each peak, it 
        gives the number of points to consider on either 
        side of the peak maximum, to fit each gaussian 
        function. I.e. if i is the iterator value for
        calibration_histogram.counts of the i-th peak, 
        then the histogram bins which will be considered 
        for the fit are given by the slice 
        calibration_histogram.counts[i - half_points_to_fit : i + half_points_to_fit + 1].
    initial_percentage: float
        It must be greater than 0.0 and smaller than 1.0.
        This parameter is passed to the 'initial_percentage' 
        parameter of the 
        wuff.__spot_first_peaks_in_CalibrationHistogram()
        function. For more information, check the 
        documentation of such function.
    percentage_step: float
        It must be greater than 0.0 and smaller than 1.0.
        This parameter is passed to the 'percentage_step' 
        parameter of the 
        wuff.__spot_first_peaks_in_CalibrationHistogram() 
        function. For more information, check the 
        documentation of such function.

    Returns
    -------
    bool
        True if the number of found peaks matches the given
        max_peaks parameter, and False if it is smaller than
        max_peaks.
    """

    if max_peaks < 1:
        raise Exception(GenerateExceptionMessage(
            1,
            'fit_peaks_of_CalibrationHistogram()',
            f"The given max_peaks ({max_peaks}) "
            "must be greater than 0."))
    
    if prominence <= 0.0 or prominence >= 1.0:
        raise Exception(GenerateExceptionMessage( 
            2,
            'fit_peaks_of_CalibrationHistogram()',
            f"The given prominence ({prominence}) "
            "must be greater than 0.0 and smaller than 1.0."))
    
    if initial_percentage <= 0.0 or initial_percentage >= 1.0:
        raise Exception(GenerateExceptionMessage( 
            3,
            'fit_peaks_of_CalibrationHistogram()',
            f"The given initial_percentage ({initial_percentage})"
            " must be greater than 0.0 and smaller than 1.0."))

    if percentage_step <= 0.0 or percentage_step >= 1.0:
        raise Exception(GenerateExceptionMessage(
            4,
            'fit_peaks_of_CalibrationHistogram()',
            f"The given percentage_step ({percentage_step})"
            " must be greater than 0.0 and smaller than 1.0."))
    
    calibration_histogram._CalibrationHistogram__reset_gaussian_fit_parameters()

    fFoundMax, spsi_output = wuff.__spot_first_peaks_in_CalibrationHistogram(   
        calibration_histogram,
        max_peaks,
        prominence,
        initial_percentage,
        percentage_step)
    
    peaks_n_to_fit = len(spsi_output[0])

    for i in range(peaks_n_to_fit):
            
        aux_idx  = spsi_output[0][i]

        aux_seeds = [
            # Scale seed
            calibration_histogram.counts[aux_idx],
            # Mean seed
            (calibration_histogram.edges[aux_idx] \
             + calibration_histogram.edges[aux_idx + 1]) / 2.,
            # Std seed : Note that 
            # wuff.__spot_first_peaks_in_CalibrationHistogram()
            # is computing the widths of the peaks, in
            # number of samples, at half of their height 
            # (rel_height = 0.5). 2.355 is approximately 
            # the conversion factor between the standard 
            # deviation and the FWHM. Also, note that here
            # we are assuming that the binning is uniform.
            spsi_output[1]['widths'][i] * calibration_histogram.mean_bin_width / 2.355]                             

        # Restrict the fit lower limit to 0
        aux_lower_lim = max(
            0,
            aux_idx - half_points_to_fit)
        
        # The upper limit should be restricted to
        # len(calibration_histogram.counts). Making it
        # be further restricted to 
        # len(calibration_histogram.counts) - 1 so that
        # there is always available data to compute
        # the center of the bins, in the following line.
        
        aux_upper_lim = min(
            len(calibration_histogram.counts) - 1,
            aux_idx + half_points_to_fit + 1)
        
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
                p0=aux_seeds)
            
        # Happens if scipy.optimize.curve_fit()
        # could not converge to a solution

        except RuntimeError:

            # In this case, we will skip this peak
            # (so, in case fFoundMax was True, now 
            # it must be false) and we will continue 
            # with the next one, if any

            fFoundMax = False

            continue    

        aux_errors = np.sqrt(np.diag(aux_covariance_matrix))

        calibration_histogram._CalibrationHistogram__add_gaussian_fit_parameters(   
            aux_optimal_parameters[0],
            aux_errors[0],
            aux_optimal_parameters[1],
            aux_errors[1],
            aux_optimal_parameters[2],
            aux_errors[2])
        
    return fFoundMax

def fit_peaks_of_ChannelWsGrid( 
    channel_ws_grid: ChannelWsGrid,
    max_peaks: int,
    prominence: float,
    half_points_to_fit: int,
    initial_percentage = 0.1,
    percentage_step = 0.1
) -> bool:
    """For each ChannelWs object, say chws, contained in
    the ChWfSets attribute of the given ChannelWsGrid
    object, channel_ws_grid, whose channel is present
    in the ch_map attribute of the channel_ws_grid, this
    function calls the
    
        fit_peaks_of_CalibrationHistogram(chws.calib_histo, ...)

    function. It returns False if at least one 
    of the fit_peaks_of_CalibrationHistogram() calls 
    returns False, and True if every 
    fit_peaks_of_CalibrationHistogram() call returned 
    True. I.e. it returns True if max_peaks peaks 
    were successfully found for each histogram, and
    False if only n peaks were found for at least one 
    of the histograms, where n < max_peaks.

    Parameters
    ----------
    channel_ws_grid: ChannelWsGrid
        The ChannelWsGrid object to fit peaks on
    max_peaks: int
        The maximum number of peaks which will be
        searched for in each calibration histogram.
        It is given to the 'max_peaks' parameter of
        the fit_peaks_of_CalibrationHistogram()
        function for each calibration histogram.    
    prominence: float
        It must be greater than 0.0 and smaller than 
        1.0. It gives the minimal prominence of the 
        peaks to spot. This parameter is passed to the 
        'prominence' parameter of the 
        fit_peaks_of_CalibrationHistogram() function 
        for each calibration histogram. For more 
        information, check the documentation of such 
        function.
    half_points_to_fit: int
        It must be a positive integer. For each peak in
        each calibration histogram, it gives the number 
        of points to consider on either side of the peak 
        maximum, to fit each gaussian function. It is
        given to the 'half_points_to_fit' parameter of
        the fit_peaks_of_CalibrationHistogram() function 
        for each calibration histogram. For more information, 
        check the documentation of such function.
    initial_percentage: float
        It must be greater than 0.0 and smaller than 1.0.
        This parameter is passed to the 'initial_percentage' 
        parameter of the fit_peaks_of_CalibrationHistogram()
        function for each calibration histogram. For more 
        information, check the documentation of such function.
    percentage_step: float
        It must be greater than 0.0 and smaller than 1.0.
        This parameter is passed to the 'percentage_step'
        parameter of the fit_peaks_of_CalibrationHistogram()
        function for each calibration histogram. For more 
        information, check the documentation of such function.

    Returns
    ----------
    output: bool
        True if max_peaks peaks were successfully found for 
        each histogram, and False if only n peaks were found 
        for at least one of the histograms, where n < max_peaks.
    """

    output = True

    for i in range(channel_ws_grid.ch_map.rows):
        for j in range(channel_ws_grid.ch_map.columns):

            try:
                channel_ws = channel_ws_grid.ch_wf_sets[
                    channel_ws_grid.ch_map.data[i][j].endpoint][
                        channel_ws_grid.ch_map.data[i][j].channel]

            except KeyError:
                continue

            output *= fit_peaks_of_CalibrationHistogram(
                channel_ws.calib_histo,
                max_peaks,
                prominence,
                half_points_to_fit,
                initial_percentage=initial_percentage,
                percentage_step=percentage_step)
            
    return output