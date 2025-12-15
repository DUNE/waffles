from waffles.data_classes.CalibrationHistogram import CalibrationHistogram
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid

import waffles.utils.fit_peaks.fit_peaks_utils as wuff

from waffles.Exceptions import GenerateExceptionMessage

from iminuit import Minuit
from iminuit.cost import LeastSquares

import numpy as np
import waffles.utils.numerical_utils as wun

def fit_peaks_of_CalibrationHistogram(
    calibration_histogram: CalibrationHistogram,
    max_peaks: int,
    prominence: float,
    initial_percentage: float = 0.1,
    percentage_step: float = 0.1,
    return_last_addition_if_fail: bool = False,
    fit_type: str = 'independent_gaussians',
    weigh_fit_by_poisson_sigmas: bool = False,
    half_points_to_fit: int = 2,
    std_increment_seed_fallback: float = 1e+2,
    ch_span_fraction_around_peaks: float = 0.05
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
    return_last_addition_if_fail: bool
        This parameter is given to the
        return_last_addition_if_fail parameter of the
        wuff.__spot_first_peaks_in_CalibrationHistogram()
        function. It makes a difference only if the
        specified number of peaks (max_peaks) is not
        found. For more information, check the
        documentation of the 
        wuff.__spot_first_peaks_in_CalibrationHistogram()
        function.
    fit_type: str
        The supported values are 'independent_gaussians',
        'correlated_gaussians' and 'multigauss_iminuit'. If
        any other value is given, the
        'independent_gaussians' value will be used instead.
        If the 'independent_gaussians' value is used, the
        function will fit each peak independently, i.e. it
        will fit a gaussian function to each peak
        independently of the others. For more information on
        this type of fit, check the documentation of the
        wuff.__fit_independent_gaussians_to_calibration_histogram()
        function. If the 'correlated_gaussians' value is
        given, the function will fit all of the peaks at
        once using a fitting function which is a sum of
        gaussians whose means and standard deviations are
        correlated. For more information on this type of
        fit, check the documentation of the
        wuff.__fit_correlated_gaussians_to_calibration_histogram()
        function.
    weigh_fit_by_poisson_sigmas: bool
        If it is set to True, and fit_type is set to
        'independent_gaussians' or 'correlated_gaussians',
        then the gaussian least squares fit will be weighed
        by the Poisson standard deviation of each bin. If
        fit_type is set to 'multigauss_iminuit', this
        parameter only affects the initial seed values
        which are calculated using the
        wuff.__fit_independent_gaussians_to_calibration_histogram()
        function.
    half_points_to_fit: int
        This parameter is only used if the fit_type
        parameter is set to 'independent_gaussians'.
        It must be a positive integer. For each peak, it 
        gives the number of points to consider on either 
        side of the peak maximum, to fit each gaussian 
        function. I.e. if i is the iterator value for
        calibration_histogram.counts of the i-th peak, 
        then the histogram bins which will be considered 
        for the fit are given by the slice 
        calibration_histogram.counts[i - half_points_to_fit : i + half_points_to_fit + 1].
    std_increment_seed_fallback: float
        This parameter is only used if the fit_type
        parameter is set to 'correlated_gaussians'.
        In that case, it is given to the
        std_increment_seed_fallback parameter of the
        wuff.__fit_correlated_gaussians_to_calibration_histogram()
        function. For more information, check the
        documentation of such function.
    ch_span_fraction_around_peaks: float
        This parameter is only used if the fit_type
        parameter is set to 'correlated_gaussians'.
        In that case, it is given to the
        ch_span_fraction_around_peaks parameter of the
        wuff.__fit_correlated_gaussians_to_calibration_histogram()
        function. For more information, check the
        documentation of such function.

    Returns
    -------
    bool
        True if the number of found-and-fitted peaks matches
        the given max_peaks parameter, and False if it is
        smaller than max_peaks.
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
        initial_percentage=initial_percentage,
        percentage_step=percentage_step,
        return_last_addition_if_fail=return_last_addition_if_fail
    )
    
    if fit_type == 'correlated_gaussians':
        fFitAll = wuff.__fit_correlated_gaussians_to_calibration_histogram(
            spsi_output,
            calibration_histogram,
            std_increment_seed_fallback=std_increment_seed_fallback,
            ch_span_fraction_around_peaks=ch_span_fraction_around_peaks,
            weigh_fit_by_poisson_sigmas=weigh_fit_by_poisson_sigmas
        )
    else:
        fFitAll = wuff.__fit_independent_gaussians_to_calibration_histogram(
            spsi_output,
            calibration_histogram,
            half_points_to_fit,
            weigh_fit_by_poisson_sigmas=weigh_fit_by_poisson_sigmas
        )

    n_peaks_found = len(calibration_histogram.gaussian_fits_parameters['scale'])

    if fit_type != 'multigauss_iminuit' or n_peaks_found < 1:
        return fFoundMax & fFitAll


    # initialize parameters for iminuit
    paramnames = ['scale_baseline', 'mean_baseline', 'std_baseline', 'gain', 'propstd']
    iminuitparams = []
    iminuitparams.append(calibration_histogram.gaussian_fits_parameters['scale'][0][0])
    iminuitparams.append(calibration_histogram.gaussian_fits_parameters['mean'][0][0])
    iminuitparams.append(calibration_histogram.gaussian_fits_parameters['std'][0][0])

    # in case there is is no 1pe fitted, still try our best
    onepe_scale = iminuitparams[0]
    onepe_std = iminuitparams[2]
    onepe_mean = iminuitparams[1] + 2*onepe_std # mean of the baseline + 2*sigma

    if n_peaks_found > 1: # in case peak of 1 pe was found, use it...
        onepe_scale = calibration_histogram.gaussian_fits_parameters['scale'][1][0]
        onepe_mean = calibration_histogram.gaussian_fits_parameters['mean'][1][0]
        onepe_std = calibration_histogram.gaussian_fits_parameters['std'][1][0]

    # std dev of 1, 2, n-th peak is are proportional
    estimated_stdprop = np.sqrt(abs(onepe_std**2 - iminuitparams[2]**2)) # abs just in case
    iminuitparams.append(onepe_mean)
    iminuitparams.append(estimated_stdprop)

    # Estimates the number of peaks that should be fitted based on the histogram
    # Starts with a huge number of peaks, and then reduces it
    n_peaks_to_fit_iminuit = n_peaks_found + 15 
    searchdone=False
    for i in range(1, n_peaks_to_fit_iminuit):
        ipeakscale = np.argmin( np.abs( calibration_histogram.edges - onepe_mean*i ))
        if ipeakscale >= len(calibration_histogram.counts):
            ipeakscale = len(calibration_histogram.counts) - 1
            searchdone = True # accept last extra peak
        iminuitparams.append(calibration_histogram.counts[ipeakscale]*0.95)
        paramnames.append(f'scale_{i}pe')
        if searchdone:
            n_peaks_to_fit_iminuit = i + 1
            break

    data_x = ( calibration_histogram.edges[:-1] + calibration_histogram.edges[1:] )*0.5
    data_y = calibration_histogram.counts
    data_err = np.sqrt(data_y)
    data_err[data_err == 0] = 1 # avoid division by zero
    chi2 = LeastSquares(data_x, data_y, data_err, wun.multigaussfit)
    mm = Minuit(chi2, *iminuitparams, name=paramnames)
    mm.fixed['scale_baseline'] = True
    mm.fixed['mean_baseline'] = True
    mm.fixed['std_baseline'] = True
    mm.fixed['gain'] = True
    mm.fixed['propstd'] = False
    mm.fixed['scale_1pe'] = True
    mm.migrad()

    for p in mm.parameters: # we are now free
        mm.fixed[p] = False
        if p != "mean_baseline":
            mm.limits[p] = (1e-6, None) # They are all positive btw
    mm.migrad() # second call to ensure convergence now with free parameters
    mm.migrad() # try hard call 
    mm.hesse() # compute errors
    fitstatus = mm.fmin.is_valid if mm.fmin else False
        
    # Resize the gaussian_fits_parameters
    calibration_histogram._CalibrationHistogram__reset_gaussian_fit_parameters()

    calibration_histogram._CalibrationHistogram__add_gaussian_fit_parameters(
        mm.params[0].value,mm.params[0].error,
        mm.params[1].value,mm.params[1].error,
        mm.params[2].value,mm.params[2].error,
    )

    gain = mm.params[3].value
    errgain = mm.params[3].error

    propstd = mm.params[4].value
    errpropstd = mm.params[4].error

    for i in range(1, n_peaks_to_fit_iminuit):
        calibration_histogram._CalibrationHistogram__add_gaussian_fit_parameters(   
            mm.params[4 + i].value,
            mm.params[4 + i].error, 

            gain * i + mm.params[1].value,
            np.sqrt(errgain**2 * i + mm.params[1].error**2) ,

            np.sqrt( mm.params[2].value**2 + propstd**2 * i ), 
            np.sqrt( mm.params[2].value**2 * mm.params[2].error**2
                + propstd**2 * errpropstd**2 * i ) / np.sqrt(
                    mm.params[2].value**2 + propstd**2 * i )
        )

    n_peaks_found = len(calibration_histogram.gaussian_fits_parameters['mean'])

    setattr(calibration_histogram, 'n_peaks_found', n_peaks_found)
    setattr(calibration_histogram, 'iminuit', mm)

    calibration_histogram.compute_cross_talk()
        
    return fitstatus

def fit_peaks_of_ChannelWsGrid( 
    channel_ws_grid: ChannelWsGrid,
    max_peaks: int,
    prominence: float,
    initial_percentage: float = 0.1,
    percentage_step: float = 0.1,
    return_last_addition_if_fail: bool = False,
    fit_type: str = 'independent_gaussians',
    weigh_fit_by_poisson_sigmas: bool = False,
    half_points_to_fit: int = 2,
    std_increment_seed_fallback: float = 1e+2,
    ch_span_fraction_around_peaks: float = 0.05,
    verbose: bool = False
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
    return_last_addition_if_fail: bool
        This parameter is given to the
        return_last_addition_if_fail parameter of the
        fit_peaks_of_CalibrationHistogram() function. It
        makes a difference only if the specified number
        of peaks (max_peaks) is not found. For more
        information, check the documentation of the
        fit_peaks_of_CalibrationHistogram() function.
    fit_type: str
        The only supported values are 'independent_gaussians'
        and 'correlated_gaussians'. If any other value is
        given, the 'independent_gaussians' value will be
        used instead. This parameter is passed to the
        'fit_type' parameter of the fit_peaks_of_CalibrationHistogram()
        function for each calibration histogram. For more 
        information, check the documentation of such function.
    weigh_fit_by_poisson_sigmas: bool
        It is given to the 'weigh_fit_by_poisson_sigmas'
        parameter of the fit_peaks_of_CalibrationHistogram()
        function for each calibration histogram. For more 
        information, check the documentation of such function.
    half_points_to_fit: int
        This parameter is only used if the fit_type
        parameter is set to 'independent_gaussians'.
        It must be a positive integer. For each peak in
        each calibration histogram, it gives the number 
        of points to consider on either side of the peak 
        maximum, to fit each gaussian function. It is
        given to the 'half_points_to_fit' parameter of
        the fit_peaks_of_CalibrationHistogram() function 
        for each calibration histogram. For more information, 
        check the documentation of such function.
    std_increment_seed_fallback: float
        This parameter is only used if the fit_type
        parameter is set to 'correlated_gaussians'.
        For more information, check the documentation
        of the fit_peaks_of_CalibrationHistogram()
        function.
    ch_span_fraction_around_peaks: float
        This parameter is only used if the fit_type
        parameter is set to 'correlated_gaussians'.
        For more information, check the documentation
        of the fit_peaks_of_CalibrationHistogram()
        function.
    verbose: bool = False
        Whether to print functioning related messages

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

            if channel_ws.calib_histo is not None:
                output *= fit_peaks_of_CalibrationHistogram(
                    channel_ws.calib_histo,
                    max_peaks,
                    prominence,
                    initial_percentage=initial_percentage,
                    percentage_step=percentage_step,
                    return_last_addition_if_fail=return_last_addition_if_fail,
                    fit_type=fit_type,
                    weigh_fit_by_poisson_sigmas=weigh_fit_by_poisson_sigmas,
                    half_points_to_fit=half_points_to_fit,
                    std_increment_seed_fallback=std_increment_seed_fallback,
                    ch_span_fraction_around_peaks=ch_span_fraction_around_peaks
                )
            elif verbose:
                print(
                    f"In function fit_peaks_of_ChannelWsGrid(): "
                    f"Skipping the peak-fitting process for channel "
                    f"{channel_ws_grid.ch_map.data[i][j].endpoint}-"
                    f"{channel_ws_grid.ch_map.data[i][j].channel}, "
                    f"because its calibration histogram is not available"
                )

    return output
