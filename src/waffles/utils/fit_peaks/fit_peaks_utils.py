import math

import numpy as np
from scipy import signal as spsi
from typing import Tuple, Dict

from waffles.data_classes.CalibrationHistogram import CalibrationHistogram


def trim_spsi_find_peaks_output_to_max_peaks(spsi_output: Tuple[np.ndarray, Dict[str, np.ndarray]],
                                             max_peaks: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    This function gets the output of a certain call to 
    scipy.signal.find_peaks() and returns the same output, 
    but truncated to the first max_peaks found peaks. 

    Parameters
    ----------
    spsi_output : tuple of ( np.ndarray,  dict, )
        The output of a call to scipy.signal.find_peaks().
        No checks are performed here regarding the
        well-formedness of this input.
    max_peaks : int
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
    output : tuple of ( np.ndarray, dict, )
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


def __spot_first_peaks_in_CalibrationHistogram(CalibrationHistogram: CalibrationHistogram,
                                               max_peaks: int,
                                               prominence: float,
                                               initial_percentage=0.1,
                                               percentage_step=0.1) -> Tuple[bool, Tuple]:
    """
    This function is not intended for user usage. It 
    must be only called by fit_peaks_of_CalibrationHistogram(),
    where the well-formedness checks of the input
    parameters have been performed. This function tries 
    to find peaks over the signal which is computed as

        signal = (CalibrationHistogram.Counts - np.min(CalibrationHistogram.Counts))/np.max(CalibrationHistogram.Counts)

    This function iteratively calls

        scipy.signal.find_peaks(signal[0:points], 
                                prominence = prominence)

    to spot, at most, max_peaks peaks. To do so, at the 
    first iteration, points is computed as 
    math.floor(initial_percentage * len(signal)). If the 
    number of spotted peaks is less than max_peaks, then 
    points is increased by 
    math.floor(percentage_step * len(signal)) and the 
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
    where peaks is the output of the last call to 
    scipy.signal.find_peaks(). If the last call found a
    number of peaks greater than or equal to max_peaks, 
    then the function returns (True, peaks), where peaks 
    is the output of scipy.signal.find_peaks() but 
    truncated to the first max_peaks found peaks.

    Parameters
    ----------
    CalibrationHistogram : CalibrationHistogram
        The CalibrationHistogram object to spot peaks on
    max_peaks : int
        The maximum number of peaks to spot. It must 
        be a positive integer. This is not checked here, 
        it is the caller's responsibility to ensure this.
    prominence : float
        The prominence parameter to pass to the
        scipy.signal.find_peaks() function. Since the
        signal is normalized, this prominence can be
        understood as a fraction of the total amplitude
        of the signal. P.e. setting prominence to 0.5,
        will prevent scipy.signal.find_peaks() from
        spotting peaks whose amplitude is less than
        half of the total amplitude of the signal.
    initial_percentage : float
        The initial percentage of the signal to
        consider. It must be greater than 0.0
        and smaller than 1.0.
    percentage_step : float
        The percentage step to increase the signal
        to consider in successive calls of 
        scipy.signal.find_peaks(). It must be greater 
        than 0.0 and smaller than 1.0.

    Returns
    -------
    output : tuple of ( bool,  tuple, )
        The first entry is a boolean which is True if 
        the number of peaks found is greater than or 
        equal to max_peaks, and False otherwise. The
        second entry is the output of the last call to
        scipy.signal.find_peaks() if the first entry
        of the tuple is False. If the first entry of
        the tuple is True, then the second entry is
        the output of scipy.signal.find_peaks() but
        truncated to the first max_peaks found peaks. 
        For more information, check the 
        scipy.signal.find_peaks() documentation.
    """

    signal = (CalibrationHistogram.Counts - np.min(CalibrationHistogram.Counts)
              ) / np.max(CalibrationHistogram.Counts)

    fFoundMax = False
    fReachedEnd = False
    points = math.floor(initial_percentage * len(signal))

    while not fFoundMax and not fReachedEnd:

        points = min(points, len(signal))

        spsi_output = spsi.find_peaks(signal[0:points],
                                      prominence=prominence,
                                      width=0,                  # Adding a minimal 0 width, which constraints nothing,
                                      # but which makes scipy.signal.find_peaks() return
                                      rel_height=0.5)
        # information about each peak-width at half its height.
        if len(spsi_output[0]) >= max_peaks:

            spsi_output = trim_spsi_find_peaks_output_to_max_peaks(spsi_output,
                                                                   max_peaks)
            fFoundMax = True

        if points == len(signal):
            fReachedEnd = True

        points += math.floor(percentage_step * len(signal))

    if fFoundMax:
        return (True, spsi_output)
    else:
        return (False, spsi_output)
