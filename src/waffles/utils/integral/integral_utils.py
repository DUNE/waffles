import numpy as np
from typing import Tuple

from waffles.Exceptions import GenerateExceptionMessage

def get_pulse_window_limits(
    adcs_array: np.ndarray,
    baseline: float,
    deviation_from_baseline: float,
    lower_limit_correction: int = 0,
    upper_limit_correction: int = 0
) -> Tuple[int, int]:
    """This function takes an unidimensional numpy array
    representing the pulse signal, in ADCs, coming from an SiPM.
    It returns a tuple of two integers representing the lower
    and upper limits, respectively, of the SiPM pulse in the
    given array. The lower limit is computed as the first point
    which negatively deviates from the given baseline by a
    certain given amount. The upper limit is computed as the
    last point, after the minimum of the pulse, which deviates
    from the given baseline by more than the given amount.
    It is assumed that the pulse is negative.

    Parameters
    ----------
    adcs_array: numpy.ndarray
        The input signal to analyze
    baseline: float
        The baseline value of the signal. It must be greater
        than the minimum of the signal, i.e. np.min(adcs_array).
    deviation_from_baseline: float
        It must be greater than 0.0 and smaller than 1.0. It is
        interpreted as a fraction of baseline - np.min(adcs_array).
        I.e. the lower limit of the pulse window is assumed to
        be the first point which drops below x, while the upper
        limit is assumed to be the last point, after the
        minimum of the pulse, which stays below x, where

            x = baseline - \
                (deviation_from_baseline * (baseline - np.min(adcs_array))).

    lower_limit_correction (resp. upper_limit_correction): int
        This is an optional parameter that can be used to apply
        a correction, to the lower (resp. upper) limit of the
        pulse window. I.e. if this parameter is set to N, then
        the final iterator value for the lower (resp. upper)
        limit is the one inferred by the deviation-from-baseline
        analysis plus N. Note that N may be positive or negative.

    Returns
    -------
    Tuple[int, int]
        A tuple containing the lower and upper limits of the
        pulse window.
    """

    idx_min = np.argmin(adcs_array)

    if baseline <= adcs_array[idx_min]:
        raise Exception(
            GenerateExceptionMessage(
                1,
                'get_pulse_window_limits()',
                f"The baseline ({baseline}) must be greater "
                f"than the minimum of the signal ({adcs_array[idx_min]})."
            )
        )
    
    if deviation_from_baseline <= 0.0 or deviation_from_baseline >= 1.0:
        raise Exception(
            GenerateExceptionMessage(
                2,
                'get_pulse_window_limits()',
                "The deviation_from_baseline parameter "
                f"({deviation_from_baseline}) must belong to the (0.0, 1.0)"
                " range."
            )
        )

    threshold = baseline - \
        (deviation_from_baseline * (baseline - adcs_array[idx_min]))

    # Compute the lower limit of the pulse window
    lower_limit = None
    for i, adc in enumerate(adcs_array):
        if adc < threshold:
            lower_limit = i
            break

    # Since we've made sure that baseline > adcs_array[idx_min],
    # lower_limit must be defined, i.e. an adcs value smaller
    # than the threshold must have been found (even if it matches
    # the minimum of the signal). The fact that idx_min is bigger
    # or equal to lower_limit is also guaranteed.

    upper_limit = None
    for i in range(idx_min+1, len(adcs_array)):
        if adcs_array[i] > threshold:
            upper_limit = i - 1
            break

    if upper_limit is None:
        raise Exception(
            GenerateExceptionMessage(
                3,
                'get_pulse_window_limits()',
                "The upper limit of the pulse window could not be "
                "found. I.e. the signal never raised above the threshold "
                f"({threshold}) after reaching its minimum value "
                f"({adcs_array[idx_min]})."
            )
        )

    # Apply the corrections to the limits
    lower_limit += lower_limit_correction
    upper_limit += upper_limit_correction

    # Make sure that the limits are within the bounds of the array
    lower_limit, upper_limit = max(0, lower_limit), max(0, upper_limit)
    lower_limit, upper_limit = min(len(adcs_array)-1, lower_limit), \
        min(len(adcs_array)-1, upper_limit)

    # Make sure that the corrections did not make the lower limit
    # greater than or equal to the upper limit
    if lower_limit >= upper_limit:
        raise Exception(
            GenerateExceptionMessage(
                4,
                'get_pulse_window_limits()',
                "The corrected lower limit of the pulse window "
                f"({lower_limit}) is greater than or equal to the "
                f"corrected upper limit ({upper_limit})."
            )
        )

    return (
        lower_limit,
        upper_limit
    )