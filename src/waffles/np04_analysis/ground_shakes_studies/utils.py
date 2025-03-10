import numpy as np
from waffles.data_classes.WaveformAdcs import WaveformAdcs

def get_baseline(
        WaveformAdcs_object: WaveformAdcs,
        lower_time_tick_for_median: int = 0,
        upper_time_tick_for_median: int = 100
) -> float:
    """This function returns the baseline of a WaveformAdcs object
    by computing the median of the ADC values in the time range
    defined by the inclusive limits [lower_time_tick_for_median, 
    upper_time_tick_for_median].

    Parameters
    ----------
    WaveformAdcs_object: WaveformAdcs
        The baseline is computed for the data in the adcs attribute
        of this WaveformAdcs object

    lower_time_tick_for_median (resp. upper_time_tick_for_median): int
        Iterator value for the time tick which is the inclusive lower
        (resp. upper) limit of the time range in which the median is
        computed.

    Returns
    ----------
    baseline: float
        The baseline of the WaveformAdcs object, which is computed
        as the median of the ADC values in the defined time range.
        """
    
    # For the sake of efficiency, no well-formedness checks for the
    # time limits are performed here. The caller must ensure that
    # the limits are well-formed.

    return np.median(WaveformAdcs_object.adcs[
        lower_time_tick_for_median:upper_time_tick_for_median
    ])