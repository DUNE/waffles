import numpy as np
from waffles.data_classes.WaveformAdcs import WaveformAdcs
from waffles.core.utils import build_parameters_dictionary
from waffles.data_classes.IPDict import IPDict

input_parameters = build_parameters_dictionary('params.yml')

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
        as the median of the ADC values in the defined time range."""
    
    # For the sake of efficiency, no well-formedness checks for the
    # time limits are performed here. The caller must ensure that
    # the limits are well-formed.

    return np.median(WaveformAdcs_object.adcs[
        lower_time_tick_for_median:upper_time_tick_for_median
    ])
    
def get_analysis_params(
        apa_no: int,
        run: int = None
    ):

    if apa_no == 1:
        if run is None:
            raise Exception(
                "In get_analysis_params(): A run number "
                "must be specified for APA 1"
            )
        else:
            int_ll = input_parameters['starting_tick'][1][run]
    else:
        int_ll = input_parameters['starting_tick'][apa_no]

    analysis_input_parameters = IPDict(
        baseline_limits=\
            input_parameters['baseline_limits'][apa_no]
    )
    analysis_input_parameters['int_ll'] = int_ll
    analysis_input_parameters['int_ul'] = \
        int_ll + input_parameters['integ_window']
    analysis_input_parameters['amp_ll'] = int_ll
    analysis_input_parameters['amp_ul'] = \
        int_ll + input_parameters['integ_window']

    return analysis_input_parameters

def get_nbins_for_charge_histo(
        pde: float,
        apa_no: int
    ):

    if apa_no in [2, 3, 4]:
        if pde == 0.4:
            bins_number = 125
        elif pde == 0.45:
            bins_number = 110 # [100-110]
        else:
            bins_number = 90
    else:
        # It is still required to
        # do this tuning for APA 1
        bins_number = 125

    return bins_number