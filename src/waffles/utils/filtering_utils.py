import inspect
import numpy as np
from typing import Optional

from waffles.data_classes.WaveformAdcs import WaveformAdcs
from waffles.data_classes.Waveform import Waveform

from waffles.Exceptions import GenerateExceptionMessage


def check_well_formedness_of_generic_waveform_function(
    wf_function_signature: inspect.Signature
) -> None:
    """This function gets an argument, wf_function_signature, 
    and returns None if the following conditions are met:

        -   such signature takes at least one argument
        -   the first argument of such signature
            is called 'waveform'
        -   the type annotation of such argument 
            must be either the WaveformAdcs class,
            the Waveform class, the 'WaveformAdcs'
            string literal or the 'Waveform' string
            literal

    If any of these conditions are not met, this
    function raises an exception.

    Parameters
    ----------
    wf_function_signature: inspect.Signature

    Returns
    ----------
    bool
    """

    try:
        if list(wf_function_signature.parameters.keys())[0] != 'waveform':
            raise Exception(GenerateExceptionMessage(
                1,
                "check_well_formedness_of_generic_waveform_function()",
                "The name of the first parameter of the given signature "
                "must be 'waveform'."))
        
    except IndexError:
        raise Exception(GenerateExceptionMessage(
            2,
            "check_well_formedness_of_generic_waveform_function()",
            'The given signature must take at least one parameter.'))

    if wf_function_signature.parameters['waveform'].annotation not in [
            WaveformAdcs, 'WaveformAdcs', Waveform, 'Waveform']:
        
        raise Exception(GenerateExceptionMessage(
            3,
            "check_well_formedness_of_generic_waveform_function()",
            "The 'waveform' parameter of the given signature must be "
            "hinted as a WaveformAdcs (or an inherited class) object."))
    return


def check_well_formedness_of_waveform_filter_function(
    wf_function_signature: inspect.Signature
) -> None:
    """This function gets an argument, wf_function_signature, 
    and returns None if the following conditions are met:

        -   such signature takes at least one argument
        -   the first argument of such signature
            is called 'waveform'
        -   the type annotation of such argument 
            must be either the WaveformAdcs class,
            the Waveform class, the 'WaveformAdcs'
            string literal or the 'Waveform' string
            literal
        -   the return type of such signature 
            is annotated as a boolean value

    If any of these conditions are not met, this
    function raises an exception.

    Parameters
    ----------
    wf_function_signature: inspect.Signature

    Returns
    ----------
    bool
    """

    check_well_formedness_of_generic_waveform_function(
        wf_function_signature
    )
    
    if wf_function_signature.return_annotation != bool:
        raise Exception(GenerateExceptionMessage(
            1,
            "check_well_formedness_of_waveform_filter_function()",
            "The return type of the given signature must be hinted as a "
            "boolean."))

    return


def match_run(
    waveform: Waveform,
    run: int
) -> bool:
    """This function returns True if the run_number attribute
    of the given Waveform object matches run. It returns 
    False if else.

    Parameters
    ----------
    waveform: Waveform
    run: int

    Returns
    ----------
    bool
    """

    return waveform.run_number == run


def match_endpoint(
    waveform: Waveform,
    endpoint: int
) -> bool:
    """This function returns True if the endpoint attribute
    of the given Waveform object matches endpoint, and 
    False if else.

    Parameters
    ----------
    waveform: Waveform
    endpoint: int

    Returns
    ----------
    bool
    """

    return waveform.endpoint == endpoint


def match_channel(
    waveform: Waveform,
    channel: int
) -> bool:
    """This function returns True if the channel attribute
    of the given Waveform object matches channel, and 
    False if else.

    Parameters
    ----------
    waveform: Waveform
    channel: int

    Returns
    ----------
    bool
    """

    return waveform.channel == channel


def match_endpoint_and_channel(
    waveform: Waveform,
    endpoint: int,
    channel: int
) -> bool:
    """This function returns True if the endpoint and channel
    attributes of the given Waveform object match endpoint 
    and channel, respectively.

    Parameters
    ----------
    waveform: Waveform
    endpoint: int
    channel: int

    Returns
    ----------
    bool
    """

    return waveform.endpoint == endpoint and waveform.channel == channel


def truncate_waveforms_in_WaveformSet(
    # Avoid circular import
    input_WaveformSet: 'WaveformSet',
    starting_tick: int,
    points_number: Optional[int] = None,
    ending_tick: Optional[int] = None,
) -> None:
    """This function gets a WaveformSet object and truncates
    all the Waveform objects in its 'waveforms' attribute
    according to the given starting_tick and 
    points_number/ending_tick. This function preserves the
    length homogeneity of the Waveform objects in the input
    WaveformSet object. These changes are applied in place.
    I.e. if the original data should be preserved, it is the
    user's responsiblity to have performed a deep copy of
    the input WaveformSet object before calling this function.
    
    Parameters
    ----------
    input_WaveformSet: WaveformSet
        The WaveformSet object whose Waveform objects will be
        truncated
    starting_tick: int
        The starting tick of the truncation. It must belong to
        the [0, input_WaveformSet.points_per_wf - 1]
    points_number: int
        If it is defined, then the input given to the
        ending_tick parameter is ignored and the Waveform
        objects in the input WaveformSet object will be
        truncated to have this number of points as of the
        specified starting tick. It must belong to the 
        [1, input_WaveformSet.points_per_wf - starting_tick]
        range. Either this parameter or the ending_tick
        parameter must be defined, otherwise an exception
        is raised.
    ending_tick: int
        If it is defined and points_number is not, then the
        Waveform objects in the input WaveformSet object will
        be truncated up to this tick. This limit is exclusive.
        It must belong to the
        [starting_tick + 1, input_WaveformSet.points_per_wf]
        range.
    """

    if starting_tick < 0 or \
        starting_tick >= input_WaveformSet.points_per_wf:

        raise Exception(
            GenerateExceptionMessage(
                1,
                'truncate_waveforms_in_WaveformSet()',
                f"The given starting_tick ({starting_tick}) must belong "
                f"to the [0, {input_WaveformSet.points_per_wf - 1}] range."
            )
        )
    
    # Use the points_number parameter if it is defined
    if points_number is not None:
        if points_number < 1 or \
            points_number > input_WaveformSet.points_per_wf - starting_tick:

            raise Exception(
                GenerateExceptionMessage(
                    2,
                    'truncate_waveforms_in_WaveformSet()',
                    f"The given points_number ({points_number}) must belong "
                    f"to the [1, {input_WaveformSet.points_per_wf - starting_tick}]"
                    " range."
                )
            )
        
        # Not adding a -1 here because ending_tick_ is exclusive
        ending_tick_ = starting_tick + points_number

    # Use the ending_tick parameter if
    # points_number is not defined
    elif ending_tick is not None:
        if ending_tick <= starting_tick or \
            ending_tick > input_WaveformSet.points_per_wf: 

            raise Exception(
                GenerateExceptionMessage(
                    3,
                    'truncate_waveforms_in_WaveformSet()',
                    f"The given ending_tick ({ending_tick}) must belong to "
                    f"the [{starting_tick + 1}, {input_WaveformSet.points_per_wf}]"
                    " range."
                )
            )

        ending_tick_ = ending_tick

    # If neither points_number nor ending_tick
    # is defined, raise an exception
    else:
        raise Exception(
            GenerateExceptionMessage(
                4,
                'truncate_waveforms_in_WaveformSet()',
                "Either the points_number or "
                "the ending_tick parameter must be defined."
            )
        )

    # Looping over every waveform for the slice ensures that
    # the length homogeneity of the Waveform objects in the
    # input WaveformSet is preserved, so there's no need to
    # call input_WaveformSet.check_length_homogeneity()
    for wf in input_WaveformSet.waveforms:
        wf._Waveform__slice_adcs(
            starting_tick,
            ending_tick_
        )

    # Update the points_per_wf attribute which has no setter method
    input_WaveformSet._WaveformSet__points_per_wf = ending_tick_ - starting_tick

    # Reset the mean waveform information
    input_WaveformSet.reset_mean_waveform()

    return


def selection_for_led_calibration(
    waveform: Waveform,
    baseline_analysis_label: str,
    baseline_i_up: int,
    signal_i_up: int,
    baseline_std: float,
    baseline_allowed_dev: float,
    signal_allowed_dev: float,
) -> bool:
    """This function returns True if the following two
    conditions are met simultaneously:

        - the waveform adcs do not deviate from the
          baseline by more than baseline_allowed_dev
          times the baseline_std in the baseline region,

        - and the waveform adcs do not drop below
          the baseline by more than signal_allowed_dev
          times the baseline_std in the signal region.

    If any of them is not met, it returns False. For more
    information on how the regions are defined, see the
    parameters documentation below.

    Parameters
    ----------
    waveform: Waveform
        The waveform to apply the selection to
    baseline_analysis_label: str
        The baseline of the filtered waveform must be available
        under the 'baseline' key of the result of the analysis
        whose label is given by this parameter, i.e. in
        waveform.analyses[analysis_label].result['baseline']
    baseline_i_up: int
        Iterator value for the waveform.adcs array which gives
        the upper limit of the baseline region
    signal_i_up: int
        Iterator value for the waveform.adcs array which gives
        the upper limit of the signal region. It may match the
        integration upper limit of the charge integration.
    baseline_std: float
        Its absolute value is assumed to be the standard
        deviation of the baseline
    baseline_allowed_dev: float
        Its absolute value is assumed to be the maximum multiple
        of baseline_std which the signal can deviate from the
        baseline in the baseline region, i.e. in the
        waveform.adcs[0:baseline_i_up] region
    signal_allowed_dev: float
        Its absolute value is assumed to be the maximum multiple
        of baseline_std which the signal can deviate, negatively,
        from the baseline in the signal region, i.e. in the
        waveform.adcs[baseline_i_up:signal_i_up] region.

    Returns
    ----------
    bool
    """

    try:
        baseline_samples = waveform.adcs[:baseline_i_up] - \
            waveform.analyses[baseline_analysis_label].result['baseline']

    except KeyError:
        raise Exception(
            GenerateExceptionMessage(
                1,
                "selection_for_led_calibration()",
                f"The given waveform does not have the analysis"
                f" '{baseline_analysis_label}' in its analyses "
                "attribute, or it does, but the 'baseline' key "
                "is not present in its result."
            )
        )
    
    if np.max(np.abs(baseline_samples)) > abs(baseline_allowed_dev * baseline_std):
        return False
    
    # The baseline is available, otherwise exception #1 would have been raised
    signal_samples = waveform.adcs[baseline_i_up:signal_i_up] - \
        waveform.analyses[baseline_analysis_label].result['baseline']
    
    if np.min(signal_samples) < -1.*abs(signal_allowed_dev * baseline_std):
        return False
    
    return True