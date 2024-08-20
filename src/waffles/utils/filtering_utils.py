import inspect

from waffles.data_classes.WaveformAdcs import waveform_adcs
from waffles.data_classes.Waveform import waveform

from waffles.Exceptions import generate_exception_message


def check_well_formedness_of_generic_waveform_function(
        wf_function_signature: inspect.Signature) -> None:
    """
    This function gets an argument, wf_function_signature, 
    and returns None if the following conditions are met:

        -   such signature takes at least one argument
        -   the first argument of such signature
            is called 'waveform'
        -   the type annotation of such argument 
            must be either the waveform_adcs class,
            the waveform class, the 'waveform_adcs'
            string literal or the 'waveform' string
            literal
        -   the return type of such signature 
            is annotated as a boolean value

    If any of these conditions are not met, this
    function raises an exception.

    Parameters
    ----------
    wf_function_signature : inspect.Signature

    Returns
    ----------
    bool
    """

    try:
        if list(wf_function_signature.parameters.keys())[0] != 'waveform':
            raise Exception(generate_exception_message(
                1,
                "check_well_formedness_of_generic_waveform_function()",
                "The name of the first parameter of the given signature "
                "must be 'waveform'."))
    except IndexError:
        raise Exception(generate_exception_message(
            2,
            "check_well_formedness_of_generic_waveform_function()",
            'The given signature must take at least one parameter.'))

    if wf_function_signature.parameters['waveform'].annotation not in [
            waveform_adcs, 'waveform_adcs', waveform, 'waveform']:
        raise Exception(generate_exception_message(
            3,
            "check_well_formedness_of_generic_waveform_function()",
            "The 'waveform' parameter of the given signature must be "
            "hinted as a waveform_adcs (or an inherited class) object."))
    if wf_function_signature.return_annotation != bool:
        raise Exception(generate_exception_message(
            4,
            "check_well_formedness_of_generic_waveform_function()",
            "The return type of the given signature must be hinted as a "
            "boolean."))
    return


def match_run(
        waveform: waveform,
        run: int) -> bool:
    """
    This function returns True if the RunNumber attribute
    of the given waveform object matches run. It returns 
    False if else.

    Parameters
    ----------
    waveform : waveform
    run : int

    Returns
    ----------
    bool
    """

    return waveform.run_number == run


def match_endpoint(
        waveform: waveform,
        endpoint: int) -> bool:
    """
    This function returns True if the endpoint attribute
    of the given waveform object matches endpoint, and 
    False if else.

    Parameters
    ----------
    waveform : waveform
    endpoint : int

    Returns
    ----------
    bool
    """

    return waveform.endpoint == endpoint


def match_channel(
        waveform: waveform,
        channel: int) -> bool:
    """
    This function returns True if the channel attribute
    of the given waveform object matches channel, and 
    False if else.

    Parameters
    ----------
    waveform : waveform
    channel : int

    Returns
    ----------
    bool
    """

    return waveform.channel == channel


def match_endpoint_and_channel(
        waveform: waveform,
        endpoint: int,
        channel: int) -> bool:
    """
    This function returns True if the endpoint and channel
    attributes of the given waveform object match endpoint 
    and channel, respectively.

    Parameters
    ----------
    waveform : waveform
    endpoint : int
    channel : int

    Returns
    ----------
    bool
    """

    return waveform.endpoint == endpoint and waveform.channel == channel
