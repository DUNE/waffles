import inspect

from waffles.data_classes.WaveformAdcs import WaveformAdcs
from waffles.data_classes.Waveform import Waveform

from waffles.Exceptions import GenerateExceptionMessage


def check_well_formedness_of_generic_waveform_function(
        wf_function_signature: inspect.Signature) -> None:
    """
    This function gets an argument, wf_function_signature, 
    and returns None if the following conditions are met:

        -   such signature takes at least one argument
        -   the first argument of such signature
            is called 'Waveform'
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
    wf_function_signature : inspect.Signature

    Returns
    ----------
    bool
    """

    try:
        if list(wf_function_signature.parameters.keys())[0] != 'Waveform':
            raise Exception(GenerateExceptionMessage(
                1,
                "check_well_formedness_of_generic_waveform_function()",
                "The name of the first parameter of the given signature "
                "must be 'Waveform'."))
    except IndexError:
        raise Exception(GenerateExceptionMessage(
            2,
            "check_well_formedness_of_generic_waveform_function()",
            'The given signature must take at least one parameter.'))

    if wf_function_signature.parameters['Waveform'].annotation not in [
            WaveformAdcs, 'WaveformAdcs', Waveform, 'Waveform']:
        raise Exception(GenerateExceptionMessage(
            3,
            "check_well_formedness_of_generic_waveform_function()",
            "The 'Waveform' parameter of the given signature must be "
            "hinted as a WaveformAdcs (or an inherited class) object."))
    if wf_function_signature.return_annotation != bool:
        raise Exception(GenerateExceptionMessage(
            4,
            "check_well_formedness_of_generic_waveform_function()",
            "The return type of the given signature must be hinted as a "
            "boolean."))
    return


def match_run(
        Waveform: Waveform,
        run: int) -> bool:
    """
    This function returns True if the RunNumber attribute
    of the given Waveform object matches run. It returns 
    False if else.

    Parameters
    ----------
    Waveform : Waveform
    run : int

    Returns
    ----------
    bool
    """

    return Waveform.run_number == run


def match_endpoint(
        Waveform: Waveform,
        endpoint: int) -> bool:
    """
    This function returns True if the endpoint attribute
    of the given Waveform object matches endpoint, and 
    False if else.

    Parameters
    ----------
    Waveform : Waveform
    endpoint : int

    Returns
    ----------
    bool
    """

    return Waveform.endpoint == endpoint


def match_channel(
        Waveform: Waveform,
        channel: int) -> bool:
    """
    This function returns True if the channel attribute
    of the given Waveform object matches channel, and 
    False if else.

    Parameters
    ----------
    Waveform : Waveform
    channel : int

    Returns
    ----------
    bool
    """

    return Waveform.channel == channel


def match_endpoint_and_channel(
        Waveform: Waveform,
        endpoint: int,
        channel: int) -> bool:
    """
    This function returns True if the endpoint and channel
    attributes of the given Waveform object match endpoint 
    and channel, respectively.

    Parameters
    ----------
    Waveform : Waveform
    endpoint : int
    channel : int

    Returns
    ----------
    bool
    """

    return Waveform.endpoint == endpoint and Waveform.channel == channel
