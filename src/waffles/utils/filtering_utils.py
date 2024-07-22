import inspect

from waffles.data_classes.WaveformAdcs import WaveformAdcs
from waffles.data_classes.Waveform import Waveform

from waffles.Exceptions import generate_exception_message

def check_well_formedness_of_generic_waveform_function(wf_function_signature : inspect.Signature) -> None:

    """
    This function gets an argument, wf_function_signature, 
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
    wf_function_signature : inspect.Signature

    Returns
    ----------
    bool
    """

    try:
        if list(wf_function_signature.parameters.keys())[0] != 'waveform':
            raise Exception(generate_exception_message( 1,
                                                        "check_well_formedness_of_generic_waveform_function()",
                                                        "The name of the first parameter of the given signature must be 'waveform'."))
    except IndexError:
        raise Exception(generate_exception_message( 2,
                                                    "check_well_formedness_of_generic_waveform_function()",
                                                    'The given signature must take at least one parameter.'))
    
    if wf_function_signature.parameters['waveform'].annotation not in [WaveformAdcs, 'WaveformAdcs', Waveform, 'Waveform']:
        raise Exception(generate_exception_message( 3,
                                                    "check_well_formedness_of_generic_waveform_function()",
                                                    "The 'waveform' parameter of the given signature must be hinted as a WaveformAdcs (or an inherited class) object."))
    if wf_function_signature.return_annotation != bool:
        raise Exception(generate_exception_message( 4,
                                                    "check_well_formedness_of_generic_waveform_function()",
                                                    "The return type of the given signature must be hinted as a boolean."))
    return