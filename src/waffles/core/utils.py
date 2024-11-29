import argparse
from typing import Optional
import waffles.Exceptions as we

def add_arguments_to_parser(
        parser: argparse.ArgumentParser 
) -> None:
    """This function defines the arguments that the main program
    should accept. The arguments are the following:
    
    -s, --steering: str
        Name of the steering file.
    -a, --analysis: str
        The name of the analysis class to be
        executed
    -p, --params: str
        Name of the parameters file.
    -v, --verbose: bool
        Whether to run with verbosity.
        
    Parameters
    ----------
    parser: argparse.ArgumentParser
        The argparse.ArgumentParser instance to which the
        arguments will be added.

    Returns
    ----------
    None    
    """
    
    parser.add_argument(
        "-s",
        "--steering",
        type=str,
        default=None,
        help="Name of the steering file. It should be a YAML"
        " file which orders the different analysis stages "
        "and sets a parameters file for each stage."
    )

    parser.add_argument(
        "-a",
        "--analysis",
        type=str,
        default=None,
        help="The name of the analysis class to be executed. "
        "The '.py' extension may not be included."
    )

    parser.add_argument(
        "-p",
        "--params",
        type=str,
        default=None,
        help="Name of the parameters file."
    )

    parser.add_argument(
        "-v", 
        "--verbose",
        action="store_true",
        help="Whether to run with verbosity."
    )

    return

def use_steering_file(
    steering: Optional[str],
    analysis: Optional[str],
    params: Optional[str]
) -> bool:
    """This function gets three of the arguments passed to the
    waffles main program, namely steering (caught from -s, --steering),
    analysis (caught from -a, --analysis) and params (caught from -p,
    --params). This function raises a 
    waffles.Exceptions.IncompatibleInput exception if the given input
    is not valid (meaning the given arguments are not compatible
    with each other). If the given input is valid, then the function
    ends execution normally, returning a boolean value which means
    whether the main program should be run using an steering file or
    not. To this end, this function only checks whether the given
    arguments are defined or not, but their value (if they are defined)
    is irrelevant.
    
    Parameters
    ----------
    steering: None or str
        The path to the steering file. The input given to this
        parameter should be the input given to the -s, --steering
        flag of the main program.
    analysis: None or str
        The name of the analysis class to be executed. The input
        given to this parameter should be the input given to the
        -a, --analysis flag of the main program.
    params: None or str
        The name of the parameters file. The input given to this
        parameter should be the input given to the -p, --params
        flag of the main program.

    Returns
    -------
    fUseSteeringFile: bool
        Indicates whether the main program should be run using
        an steering file
    """

    fUseSteeringFile = None

    # args.steering is defined
    if steering is not None:

        # args.analysis and/or args.params
        # are defined as well
        if analysis is not None or \
            params is not None:

            raise we.IncompatibleInput(
                we.GenerateExceptionMessage(
                    1,
                    'use_steering_file()',
                    reason="The given input is not valid since the "
                    "'steering' parameter was defined along with the "
                    "'analysis' and/or 'params' parameter. Note that "
                    "the 'steering' parameter is mutually exclusive "
                    "with the 'analysis' parameter, and the 'params' "
                    "parameter."
                )
            )

        # args.analysis and args.params
        # are not defined
        else:
            fUseSteeringFile = True
    
    # args.steering is not defined
    else:

        # Neither args.steering, args.analysis
        # nor args.params are defined
        if analysis is None and \
            params is None:
            fUseSteeringFile = True

        # args.steering is not defined, but
        # args.analysis or args.params are
        else:
            fUseSteeringFile = False

    return fUseSteeringFile
