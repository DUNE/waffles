import argparse
import pathlib
import yaml
from typing import Optional

from waffles.data_classes.WafflesAnalysis import WafflesAnalysis
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

def get_ordered_list_of_analyses(
        args: argparse.Namespace,
        remaining_args: list,
        verbose: bool = False
) -> list:
    """This function gets the arguments parsed by the main program
    and the remaining arguments that were not recognized by the parser.
    It returns a list of the analyses to be executed, whose order
    match the execution order.

    Parameters
    ----------
    args: argparse.Namespace
        The arguments parsed by the main program. It should be
        the first output of the parse_known_args() method of
        the used argparse.ArgumentParser instance.
    remaining_args: list
        The remaining arguments that were not recognized by the
        parser. It should be the second output of the parse_known_args()
        method of the used argparse.ArgumentParser instance.
    verbose: bool
        Whether to run with verbosity.

    Returns
    ----------
    analyses: list
        The ordered list of analyses to be executed. Each
        element of the list is a dictionary with the following
        keys:
    
        - name: str
            The name of the analysis class to be executed
        - parameters: str
            Either the name of the parameters file to be used or
            a string which represents the parameters to be used,
            in the format which is normally given to a python
            shell command.
        - parameters_is_file: bool
            Whether the 'parameters' key is a file name or not.
        - preferred_parameters: str
            Parameters which may overwrite those which are
            fetched from the 'parameters' entry. This key is
            only present in the following case:
                - An steering file is not used
                - The -p, --params argument is defined
                - Additional (a priori unrecognized) arguments
                were given to the main program.
            The value of this key is the string which represents
            these unrecognized arguments, following the same
            format in which they appeared in the python command
            which called the main program.
    """

    fUseSteeringFile = use_steering_file(
        args.steering,
        args.analysis,
        args.params
    )

    if verbose:
        if fUseSteeringFile:
            print(
                "In function get_ordered_list_of_analyses(): "
                "Running with an steering file"
            )
        else:
            print(
                "In function get_ordered_list_of_analyses(): "
                "Running without an steering file"
            )

    if fUseSteeringFile:

        # If an steering file other than the default one is used,
        # we still need to check that it exists in the analysis folder
        # (in the CWD) and that it meets the requirements. I.e. 
        # WafflesAnalysis.analysis_folder_meets_requirements()
        # only cares about the default steering file.
        if args.steering is not None:
            WafflesAnalysis._WafflesAnalysis__steering_file_meets_requirements(
                pathlib.Path(
                    pathlib.Path.cwd(),
                    args.steering
                )
            )
            aux = args.steering

            if verbose:
                print(
                    "In function get_ordered_list_of_analyses(): "
                    f"Using specified steering file '{aux}'"
                )
        else:
            aux = 'steering.yml'

            if verbose:
                print(
                    "In function get_ordered_list_of_analyses(): "
                    "An steering file was not specified. The default "
                    f"one ('{aux}') will be used."
                )

        with open(aux, 'r') as file:

            analyses = yaml.load(
                file,
                Loader=yaml.Loader
            )

            # The 'preferred_parameters' key must not be present
            # in the steering file. We are adding it here for
            # consistency with the case where an steering file
            # is not used but the -p, --params argument is given
            # simultaneously with some spare arguments that are
            # appended to the shell command.
            for key in analyses:
                analyses[key]['preferred_parameters'] = ''
    else:
        if args.analysis is not None:
            WafflesAnalysis._WafflesAnalysis__check_analysis_class(
                args.analysis,
                pathlib.Path.cwd()
            )
            aux_name = args.analysis

            if verbose:
                print(
                    "In function get_ordered_list_of_analyses(): "
                    f"Using specified analysis class '{aux_name}'"
                )
        else:
            aux_name = 'Analysis1'

            if verbose:
                print(
                    "In function get_ordered_list_of_analyses(): "
                    "An analysis class was not specified. The default "
                    f"one ('{aux_name}') will be used."
                )

        # Means that a -p, --params argument was given
        # which gives the name of the parameters file 
        if args.params is not None:

            # In this case, check that the given parameters
            # file exists in the analysis folder
            check_file_or_folder_exists(
                pathlib.Path.cwd(),
                args.params,
                is_file=True
            )

            aux_params = args.params
            aux_parameters_is_file = True
            aux_preferred_parameters = " ".join(remaining_args)

            if verbose:
                print(
                    "In function get_ordered_list_of_analyses(): "
                    f"Using specified parameters file '{aux_params}'"
                )

                if len(aux_preferred_parameters) > 0:
                    print(
                        "In function get_ordered_list_of_analyses(): "
                        f"Using the additionally given arguments "
                        f"({aux_preferred_parameters}) as preferred parameters"
                    ) 

        # If no parameters file was given, then
        # assume that the unrecognized arguments
        # are the analysis parameters
        else:
            aux_params = " ".join(remaining_args)
            aux_parameters_is_file = False
            aux_preferred_parameters = ""

            if verbose:
                print(
                    "In function get_ordered_list_of_analyses(): "
                    "No parameters file was given."
                )

                if len(aux_params) > 0:
                    print(
                        "In function get_ordered_list_of_analyses(): "
                        "Using the additionally given arguments "
                        f"({aux_params}) as default parameters."
                    )

        # Arrange an unique-entry dictionary just to be
        # consistent with the dictionary that is returned
        # when an steering file is used
        analyses = {
            1:{
                'name': aux_name,
                'parameters': aux_params,
                'parameters_is_file': aux_parameters_is_file,
                'preferred_parameters': aux_preferred_parameters
            }
        }

    # The steering file should have been checked to have
    # keys which are consecutive integers starting from 1
    ordered_list_of_analyses = [
        analyses[i] for i in range(1, 1 + len(analyses))
    ]
    
    return ordered_list_of_analyses

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
                    "'steering' parameter (-s, --steering) was "
                    "defined along with the 'analysis' (-a, --analysis)"
                    " and/or 'params' (-p, --params) parameter. Note "
                    "that the 'steering' parameter is mutually exclusive "
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

def build_parameters_dictionary(
        parameters_file_name: Optional[str] = None,
        parameters_shell_string: Optional[str] = None,
        prioritize_string_parameters: bool = True,
        verbose: bool = False
) -> dict:
    """This function gets the name of a .yml parameters file
    and/or an string which defines some input parameters with
    the usual shell-commands format. This function returns a
    dictionary which contains both: the variables defined in
    the given YAML parameters file, and the variables which
    were parsed from the shell string. If the same variable is
    found in both the parameters file and the shell string,
    then the value of the variable in the shell string will
    overwrite the value of the variable in the parameters file
    if prioritize_string_parameters is set to True. Otherwise,
    the value of the variable in the parameters file will
    overwrite that of the variable in the shell string. If none
    of the optional parameters are given, then an empty
    dictionary is returned.

    Parameters
    ----------
    parameters_file_name: None or str
        The name of the parameters file. Its extension must match
        '.yml'. It is assumed that this file is present in the
        current working directory.
    parameters_shell_string: None or str
        A string which defines some input parameters with the
        usual shell-commands format
    prioritize_string_parameters: bool
        Whether to prioritize the parameters which are defined
        in the shell string over those which are defined in the
        parameters file. 
    verbose: bool
        Whether to run with verbosity
    
    Returns
    ----------
    parameters_dict: dict
        A dictionary which contains all of the variables which
        were found in the parameters file and all of the variables
        which were parsed from the shell string.
    """

    file_dict = __build_parameters_dictionary_from_file(
        parameters_file_name
    ) if parameters_file_name is not None else {}

    if len(file_dict) > 0:
        if verbose:
            print(
                "In function build_parameters_dictionary(): "
                f"Collected the following parameters from the"
                f" file '{parameters_file_name}': {file_dict}"
            )

    shell_dict = __build_parameters_dictionary_from_shell_string(
        parameters_shell_string,
        verbose=verbose
    ) if parameters_shell_string is not None else {}

    if len(shell_dict) > 0:
        if verbose:
            print(
                "In function build_parameters_dictionary(): "
                f"Collected the following parameters from the"
                f" string '{parameters_shell_string}': {shell_dict}"
            )

    if prioritize_string_parameters:
        return {**file_dict, **shell_dict}
    else:
        return {**shell_dict, **file_dict}

def __build_parameters_dictionary_from_file(
        parameters_file_name: str
) -> dict:
    """This helper function gets the name of a .yml 
    parameters file and creates a dictionary which contains
    all of the variables which were found in the parameters
    file.
    
    Parameters
    ----------
    parameters_file_name: str
        The name of the parameters file. This file must be
        located in the current working directory, and its
        extension must match '.yml'.
        
    Returns
    ----------
    loaded_dict: dict
        A dictionary which contains all of the variables which
        were found in the parameters file
    """

    check_file_or_folder_exists(
        pathlib.Path.cwd(),
        parameters_file_name,
        is_file=True
    )

    if not parameters_file_name.endswith('.yml'):
        raise we.IllFormedParametersFile(
            we.GenerateExceptionMessage(
                1,
                '__build_parameters_dictionary_from_file()',
                reason="The given parameters file must have a '.yml' "
                "extension."
            )
        )

    with open(parameters_file_name, 'r') as file:
        loaded_dict = yaml.load(
            file, 
            Loader=yaml.Loader
        )

    return loaded_dict

def __build_parameters_dictionary_from_shell_string(
        parameters_shell_string: str,
        verbose: bool = False
) -> dict:
    """This helper function gets an string which defines some
    input parameters with the usual shell-commands format. It
    returns a dictionary which contains all of the variables
    which were parsed from the shell string. 

    Parameters
    ----------
    parameters_shell_string: str
        A string which defines some input parameters with the
        usual shell-commands format. It must have at least
        length 2, and the first word must start with a dash
        ('-'). Otherwise, an exception is raised. If a parameter
        is repeated, then it is overwritten with the last given
        value. If two or more consecutive words do not start
        with a dash, then all of them except for the first one
        are ignored.
    
    Returns
    ----------
    parameters_dict: dict
        A dictionary which contains all of the variables which
        were parsed from the shell string. The keys are strings.
        The values are either boolean or strings.
    """

    if len(parameters_shell_string) < 2:
        raise we.WafflesBaseException(
            we.GenerateExceptionMessage(
                1,
                '__build_parameters_dictionary_from_shell_string()',
                reason="The given shell string "
                f"('{parameters_shell_string}') must have at least "
                "length 2, p.e. '-v'."
            )
        )

    chunks = parameters_shell_string.split()

    if not chunks[0].startswith('-'):
        raise Exception(
            we.WafflesBaseException(
                2,
                '__build_parameters_dictionary_from_shell_string()',
                reason=f"The first word ('{chunks[0]}') of the given "
                f"shell string ('{parameters_shell_string}') must start "
                "with a dash ('-')."
            )
        )

    parameters_dict = {}
    key = chunks[0].strip('-')
    fGotKey = True
    i = 1
    fReachedEnd = (len(chunks) == 1)

    while not fReachedEnd:

        if chunks[i].startswith('-'):

            if fGotKey:
                parameters_dict[key] = True
                key = chunks[i].strip('-')
                fGotKey = True
            else:
                key = chunks[i].strip('-')
                fGotKey = True
        else:

            if fGotKey:
                parameters_dict[key] = chunks[i]
                fGotKey = False
            else:
                if verbose:
                    print(
                        "In function __build_parameters_dictionary_from_shell_string():"
                        f" Ignoring value ('{chunks[i]}') with no key."
                    )
        i += 1
        fReachedEnd = (i == len(chunks))

    if fGotKey:
        parameters_dict[key] = True

    # Ill-formed inputs can yield empty strings
    # as keys or values: Filter those out
    return __purge_parameters_dictionary(parameters_dict)

def __purge_parameters_dictionary(
        input_: dict
) -> dict:
    """This helper function takes an input dictionary and
    deletes from it any key-value pair for which its key
    or value matches an empty string.
    
    Parameters
    ----------
    input_: dict
    
    Returns
    ----------
    dict
    """

    keys_to_delete = []
    for key in input_.keys():
        if key == '' or input_[key] == '':
            keys_to_delete.append(key)

    for key in keys_to_delete:
        del input_[key]

    return input_

def empty_string_to_None(
        input_: str
) -> Optional[str]:
    """This function takes an string as input and returns
    None if the input is an empty string. Otherwise, it
    returns the input string.
    
    Parameters
    ----------
    input_: str
    
    Returns
    ----------
    Optional[str]
    """

    return None if input_ == '' else input_

def check_file_or_folder_exists(
    folder_path: pathlib.Path,
    name: str,
    is_file: bool = True
) -> None:
    """This helper static method checks that the given
    folder contains a file or folder with the given
    name, up to the input given to the is_file parameter.
    If it is not found, a FileNotFoundError is raised.
    If it is found, then the method ends execution
    normally.
    
    Parameters
    ----------
    folder_path: pathlib.Path
        The path to the folder to be checked
    name: str
        The name of the file (resp. folder) 
        to be checked, if is_file is True 
        (resp. False)
    is_file: bool
        If True (resp. False), the method
        checks for the existence of a file
        (resp. folder) with the given name
        in the given folder path.

    Returns
    ----------
    None
    """

    if is_file:
        if not pathlib.Path(
            folder_path,
            name
        ).is_file():
        
            raise FileNotFoundError(
                we.GenerateExceptionMessage(
                    1,
                    'check_file_or_folder_exists()',
                    reason=f"The file '{name}' is not found in the "
                    f"folder '{folder_path}'."
                )
            )
    else:
        if not pathlib.Path(
            folder_path,
            name
        ).is_dir():
            
            raise FileNotFoundError(
                we.GenerateExceptionMessage(
                    2,
                    'check_file_or_folder_exists()',
                    reason=f"The folder '{name}' is not found in the "
                    f"folder '{folder_path}'."
                )
            )
    return

