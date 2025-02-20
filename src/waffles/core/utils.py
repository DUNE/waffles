import argparse
import pathlib
import yaml
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

def get_ordered_list_of_analyses(
        args: argparse.Namespace,
        remaining_args: list,
        verbose: bool = False
) -> list:
    """This function gets the arguments parsed by the main program
    and the remaining arguments that were not recognized by the parser.
    It returns a list of the analyses to be executed, whose order
    match the execution order, and which follow an unified format
    regardless of whether an steering file is used or not.

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
            The name of the analysis class to be executed. If
            an steering file is used, then the analysis name
            comes from the value of the 'name' sub-key, for each
            analysis. If an steering file is not used, then the
            analysis name comes from the value given to the -a,
            --analysis argument, if such argument is defined.
            It is set to 'Analysis1' by default, if an steering
            file is not used and the -a, --analysis argument
            is not defined.
        - parameters_file: str
            If it is a non-empty string, then is interpreted
            as the name of the file from where to gather the
            parameters to be used. If it matches "", then no
            parameters file is used. If an steering file is
            used, then the parameters-file name comes from
            the value of the 'parameters_file' sub-key, for
            each analysis. If an steering file is not used,
            then the parameters-file name comes from the value
            given to the -p, --params argument, if defined,
            or set to "" otherwise.
        - overwriting_parameters: str
            An string, in the format which is normally given
            to a python shell comand, containing the parameters
            to be used. These parameters should overwrite those
            which are gotten from the parameters file, if any.
            If an steering file is used, these parameters
            come from the value of the 'overwriting_parameters'
            sub-key, for each analysis. If an steering file is
            not used, then these parameters are only defined
            if additional (a priori unrecognized) arguments
            were given to the main program. In this case, the
            value of this key is the string which represents
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
            
            if len(remaining_args) > 0:
                print(
                    "In function get_ordered_list_of_analyses(): "
                    "Note that the given additional arguments "
                    f"({remaining_args}) will be ignored"
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
        # analysis_folder_meets_requirements() only cares about
        # the default steering file.
        if args.steering is not None:
            steering_file_meets_requirements(
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

    else:
        if args.analysis is not None:
            check_analysis_class(
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

            aux_parameters_file = args.params
            if verbose:
                print(
                    "In function get_ordered_list_of_analyses(): "
                    f"Using specified parameters file '{aux_parameters_file}'"
                )
        else:
            aux_parameters_file = ""
            if verbose:
                print(
                    "In function get_ordered_list_of_analyses(): "
                    "No parameters file was given"
                )

        aux_overwriting_parameters = " ".join(remaining_args)
        if len(aux_overwriting_parameters) > 0:
            print(
                "In function get_ordered_list_of_analyses(): "
                f"Using the additionally given arguments "
                f"({aux_overwriting_parameters}) as preferred parameters"
            )

        # Arrange an unique-entry dictionary just to be
        # consistent with the dictionary that is returned
        # when an steering file is used
        analyses = {
            1:{
                'name': aux_name,
                'parameters_file': aux_parameters_file,
                'overwriting_parameters': aux_overwriting_parameters
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
    file. If the given YAML file is empty, then an
    empty dictionary is returned.
    
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

    try:
        with open(parameters_file_name, 'r') as file:
            loaded_dict = yaml.load(
                file, 
                Loader=yaml.Loader
            )
    except yaml.parser.ParserError as e:
        raise we.WafflesBaseException(
            we.GenerateExceptionMessage(
                2,
                '__build_parameters_dictionary_from_file()',
                reason=f"The YAML module threw the following "
                f"error while parsing file '{parameters_file_name}'."
                f" \n {e}"
            )
        )

    if loaded_dict is None:
        loaded_dict = {}

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
        raise we.WafflesBaseException(
            we.GenerateExceptionMessage(
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
    """This function checks that the given folder contains
    a file or folder with the given name, up to the input
    given to the is_file parameter. If it is not found,
    a FileNotFoundError is raised. If it is found, then
    this function ends execution normally.
    
    Parameters
    ----------
    folder_path: pathlib.Path
        The path to the folder to be checked
    name: str
        The name of the file (resp. folder) 
        to be checked, if is_file is True 
        (resp. False)
    is_file: bool
        If True (resp. False), the function
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

def check_analysis_class(
    analysis_name: str,
    analysis_folder_path: pathlib.Path
) -> None:
    """This function gets an analysis name and the
    path to the folder from which the analysis is
    being run. It checks that the analysis name
    follows the format 'Analysis<i>', where i is an
    integer >=1, and that the file 'Analysis<i>.py'
    exists in the given folder. If any of these
    conditions is not met, a
    waffles.Exceptions.IllFormedAnalysisClass exception
    is raised. If the given analysis class meets the
    specified requirements, then this function ends
    execution normally.

    Parameters
    ----------
    analysis_name: str
        The name of the analysis class to be checked
    analysis_folder_path: pathlib.Path
        The path to the folder from which the analysis
        is being run

    Returns
    ----------
    None
    """

    if not analysis_name.startswith('Analysis'):
        raise we.IllFormedAnalysisClass(
            we.GenerateExceptionMessage(
                1,
                'check_analysis_class()',
                reason=f"The analysis class name ({analysis_name}) "
                "must start with 'Analysis'."
            )
        )
    
    try:
        i = int(analysis_name[8:])

    except ValueError:
        raise we.IllFormedAnalysisClass(
            we.GenerateExceptionMessage(
                2,
                'check_analysis_class()',
                reason=f"The analysis class name ({analysis_name}) "
                "must follow the 'Analysis<i>' format, with i being "
                "an integer."
            )
        )
    else:
        if i < 1:
            raise we.IllFormedAnalysisClass(
                we.GenerateExceptionMessage(
                    3,
                    'check_analysis_class()',
                    reason=f"The integer ({i}) at the end of the "
                    f"analysis class name ({analysis_name}) must be >=1."
                )
            )

    if not pathlib.Path(
        analysis_folder_path,
        analysis_name + '.py'
    ).exists():
        
        raise we.IllFormedAnalysisClass(
            we.GenerateExceptionMessage(
                4,
                'check_analysis_class()',
                reason=f"The file '{analysis_name}.py' must exist "
                f"in the analysis folder ({analysis_folder_path})."
            )
        )
    
    return

def steering_file_meets_requirements(
    steering_file_path: pathlib.Path
) -> None:
    """This function checks that the given path points
    to an existing file, whose name ends with '.yml' and
    that this (assumed YAML) file abides by the following
    structure:

        - It contains at least one key
        - Its keys are consecutive integers starting
        from 1
        - The sub-keys of each key are 'name',
        'parameters_file' and 'overwriting_parameters'
        - The value for each 'name' sub-keys is an
        string, say x, that meets the following
        sub-requirements:
            - x follows the format "Analysis<i>", where
            i is an integer >=1
            - the file 'x.py' exists alongside the
            steering file
        - The value for each 'parameters_file' sub-keys
        is an string. If it is different from an emtpy
        string, then it is interpreted as the name
        of a parameters file which must exist alongside
        the steering file. If it is an empty string,
        then it is assumed that no parameters file was
        given, and this parameter is ignored.
        - The value for each 'overwriting_parameters'
        sub-keys is an string, which is interpreted as
        the string that would be given as part of a
        shell command. The parameter values extracted
        from this string should overwrite those gotten
        from the parameters file, if any. If this value
        is an empty string, then no parameters are
        overwritten.

    If any of these conditions is not met, a
    waffles.Exceptions.IllFormedSteeringFile exception
    is raised. If the given steering file meets the
    specified requirements, then this function ends
    execution normally.

    Parameters
    ----------
    steering_file_path: pathlib.Path
        The path to the steering file to be checked.
        It is assumed to be a YAML file.

    Returns
    ----------
    None
    """

    if not steering_file_path.exists():
        raise we.IllFormedSteeringFile(
            we.GenerateExceptionMessage(
                1,
                'steering_file_meets_requirements()',
                reason=f"The file '{steering_file_path}' does not exist."
            )
        )

    if steering_file_path.suffix != '.yml':
        raise we.IllFormedSteeringFile(
            we.GenerateExceptionMessage(
                2,
                'steering_file_meets_requirements()',
                reason=f"The file '{steering_file_path}' must have a '.yml' "
                "extension."
            )
        )

    with open(
        steering_file_path,
        'r'
    ) as file:
        
        content = yaml.load(
            file, 
            Loader=yaml.Loader
        )

    if not isinstance(content, dict):
        raise we.IllFormedSteeringFile(
            we.GenerateExceptionMessage(
                3,
                'steering_file_meets_requirements()',
                reason="The content of the given steering file must be a "
                "dictionary."
            )
        )
    
    if len(content) == 0:
        raise we.IllFormedSteeringFile(
            we.GenerateExceptionMessage(
                4,
                'steering_file_meets_requirements()',
                reason="The given steering file must contain at "
                "least one key."
            )
        )
    
    keys = list(content.keys())
    keys.sort()

    if keys != list(range(1, len(keys) + 1)):
        raise we.IllFormedSteeringFile(
            we.GenerateExceptionMessage(
                5,
                'steering_file_meets_requirements()',
                reason="The keys of the given steering file must "
                "be consecutive integers starting from 1."
            )
        )
    
    for key in keys:
        if not isinstance(content[key], dict):
            raise we.IllFormedSteeringFile(
                we.GenerateExceptionMessage(
                    6,
                    'steering_file_meets_requirements()',
                    reason=f"The value of the key {key} must be a "
                    "dictionary."
                )
            )

        for aux in ('name', 'parameters_file', 'overwriting_parameters'):

            if aux not in content[key].keys():
                raise we.IllFormedSteeringFile(
                    we.GenerateExceptionMessage(
                        7,
                        'steering_file_meets_requirements()',
                        reason=f"The key {key} must contain a '{aux}' key."
                    )
                )

            if not isinstance(
                content[key][aux],
                str
            ):
                raise we.IllFormedSteeringFile(
                    we.GenerateExceptionMessage(
                        8,
                        'steering_file_meets_requirements()',
                        reason=f"The value of the '{aux}' sub-key of the key "
                        f"{key} must be an string."
                    )
                )
            
        check_analysis_class(
            content[key]['name'],
            steering_file_path.parent
        )

        if content[key]['parameters_file'] != '':
            check_file_or_folder_exists(
                steering_file_path.parent,
                content[key]['parameters_file'],
                is_file=True
            )

    return

def analysis_folder_meets_requirements():
    """This function checks that the folder structure of
    the folder from which the analysis is being executed
    follows the required structure. It will raise a 
    waffles.Exceptions.IllFormedAnalysisFolder exception
    otherwise. The list of the checked requirements is
    the following:

    1) The folder contains a file called 'steering.yml',
    which specifies, by default, the order in which
    different analysis (if many) should be executed and
    which parameters to use for each analysis stage. This
    file must be a YAML file which must follow the
    structure described in the
    steering_file_meets_requirements() function docstring.
    2) The folder contains a file called 'utils.py',
    which may contain utility functions used by the
    analysis.
    3) The folder contains a file called 'params.yml',
    which contains the input parameters used, by default,
    by the analysis.
    4) The folder contains a file called 'imports.py',
    which contains the imports needed by the analysis.
    5) The folder contains a file called 'Analysis1.py',
    where 'Analysis1' is the name of the analysis class
    which implements the first (and possibly the unique)
    analysis stage. It gives the analysis to be executed
    by default.
    6) The folder contains a sub-folder called 'configs',
    which may contain configuration files which are not
    as volatile as the input parameters.
    7) The folder contains a sub-folder called 'output',
    which is meant to store the output of the first
    (and possibly unique) analysis stage, and possibly
    the inputs and outputs for the rest of the analysis
    stages.

    The function also checks whether sub-folders called
    'data' and 'scripts' exist. If they don't exist
    an exception is not raised, but a warning message
    is printed.
    """

    analysis_folder_path = pathlib.Path.cwd()

    steering_file_meets_requirements(
        pathlib.Path(
            analysis_folder_path,
            'steering.yml'
        )
    )

    check_file_or_folder_exists(
        analysis_folder_path,
        'utils.py',
        is_file=True
    )

    check_file_or_folder_exists(
        analysis_folder_path,
        'params.yml',
        is_file=True
    )

    check_file_or_folder_exists(
        analysis_folder_path,
        'imports.py',
        is_file=True
    )

    check_file_or_folder_exists(
        analysis_folder_path,
        'Analysis1.py',
        is_file=True
    )

    check_file_or_folder_exists(
        analysis_folder_path,
        'configs',
        is_file=False
    )

    check_file_or_folder_exists(
        analysis_folder_path,
        'output',
        is_file=False
    )

    try:
        check_file_or_folder_exists(
            analysis_folder_path,
            'data',
            is_file=False
        )
    except FileNotFoundError:
        print(
            "In function analysis_folder_meets_requirements(): "
            "A 'data' folder does not exist in the analysis folder."
        )

    try:
        check_file_or_folder_exists(
            analysis_folder_path,
            'scripts',
            is_file=False
        )
    except FileNotFoundError:
        print(
            "In function analysis_folder_meets_requirements(): "
            "An 'scripts' folder does not exist in the analysis folder."
        )
    
    return