from abc import ABC, abstractmethod
import pathlib
import yaml
from pydantic import BaseModel, Field
import waffles.core.utils as wcu
import waffles.Exceptions as we

class BaseInputParams(BaseModel):

    input_path: str = Field(
        ..., 
        description="Path to the input file or folder",
        example='input/'
    )

    output_path: str = Field(
        default='output/', 
        description="Path to the output file or folder"
    )

class WafflesAnalysis(ABC):
    """This abstract class implements a Waffles Analysis.
    It fixes a common interface and workflow for all
    Waffles analyses.

    Attributes
    ----------
    read_input_loop: list
        # Add description of this parameter
    analyze_loop: list
        # Add description of this parameter
    analyze_itr: list
        # Add description of this parameter
    read_input_itr: list
        # Add description of this parameter

    Methods
    ----------
    get_input_params_model():
        Class abstract method which is responsible for
        defining and returning a validation model for the
        input parameters of the analysis. This method must
        be implemented by each derived analysis class.
    initialize(input_parameters: BaseInputParams):
        Abstract method which is responsible for defining
        both, the common instance attributes (namely
        self.read_input_loop, self.analyze_loop,
        self.analyze_itr and self.read_input_itr) and
        whichever further attributes are required by the
        analysis. The defined attributes are potentially
        used by the read_input(), analyze() and write_output()
        methods.
    read_input():
        Abstract method which is responsible for reading
        the input data for the analysis, p.e. Waffles
        objects such as WaveformSet's. For more information,
        refer to its docstring.
    analyze():
        Abstract method which is responsible for performing
        the analysis on the input data. For more information,
        refer to its docstring.
    write_output():
        Abstract method which is responsible for writing
        the output of the analysis. For more information,
        refer to its docstring.
    """

    def __init__(self):
        """Initializer of the WafflesAnalysis class. It
        initializes each attribute of the class to a
        dummy None.
        """
    
        # Logically useless. Just to hint the
        # user that these attributes are meant
        # to be defined in the initialize() method.

        self.read_input_loop = None
        self.analyze_loop = None

        self.analyze_itr    = None 
        self.read_input_itr = None
        
        pass

    @classmethod
    @abstractmethod
    def get_input_params_model(
        cls
    ) -> type:
        """This class method must be implemented by each
        derived analysis class. It must define and return
        a Pydantic model which will be used to validate
        the input parameters given to the analysis. The
        model must inherit from the BaseInputParams
        class."""
        pass

    @abstractmethod
    def initialize(
            self,
            input_parameters: BaseInputParams
        ) -> None:
        """This method must be implemented by each derived
        analysis class. It is responsible for defining the
        instance attributes of the analysis class out of
        the given input parameters, which will abide by
        the model defined in the get_input_params_model()
        class method. Note that these instance attributes
        include both the common attributes (namely
        self.read_input_loop, self.analyze_loop,
        self.analyze_itr and self.read_input_itr) and
        whichever further attribute is required by the
        analysis. The defined attributes are potentially
        used by the read_input(), analyze() and
        write_output() methods.
        
        Parameters
        ----------
        input_parameters: BaseInputParams
            The input parameters given to the analysis
        
        Returns
        ----------
        None
        """
        pass

    @abstractmethod
    def read_input(self) -> bool:
        """This method must be implemented by each derived
        analysis class. It is responsible for reading the
        input data for the analysis, p.e. Waffles objects
        such as WaveformSet's. The execute() method will
        potentially loop this method over the
        self.read_input_loop attribute, which should have
        been defined in the initialize() method.
        
        Returns
        ----------
        bool
            True if the reading process ended normally, 
            False otherwise"""
        pass

    @abstractmethod
    def analyze(self) -> bool:
        """This method must be implemented by each derived
        analysis class. It is responsible for performing
        the analysis on the input data. The execute() method
        will potentially loop this method over the
        self.analyze_loop attribute, which should have been
        defined in the initialize() method.

        Returns
        ----------
        bool
            True if the analysis process ended normally,
            False otherwise
        """
        pass

    @abstractmethod    
    def write_output(self) -> bool:
        """This method must be implemented by each derived
        analysis class. It is responsible for writing the
        output of the analysis. The execute() method may
        call this method every time the analyze() method
        have been called and returned True.
        
        Returns
        ----------
        bool
            True if the writing process ended normally,
            False otherwise
        """
        pass

    def execute(
            self,
            input_parameters: BaseInputParams,
            verbose: bool = False
        ) -> None:
        """This method is responsible for executing the
        analysis. It serves as a hook for the main program.
        It should call the initialize() method once, then 
        iterate the read_input(), analyze() and write_output()
        methods according to the self.read_input_loop and
        self.analyze_loop attributes.
        
        Parameters
        ----------
        input_parameters: BaseInputParams
            The input parameters given to the analysis.
            This input should match the output of the
            validation of the input-parameters model
            returned by the get_input_params_model()
            class method.
        verbose: bool
            Whether to print verbose messages or not.

        Returns
        ----------
        None
        """

        self.initialize(input_parameters)

        for self.read_input_itr in self.read_input_loop:

            if verbose:
                print(
                    "In function WafflesAnalysis.execute(): "
                    "Executing iteration of the read-input loop "
                    f"with its iterator set to {self.read_input_itr}"
                )

            if not self.read_input():
                continue

            for self.analyze_itr in self.analyze_loop:

                if verbose:
                    print(
                        "In function WafflesAnalysis.execute(): "
                        "Executing iteration of the analysis loop "
                        f"with its iterator set to {self.analyze_itr}"
                    )

                if not self.analyze():
                    continue

                self.write_output()

    @staticmethod
    def analysis_folder_meets_requirements():
        """This static method checks that the folder structure
        of the folder from which the analysis is being executed
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
        __steering_file_meets_requirements() method docstring.
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

        WafflesAnalysis.__steering_file_meets_requirements(
            pathlib.Path(
                analysis_folder_path,
                'steering.yml'
            )
        )

        wcu.check_file_or_folder_exists(
            analysis_folder_path,
            'utils.py',
            is_file=True
        )

        wcu.check_file_or_folder_exists(
            analysis_folder_path,
            'params.yml',
            is_file=True
        )

        wcu.check_file_or_folder_exists(
            analysis_folder_path,
            'imports.py',
            is_file=True
        )

        wcu.check_file_or_folder_exists(
            analysis_folder_path,
            'Analysis1.py',
            is_file=True
        )

        wcu.check_file_or_folder_exists(
            analysis_folder_path,
            'configs',
            is_file=False
        )

        wcu.check_file_or_folder_exists(
            analysis_folder_path,
            'output',
            is_file=False
        )

        try:
            wcu.check_file_or_folder_exists(
                analysis_folder_path,
                'data',
                is_file=False
            )
        except FileNotFoundError:
            print(
                "In function WafflesAnalysis.analysis_folder_meets_requirements(): "
                "A 'data' folder does not exist in the analysis folder."
            )

        try:
            wcu.check_file_or_folder_exists(
                analysis_folder_path,
                'scripts',
                is_file=False
            )
        except FileNotFoundError:
            print(
                "In function WafflesAnalysis.analysis_folder_meets_requirements(): "
                "An 'scripts' folder does not exist in the analysis folder."
            )
        
        return

    @staticmethod
    def __steering_file_meets_requirements(
        steering_file_path: pathlib.Path
    ) -> None:
        """This helper static method checks that the given
        path points to an existing file, whose name ends with
        '.yml' and that this (assumed YAML) file abides by
        the following structure:

            - It contains at least one key
            - Its keys are consecutive integers starting
            from 1
            - The sub-keys of each key are 'name',
            'parameters' and 'parameters_is_file'
            - The value for each 'name' sub-keys is an
            string, say x, that meets the following
            sub-requirements:
                - x follows the format "Analysis<i>", where
                i is an integer >=1
                - the file 'x.py' exists alongside the
                steering file
            - The value for each 'parameters' sub-keys is
            an string
            - The value for each 'parameters_is_file'
            sub-keys is a boolean. If it is True, then the
            value of the 'parameters' sub-key is interpreted
            as the name of a parameters file which must exist
            alongside the steering file. If it is False, then
            the value of the 'parameters' sub-key is
            interpreted as the string that would be given as
            part of a shell command.

        If any of these conditions is not met, a
        waffles.Exceptions.IllFormedSteeringFile exception
        is raised. If the given steering file meets the
        specified requirements, then this method ends
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
                    'WafflesAnalysis.__steering_file_meets_requirements()',
                    reason=f"The file '{steering_file_path}' does not exist."
                )
            )

        if steering_file_path.suffix != '.yml':
            raise we.IllFormedSteeringFile(
                we.GenerateExceptionMessage(
                    2,
                    'WafflesAnalysis.__steering_file_meets_requirements()',
                    reason=f"The file '{steering_file_path}' must have a '.yml' "
                    "extension."
                )
            )

        with open(
            steering_file_path,
            'r'
        ) as archivo:
            
            content = yaml.load(
                archivo, 
                Loader=yaml.Loader
            )

        if not isinstance(content, dict):
            raise we.IllFormedSteeringFile(
                we.GenerateExceptionMessage(
                    3,
                    'WafflesAnalysis.__steering_file_meets_requirements()',
                    reason="The content of the given steering file must be a "
                    "dictionary."
                )
            )
        
        if len(content) == 0:
            raise we.IllFormedSteeringFile(
                we.GenerateExceptionMessage(
                    4,
                    'WafflesAnalysis.__steering_file_meets_requirements()',
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
                    'WafflesAnalysis.__steering_file_meets_requirements()',
                    reason="The keys of the given steering file must "
                    "be consecutive integers starting from 1."
                )
            )
        
        for key in keys:
            if not isinstance(content[key], dict):
                raise we.IllFormedSteeringFile(
                    we.GenerateExceptionMessage(
                        6,
                        'WafflesAnalysis.__steering_file_meets_requirements()',
                        reason=f"The value of the key {key} must be a "
                        "dictionary."
                    )
                )
            
            for aux in ('name', 'parameters', 'parameters_is_file'):

                if aux not in content[key].keys():
                    raise we.IllFormedSteeringFile(
                        we.GenerateExceptionMessage(
                            7,
                            'WafflesAnalysis.__steering_file_meets_requirements()',
                            reason=f"The key {key} must contain a '{aux}' key."
                        )
                    )
                
                aux_map =  {
                    'name': str, 
                    'parameters': str, 
                    'parameters_is_file': bool
                }

                if not isinstance(
                    content[key][aux],
                    aux_map[aux]
                ):
                    raise we.IllFormedSteeringFile(
                        we.GenerateExceptionMessage(
                            8,
                            'WafflesAnalysis.__steering_file_meets_requirements()',
                            reason=f"The value of the '{aux}' sub-key of the key "
                            f"{key} must be of type {aux_map[aux]}."
                        )
                    )
                
            WafflesAnalysis.__check_analysis_class(
                content[key]['name'],
                steering_file_path.parent
            )

            if content[key]['parameters_is_file']:
                wcu.check_file_or_folder_exists(
                    steering_file_path.parent,
                    content[key]['parameters'],
                    is_file=True
                )

        return
    
    @staticmethod
    def __check_analysis_class(
        analysis_name: str,
        analysis_folder_path: pathlib.Path
    ) -> None:
        """This helper static method gets an analysis name
        and the path to the folder from which the analysis
        is being run. It checks that the analysis name
        follows the format 'Analysis<i>', where i is an
        integer >=1, and that the file 'Analysis<i>.py'
        exists in the given folder. If any of these
        conditions is not met, a
        waffles.Exceptions.IllFormedAnalysisClass exception
        is raised. If the given analysis class meets the
        specified requirements, then this method ends
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
                    'WafflesAnalysis.__check_analysis_class()',
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
                    'WafflesAnalysis.__check_analysis_class()',
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
                        'WafflesAnalysis.__check_analysis_class()',
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
                    'WafflesAnalysis.__check_analysis_class()',
                    reason=f"The file '{analysis_name}.py' must exist "
                    f"in the analysis folder ({analysis_folder_path})."
                )
            )
        
        return
