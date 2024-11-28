from abc import ABC, abstractmethod
import waffles.Exceptions as we
import argparse
import yaml
import pathlib

class WafflesAnalysis(ABC):
    """This abstract class implements a Waffles Analysis.
    It fixes a common interface and workflow for all
    Waffles analyses.

    Attributes
    ----------
    ## Add the attributes here. Example: 
    time_step_ns: float (inherited from WaveformAdcs)
    daq_window_timestamp: int
        The timestamp value for the DAQ window in which
        this Waveform was acquired

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    def __init__(self):

        # self.path_to_input_file 
        # self.path_to_output_file 
        
        # self.read_input_loop = self.define_read_input_loop()
        # self.analyze_loop

        # self.analyze_itr    = None 
        # self.read_input_itr = None
        
        pass

    # @abstractmethod
    # def define_read_input_loop():
    #     pass
    
    @abstractmethod
    def arguments(
            parse: argparse.ArgumentParser
        ):
        pass

    @abstractmethod
    def initialize(
            args: list
        ):
        pass

    @abstractmethod
    def read_input() -> bool:
        pass

    @abstractmethod
    def analyze() -> bool:
        pass

    @abstractmethod    
    def write_output():
        pass

    def base_arguments(
            self,
            first_arg: str
        ):

        parse = argparse.ArgumentParser()

        if first_arg != '-i' and first_arg != '-h':
            parse.add_argument(
                'analysis',
                type=str,
                help="name of the analysis to process"
            )

        # input and output files
        parse.add_argument(
            '-i',
            '--input_file',
            type=str,
            required=False,
            help="path of input file or folder"
        )
        parse.add_argument(
            '-o',
            '--output_file',
            type=str,
            required=False,
            help="path to output file",
            default=None)

        # call the method of the derived class to add additional arguments
        self.arguments(parse)

        # parse the arguments 
        args = vars(parse.parse_args())

        return args

    def base_initialize(self, args):        

        # asign the input and output files to data members
        self.path_to_input_file = args['input_file']
        self.path_to_output_file = args['output_file']

        print(
            'Input file:',
            self.path_to_input_file
        )
        print (
            'Output file:',
            self.path_to_output_file
        )

        # by default there is no iteration (only one elemet in list)
        self.read_input_loop =[0]
        self.analyze_loop =[0]

        # call the method of the derived class
        self.initialize(args)

    def execute(
            self,
            first_arg: str
        ):

        args = self.base_arguments(first_arg)
        self.base_initialize(args)

        for self.read_input_itr in self.read_input_loop:
            print(f"read_input loop: {self.read_input_itr}")
            if not self.read_input():
                continue

            for self.analyze_itr in self.analyze_loop:
                print(f'analyze loop: {self.analyze_itr}')
                if not self.analyze():
                    continue

                if self.path_to_output_file != None: 
                    self.write_output()
        

    @staticmethod
    def __steering_file_meets_requirements(
        steering_file_path: pathlib.Path
    ) -> None:
        """This helper static method should only be called
        by the analysis_folder_meets_requirements() method.
        It checks that the given path points to an existing
        file, whose name ends with '.yml' and that this
        (assumed YAML) file abides by the following structure:

            - It contains at least one key
            - Its keys are consecutive integers starting from 1
            - The sub-keys of each key are 'name' and 'parameters'
            - The value for each 'name' sub-keys is an string, say
            x, such that the file 'x.py' exists alongside the
            steering file
            - The value for each 'parameter' sub-keys is an string

        If any of these conditions is not met, a
        waffles.Exceptions.IllFormedSteeringFile exception
        is raised. If the given steering file meets the specified
        requirements, then this method ends execution normally.

        Parameters
        ----------
        steering_file_path: pathlib.Path
            The path to the steering file to be checked. It is
            assumed to be a YAML file.

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
            
            for aux in ('name', 'parameters'):

                if aux not in content[key].keys():
                    raise we.IllFormedSteeringFile(
                        we.GenerateExceptionMessage(
                            7,
                            'WafflesAnalysis.__steering_file_meets_requirements()',
                            reason=f"The key {key} must contain a '{aux}' key."
                        )
                    )
                
                if not isinstance(content[key][aux], str):
                    raise we.IllFormedSteeringFile(
                        we.GenerateExceptionMessage(
                            8,
                            'WafflesAnalysis.__steering_file_meets_requirements()',
                            reason=f"The value of the '{aux}' key of the key "
                            f"{key} must be an string."
                        )
                    )
        
            if not pathlib.Path(
                steering_file_path.parent,
                content[key]['name'] + '.py'
            ).exists():
                
                raise we.IllFormedSteeringFile(
                    we.GenerateExceptionMessage(
                        9,
                        'WafflesAnalysis.__steering_file_meets_requirements()',
                        reason=f"The file '{content[key]['name']}.py' must "
                        "exist alongisde the steering file."
                    )
                )

        return
    
    @staticmethod
    def __check_file_or_folder_exists(
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
                        'WafflesAnalysis.__check_file_or_folder_exists()',
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
                        'WafflesAnalysis.__check_file_or_folder_exists()',
                        reason=f"The folder '{name}' is not found in the "
                        f"folder '{folder_path}'."
                    )
                )
        return
