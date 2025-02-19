# import all necessary files and classes
from waffles.np04_analysis.beam_example.imports import *

class Analysis1(WafflesAnalysis):

    def __init__(self):
        pass        

    ##################################################################
    @classmethod
    def get_input_params_model(
        cls
    ) -> type:
        """Implements the WafflesAnalysis.get_input_params_model()
        abstract method. Returns the InputParams class, which is a
        Pydantic model class that defines the input parameters for
        this analysis.
        
        Returns
        -------
        type
            The InputParams class, which is a Pydantic model class"""
        
        class InputParams(BaseInputParams):
            """Validation model for the input parameters of the LED
            calibration analysis."""

            beam_input_path:      str = Field(...,    description="work in progress")
            wfs_input_path:       str = Field(...,    description="work in progress")
            events_output_path:   str = Field(...,    description="work in progress")
            delta_time:           float = Field(...,  description="work in progress")

        return InputParams

    ##################################################################
    def initialize(
        self,
        input_parameters: BaseInputParams
    ) -> None:
    
        self.analyze_loop = [None,]
        self.params = input_parameters

        self.read_input_loop_1 = [None,]
        self.read_input_loop_2 = [None,]
        self.read_input_loop_3 = [None,]
        
    ##################################################################
    def read_input(self) -> bool:

        np04_file = f'{self.params.input_path}/{self.params.wfs_input_path}'
        beam_file = f'{self.params.input_path}/{self.params.beam_input_path}'
        
        print("Read np04 and beam information from: ")
        print("  np04 PDS pickle file: ", np04_file)
        print("  beam root file:       ", beam_file)

        # Read the two files and create BeamEvents combining their information
        self.events = events_from_pickle_and_beam_files(np04_file, beam_file, self.params.delta_time)

        # sort events by timing
        self.events.sort(key=lambda x: x.ref_timestamp, reverse=False)

        print(f"{len(self.events)} events created fro NP04 PDS and beam info")
        
        return True

    ##################################################################
    def analyze(self) -> bool:

        return True

    ##################################################################
    def write_output(self) -> bool:
            
        with open(self.params.events_output_path, "wb") as f:
            pickle.dump(self.events, f)
        print(f'Events saved in file: {self.params.events_output_path}')

        return True


       
