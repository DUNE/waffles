# import all necessary files and classes
from waffles.np04_analysis.beam_example.imports import *

class Analysis2(WafflesAnalysis):

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

            events_output_path:      str = Field(...,          description="work in progress")
            
        return InputParams

    ##################################################################
    def initialize(
        self,
        input_parameters: BaseInputParams
    ) -> None:
    
        self.analyze_loop = [None,]
        self.params = input_parameters

        self.read_input_loop = [None,]
#        self.read_input_loop_2 = [None,]
#        self.read_input_loop_3 = [None,]

        
    ##################################################################
    def read_input(self) -> bool:

        print(f"Reading events from pickle file: ", self.params.events_output_path)

        self.events = events_from_pickle_file(self.params.events_output_path)

        print(f"  {len(self.events)} events read")
        
        return True

    ##################################################################
    def analyze(self) -> bool:

        print(f"Dump information about events:")

        t0 = self.events[0].ref_timestamp

        # loop over events
        for e in self.events:

            # get the number of waveforms
            nwfs = len(e.wfset.waveforms) if e.wfset else 0            

            # print information about the event
            print (e.record_number,
                   e.event_number,
                   e.first_timestamp-t0,
                   (e.last_timestamp-e.first_timestamp)*0.016,
                   ', p =', e.beam_info.p,
                   ', nwfs =', nwfs,
                   ', c0 =', e.beam_info.c0,
                   ', c1 =', e.beam_info.c1,
                   ', tof =', e.beam_info.tof)
        
        return True

    ##################################################################
    def write_output(self) -> bool:
            
        return True


       
