from waffles.np04_analysis.light_vs_hv.imports import *

class Analysis1(WafflesAnalysis):

    def __init__(self):
        pass

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

            endpoints:      list = Field(default=[],          
                            description="list of the endpoints (note: must be te same order of the channels)")
            channels:       list = Field(default=[],          
                                description="list of the channels (note: must be te same order of the endpoints)")
            input_path:     str =  Field(default=".output/data_filtered.pkl",          
                                description= "File with the list of files to search for the data. In each each line must be only a file name, and in that file must be a collection of .fcls from the same run")
            output:         str =  Field(default="./output",          
                                description= "Output folder to save the filtered data")
            baseline_limits:    list = Field (... , description= "limits in clock ticks for calculate the baseline")
            integral_limits:    list = Field (... , description= "limits in clock ticks for calculate the integral/charge")
            amplitude_limits:   list = Field (... , description= "limits in clock ticks for calculate the amplitude")
            zero_crossing_limits:   list = Field (..., description= "limits in clock ticks for calculate the zero_crossing")
            t0:             int = Field(..., description="Max limit to calculate the start of waveform")
            fprompt:        int = Field(..., description="Max limit to calculate the fast charge of waveform")      
         

        return InputParams
    
    def initialize(
        self,
        input_parameters: BaseInputParams
    ) -> None:
    
        self.params = input_parameters

        endpoints_len=len(self.params.endpoints)
        chs_len=len(self.params.channels)

        if endpoints_len != chs_len:
            raise ValueError("The size of the endpoints list is different from the size of the channels list")
        if endpoints_len == 0:
            raise ValueError("Endpoint list is empty")
        if chs_len == 0:
            raise ValueError("Channel list is empty")

        self.list_endpoints=self.params.endpoints
        self.list_channels=self.params.channels

        self.file_name=self.params.input_path
        print(f"File name: {self.file_name}")

        self.baseline_limits=self.params.baseline_limits
        self.integral_limits=self.params.integral_limits
        self.amplitude_limits=self.params.amplitude_limits
        self.zero_crossing_limits=self.params.zero_crossing_limits
        self.t0 = self.params.t0
        self.fprompt = self.params.fprompt

        self.output = self.params.output


    ##################################################################
    def read_input(self) -> bool:

        with open(self.file_name, "rb") as file:
            self.wfset = pickle.load(file)
        
        return True
    #############################################################

    def analyze(self) -> bool:

       
        return True
    
    def write_output(self) -> bool:
       
        return True
