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

            endpoints:      list = Field(defaut=[],          
                            description="list of the endpoints (note: must be te same order of the channels)")
            channels:       list = Field(default=[],          
                                description="list of the channels (note: must be te same order of the endpoints)")
            main_channel:   int =  Field(default=-1,          
                                description= "Main channel that the code will search for coincidences in the other channels")
            main_endpoint:  int =  Field(default=-1,          
                                description= "Main endpoin that the code will search for coincidences in the other channels")
            file_name:      str =  Field(default="data/runs.txt",          
                                description= "File with the list of folder to search for the .fcl files. In each folder must be only .fcls for the same run")
            output:         str =  Field(default="output",          
                                description= "Output folder to save the correlated channels")
            time_window:    int =  Field(default= 15,  
                                description="Time window in the search of coincidences")

        return InputParams
    
    def initialize(
        self,
        input_parameters: BaseInputParams
    ) -> None:
    
        self.params = input_parameters

        endpoints_len=len(self.params.endpoints)
        chs_len=len(self.params.channel)

        if endpoints_len != chs_len:
            raise ValueError("The size of the endpoints list is different from the size of the channels list")
        if endpoints_len == 0:
            raise ValueError("Endpoint list is empty")
        if chs_len == 0:
            raise ValueError("Channel list is empty")

        self.list_endpoints=self.params.endpoints
        self.list_channels=self.params.channels

        print("Channels that will read:")
        for endpoint,ch in zip(self.list_endpoints,self.list_channels):
            print(f"{endpoint}-{ch}")

        if self.params.main_channel==-1:
            self.main_channel=self.list_channels[0]
        else:
            self.main_channel=self.params.main_channel
        if self.params.main_endpoint==-1:
            self.main_endpoint=self.list_endpoints[0]
        else:
            self.main_endpoint=self.params.main_endpoint

        print(f"Master channel to search for coincidences: {self.main_endpoint}-{self.main_endpoint}")

        self.file_name=self.params.file_name
        print(f"File name: {self.file_name}")


        self.time_window=self.params.time_window

    ##################################################################
    def read_input(self) -> bool:
        
