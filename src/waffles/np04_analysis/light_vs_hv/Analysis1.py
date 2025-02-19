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
            main_channel:   int =  Field(...,          
                                description= "Main channel that the code will search for coincidences in the other channels")
            file_name:      str =  Field(...,          
                                description= "File with the list of folder to search for the .fcl files. In each folder must be only .fcls for the same run")
            output:         str =  Field(...,          
                                description= "Output folder to save the correlated channels")

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
