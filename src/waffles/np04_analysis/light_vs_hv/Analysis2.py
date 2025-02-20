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
         
            max_noise: float = Field(..., description= "max noise allowed")

            min_baseline: float = Field(..., description= "min baseline allowed")
            max_baseline: float = Field(..., description= "max baseline allowed")

            min_amplitude: float = Field(..., description= "min amplitude allowed")
            max_amplitude: float = Field(..., description= "max amplitude allowed")

            min_t0: float = Field(..., description= "min t0 allowed")
            max_t0: float = Field(..., description= "max t0 allowed")

            min_zero_crossing: float = Field(..., description= "min zero crossing allowed")
            max_zero_crossing: float = Field(..., description= "max zero crossing allowed")

            min_fprompt: float = Field(..., description= "min fprompt allowed")
            max_fprompt: float = Field(..., description= "max fprompt allowed")
            
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

        self.max_noise=self.params.max_noise

        self.min_baseline = self.params.min_baseline
        self.max_baseline = self.params.max_baseline

        self.min_amplitude = self.params.min_amplitude
        self.max_amplitude = self.params.max_amplitude

        self.min_t0 = self.params.min_t0
        self.max_t0 = self.params.max_t0

        self.min_zero_crossing = self.params.min_zero_crossing
        self.max_zero_crossing = self.params.max_zero_crossing

        self.min_fprompt = self.params.min_fprompt
        self.max_fprompt = self.params.max_fprompt



    ##################################################################
    def read_input(self) -> bool:

        self.n_run=len(self.file_path)
        self.n_channel=len(self.list_channels)

        with open(self.file_name, "rb") as file:
            self.wfset = pickle.load(file)
        
        return True
    #############################################################

    def analyze(self) -> bool:

        input_parameters = IPDict()

        input_parameters['baseline_ll'] =  self.baseline_limits[0]
        input_parameters['baseline_ul'] =  self.baseline_limits[1]
        input_parameters['zero_ll'] =  self.zero_crossing_limits[0]
        input_parameters['t0_wf_ul'] =  self.t0
        input_parameters['zero_ul'] =  self.zero_crossing_limits[0]
        input_parameters['int_ll'] =  self.integral_limits[0]
        input_parameters['int_ul'] =  self.integral_limits[1]
        input_parameters['amp_ll'] =  self.amplitude_limits[0]
        input_parameters['amp_ul'] =  self.amplitude_limits[1]
        input_parameters['fprompt_ul'] =  self.fprompt
       
        
        checks_kwargs = IPDict()

        #calculate the desired parameters
        for wfset in self.wfsets:
            for wfset_ch in wfset:
                _ = wfset_ch.analyse("minha_analise",
                                    ZeroCrossingAna,
                                    input_parameters,
                                    *[], # *args,
                                    analysis_kwargs = {},
                                    checks_kwargs = checks_kwargs,
                                    overwrite = True)

       
        #filter based on the parameters
        for i,wfset in enumerate(self.wfsets):
            for channel in range(self.n_channel):
                try:
                
                    wfset[channel] = WaveformSet.from_filtered_WaveformSet( wfset[channel], from_generic , max=self.max_noise, analysis_label="minha_analise",parameter_label="noise")
                    wfset[channel] = WaveformSet.from_filtered_WaveformSet( wfset[channel], from_generic , max=self.max_baseline, min=self.min_baseline, analysis_label="minha_analise",parameter_label="baseline")
                    wfset[channel] = WaveformSet.from_filtered_WaveformSet( wfset[channel], from_generic , max=self.max_amplitude, min=self.min_amplitude, analysis_label="minha_analise",parameter_label="amplitude")
                    wfset[channel] = WaveformSet.from_filtered_WaveformSet( wfset[channel], from_generic , min=self.min_t0,max=self.max_t0, analysis_label="minha_analise",parameter_label="t0")
                    wfset[channel] = WaveformSet.from_filtered_WaveformSet( wfset[channel], from_generic , max=self.max_zero_crossing, min=self.min_zero_crossing, analysis_label="minha_analise",parameter_label="zero_crossing")
                    wfset[channel] = WaveformSet.from_filtered_WaveformSet( wfset[channel], from_generic , min=self.min_fprompt, max=self.max_fprompt, analysis_label="minha_analise",parameter_label="fprompt")
                    #wfset = WaveformSet.from_filtered_WaveformSet( wfset, from_generic , max=second_peak_max, analysis_label="minha_analise",parameter_label="second_peak")
                    print(f"filtering-{i}:{self.list_endpoints[i]}-{self.list_channels[i]}")
                    self.wfsets[i][channel] = wfset[channel]
                except:
                    pass

        return True
    
    def write_output(self) -> bool:
        output_file=self.output + "/data_filtered_2.pkl"       
        with open(output_file, "wb") as file:
            pickle.dump(self.wfsets, file)
        return True
