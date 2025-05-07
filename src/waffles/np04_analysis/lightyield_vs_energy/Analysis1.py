# import all necessary files and classes
from waffles.np04_analysis.lightyield_vs_energy.imports import *
from waffles.np04_analysis.lightyield_vs_energy.utils import *

class Analysis1(WafflesAnalysis):

    def __init__(self):
        pass        

    ######################################################################################################################################################################################################
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
            """Validation model for the input parameters of the light yield analysis."""

            set_name: Literal["A", "B"] = Field(
                ..., 
                description="Run set to analyze ('A' or 'B')",
                example="A"
            )
            
            apa_list: Union[conlist(Literal[1, 2, 3, 4]), Literal["all"]] = Field(
                ..., 
                description="APA list to analyze (1, 2, 3, 4 or 'all')", 
                example=[2]
            )

            # Correct usage of conlist without min_items/max_items
            endpoint_list: Union[conlist(Literal[104, 105, 107, 109, 111, 112, 113]), Literal["all"]] = Field(
                ..., 
                description="Endpoint list to analyze (104, 105, 107, 109, 111, 112, 113 or 'all')", 
                example=[109]
            )
            
            input_pickles_wf_filename: str = Field(
                ..., 
                description="Filename of input file (no folder path)",
                example="set_A_self_15files109.pkl"
            )
            
            pickles_folder: str = Field(
                ..., 
                description="Path to folder containing pickles files (input)",
                example="/afs/cern.ch/work/a/anbalbon/public/np04_beam_pickles"
            )
            
            output_folder: str = Field(
                ..., 
                description="Path to folder where output files are saved (output)",
                example="output"
            )
            
            additional_output_filename:  str = Field(
                ..., 
                description="Additional string to add to the output filename",
                example=""
            )
            
            input_path: str = Field(
                ..., 
                description="Path to folder with json input file",
                example="data"
            )
            
            beam_run_dic_info_filename: str = Field(
                ..., 
                description="Filename of the json file with beam run informationv(in data folder)",
                example="beam_run_info.json"
            )
            
            full_streaming: bool = Field(
                ..., 
                description="Are input data full-streaming data (True/False)?",
                example=False
            )
            
            beam_data: bool = Field(
                ..., 
                description="Are input data selected beam data (True/False)?",
                example=False
            )
            
            searching_beam_timeoffset: bool = Field(
                ..., 
                description="Do you want to search for beam timeoffset range? (True/False)?",
                example=False
            )
            
            searching_integration_range: bool = Field(
                ..., 
                description="Do you want to search for integration range? (True/False)?",
                example=False
            )
            
            overwrite: bool = Field(
                ..., 
                description="Do you want to overwrite output file (True/False)?",
                example=False
            )
            
            save_file: bool = Field(
                ..., 
                description="Do you want to save the output file (True/False)?",
                example=False
            )
            
            LED_calibration_file: str = Field(
                ..., 
                description="Path to the pickles file with LED calibration info (BATCH 1 from 2024/07/09)",
                example="batch_1_LED_calibration_data.pkl"
            )

        return InputParams

    ######################################################################################################################################################################################################
    def initialize(
        self,
        input_parameters: BaseInputParams
    ) -> None:
    
        self.params = input_parameters
        self.read_input_loop_1 = [None,] # ??
        self.read_input_loop_2 = [None,] # ??
        self.read_input_loop_3 = [None,] # ??
        self.analyze_loop = [None,] # ??
        
        if self.params.apa_list == 'all':
            self.params.apa_list = [1, 2, 3, 4]
        if self.params.endpoint_list == 'all':
            self.params.endpoint_list = [104, 105, 107, 109, 111, 112, 113]
            
        compatible_apa_list = []
        compatible_endpoint_list = []
        for end in self.params.endpoint_list:
            for apa in self.params.apa_list:
                if end in which_endpoints_in_the_APA(apa):
                    compatible_apa_list.append(apa)
                    compatible_endpoint_list.append(end)

        if len(compatible_apa_list) == 0 and len(compatible_endpoint_list) == 0:
            print("ValueError: No compatibility between APA and ENDPOINT chosen.")
            sys.exit()
            
    
        self.params.apa_list = sorted(compatible_apa_list)
        self.params.endpoint_list = sorted(compatible_endpoint_list)
        
        if self.params.additional_output_filename == '':
            self.output_filepath = f"{self.params.output_folder}/{self.params.input_pickles_wf_filename.split('.')[0]}_lightyield.json"
        else: 
            self.output_filepath = f"{self.params.output_folder}/{self.params.input_pickles_wf_filename.split('.')[0]}_{self.params.additional_output_filename}_lightyield.json" 
        if self.params.save_file:        
            if self.params.overwrite:
                print('\nAttention: if the output filename already exists, it will be overwritten!!\n')
            else:
                if os.path.exists(self.output_filepath):
                    print(f"The filename {self.output_filepath} exists, please select a new additional output filename and try again")
                    sys.exit()
              
                  

        
    ######################################################################################################################################################################################################
    def read_input(self) -> bool:
        
        print('\nReading beam run info...')
        with open(f"{self.params.input_path}/{self.params.beam_run_dic_info_filename}", "r") as file:
            self.run_set = json.load(file)[self.params.set_name]
            print('done\n')
            
        # print('\nReading led calibration info...')
        # LED_calibration_info = pd.read_pickle(f"{self.params.input_path}/{self.params.LED_calibration_file}")
        # print('done\n')
        
        print('Reading waveform pickles file...')
        with open(f"{self.params.pickles_folder}/set_{self.run_set['Name']}/{self.params.input_pickles_wf_filename}", 'rb') as f:
            self.wfset = pickle.load(f) 
            print('done\n')
            
        print('Checking runs numbers...')
        available_run = set(self.wfset.runs)
        run_set_values = set(self.run_set['Runs'].values())
        if available_run != run_set_values:
            if missing := run_set_values - available_run:
                print(f"Missing run: {missing} \n")
            if missing := available_run - run_set_values:
                print(f"ERROR, your set has different runs from the requested: {missing} should not be present - please update your pickle file or change the map!!\n")
                sys.exit()
        else:
            ('fine\n')
        
        return True

    ######################################################################################################################################################################################################
    def analyze(self) -> bool:

        ######################### BEAM EVENT OFFSET RANGE ######################### 
        if not self.params.beam_data:  
            if self.params.searching_beam_timeoffset:
                print('Searching for beam time-offset range...')
                answer = 'yes'
                searching_for_beam_events(self.wfset, show=True, output_folder = self.params.output_folder)
                while(answer == 'yes'):
                    beam_min = int(input('Minimum time-offset: '))  
                    beam_max = int(input('Maximum time-offset: ')) 
                    searching_for_beam_events(self.wfset, show=True, beam_min=beam_min, beam_max=beam_max, x_min = -10000, x_max = 10000, output_folder = self.params.output_folder)
                    answer = input('Do you want to CHANGE time-offset range? (yes/no) ').strip().lower()  
                print('done\n\n') 
            else:
                beam_min = -120
                beam_max = -90     
            
            print('Selecting beam data...')
            self.beam_wfset = WaveformSet.from_filtered_WaveformSet(self.wfset, beam_self_trigger_filter, timeoffset_min = beam_min, timeoffset_max = beam_max)
            print('done\n\n')
        else:
            self.beam_wfset = self.wfset

        ######################### INTEGRAL REGION SELECTION #########################  
        if self.params.searching_integration_range:
            print('Searching for integration range...')
            answer = 'yes'
            plotting_overlap_wf(self.beam_wfset, show=True, output_folder = self.params.output_folder)
            while(answer == 'yes'):
                int_ll = int(input('Lower integration limit: '))  
                int_ul = int(input('Upper integration limit: ')) 
                plotting_overlap_wf(self.beam_wfset, show=True, int_ll=int_ll, int_ul=int_ul, output_folder = self.params.output_folder)
                answer = input('Do you want to CHANGE integration limits? (yes/no) ').strip().lower() 
            print('done\n\n')  
        else:
            int_ll = 55
            int_ul = 115
        
        ######################### WAVEFORM ANALYSIS ######################### 
        print('\nAnalysis... ')
        
        peak_finding_kwargs = dict( prominence = 20,rel_height=0.5,width=[0,75]) 
        analysis_kwargs = dict(  return_peaks_properties = False)
        checks_kwargs   = dict( points_no = self.beam_wfset.points_per_wf )
        bl = [0, 40] #not used
        ampl_ll = 10 #not used
        ampl_up = 110 #not used
        
        baseline_method = 'SBaseline'
        baseliner = SBaseline()
        baseliner.binsbase       = np.linspace(0,2**14-1,2**14) # default
        baseliner.minimumfrac    = 1/6. #default
        baseliner.threshold      = 6
        baseliner.wait           = 25
        baseliner.baselinestart  = 0
        baseliner.baselinefinish = 45 #1024 #set it to the pre-trigger.
        baseliner.filtering      = 8 
        
        # BASIC ANALYSIS
        analysis_label = 'standard'
        analysis_class=BasicWfAna
        ip = IPDict(baseline_limits=bl,
                    int_ll=int_ll,
                    int_ul=int_ul,
                    amp_ll=ampl_ll,
                    amp_ul=ampl_up,
                    points_no=10,
                    peak_finding_kwargs=peak_finding_kwargs,
                    baseline_method=baseline_method,
                    baseliner =  baseliner)
        
        b = self.beam_wfset.analyse(label=analysis_label ,analysis_class=analysis_class, input_parameters=ip, checks_kwargs = checks_kwargs, overwrite=True)
    
        # MY ANALYSIS
        analysis_label = 'my_deconvolution'
        analysis_class=myWfAna
        ip = IPDict(baseline_limits=bl,
                    int_ll=int_ll,
                    int_ul=int_ul,
                    amp_ll=ampl_ll,
                    amp_ul=ampl_up,
                    points_no=10,
                    peak_finding_kwargs=peak_finding_kwargs,
                    baseline_method=baseline_method,
                    baseliner =  baseliner,
                    deconvolution = True, 
                    save_devonvolved_wf = True, 
                    maritza_template = True,
                    gauss_filtering = True,
                    gauss_cutoff = 2.5
                    )

        a = self.beam_wfset.analyse(label=analysis_label ,analysis_class=analysis_class, input_parameters=ip, checks_kwargs = checks_kwargs, overwrite=True)
        
        print(self.beam_wfset.waveforms[0].analyses['standard'].result['integral'])
        print(self.beam_wfset.waveforms[0].analyses['my_deconvolution'].result['integral_before'])
        
        baseline = self.beam_wfset.waveforms[0].analyses[analysis_label].result['baseline']
        
        #### Includere la possibilità di disgenare le waveform deconvolute!!!!
        print()
        plotting_overlap_wf(self.beam_wfset, int_ll=int_ll, int_ul=int_ul, baseline = baseline)
        print('done\n\n')
        
        '''
        ######################### SAVING INFORMATION #########################
        self.results_info_dic = {}
        
        
        ######################### COMPUTING LY vs ENERGY BEAM X CHANNEL ######################### 
        for APA in self.params.apa_list:
            print(f'\n\n ------------------------------------\n \t       APA {APA} \n ------------------------------------\n')
            apa_info = {}
            for endpoint in self.params.endpoint_list: 
                print(f'\n --- \t ENDPOINT {endpoint} \t ---')
                end_info = {}
                for channel in sorted(which_channels_in_the_ENDPOINT(endpoint)):
                    print(f"\n\nLet's study APA {APA} - Endpoint {endpoint} - Channel {channel}")
                    ID_String = f"APA{APA}_endpoint{endpoint}_channel{channel}"
                    ch_info = {'ID_String': ID_String, 'APA': APA, 'endpoint' : endpoint, 'channel' : channel, 'Runs' : self.run_set['Runs']}
                    ly_data_dic = {} 
                    ly_result_dic = {} 
     
                                        
                    if (APA == 1) and self.params.full_streaming:
                        print('to be implemented...\n')
                                  
                    if ((APA == 2) or (APA == 3) or (APA == 4) ) and not self.params.full_streaming: #or (APA == 3) or (APA == 4)
                        
                        try:
                            ch_beam_wfset = WaveformSet.from_filtered_WaveformSet(self.beam_wfset, channel_filter, end=endpoint, ch=channel)
                            ly_data_dic, ly_result_dic = LightYield_SelfTrigger_channel_analysis(wfset = ch_beam_wfset, end = endpoint, ch = channel, run_set = self.run_set, analysis_label = analysis_label)
                                    
                        except Exception as e:
                            if "There must be at least one Waveform in the set" in str(e):
                                print(f"No beam events --> Skipped channel")
                            else:
                                print(f"Error: {e} --> Skipped channel")    
                                
                    
                    ch_info['LY data'] = ly_data_dic
                    ch_info['LY result'] = ly_result_dic  
                    
                    end_info[channel] = ch_info
                apa_info[endpoint] = end_info            
            self.results_info_dic[APA] = apa_info  
                                                              
        print('\n\nAnalysis... done\n')
        '''
        return True

    ##################################################################
    def write_output(self) -> bool:
        '''if self.params.save_file: 
            print('Saving...', end='')
            with open(self.output_filepath, "w") as file:
                json.dump(self.results_info_dic, file, indent=4)
                print(f' done: {self.output_filepath}\n')
        else:
            print(f'\n\nThe file was not saved (save_file = {self.params.save_file})')
        

        print(f"\n\nSUMMARY: \nFull-streaming data: {self.params.full_streaming}\nBeam data: {self.params.beam_data}\nSearching beam timeoffset: {self.params.searching_beam_timeoffset}\nSearching integration range: {self.params.searching_integration_range}\nSet name: {self.params.set_name}\nAPA: {self.params.apa_list}\nENDPOINT: {self.params.endpoint_list}\n# analyzed beam wavforms: {len(self.beam_wfset.waveforms)}\n\n")
        '''
        print('\n\nfatto\n\n')
        return True