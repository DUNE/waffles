# import all necessary files and classes
from waffles.np04_analysis.tau_slow_convolution.imports import *

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

            channels:                list = Field(...,          description="work in progress")
            dry:                     bool = Field(default=False,description="work in progress")
            force:                   bool = Field(default=False,description="work in progress")
            runlist:                  str = Field(...,          description="work in progress")
            runs:                    list = Field(...,          description="work in progress")
            showp:                   bool = Field(default=False,description="work in progress")
            blacklist:               list = Field(...,          description="work in progress")
            baseline_threshold:     float = Field(...,          description="work in progress")
            baseline_wait:            int = Field(...,          description="work in progress")
            baseline_start:           int = Field(...,          description="work in progress")
            baseline_finish_template: int = Field(...,          description="work in progress")
            baseline_finish_response: int = Field(...,          description="work in progress")
            baseline_minimum_frac:  float = Field(...,          description="work in progress")

            validate_items = field_validator(
                "channels",
                "runs",
                "blacklist",
                mode="before"
            )(wcu.split_comma_separated_string)

        return InputParams

    ##################################################################
    def initialize(
        self,
        input_parameters: BaseInputParams
    ) -> None:
    

        self.params = input_parameters

        self.endpoint = self.params.channels[0]//100

        self.safemode = True
        if self.params.force:
            self.safemode = False

        if self.params.runlist != "led":
            self.selection_type='response'
        else:
            self.selection_type='template'

        # ReaderCSV is in np04_data
        dfcsv = ReaderCSV()

        # these runs should be analyzed only on the last half
        try:
            runs = np.unique(dfcsv.dataframes[self.params.runlist]['Run'].to_numpy())
        except Exception as error:
            print(error)
            print('Could not open the csv file...')
            exit(0)

        if self.params.runs is not None:
            for r in self.params.runs:
                if r not in runs:
                    print(f"Run {r} is not in database... check {self.params.runlist}_runs.csv")
            runs = [ r for r in runs if r in self.params.runs ]


        self.baseliner = SBaseline()
        # Setting up baseline parameters
        self.baseliner.binsbase       = np.linspace(0,2**14-1,2**14)
        self.baseliner.threshold      = self.params.baseline_threshold
        self.baseliner.wait           = self.params.baseline_wait
        self.baseliner.minimumfrac    = self.params.baseline_minimum_frac
        self.baseliner.baselinestart  = self.params.baseline_start
        self.baseliner.baselinefinish = self.params.baseline_finish_template
        if self.selection_type=='response':
            self.baseliner.baselinefinish = self.params.baseline_finish_response


        # read_input will be iterated over run numbers
        self.read_input_loop_1 = runs

        # single element loops
        self.read_input_loop_2 = [None,]
        self.read_input_loop_3 = [None,]

        # analyze will be iterated over channels
        self.analyze_loop = self.params.channels

        print (f'Running in {self.selection_type} mode')

    ##################################################################
    def read_input(self) -> bool:

        # item for current iteration
        run = self.read_input_itr_1

        print(f"  Processing run {run}")

        # this will be the input file name
        file = f"{self.params.input_path}/{self.endpoint}/wfset_run0{run}.pkl"

        if not os.path.isfile(file):
            print("No file for run", run, "endpoint", self.endpoint)
            return False


        self.pickle_selec_name = {}
        self.pickle_avg_name   = {}
        os.makedirs(f'{self.params.output_path}/{self.selection_type}s', exist_ok=True)
        self.missingchannels = []
        for ch in self.params.channels:
            base_file_path = f'{self.params.output_path}/{self.selection_type}s/{self.selection_type}_run0{run}_ch{ch}'

            self.pickle_selec_name[ch] = f'{base_file_path}.pkl'
            self.pickle_avg_name[ch]   = f'{base_file_path}_avg.pkl'
            if self.safemode and os.path.isfile(self.pickle_avg_name[ch]):
                return False
            self.missingchannels.append(ch)

        if not self.missingchannels:
            print(f"Run {runnumber} there already for both channels...")
            return False
        elif self.params.dry:
            print(run, file)
            return False

        # read all waveforms from the pickle file
        self.wfset = 0
        try:
            self.wfset = WaveformSet_from_pickle_file(file)
        except Exception as error:
            print(error)
            print("Could not load the file... of run ", run, file)
            return False


        self.analyze_loop = self.missingchannels
        
        return True

    ##################################################################
    def analyze(self) -> bool:

        # items for current iteration
        run     = self.read_input_itr_1
        channel = self.analyze_itr

        print(f"    Processing channel {channel}")

        self.wfset_ch:WaveformSet = 0
            
        # --------- perform waveform selection  ----------- 

        # create an instance of the class with the sequence of cuts
        extractor = Extractor(self.params,self.selection_type, run) #here because I changed the baseline down..

        wch = channel
        if (self.wfset.waveforms[0].channel).astype(np.int64) - 100 < 0: # the channel stored is the short one
            wch = int(str(channel)[3:])
            extractor.channel_correction = True
                
        print ('      - #Waveforms (total):      ', len(self.wfset.waveforms))

        # select waveforms in the interesting channels
        self.wfset_ch = WaveformSet.from_filtered_WaveformSet(self.wfset, 
                                                                  extractor.allow_certain_endpoints_channels, 
                                                                  [self.endpoint], [wch], 
                                                                  show_progress=self.params.showp)

        print ('      - #Waveforms (in channel): ', len(self.wfset_ch.waveforms))
       
        try: 
            self.wfset_ch = WaveformSet.from_filtered_WaveformSet(self.wfset_ch, 
                                                                  extractor.apply_cuts,
                                                                  show_progress=self.params.showp)
        except Exception as error:
            print(error)
            print(f"No waveforms for run {run}, channel {wch}")
            return False

        print ('      - #Waveforms (selected):   ', len(self.wfset_ch.waveforms))

        # --------- compute the baseline -----------

        # Substract the baseline and invert the result
        wf_arrays = np.array([(wf.adcs.astype(np.float32) - wf.baseline)*-1 for wf in self.wfset_ch.waveforms if wf.channel == wch])
        
        # special treatment for runs in the blacklist
        if run in self.params.blacklist:
            print("Skipping first half...")
            skip = int(0.5*len(wf_arrays))
            wf_arrays = wf_arrays[skip:]

        # compute the average waveform
        avg_wf = np.mean(wf_arrays, axis=0)

        # Create an array with 500 numbers from -20 to 20
        self.baseliner.binsbase = np.linspace(-20,20,500)

        # compute the baseline again with a different method
        res0, status = self.baseliner.compute_baseline(avg_wf)

        # --------- compute final average waveform -----------

        # subtract the baseline
        avg_wf -= res0

        # save the results into the WaveformSet
        self.wfset_ch.avg_wf = avg_wf
        self.wfset_ch.nselected = len(wf_arrays)
            
        return True

    ##################################################################
    def write_output(self) -> bool:

        # get the channel number from the analyze iterator
        channel = self.analyze_itr

        # save all the waveforms contributing to the average waveform
        with open(self.pickle_selec_name[channel], "wb") as f:
            pickle.dump(self.wfset_ch, f)

        # save the average waveform, the time stamp of the first waveform and the number of selected waveforms
        output = np.array([self.wfset_ch.avg_wf, self.wfset_ch.waveforms[0].timestamp, self.wfset_ch.nselected], dtype=object)

        with open(self.pickle_avg_name[channel], "wb") as f:
            pickle.dump(output, f)
        print(f'      Average waveform saved in file: {self.pickle_avg_name[channel]}')

        return True


       
