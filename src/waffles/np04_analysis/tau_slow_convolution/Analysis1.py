# import all necessary files and classes
from waffles.np04_analysis.tau_slow_convolution.imports import *

class Analysis1(WafflesAnalysis):

    def __init__(self):
        pass        

    ##################################################################
    # def arguments(self, parse: argparse.ArgumentParser):                

    #     parse.add_argument('-runs','--runs',  type=int, nargs="+",  help="Keep empty for all, or put the runs you want to be processed")
    #     parse.add_argument('-r','--response', action="store_true",  help="Set true if response")
    #     parse.add_argument('-t','--template', action="store_true",  help="Set true if template")
    #     parse.add_argument('-rl','--runlist', type=str,             help="What run list to be used (purity or beam)", default="purity")
    #     parse.add_argument('-ch','--channels',type=int, nargs="+",  help="Channels to analyze (format: 11225)", default=[11225])
    #     parse.add_argument('-p','--showp',    action="store_true",  help="Show progress bar")
    #     parse.add_argument('-f','--force',    action="store_true",  help='Overwrite...')
    #     parse.add_argument('-n','--dry',      action="store_true",  help="Dry run")
    
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
            response:                bool = Field(default=False,description="work in progress")
            template:                bool = Field(default=False,description="work in progress")
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

        # make sure only -r or -t is chosen, not both
        if not self.params.response and not self.params.template:
            print("Please, choose one type --response or --template")
            exit(0)

        if self.params.response:
            self.selection_type='response'
        elif self.params.template:
            self.selection_type='template'

        # ReaderCSV is in np04_data
        dfcsv = ReaderCSV()


        self.raw_data_path = "data"

        # these runs should be analyzed only on the last half
        try: 
            tmptype = 'Run'
            if self.params.template:
                tmptype = 'Run LED'
            runs = np.unique(dfcsv.dataframes[self.params.runlist][tmptype].to_numpy())
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
        self.read_input_loop = runs

        # analyze will be iterated over channels
        self.analyze_loop = self.params.channels

    ##################################################################
    def read_input(self) -> bool:

        # item for current iteration
        run = self.read_input_itr

        file = f"{self.raw_data_path}/{self.endpoint}/wfset_run0{run}.pkl"

        if not os.path.isfile(file):
            print("No file for run", run, "endpoint", self.endpoint)
            return False
        if self.params.dry:
            print(run, file)
            return False

        self.wfset = 0
        try:
            self.wfset = WaveformSet_from_pickle_file(file)
        except Exception as error:
            print(error)
            print("Could not load the file... of run ", run, file)
            return False

        return True

    ##################################################################
    def analyze(self) -> bool:

        # items for current iteration
        run     = self.read_input_itr
        channel = self.analyze_itr

        """ --------- perform the analysis for channel in run ----------- """

        self.wfset_ch:WaveformSet = 0
        self.pickle_selec_name = f'{self.params.output_path}/{self.selection_type}s/{self.selection_type}_run0{run}_ch{channel}.pkl'
        self.pickle_avg_name   = f'{self.params.output_path}/{self.selection_type}s/{self.selection_type}_run0{run}_ch{channel}_avg.pkl'
        os.makedirs(f'{self.params.output_path}/{self.selection_type}s', exist_ok=True)

        if self.safemode and os.path.isfile(self.pickle_selec_name):
            val:str
            val = input('File already there... overwrite? (y/n)\n')
            val = val.lower()
            if val == "y" or val == "yes":
                pass
            else:
                return False
            
        """ ---------  ----------- """

        extractor = Extractor(self.params,self.selection_type, run) #here because I changed the baseline down..

        wch = channel
        if (self.wfset.waveforms[0].channel).astype(np.int64) - 100 < 0: # the channel stored is the short one
            wch = int(str(channel)[3:])
            extractor.channel_correction = True
        
        
        """ --------- perform waveform selection  ----------- """

        print ('#Waveforms: ', len(self.wfset.waveforms))

        # select waveforms in the interesting channels
        self.wfset_ch = WaveformSet.from_filtered_WaveformSet(self.wfset, 
                                                                  extractor.allow_certain_endpoints_channels, 
                                                                  [self.endpoint], [wch], 
                                                                  show_progress=self.params.showp)

        print ('#Waveforms in channel: ', len(self.wfset_ch.waveforms))
       
        try: 
            self.wfset_ch = WaveformSet.from_filtered_WaveformSet(self.wfset_ch, 
                                                                extractor.apply_cuts,                                                                   
                                                                show_progress=self.params.showp)
        except Exception as error:
            print(error)
            print(f"No waveforms for run {run}, channel {wch}")
            return False

        print ('#Waveforms selected: ', len(self.wfset_ch.waveforms))

        """ --------- compute the baseline ----------- """

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

        """ --------- compute final average waveform ----------- """

        # subtract the baseline
        avg_wf -= res0

        # save the results into the WaveformSet
        self.wfset_ch.avg_wf = avg_wf
        self.wfset_ch.nselected = len(wf_arrays)
        
        print(f'{run} total: {len(self.wfset.waveforms)}\t {channel}: {len(wf_arrays)}')
    
        return True

    ##################################################################
    def write_output(self) -> bool:
            
        # save all the waveforms contributing to the average waveform
        with open(self.pickle_selec_name, "wb") as f:
            pickle.dump(self.wfset_ch, f)

        # save the average waveform, the time stamp of the first waveform and the number of selected waveforms
        output = np.array([self.wfset_ch.avg_wf, self.wfset_ch.waveforms[0].timestamp, self.wfset_ch.nselected], dtype=object)

        with open(self.pickle_avg_name, "wb") as f:
            pickle.dump(output, f)
        print('Saved... ')

        return True


       