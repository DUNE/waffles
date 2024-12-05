# import all necessary files and classes
from waffles.np04_analysis.tau_slow_convolution.imports import *

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

            runs:         list = Field(...,          description="work in progress")
            channels:     list = Field(...,          description="work in progress")
            fix_template: bool = Field(...,          description="work in progress")
            the_template: int  = Field(...,          description="work in progress")
            namespace:    str  = Field(...,          description="work in progress")
            runlist:      str  = Field(...,          description="work in progress")
            print:        bool = Field(...,          description="work in progress")
            interpolate:  bool = Field(...,          description="work in progress")
            no_save:      bool = Field(...,          description="work in progress")
            scan:         int  = Field(...,          description="work in progress")

        return InputParams

    ##################################################################
    def initialize(
        self,
        input_parameters: BaseInputParams
    ) -> None:
    
        self.params = input_parameters

        if self.params.runs is None:
            print('Please give a run')
            exit(0)

        runs = [ r for r in self.params.runs ]
        
        dfcsv = ReaderCSV()
        df = dfcsv.dataframes[self.params.runlist]
        runs2    = df['Run'].to_numpy()
        led_runs = df['Run LED'].to_numpy()

        self.run_pairs = { r:lr for r, lr in zip(runs2, led_runs) }

        # use a fix template
        self.led_run_template = self.params.the_template
        
        # results subfolder 
        self.output_subfolder="results"
        if self.params.runlist != "purity":
            self.output_subfolder += f"_{self.params.runlist}"
        if self.params.namespace != "":
            self.output_subfolder += f"_{self.params.namespace}"
        if self.params.fix_template:
            self.output_subfolder += "_fixtemplate"
          
        # create the Convolution Fitter
        self.cfit = ConvFitter(threshold_align_template = 0.27, 
                                threshold_align_response = 0.1, 
                                error=10, usemplhep=True, 
                                dointerpolation=self.params.interpolate, 
                                interpolation_fraction = 8, 
                                align_waveforms = True)
                        
        if self.params.scan > 0:
            self.cfit.reduce_offset = True

        self.cfit.dosave = not self.params.no_save

        # loop over runs
        self.read_input_loop = runs

        # loop over channels
        self.analyze_loop = self.params.channels

    ##################################################################
    def read_input(self) -> bool:

        # items for current iteration (run number)
        self.run = self.read_input_itr
        
        print(f"Processing run {self.run}")
        if self.run not in self.run_pairs:
            print('Run not found in runlist, check it')
            exit(0)
        self.runled = self.run_pairs[self.run]

        # change template in the case it is fixed at 0 for endpoint 112
        if self.led_run_template == 0 and self.run > 27901:# and ch//100 == 112:
            self.led_run_template = 29177
        
        if self.params.fix_template:
            self.runled = self.led_run_template
      
        return True


    ##################################################################
    def analyze(self) -> bool:

        #-------- This block should be moved to input when a double loop is available in read_input ------

        # items for current iteration (channel number)
        self.channel = self.analyze_itr


        file_response = f"{self.params.output_path}/responses/response_run0{self.run}_ch{self.channel}_avg.pkl"
        file_template = f'{self.params.output_path}/templates/template_run0{self.runled}_ch{self.channel}_avg.pkl'

        if os.path.isfile(file_template) is not True:
                print(f'file {file_template} does not extst !!!')    
                print(f"No match of LED run {self.runled}.. using \'the_template\' instead: {self.led_run_template} ")
                self.runled = self.led_run_template
                file_template = f'templates/template_run0{self.runled}_ch{self.channel}_avg.pkl'
        
        print ('file response: ', file_response)
        print ('file template: ', file_template)

        # read the average waveforms for the template and the response
        self.cfit.read_waveforms(file_template, file_response)

        #--------------------------------------------------

        # prepare the template and response waveforms for the current iteration (run number)
        # performs interpolation and time alignment between template and response waveforms
        self.cfit.prepare_waveforms()  

        # perform the actual convolution fit
        self.cfit.fit(self.params.scan, self.params.print)

        return True
        
    ##################################################################
    def write_output(self) -> bool:

        #---------- do the convolution and fit plot ----------- 
        # do the plot
        plt = self.cfit.plot()

        #add legend to plot
        plt.legend(title=f'run {self.run}')

        # ---------- Save results and plot -----------
        
        dirout = f'{self.params.output_path}/{self.output_subfolder}/run0{self.run}'
        os.makedirs(dirout, exist_ok=True)
        
        nselected  = self.cfit.wf_response["nselected"]
        first_time = self.cfit.wf_response["firsttime"]

        with open(f"{dirout}/convolution_output_{self.run}_{self.runled}_ch{self.channel}.txt", "w") as fout:
            fout.write(f"{first_time} {self.cfit.m.values['fp']} {self.cfit.m.values['t1']}"
                       "{self.cfit.m.values['t3']} {self.cfit.m.fmin.reduced_chi2} {nselected} \n")

        with open(f"{dirout}/run_output_{self.run}_{self.runled}_ch{self.channel}.txt", "w") as fout:
            print(self.cfit.m, file=fout)

        # save the plot
        plt.savefig(f'{dirout}/convfit_data_{self.run}_template_{self.runled}_ch{self.channel}.png')

        return True
            
        

    
    



