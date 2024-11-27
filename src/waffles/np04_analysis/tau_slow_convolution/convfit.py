# import all necessary files and classes
from waffles.np04_analysis.tau_slow_convolution.imports import *

# import all tunable parameters
import waffles.np04_analysis.tau_slow_convolution.params as params


class convfit(WafflesAnalysis):

    def __init__(self):
        pass        

    ##################################################################
    def arguments(self, parse: argparse.ArgumentParser):                

        parse.add_argument('-runs','--runs',       type=int, nargs="+", help="Keep empty for all, or put the runs you want to be processed")
        parse.add_argument('-ch','--channel',      type=int,     help="Which channel to analyze", default=11225)
        parse.add_argument('-ft','--fix-template', action="store_true", help="Fix template to run 26261 (or thetemplate)")
        parse.add_argument('-tt', '--thetemplate', type=int,     help="If fix-template is set, use this to tell which template to use", default=0)
        parse.add_argument('-ns', '--namespace',   type=str,     help="Name space in case different folder", default="")
        parse.add_argument('-rl','--runlist',      type=str,     help="What run list to be used (purity or beam)", default="purity")
        parse.add_argument('-fr','--folder-responses', type=str, help="Directory of responses (just the name, default: responses)", default="responses")
        parse.add_argument('-p','--print',         action="store_true", help="If you want you can print result and not save")
        parse.add_argument('-in','--interpolate',  action="store_true", help="If you want 16 ns to be linear interpolated to 2 ns")
        parse.add_argument('--no-save',            action="store_true", help="If you want the output to be saved")
        parse.add_argument('-scan','--scan',       type=int,     help="Set maximum offset if you want to scan different offsets and get minimum. Scan is done around the default offset applied (-2, -(scan-2)). Set 0 to not scan.", default=0)    
    
    ##################################################################
    def initialize(self, args):                

        if args['runs'] is None:
            print('Please give a run')
            exit(0)

        runs = [ r for r in args['runs'] ]
        self.channel = args['channel']
        self.use_fix_template = args['fix_template']
        
        dfcsv = ReaderCSV()
        df = dfcsv.dataframes[args['runlist']]
        runs2    = df['Run'].to_numpy()
        led_runs = df['Run LED'].to_numpy()

        self.run_pairs = { r:lr for r, lr in zip(runs2, led_runs) }

        # use a fix template
        self.led_run_template = args['thetemplate']
        
        # results subfolder 
        self.output_subfolder="results"
        if args['runlist'] != "purity":
            self.output_subfolder += f"_{args['runlist']}"
        if args['namespace'] != "":
            self.output_subfolder += f"_{args['namespace']}"
        if self.use_fix_template:
            self.output_subfolder += "_fixtemplate"

        self.folder_responses = args['folder_responses']
        self.no_save = args['no_save']
        self.scan = args['scan']
        self.print = args['print']
        interpolate = args['interpolate']

        # create the Convolution Fitter
        self.cfit = ConvFitter(threshold_align_template = 0.27, 
                                threshold_align_response = 0.1, 
                                error=10, usemplhep=True, 
                                dointerpolation=interpolate, 
                                interpolation_fraction = 8, 
                                align_waveforms = True)
                        
        if self.scan > 0:
            self.cfit.reduce_offset = True

        self.cfit.dosave = not self.no_save

        # loop over runs
        self.read_input_loop = runs

    ##################################################################
    def read_input(self):

        # items for current iteration (run number)
        self.run = self.read_input_itr
        
        """ ---------  ----------- """

        print(f"Processing run {self.run}")
        if self.run not in self.run_pairs:
            print('Run not found in runlist, check it')
            exit(0)
        self.runled = self.run_pairs[self.run]

        # change template in the case it is fixed at 0 for endpoint 112
        if self.led_run_template == 0 and self.run > 27901:# and ch//100 == 112:
            self.led_run_template = 29177
        
        if self.use_fix_template:
            self.runled = self.led_run_template

        file_response = f"{params.output_folder}/{self.folder_responses}/response_run0{self.run}_ch{self.channel}_avg.pkl"

        file_template = f'{params.output_folder}/templates/template_run0{self.runled}_ch{self.channel}_avg.pkl'

        if os.path.isfile(file_template) is not True:
                print(f'file {file_template} does not extst !!!')    
                print(f"No match of LED run {self.runled}.. using \'thetemplate\' instead: {self.led_run_template} ")
                self.runled = self.led_run_template
                file_template = f'templates/template_run0{self.runled}_ch{self.channel}_avg.pkl'
        
        print ('file response: ', file_response)
        print ('file template: ', file_template)

        # read the average waveforms for the template and the response
        self.cfit.read_waveforms(file_template, file_response)
      
        return True


    ##################################################################
    def analyze(self):

        # prepare the template and response waveforms for the current iteration (run number)
        # performs interpolation and time alignment between template and response waveforms
        self.cfit.prepare_waveforms()  

        # perform the actual convolution fit
        self.cfit.fit(self.scan, self.print)

        return True
        
    ##################################################################
    def write_output(self):

        """ ---------- do the convolution and fit plot ----------- """

        # do the plot
        plt = self.cfit.plot()

        #add legend to plot
        plt.legend(title=f'run {self.run}')

        """ ---------- Save results and plot ----------- """
        
        dirout = f'{params.output_folder}/{self.output_subfolder}/run0{self.run}'
        os.makedirs(dirout, exist_ok=True)
        
        nselected  = self.cfit.wf_response["nselected"]
        first_time = self.cfit.wf_response["firsttime"]

        with open(f"{dirout}/convolution_output_{self.run}_{self.runled}_ch{self.channel}.txt", "w") as fout:
            fout.write(f"{first_time} {self.cfit.m.values['fp']} {self.cfit.m.values['t1']} {self.cfit.m.values['t3']} {self.cfit.m.fmin.reduced_chi2} {nselected} \n")

        with open(f"{dirout}/run_output_{self.run}_{self.runled}_ch{self.channel}.txt", "w") as fout:
            print(self.cfit.m, file=fout)

        # save the plot
        plt.savefig(f'{dirout}/convfit_data_{self.run}_template_{self.runled}_ch{self.channel}.png')


            
        

    
    



