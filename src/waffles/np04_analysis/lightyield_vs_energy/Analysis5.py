# Nuova versione di Analysis 2 per fare diverse analisi sugli histogrammi 

# import all necessary files and classes
from waffles.np04_analysis.lightyield_vs_energy.imports import *
from waffles.np04_analysis.lightyield_vs_energy.utils import *

class Analysis5(WafflesAnalysis):

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
            """Validation model for the input parameters of the light yield analysis."""

            set_name: Literal["A", "B"] = Field(
                ..., 
                description="Run set to analyze ('A' or 'B')",
                example="A"
            )
            
            # Correct usage of conlist without min_items/max_items
            apa_list: Union[conlist(Literal[1, 2, 3, 4]), Literal["all"]] = Field(
                ..., 
                description="APA list to analyze (1, 2, 3, 4 or 'all')", 
                example=[2]
            )

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
            
            output_folder: str = Field(
                ..., 
                description="Path to folder where output files are saved (output)",
                example="output"
            )
            
            additional_output_filename: str = Field(
                ..., 
                description="Additional string to add to the output filename",
                example=""
            )
            
            beam_run_dic_info_filename: str = Field(
                ..., 
                description="Path to the json file with beam run information",
                example="data/beam_run_info.json"
            )
            
            full_streaming: bool = Field(
                ..., 
                description="Are input data full-streaming data (True/False)?",
                example=False
            )
            
            overwrite: bool = Field(
                ..., 
                description="Do you want to overwrite output file (True/False)?",
                example=False
            )
            
        return InputParams

    ##################################################################
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
            self.input_output_file = f"{self.params.output_folder}/{self.params.input_pickles_wf_filename.split('.')[0]}_lightyield"
        else: 
            self.input_output_file = f"{self.params.output_folder}/{self.params.input_pickles_wf_filename.split('.')[0]}_{self.params.additional_output_filename}_lightyield" 
            
        if self.params.overwrite:
                print('\nAttention: if the output filename already exists, it will be overwritten!!\n')
        else:
            if os.path.exists(f"{self.input_output_file}.pdf"):
                print(f"The filename {self.output_filepath}.pdf exists, please select a new additional output filename and try again")
                sys.exit()

        
    ##################################################################
    def read_input(self) -> bool:

        print('\nReading beam run info...')
        with open(f"{self.params.input_path}/{self.params.beam_run_dic_info_filename}", "r") as file:
            self.run_set = json.load(file)[self.params.set_name]
            print('done\n')
            
        
        print('\nReading analysis results...')
        with open(f"{self.input_output_file}.json", "r") as file:
            self.result_info = json.load(file)
            print('done\n')
        
        return True

    ##################################################################
    def analyze(self) -> bool:
        
        gaussian_fit = False    # Set to True if you want to fit a Gaussian distribution
        gaussian_around_max = True  # Set to True if you want to fit a Gaussian around the maximum of the histogram
        landau_gauss = False # Set to True if you want to fit a Landau-Gauss distribution
        
        
        
        print('Analysis...')
        self.analysis_results = {}
        
        for key in find_analysis_keys(self.result_info):
            self.analysis_results[key] = {}
        
        for apa, apa_dic in self.result_info.items():
            
            for key in find_analysis_keys(self.result_info):
                self.analysis_results[key][apa] = {}
            
            print(f' ----------------------------\n\n\t APA {apa}\n')
            
            for end, end_dic in apa_dic.items():
                for key in find_analysis_keys(self.result_info):
                    self.analysis_results[key][apa][end] = {}
                for ch, ch_dic in end_dic.items():
                    if int(ch) < 10:
                        print(f"Endpoint {end} - Channel {ch}")
                                            
                        for integral_label, integral_info in ch_dic['Analysis'].items():
                            fig, ax = plt.subplots(3, 2, figsize=(12, 10))
                            plt.suptitle(f'Endpoint {end} - Channel {ch}')
                            ax = ax.flatten()
                            i = 0
                            if integral_info:
                                for energy, histo_info in integral_info['LY data'].items():
                                    if histo_info:
                                        ax[i].hist(histo_info['histogram data'], bins=histo_info['gaussian fit']['bins'], density=True, alpha=0.6, color='blue', label="Data")
                                        ax[i].set_xlabel("Integrated Charge")
                                        ax[i].set_ylabel("Density")
                                        ax[i].set_title(f"Energy: {energy} GeV")
                                        ax[i].legend(fontsize='small')
                                        
                                        if gaussian_around_max:
                                            filtered_data = np.array(histo_info['histogram data'])[(np.array(histo_info['histogram data']) >= histo_info['gaussian fit']['mean']['value'] - histo_info['gaussian fit']['sigma']['value']) &  (np.array(histo_info['histogram data']) <= histo_info['gaussian fit']['mean']['value'] + histo_info['gaussian fit']['sigma']['value'])]

                                            bin_heights, bin_edges = np.histogram(filtered_data, bins=10, density=True)
                                            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  
                                            p0 = [np.mean(filtered_data), np.std(filtered_data), max(bin_heights)]
                                            popt, pcov = curve_fit(gaussian, bin_centers, bin_heights, p0=p0)
                                            perr = np.sqrt(np.diag(pcov))   
                                            
                                        if histo_info['gaussian fit']:
                                            x_fit = np.linspace(min(histo_info['histogram data']), max(histo_info['histogram data']), 1000)
                                            y_fit = gaussian(x_fit, histo_info['gaussian fit']['mean']['value'], histo_info['gaussian fit']['sigma']['value'], histo_info['gaussian fit']['normalized amplitude']['value'])
                                            ax[i].plot(x_fit, y_fit, color='red', lw=2, label=f"Gaussian fit: \n$\mu= {to_scientific_notation(histo_info['gaussian fit']['mean']['value'], histo_info['gaussian fit']['mean']['error'])}$ \n$\sigma= {to_scientific_notation(histo_info['gaussian fit']['sigma']['value'], histo_info['gaussian fit']['sigma']['error'])}$ \n$A= {to_scientific_notation(histo_info['gaussian fit']['normalized amplitude']['value'], histo_info['gaussian fit']['normalized amplitude']['error'])} $") 
                                    i += 1
                                    
                                if integral_info['LY result']:
                                    ax[i].errorbar(integral_info['LY result']['x'], integral_info['LY result']['y'], yerr=np.abs(integral_info['LY result']['e_y']), fmt='o', label='Data')
                                    ax[i].plot(integral_info['LY result']['x'], linear_fit(np.array(integral_info['LY result']['x']), integral_info['LY result']['slope']['value'], integral_info['LY result']['intercept']['value']), 'r-', label=f"Linear fit: y=a+bx\n$a = {to_scientific_notation(integral_info['LY result']['intercept']['value'], integral_info['LY result']['intercept']['error'])}$ \n$b = {to_scientific_notation(integral_info['LY result']['slope']['value'], integral_info['LY result']['slope']['error'])}$") 
                                    ax[i].set_xlabel('Beam energy (GeV)')
                                    ax[i].set_ylabel('Integrated charge')
                                    ax[i].set_title('Charge vs energy with linear fit')
                                    ax[i].legend()
                        
                            plt.tight_layout()
                            self.analysis_results[integral_label][apa][end][ch] = fig
                            #plt.close(fig)
        
        print('\nAnalysis... done\n')
        return True


    def write_output(self) -> bool:
        print('Saving...')
        
        for key in find_analysis_keys(self.result_info):
            for apa, apa_dic in self.analysis_results[key].items():
                APA_pdf_file = PdfPages(f"{self.input_output_file}_APA{apa}_{key}_NEW.pdf")
                for end, end_dic in apa_dic.items():
                    for ch, fig in end_dic.items():
                        APA_pdf_file.savefig(fig)
                        plt.close(fig)
                APA_pdf_file.close()
        
        print('\nPDFs saved successfully!\n')
        return True

       