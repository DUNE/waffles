# import all necessary files and classes
from waffles.np04_analysis.lightyield_vs_energy.imports import *
from waffles.np04_analysis.lightyield_vs_energy.utils import *

class Analysis3(WafflesAnalysis):

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
            
            save_file: bool = Field(
                ..., 
                description="Do you want to save the output file (True/False)?",
                example=False
            )
            
        return InputParams

    ##################################################################
    def initialize(
        self,
        input_parameters: BaseInputParams
    ) -> None:
    
        self.params = input_parameters
        self.read_input_loop = [None,] # ??
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
            self.output_file = f"{self.params.output_folder}/{self.params.input_pickles_wf_filename.split('.')[0]}_lightyield_wholeAPA"
            self.input_file = f"{self.params.output_folder}/{self.params.input_pickles_wf_filename.split('.')[0]}_lightyield.json"
        else: 
            self.output_file = f"{self.params.output_folder}/{self.params.input_pickles_wf_filename.split('.')[0]}_{self.params.additional_output_filename}_lightyield_wholeAPA" 
            self.input_file = f"{self.params.output_folder}/{self.params.input_pickles_wf_filename.split('.')[0]}_{self.params.additional_output_filename}_lightyield.json" 

        if self.params.overwrite:
                print('\nAttention: if the output filename already exists, it will be overwritten!!\n')
        else:
            if os.path.exists(f"{self.output_file}.pdf") or os.path.exists(f"{self.output_file}.json"):
                print(f"The filename {self.output_file}.pdf exists, please select a new additional output filename and try again")
                sys.exit()

        self.bins = 500
        
    ##################################################################
    def read_input(self) -> bool:

        print('\nReading beam run info...')
        with open(f"{self.params.input_path}/{self.params.beam_run_dic_info_filename}", "r") as file:
            self.run_set = json.load(file)[self.params.set_name]
            print('done\n')
            
        
        print('\nReading analysis results...')
        with open(f"{self.input_file}", "r") as file:
            self.result_info = json.load(file)
            print('done\n')
        
        return True

    ##################################################################
    def analyze(self) -> bool:
        print('Analysis...')
 
        whole_histogram_data =  {}
        for apa, apa_dic in self.result_info.items():
            whole_apa_dic = {"1":[], "2":[], "3":[], "5":[], "7":[]}
            print(f'\n\n ----------------------------\n\n\t APA {apa}\n')
            for end, end_dic in apa_dic.items():
                for ch, ch_dic in end_dic.items():
                    print(f"Endpoint {end} - Channel {ch}")        
                    for energy, histo_info in ch_dic['LY data'].items():
                        if histo_info:
                            whole_apa_dic[energy].extend(histo_info['histogram data'])
            
            whole_histogram_data[apa] = whole_apa_dic
        
        self.analysis_results = {}
        self.figure_data = {}
        
        for apa, apa_info in whole_histogram_data.items():    
            apa_dic_info = {'APA': apa, 'Runs' : self.run_set['Runs']}
            ly_data_dic = {"1" : {'histogram data' :{}, 'gaussian fit': {}}, "2" : {'histogram data' :{}, 'gaussian fit': {}}, "3" : {'histogram data' :{}, 'gaussian fit': {}}, "5" : {'histogram data' :{}, 'gaussian fit': {}}, "7" : {'histogram data' :{}, 'gaussian fit': {}}} 
            ly_result_dic = {'x':[], 'y': [], 'e_y' : []} 
            
            fig, ax = plt.subplots(3, 2, figsize=(12, 10))
            plt.suptitle(f'APA {apa}')
            ax = ax.flatten()
            i = 0
               
            for energy, histo_data in apa_info.items():
                if len(histo_data)>0:
                    ax[i].hist(histo_data, bins=self.bins, density=True, alpha=0.6, color='blue', label="Data")
                    ly_data_dic[energy]['histogram data'] = histo_data
                    
                    try:
                        bin_heights, bin_edges = np.histogram(histo_data, bins=self.bins, density=True)
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  
                        p0 = [np.mean(histo_data), np.std(histo_data), max(bin_heights)]

                        popt, pcov = curve_fit(gaussian, bin_centers, bin_heights, p0=p0)
                        perr = np.sqrt(np.diag(pcov))    
                        
                        x_fit = np.linspace(min(histo_data), max(histo_data), 1000)
                        y_fit = gaussian(x_fit, popt[0], popt[1], popt[2])
                        ax[i].plot(x_fit, y_fit, color='red', lw=2, label=f"Gaussian fit: \n$\mu= {to_scientific_notation(popt[0], perr[0])}$ \n$\sigma= {to_scientific_notation(popt[1], perr[1])}$ \n$A= {to_scientific_notation(popt[2], perr[2])} $")
                    
                        
                        ly_data_dic[energy]['gaussian fit'] = {"mean": { "value": popt[0],"error": perr[0]}, "sigma": { "value": popt[1],"error": perr[1]}, "normalized amplitude": { "value": popt[2],"error": perr[2]}, "bins": self.bins}
                    
                        ly_result_dic['x'].append(int(energy))
                        ly_result_dic['y'].append(popt[0])
                        ly_result_dic['e_y'].append(popt[1])
                            
                    except Exception as e:
                        print(f'Fit error: {e} --> skipped')
                    
                    ax[i].set_xlabel("Integrated Charge")
                    ax[i].set_ylabel("Density")
                    ax[i].set_title(f"Energy: {energy} GeV")
                    ax[i].legend(fontsize='small')
                i += 1
            
  
            if len(ly_result_dic['x']) >1:
                x = np.array(ly_result_dic['x'])
                y = np.array(ly_result_dic['y'])
                y_err = np.array(ly_result_dic['e_y'])
                
                popt, pcov = curve_fit(linear_fit, x, y, sigma=y_err, absolute_sigma=True)
                slope, intercept = popt
                slope_err, intercept_err = np.sqrt(np.diag(pcov))
                
                ax[i].errorbar(x, y, yerr=y_err, fmt='o', label='Data')
                ax[i].plot(x, linear_fit(x, *popt), 'r-', label=f"Linear fit: y=a+bx\n$a = {to_scientific_notation(intercept, intercept_err)}$ \n$b = {to_scientific_notation(slope, slope_err)}$") 
                ax[i].set_xlabel('Beam energy (GeV)')
                ax[i].set_ylabel('Integrated charge')
                ax[i].set_title('Charge vs energy with linear fit')
                ax[i].legend()
                
                ly_result_dic['slope'] = {'value': slope, 'error': slope_err}
                ly_result_dic['intercept'] = {'value': intercept, 'error': intercept_err}
                
                                
            plt.tight_layout()
            self.figure_data[apa] = fig
            plt.close(fig)
            
            apa_dic_info['LY data'] = ly_data_dic
            apa_dic_info['LY result'] = ly_result_dic
            
            self.analysis_results[apa] = apa_dic_info

        print('\nAnalysis... done\n')
        return True


    def write_output(self) -> bool:
        if self.params.save_file: 
            print('Saving (json and pdf)...', end='')
            
            with open(f"{self.output_file}.json", "w") as file:
                json.dump(self.analysis_results, file, indent=4)
            
            pdf_file = PdfPages(f"{self.output_file}.pdf")
            for apa, fig in self.figure_data.items():
                pdf_file.savefig(fig)
                plt.close(fig)
            pdf_file.close()
                
            print(f' done: {self.output_file}\n')
        else:
            print(f'\n\nThe file was not saved (save_file = {self.params.save_file})')
                
        return True

       