from waffles.np04_analysis.light_vs_hv.imports import *

class Analysis4(WafflesAnalysis):

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

            input_path:     str =  Field(default="./output/data_avg_wf.root",          
                                description= "File with the list of files to search for the data. In each each line must be only a file name, and in that file must be a collection of .fcls from the same run")
            output:         str =  Field(default="./output",          
                                description= "Output folder to save the filtered data")
            hv:         list =  Field(... ,          
                                description= "HV vector")
            channels:       list = Field(default=[],          
                                description="list of the channels (note: must be te same order of the endpoints)")
          

        return InputParams
    
    def initialize(
        self,
        input_parameters: BaseInputParams
    ) -> None:
    
        self.params = input_parameters

        self.file_name=self.params.input_path
        print(f"File name: {self.file_name}")

        self.hv=self.params.hv
        self.n_run=(len(self.hv))

        self.output = self.params.output

        self.read_input_loop=[None,]

        self.list_channels=self.params.channels
        self.n_ch=len(self.list_channels)

    ##################################################################
    def read_input(self) -> bool:
        # open root file
        self.file = root.TFile(self.file_name, "READ")
        self.tree = self.file.Get("my_tree")
        
        return True
    #############################################################

    def analyze(self) -> bool:

        # number of events in ttree
        self.n_entries = self.tree.GetEntries()
        
        self.waveform=[[None for _ in range(self.n_ch)] for _  in range(self.n_run)]
        

        # get all waveforms
        for i in range (self.n_entries):
            self.tree.GetEntry(i)
            ch=self.tree.ch
            hv=self.tree.hv
          
            self.waveform[hv][ch] = (np.array(self.tree.avg_wf_dec_filt))  

        self.s3 = [np.zeros(len(self.hv)) for _ in range(self.n_ch)]
        self.s3_error = [np.zeros(len(self.hv)) for _ in range(self.n_ch)]
        s3_error_max = [np.zeros(len(self.hv)) for _ in range(self.n_ch)]
        s3_error_min = [np.zeros(len(self.hv)) for _ in range(self.n_ch)]

         
        #fit slow and fast component:
        self.tau_slow = [np.zeros(len(self.hv)) for _ in range(self.n_ch)]
        self.tau_fast = [np.zeros(len(self.hv)) for _ in range(self.n_ch)]
        self.par_slow = [[None for _ in range(len(self.hv)) ] for _ in range(self.n_ch)]
        self.cov_slow = [[None for _ in range(len(self.hv)) ] for _ in range(self.n_ch)]
        self.y_fit= [[None for _ in range(len(self.hv)) ] for _ in range(self.n_ch)]

        #self.max = [np.zeros(len(self.hv)) for _ in range(self.n_ch)]
              
        self.waveform_total=[ np.zeros(len(self.waveform[0][0])) for _  in range(self.n_run)]

        for j in range(len(self.hv)):
            for i in range(self.n_ch):
                self.waveform_total[j]=self.waveform_total[j]+self.waveform[j][i]
        
        
        for i in range(self.n_ch):
            for j in range(len(self.hv)):
                aux=self.waveform[j][i][0:600]
                max_aux=np.argmax(aux)
                mean = 0# np.mean(self.waveform[i][0:30])
                #self.max[i]=max_aux
                until=np.argmin(np.abs(self.waveform[j][i]-mean)[max_aux:])
                #print(max_aux,until)
                until=400
                y_sum=(self.waveform[j][i]-mean)[max_aux-10:until]
                y_sum_mais=(self.waveform[j][i]-mean)[max_aux-10:until+400]
                y_sum_menos=(self.waveform[j][i]-mean)[max_aux-10:until-100]
               
                until=500
                self.y_fit[i][j]=(self.waveform[j][i]-mean)[max_aux+5:max_aux+5+until]
                
                x_fit=np.arange(0,len(self.y_fit[i][j]),1)
                
                self.par_slow[i][j], self.cov_slow[i][j] = curve_fit(func_tau,x_fit,self.y_fit[i][j],maxfev=20000,p0=[1,100,1,10,0])      
                self.tau_slow[i][j]=np.max([self.par_slow[i][j][0],self.par_slow[i][j][1]])*16
                self.tau_fast[i][j]=np.min([self.par_slow[i][j][0],self.par_slow[i][j][1]])*16

                self.s3[i][j]=np.sum(y_sum)
                s3_error_max[i][j] = np.sum(y_sum_mais)
                s3_error_min[i][j] = np.sum(y_sum_menos) 
 
            self.s3[i] = np.array(self.s3[i])/self.s3[i][0]
            s3_error_max[i] = np.array(s3_error_max[i])/s3_error_max[i][0]
            s3_error_min[i] = np.array(s3_error_min[i])/s3_error_min[i][0]
            #print(len(self.s3_error[i]))
            self.s3_error[i]=np.abs((s3_error_max[i]-s3_error_min[i])/2)

        self.s=self.s3
        self.hv=np.array(self.hv)/360
        self.params_s=[]
        self.params_covariance_s=[] 

        self.s_mean=np.zeros(self.n_run)
        self.s_var=np.zeros(self.n_run)
        self.s_total=[]

        for i in range(self.n_ch):
            param,cov=curve_fit(birks_law,self.hv,self.s[i],maxfev=20000)
            self.params_s.append(param)
            self.params_covariance_s.append(cov)
            self.s_mean=self.s_mean+self.s[i]
            self.s_var=self.s_var+self.s[i]*self.s[i]

        self.s_mean = self.s_mean/self.n_ch
        self.s_var = np.sqrt(self.s_var/self.n_ch - self.s_mean*self.s_mean)

        self.param_mean,self.cov_mean=curve_fit(birks_law,self.hv,self.s_mean,maxfev=20000)


        print(4)
        for i in range(self.n_run):
            aux=self.waveform_total[i][0:600]
            max_aux=np.argmax(aux)
            self.s_total.append(np.sum((self.waveform_total[i])[max_aux-10:max_aux+300]))
        self.s_total = np.array(self.s_total)/self.s_total[0]
        print(5)
        self.param_total,self.cov_total=curve_fit(birks_law,self.hv,self.s_total,maxfev=20000)

        return True
    
    def write_output(self) -> bool:
        output_file_1=self.output + "/fit/"   
        
        for i in range(self.n_ch):

            plot_scint(self.s[i],self.hv,self.params_s[i],output_file_1+f"fit_light_{i}.png",self.s3_error[i])
            
        print(self.s_var,len(self.s_var))
        plot_scint(self.s_mean,self.hv,self.param_mean,output_file_1+f"fit_light_mean.png",self.s_var)
        plot_scint(self.s_total,self.hv,self.param_total,output_file_1+f"fit_light_total.png")
            
        output_file = self.output + "/data_total_wf.root"   

        # Create a root file
        file = root.TFile(output_file, "RECREATE")
        
        # Create a TTree
        tree = root.TTree("my_tree", "Tree with waveforms")

        length=len(self.waveform_total[0])
        waveform_array = np.zeros(length, dtype=np.float32)  
    
        branch1 = tree.Branch("avg_wf", waveform_array, f"avg_wf[{length}]/F")
      
        for run_index in range(self.n_run):
            
            waveform_array[:length] = self.waveform_total[run_index]
               
            tree.Fill()  # Fill the ttree

        # Save and close
        tree.Write()
        file.Close()
        output_file_2=self.output + "/slow_comp/" 
        for i in range(self.n_ch):
            fig, axs = plt.subplots(1, 2, figsize=(15, 5))
            
            axs[0].scatter(self.hv, self.tau_slow[i], marker="o", color="red", label="Slow Component")
            axs[1].scatter(self.hv, self.tau_fast[i], marker="o", color="blue", label="Fast Component")

            axs[0].set_xlabel("E Field [kV/cm]")
            axs[1].set_xlabel("E Field [kV/cm]")

            axs[0].set_ylabel("Slow Comp [ns]")
            axs[1].set_ylabel("Fast Comp [ns]")  

            # Grid
            axs[0].grid()
            axs[1].grid()

            axs[0].legend()
            axs[1].legend()

            fig.tight_layout()

            fig.savefig(output_file_2 + f"fit_slow_fast_{i}.png")

            plt.close(fig)
            
            for j in range(self.n_run):
                plt.plot((self.y_fit[i][j]-self.par_slow[i][j][4]))
                x_fit=np.arange(0,len(self.y_fit[i][j]),1)

                plt.plot(func_tau(x_fit,*self.par_slow[i][j]))
                plt.grid()
                plt.savefig(output_file_2+f"waveform_hv_{j}_ch_{i}")
                plt.close()
        return True
