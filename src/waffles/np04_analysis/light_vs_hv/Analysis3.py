from waffles.np04_analysis.light_vs_hv.imports import *

class Analysis3(WafflesAnalysis):

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
            input_path:     str =  Field(default="./output/data_filtered_2.pkl",          
                                description= "File with the list of files to search for the data. In each each line must be only a file name, and in that file must be a collection of .fcls from the same run")
            output:         str =  Field(default="./output",          
                                description= "Output folder to save the filtered data")
            
            
            template_file:  str =  Field(... ,  description= "Template file path")

            avg_channel:   int =  Field(default=-1,          
                                description= "channel to calculate average waveform")
            avg_endpoint:  int =  Field(default=-1,          
                                description= "channel to calculate average waveform")
            
            filter_length:  int =  Field(default=10,          
                                description= "window of of the avg_filter")
          

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

        self.template_file= self.params.template_file
        
        if self.params.avg_channel==-1:
            self.avg_channel=self.list_channels[0]
        else:
            self.avg_channel=self.params.avg_channel
        if self.params.avg_endpoint==-1:
            self.avg_endpoint=self.list_endpoints[0]
        else:
            self.avg_endpoint=self.params.avg_endpoint

        for k in range(len(self.list_channels)):
            if self.avg_channel == self.list_channels[k] and self.avg_endpoint == self.list_endpoints[k]:
                self.channel_index = k
                break 

        self.filter_length = self.params.filter_length

        self.output = self.params.output
    
        self.read_input_loop=[None,]


    ##################################################################
    def read_input(self) -> bool:

        with open(self.file_name, "rb") as file:
            self.wfsets = pickle.load(file)

        self.n_run=len(self.wfsets)
        self.n_channel=len(self.list_channels)

        #load_template    
        with open(self.template_file, "r") as file:
            values = [float(line.strip()) for line in file]

        self.template = np.array(values)

        return True
    #############################################################

    def analyze(self) -> bool:

        #calculate avg_wf and deconvoluted avg_wf
        self.mean_waveform = [[] for _  in range(self.n_run)]
        self.mean_deconvolved= [[] for _ in range(self.n_run)]
        self.mean_deconvolved_sin= [[] for _ in range(self.n_run)]
        self.mean_deconvolved_filtered= [[] for _ in range(self.n_run)]

        t0_mean=50
        n=len(self.wfsets[0][0].waveforms[0].adcs)
        ch=self.channel_index

        for file in range(self.n_run):      
           
            self.mean_waveform[file] = np.zeros(n)

            wfset_aux = self.wfsets[file][ch]
            n_wfs=0                                                                         
            for k in range(len(wfset_aux.waveforms)):
                aux_t=wfset_aux.waveforms[k].analyses["minha_analise"].result["t0"]
                deltat=aux_t-t0_mean
                deltat=0
                baseline=wfset_aux.waveforms[k].analyses["minha_analise"].result["baseline"]
                self.mean_waveform[file] = self.mean_waveform[file]+np.concatenate([(wfset_aux.waveforms[k].adcs-baseline)[deltat:],np.zeros(deltat)])
                n_wfs=n_wfs+1
            self.mean_waveform[file]=self.mean_waveform[file]/n_wfs


        window = signal.windows.tukey(1024,0.16)
        wn=0.25
        b,a = signal.butter(4, wn, btype='low', analog=False)

        K = 0  # Fator de suavização
        x_axis=np.concatenate([np.arange(0,500,1),np.arange(1100,2000,1)])
        
        for file in range(self.n_run):
            signall=np.concatenate([np.zeros(int(len(self.mean_waveform[file])/2)),self.mean_waveform[file]*window,np.zeros(int(len(self.mean_waveform[file])/2))])
            template_menos=-self.template
            signal_fft = np.fft.fft(signall)
            template_menos_fft = np.fft.fft(template_menos, n=len(signall))  # Match signal length
            #deconvolved_fft = signal_fft * np.conj(template_menos_fft)/  (template_menos_fft * np.conj(template_menos_fft) + K)     # Division in frequency domain
            deconvolved_fft = signal_fft/ (template_menos_fft )     # Division in frequency domain
            deconvolved_aux = np.fft.ifft(deconvolved_fft)      # Transform back to time domain
            
            # Take the real part (to ignore small imaginary errors)
            self.mean_deconvolved[file] = np.real(deconvolved_aux)
          
            self.mean_deconvolved_filtered[file] = signal.filtfilt(b,a,self.mean_deconvolved[file])

            y=np.concatenate([self.mean_deconvolved_filtered[file][:500],self.mean_deconvolved_filtered[file][1100:2000]])
            params,cov=curve_fit(my_sin,x_axis,y, maxfev=20000,p0=[0.5,1000,-0.5,0.5,0.5,-5])
            x_minus=np.arange(0,len(self.mean_deconvolved_filtered[file]),1)
            self.mean_deconvolved_sin[file]=self.mean_deconvolved_filtered[file]-my_sin(x_minus,*params)
            #mean=np.mean(np.concatenate([self.mean_deconvolved_sin[file][:400],self.mean_deconvolved_sin[file][1400:2000]]))
            #self.mean_deconvolved_sin[file]=self.mean_deconvolved_sin[file]-mean
            #self.mean_deconvolved_filtered[file]=np.convolve(self.mean_deconvolved[file],np.ones(2))
    

        return True
    
    def write_output(self) -> bool:
        
        output_file = self.output + "/data_avg_wf.root"   

        # Create a root file
        
        file = root.TFile(output_file, "RECREATE")
        
        # Create a TTree
        tree = root.TTree("my_tree", "Tree with waveforms")

        length=len(self.mean_waveform[0])
        waveform_array = np.zeros(length, dtype=np.float32)  
        
        length_dec=len(self.mean_deconvolved[0])
        waveform_array_dec = np.zeros(length_dec, dtype=np.float32)  

        length_filt=len(self.mean_deconvolved_filtered[0])
        waveform_array_filt = np.zeros(length_filt, dtype=np.float32)  
        
        length_no_dec=len(self.mean_waveform[0])
        waveform_array_no_dec = np.zeros(length_no_dec, dtype=np.float32)  

        length_sin=len(self.mean_deconvolved_sin[0])
        waveform_array_sin = np.zeros(length_sin, dtype=np.float32)  


        branch1 = tree.Branch("avg_wf", waveform_array, f"avg_wf[{length}]/F")
        branch2 = tree.Branch("avg_wf_dec", waveform_array_dec, f"avg_wf[{length_dec}]/F")
        branch3 = tree.Branch("avg_wf_dec_sin", waveform_array_sin, f"avg_wf[{length_sin}]/F")
        branch4 = tree.Branch("avg_wf_dec_filt", waveform_array_filt, f"avg_wf[{length_filt}]/F")
        branch5 = tree.Branch("avg_wf_no_dec", waveform_array_no_dec, f"avg_wf[{length_no_dec}]/F")

    
        for [wf,dec,filt,means,sin] in zip(self.mean_waveform,self.mean_deconvolved,self.mean_deconvolved_filtered,self.mean_waveform,self.mean_deconvolved_sin):
            waveform_array[:len(wf)] = wf  
            waveform_array_dec[:len(dec)] = dec  
            waveform_array_filt[:len(filt)] = filt
            waveform_array_no_dec[:len(means)] = means
            waveform_array_sin[:len(sin)] = sin
            tree.Fill()  # Fill the ttree

        # Save and close
        tree.Write()
        file.Close()

        return True
