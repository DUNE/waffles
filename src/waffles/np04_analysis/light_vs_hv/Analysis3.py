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

        self.output = self.params.output
    

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
        n=len(self.wfsets[0][0].waveforms[0].adcs)
        ch=self.channel_index

        for file in range(self.n_run):      
            try:
                self.mean_waveform[file] = np.zeros(n)

                wfset_aux = self.wfsets[file][ch]
                n_wfs=0                                                                         
                for k in range(len(wfset_aux.waveforms)):
                    baseline=wfset_aux.waveforms[k].analyses["minha_analise"].result["baseline"]
                    self.mean_waveform[file] = self.mean_waveform[file]+(wfset_aux.waveforms[k].adcs-baseline)
                    n_wfs=n_wfs+1
                self.mean_waveform[file]=self.mean_waveform[file]/n_wfs

            except:
                None   

        return True
    
    def write_output(self) -> bool:
        
        output_file = self.output + "/data_avg_wf.root"   

        # Create a root file
        
        file = root.TFile(output_file, "RECREATE")
        
        # Create a TTree
        tree = root.TTree("my_tree", "Tree with waveforms")

        max_length=len(self.mean_waveform[0])
        waveform_array = np.zeros(max_length, dtype=np.float32)  # Array fixo para armazenar os dados
        
        # Criar um branch no TTree (com tamanho variável)
        branch1 = tree.Branch("avg_wf", waveform_array, f"avg_wf[{max_length}]/F")

    
        for wf in self.mean_waveform:
            waveform_array[:len(wf)] = wf  
          
            tree.Fill()  # Afill the ttree

        # Save and close
        tree.Write()
        file.Close()

        return True
