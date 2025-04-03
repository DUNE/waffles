from waffles.np04_analysis.light_vs_hv.imports import *

class Analysis1(WafflesAnalysis):

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
            main_channel:   int =  Field(default=-1,          
                                description= "Main channel that the code will search for coincidences in the other channels")
            main_endpoint:  int =  Field(default=-1,          
                                description= "Main endpoin that the code will search for coincidences in the other channels")
            input_path:      str =  Field(default="./data/list_file.txt",          
                                description= "File with the list of files to search for the data. In each each line must be only a file name, and in that file must be a collection of .fcls from the same run")
            output:         str =  Field(default="./output",          
                                description= "Output folder to save the correlated channels")
            time_window:    int =  Field(default= 5,  
                                description="Time window in the search of coincidences")
            min_coincidence:   int=  Field(default = 10,  
                                description="Mininum number of coincidences to save")

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

        print("Channels that will read:")
        for n, (endpoint, ch) in enumerate(zip(self.list_endpoints, self.list_channels)):
            if check_endpoint_and_channel(endpoint,ch):
                print(f"{endpoint}-{ch}")
            else:
                print(f"{endpoint}-{ch}: dont exist -- skipping")
                self.list_channels.pop(n)
                self.list_endpoints.pop(n)

        if self.params.main_channel==-1:
            self.main_channel=self.list_channels[0]
        else:
            self.main_channel=self.params.main_channel
        if self.params.main_endpoint==-1:
            self.main_endpoint=self.list_endpoints[0]
        else:
            self.main_endpoint=self.params.main_endpoint

        if check_endpoint_and_channel(self.main_endpoint,self.main_channel):
            if self.main_endpoint == self.list_endpoints[0] and  self.main_channel == self.list_channels[0]:
                print(f"Master channel to search for coincidences: {self.main_endpoint}-{self.main_channel}")
            else:
                raise ValueError(f"The channel {self.main_endpoint}-{self.main_channel} is not the first channel given")

        else:
            raise ValueError(f"The channel {self.main_endpoint}-{self.main_channel} to check for coincidendes dont exist")

        self.file_name=self.params.input_path
        print(f"File name: {self.file_name}")

        self.time_window=self.params.time_window

        self.min_coincidence = self.params.min_coincidence-2
        print(f"--------------checking for coincidences greater than {self.params.min_coincidence}--------------")
        self.read_input_loop=[None,]

        self.output = self.params.output

    ##################################################################
    def read_input(self) -> bool:

        #open the file to look the folders for searching the data - note: each folder must be only files of the same run
        self.file_path=[]
        with open(self.file_name, "r") as file:
            for lines in file:
                lines=lines.strip()
                self.file_path.append(lines)

        self.n_run=len(self.file_path)
        self.n_channel=len(self.list_channels)

        #open the wfsets inside each folder: wfsets[i][j] --> i is the run index, j is the channel index

        self.wfsets=[[[] for _ in range (self.n_channel)] for _ in range(self.n_run)]

        i=0
        for file_name in self.file_path:
            with open(file_name, "r") as file:
                for lines in file:
                    lines=lines.strip()
                    with open(lines, 'rb') as file_line:
                        print("Opening file: "+lines)
                        wfset_aux=pickle.load(file_line)
                        for j,ch in enumerate(self.list_channels):
                            self.wfsets[i][j].append(WaveformSet.from_filtered_WaveformSet( wfset_aux, comes_from_channel, self.list_endpoints[j], [ch]))   
            i=i+1

        for run_index in range(self.n_run):
            for j in range(self.n_channel):
                for i in range(len(self.wfsets[run_index][j])):
                    if i!=0:
                        self.wfsets[run_index][j][i]=self.wfsets[run_index][j][0].merge(self.wfsets[run_index][j][i]) 

        for run_index in range(self.n_run):
            for j in range(self.n_channel):
                aux=self.wfsets[run_index][j][0]
                self.wfsets[run_index][j]=aux

        #-------------

        return True
    #############################################################

    def analyze(self) -> bool:
        print(0)
        #get a vector of ordered timestamps per run per channel [i][j]
        self.timestamps, self.min_timestamp = get_ordered_timestamps(self.wfsets,self.n_channel,self.n_run)
        print(1)
        #return a list of double coincidences
        #coincidences[run index][goal channel][target channel][coindences index] --> [0: timestamp index of the goal channel][1: timestamp index of the target channel],[2:deltaT]]
        self.coincidences = get_all_double_coincidences(self.timestamps, 
                                                        self.n_channel, self.n_run, self.time_window)
   
        print(2)
        #return a list of all coincidences
        #mult_coincidences[run_index][coincidence_index] --> [0: list of channels, 1: list of the index related to the channel on the timestamp array, 2: delta t of each channel related to goal channel]
        self.mult_coincidences = get_all_coincidences(self.coincidences, self.timestamps, 
                                                      self.n_channel, self.n_run )

        print(3)
        self.coincidences_level = get_level_coincidences(self.mult_coincidences,self.n_channel,self.n_run)
        
        print(4)
        self.wfsets=filter_not_coindential_wf(self.wfsets,self.coincidences_level,self.timestamps,
                                              self.min_timestamp,self.n_channel,self.n_run,self.min_coincidence)
        print(5)
     
        return True
    
    def write_output(self) -> bool:
        output_file=self.output + "/data_filtered.pkl"       
        with open(output_file, "wb") as file:
            pickle.dump(self.wfsets, file)
        return True
