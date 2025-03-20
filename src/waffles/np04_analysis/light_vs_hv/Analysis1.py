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
            input_path:      str =  Field(default="./data/list_file_aux.txt",          
                                description= "File with the list of files to search for the data. In each each line must be only a file name, and in that file must be a collection of .fcls from the same run")
            output:         str =  Field(default="./output/data_filtered.pkl",          
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
        print(f"------------ Coincidences on a time window of {self.time_window} ticks ---------------")

        self.min_coincidence = self.params.min_coincidence-2
        print(f"--------------checking for coincidences greater than {self.params.min_coincidence}--------------")
        self.read_input_loop=[None,]

        self.output = self.params.output
        print(self.output)

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
                            print(ch)
                            self.wfsets[i][j].append(WaveformSet.from_filtered_WaveformSet( wfset_aux, comes_from_channel, self.list_endpoints[j], [ch]))   
            i=i+1

        for run_index in range(self.n_run):
            for j in range(self.n_channel):
                for i in range(len(self.wfsets[run_index][j])):
                    if i!=0:
                        self.wfsets[run_index][j][i]=self.wfsets[run_index][j][0].merge(self.wfsets[run_index][j][i]) 
                        self.wfsets[run_index][j][i]=None

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
        timestamps_aux, min_timestamp_aux = get_timestamps(self.wfsets,self.n_channel,self.n_run)
        self.timestamps, self.min_timestamp = get_ordered_timestamps(self.wfsets,self.n_channel,self.n_run)

        for run_index in range(self.n_run):
            for j in range(self.n_channel):
                min_t = np.min(self.timestamps[run_index][j]).astype(np.float64)
                max_t = np.max(self.timestamps[run_index][j]).astype(np.float64)
                timestamp_diff=max_t-min_t
                print(f"Run{run_index} -- Ch: {j}: Have {len(self.wfsets[run_index][j].waveforms)} waveforms -- Time: {timestamp_diff*16e-9} s")

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


        # Ordinate using the timestamps

        for run_index in range(self.n_run):
            for j in range(self.n_channel):
                # Ordinate using the timestamps
                sorted_pairs = sorted(zip(timestamps_aux[run_index][j], self.wfsets[run_index][j].waveforms), key=lambda pair: pair[0])
                self.wfsets[run_index][j].waveforms = [x for _, x in sorted_pairs]

        print(4.1)
        #calculate the amplitude of all channels for a single event
        times_array=[[] for _ in range(self.n_run)]
        max_array=[[] for _ in range(self.n_run)]
        max=[]
        for file_index in range(self.n_run):
            for k in range(len(self.coincidences_level[file_index][18])):
                max=[]
                times=[]
                for channel in range(self.n_channel):
                    index=self.coincidences_level[file_index][18][k][1][channel]
                    time_diff=self.coincidences_level[file_index][18][k][2][channel]
                    true_index=index#find_true_index(wfsets,file_index,channel,timestamps,index,min_timestamp)
                    wf=self.wfsets[file_index][channel].waveforms[true_index].adcs
                    baseline=np.mean(wf[0:50])
                    wf=wf-baseline
                    max.append(-np.min(wf[0:200]))
                    times.append(time_diff)
                max_array[file_index].append(max)
                times_array[file_index].append(times)

        print(4.2)
        #separate the values in two arrays, each for each colunm in the APA
        max1_array=[[] for _ in range(self.n_run)]
        max2_array=[[] for _ in range(self.n_run)]
        times1_array=[[] for _ in range(self.n_run)]
        times2_array=[[] for _ in range(self.n_run)]
        sigma=0.8

        for file_index in range(self.n_run):
            i=0
            for my_coin in max_array[file_index]:
                #max_value=np.max(my_coin)
                aux1 = np.array(my_coin[0:10])
                aux2 = np.array(my_coin[10:20])
                
                max_value=np.max([np.max(aux1),np.max(aux2)])
                aux1 = gaussian_filter1d(aux1, sigma=sigma)
                aux2 = gaussian_filter1d(aux2, sigma=sigma)

                max1_array[file_index].append(aux1/max_value)
                max2_array[file_index].append(aux2/max_value)

                times1_array[file_index].append(times_array[file_index][i][0:10])
                times2_array[file_index].append(times_array[file_index][i][10:20])
                i=i+1

        print(4.3)
        #calculate position of max and the min values on each half
        pos_max_1=[[] for _ in range(self.n_run)]
        pos_max_2=[[] for _ in range(self.n_run)]
        val_max_1=[[] for _ in range(self.n_run)]
        val_max_2=[[] for _ in range(self.n_run)]
        pos_min_left_1=[[] for _ in range(self.n_run)]
        pos_min_left_2=[[] for _ in range(self.n_run)]
        pos_min_rigth_1=[[] for _ in range(self.n_run)]
        pos_min_rigth_2=[[] for _ in range(self.n_run)]
        val_min_left_1=[[] for _ in range(self.n_run)]
        val_min_left_2=[[] for _ in range(self.n_run)]
        val_min_rigth_1=[[] for _ in range(self.n_run)]
        val_min_rigth_2=[[] for _ in range(self.n_run)]

        for file_index in range(self.n_run):
            for i,[my_max_1,my_max_2] in enumerate(zip(max1_array[file_index],max2_array[file_index])):

                pos_max_1[file_index].append(np.argmax(my_max_1))
                pos_max_2[file_index].append(np.argmax(my_max_2))
                val_max_1[file_index].append(my_max_1[pos_max_1[file_index][i]])
                val_max_2[file_index].append(my_max_2[pos_max_2[file_index][i]])

                aux1=my_max_1[0:5]
                aux2=my_max_2[0:5]
                pos_min_left_1[file_index].append(np.argmin(aux1))
                pos_min_left_2[file_index].append(np.argmin(aux2))
                val_min_left_1[file_index].append(aux1[0])
                val_min_left_2[file_index].append(aux2[0])

                aux1=my_max_1[5:10]
                aux2=my_max_2[5:10]
                pos_min_rigth_1[file_index].append(np.argmin(aux1)+5)
                pos_min_rigth_2[file_index].append(np.argmin(aux2)+5)
                val_min_rigth_1[file_index].append(aux1[4])
                val_min_rigth_2[file_index].append(aux2[4])  

        print(4.4)
        filter_index=[]
        
        for file_index in range(self.n_run):
            
            index2 = np.isin(pos_max_1[file_index], [4, 5])
            index3 = np.isin(pos_max_2[file_index], [4, 5])
            index4 = np.isin(pos_min_rigth_1[file_index], [9])
            index5 = np.isin(pos_min_rigth_2[file_index], [9])
            index6 = np.isin(pos_min_left_1[file_index], [0])
            index7 = np.isin(pos_min_left_2[file_index], [0])
            
            filter_index.append(np.logical_and.reduce([ index2, index3, index4, index5, index6, index7]))


        #print(filter_index[0])
        for file_index in range(self.n_run):
            for i in range(len(times1_array[file_index])):
                
                #print(times1_array[file_index][i])
                #print(times2_array[file_index][i])
                #print(is_sorted(times1_array[file_index][i],10) , is_sorted(times2_array[file_index][i],10) , are_elements_off_by_one(times1_array[file_index][i], times2_array[file_index][i], 10))
                index8 = is_sorted(times1_array[file_index][i],2) and is_sorted(times2_array[file_index][i],2) and are_elements_off_by_one(times1_array[file_index][i], times2_array[file_index][i], 2)
                #print(index8,filter_index[file_index][i])
                filter_index[file_index][i] = filter_index[file_index][i] and index8
                #input("Pressione Enter para continuar...")

        #print(filter_index[0])
        print(5)

        true_index_2=[[] for _ in range(self.n_run)]
        for file_index in range(self.n_run):
            for num,i in enumerate(filter_index[file_index]):
                if i:
                    true_index_2[file_index].append(num)

        self.wfsets=filter_not_coindential_wf(self.wfsets,self.coincidences_level,self.timestamps,
                                              self.min_timestamp,self.n_channel,self.n_run,self.min_coincidence,true_index_2)
        

        print(6)
     
        return True
    
    def write_output(self) -> bool:
        output_file=self.output     
        
        with open(output_file, "wb") as file:
            pickle.dump(self.wfsets, file)

        """ with open(self.output + "/coincidences_level.pkl","wb") as file:
            pickle.dump(self.coincidences_level,file) """
        
        return True
