<<<<<<< HEAD
from imports import *
=======
from waffles.np04_analysis.noise_studies.imports import *
>>>>>>> 264bdce2c6b35b5dd071455c3cbe62221217a107

class Analysis1(WafflesAnalysis):

    def __init__(self) -> None:
        pass

    #################################################################
    @classmethod
    def get_input_params_model(cls) -> type:
        """
        comment
        """
        class InputParams(BaseInputParams):
            """
            InputParams for FFT analysis
            """
<<<<<<< HEAD
            filepath_folder: str
            run_vgain_dict: Dict[int, int]
            channel_map_file: str
            out_writing_mode: str
            out_path: str
            full_stat: bool
            user_runs: List[int]
            all_noise_runs: List[int]
=======
            filepath_folder: str = Field(..., description="Path to the folder containing the files.txt with rucio paths")
            run_vgain_dict: Dict[int, int] = Field(..., description="Dictionary with run number as key and vgain as value")
            channel_map_file: str = Field(..., description="Path to the channel map file")
            out_writing_mode: str = Field(..., description="Writing mode for the output file")
            out_path: str = Field(..., description="Path to the output folder")
            full_stat: bool = Field(..., description="Full statistics")
            user_runs: list = Field([], description="List of runs to analyze") 
            all_noise_runs: list = Field([], description="List of all noise runs")
            integratorsON_runs: list = Field([], description="List of runs with integrators ON")
>>>>>>> 264bdce2c6b35b5dd071455c3cbe62221217a107

        return InputParams


    #################################################################
    def initialize(self, input_parameters: BaseInputParams) -> None:
        """
        comment
        """
<<<<<<< HEAD
        self.filepath_folder  = input_parameters.filepath_folder
        self.run_vgain_dict   = input_parameters.run_vgain_dict
        self.channel_map_file = input_parameters.channel_map_file
        self.out_writing_mode = input_parameters.out_writing_mode
        self.out_path         = input_parameters.out_path
=======
        self.params = input_parameters
        self.filepath_folder  = self.params.filepath_folder
        self.run_vgain_dict   = input_parameters.run_vgain_dict
        self.integratorsON_runs = input_parameters.integratorsON_runs
        self.debug_mode = input_parameters.debug_mode
        self.channel_map_file = input_parameters.channel_map_file
        df = pd.read_csv(self.channel_map_file, sep=",")
        self.daphne_channels = df['daphne_ch'].values + 100*df['endpoint'].values
        self.daphne_to_offline = dict(zip(self.daphne_channels, df['offline_ch']))
        self.out_writing_mode = input_parameters.out_writing_mode
        self.out_path         = input_parameters.out_path
        self.out_csv_file     = open(self.out_path+"Noise_Studies_Results.csv", self.out_writing_mode)
>>>>>>> 264bdce2c6b35b5dd071455c3cbe62221217a107
        self.full_stat        = input_parameters.full_stat
        self.runs             = input_parameters.user_runs
        if (len(self.runs) == 0):
            self.runs = input_parameters.all_noise_runs
            if (len(self.runs) == 0):
                print("No runs to analyze")
                exit()

<<<<<<< HEAD
=======
        # read_input loop will iter over runs
        self.read_input_loop = self.runs

        # analyze loop will iter over runs
        self.analyze_loop = self.runs



>>>>>>> 264bdce2c6b35b5dd071455c3cbe62221217a107
    #################################################################
    def read_input(self) -> bool:
        """
        comment
        """
<<<<<<< HEAD
=======
        self.run = self.read_input_itr
        
        print("Reading run: ", self.run)
        self.wfset_run = nf.read_waveformset(
                self.filepath_folder, int(self.run), full_stat=self.full_stat)
>>>>>>> 264bdce2c6b35b5dd071455c3cbe62221217a107
        return True


    #################################################################
    def analyze(self) -> bool:
        """
        comment
        """
<<<<<<< HEAD
        return True
    

    #################################################################
    def write_output(self) -> bool:
        """
        comment
        """
        return True

    #################################################################
# --- MAIN ----------------------------------------------------------
if __name__ == "__main__":

    # Setup variables according to the noise_run_info.yaml file
    with open("./noise_run_info.yaml", 'r') as stream:
        run_info = yaml.safe_load(stream)

    filepath_folder  = run_info.get("filepath_folder")
    run_vgain_dict   = run_info.get("run_vgain_dict", {})
    channel_map_file = run_info.get("channel_map_file")

    # Setup variables according to the user_config.yaml file
    with open("user_setting.yaml", 'r') as stream:
        user_config = yaml.safe_load(stream)

    out_writing_mode = user_config.get("out_writing_mode")
    out_path  = user_config.get("out_path")
    full_stat = user_config.get("full_stat")
    runs      = user_config.get("user_runs", [])
    if (len(runs) == 0):
        runs = user_config.get("all_noise_runs", [])
        if (len(runs) == 0):
            print("No runs to analyze")
            exit()

    # Read the channel map file (daphne ch <-> offline ch)
    df = pd.read_csv(channel_map_file, sep=",")
    daphne_channels = df['daphne_ch'].values + 100*df['endpoint'].values
    daphne_to_offline = dict(zip(daphne_channels, df['offline_ch']))


    # File where the results will be printed (run, vgain, endpoint, channel, offline_channel, rms)
    my_csv_file = open(out_path+"Noise_Studies_Results.csv", out_writing_mode)

    # --- LOOP OVER RUNS ----------------------------------------------
    for run in runs:
        print("Reading run: ", run)
        wfset_run = nf.read_waveformset(filepath_folder, run, full_stat=full_stat)
        endpoints = wfset_run.get_set_of_endpoints()
=======
        endpoints = self.wfset_run.get_set_of_endpoints()
        
        integrator_ON = False
        if self.run in self.integratorsON_runs:
            integrator_ON = True

>>>>>>> 264bdce2c6b35b5dd071455c3cbe62221217a107

        # --- LOOP OVER ENDPOINTS -------------------------------------
        for ep in endpoints:
            print("Endpoint: ", ep)
<<<<<<< HEAD
            wfset_ep = waffles.WaveformSet.from_filtered_WaveformSet(wfset_run, nf.allow_ep_wfs, ep)
=======
            wfset_ep = waffles.WaveformSet.from_filtered_WaveformSet(self.wfset_run, nf.allow_ep_wfs, ep)
>>>>>>> 264bdce2c6b35b5dd071455c3cbe62221217a107

            ep_ch_dict = wfset_ep.get_run_collapsed_available_channels()
            channels = list(ep_ch_dict[ep])

            # --- LOOP OVER CHANNELS ----------------------------------
            for ch in channels:
                print("Channel: ", ch)
                wfset_ch = waffles.WaveformSet.from_filtered_WaveformSet(wfset_ep, nf.allow_channel_wfs, ch)
                # check if the channel is in the daphne_to_offline dictionary
                channel = np.uint16(np.uint16(ep)*100+np.uint16(ch))
<<<<<<< HEAD
                if channel not in daphne_to_offline:
                    print(f"Channel {channel} not in the daphne_to_offline dictionary")
                    continue
                offline_ch = daphne_to_offline[channel]
=======
                if channel not in self.daphne_to_offline:
                    print(f"Channel {channel} not in the daphne_to_offline dictionary")
                    continue
                offline_ch = self.daphne_to_offline[channel]
>>>>>>> 264bdce2c6b35b5dd071455c3cbe62221217a107
        
                wfs = wfset_ch.waveforms
                nf.create_float_waveforms(wfs)
                nf.sub_baseline_to_wfs(wfs, 1024)

                norm = 1./len(wfs)
                fft2_avg = np.zeros(1024)
                rms = 0.

                # Compute the average FFT of the wfs.adcs_float
                for wf in wfs:
                    rms += np.std(wf.adcs_float)
                    fft  = np.fft.fft(wf.adcs_float)
                    fft2 = np.abs(fft)
                    fft2_avg += fft2

                fft2_avg = fft2_avg*norm
                rms = rms*norm
<<<<<<< HEAD
                vgain = run_vgain_dict[run]
                
                # print run, vgain, ep, ch, offline_ch, rms in a csv file
                my_csv_file.write(f"{run},{vgain},{ep},{ch},{offline_ch},{rms}\n")
                # print the FFT in a txt file
                np.savetxt(out_path+"/FFT_txt/fft_run_"+str(run)+"_vgain_"
                           +str(vgain)+"_ch_"+str(channel)+"_offlinech_"
                           +str(offline_ch)+".txt", fft2_avg[0:513])
=======
                vgain = self.run_vgain_dict[self.run]
                
                # print run, vgain, ep, ch, offline_ch, rms in a csv file
                self.out_csv_file.write(f"{self.run},{vgain},{ep},{ch},{offline_ch},{rms}\n")

                # Check wheter the folder FFT_txt exists in the "output" folder
                if not os.path.exists("output/FFT_txt"):
                    os.makedirs("output/FFT_txt")

                # print the FFT in a txt file
                print("Writing FFT to txt file")
                integrator = "OFF"
                if integrator_ON:
                    integrator = "ON"
                np.savetxt("output/FFT_txt/fft_run_"+str(self.run)
                           +"_int_"+integrator
                           +"_vgain_"+str(vgain)
                           +"_ch_"+str(channel)
                           +"_offlinech_"+str(offline_ch)+".txt", fft2_avg[0:513])

               
                if self.debug_mode:
                    nf.plot_heatmaps(wfs, "raw", self.run, vgain, int(channel), offline_ch)
                    nf.plot_heatmaps(wfs, "baseline_removed", self.run, vgain, int(channel), offline_ch)
                    print("done")

        return True
    

    #################################################################
    def write_output(self) -> bool:
        """
        comment
        """
        return True
>>>>>>> 264bdce2c6b35b5dd071455c3cbe62221217a107
