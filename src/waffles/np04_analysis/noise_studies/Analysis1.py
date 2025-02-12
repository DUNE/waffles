from imports import *

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
            filepath_folder: str
            run_vgain_dict: Dict[int, int]
            channel_map_file: str
            out_writing_mode: str
            out_path: str
            full_stat: bool
            user_runs: List[int]
            all_noise_runs: List[int]

        return InputParams


    #################################################################
    def initialize(self, input_parameters: BaseInputParams) -> None:
        """
        comment
        """
        self.filepath_folder  = input_parameters.filepath_folder
        self.run_vgain_dict   = input_parameters.run_vgain_dict
        self.channel_map_file = input_parameters.channel_map_file
        self.out_writing_mode = input_parameters.out_writing_mode
        self.out_path         = input_parameters.out_path
        self.full_stat        = input_parameters.full_stat
        self.runs             = input_parameters.user_runs
        if (len(self.runs) == 0):
            self.runs = input_parameters.all_noise_runs
            if (len(self.runs) == 0):
                print("No runs to analyze")
                exit()

    #################################################################
    def read_input(self) -> bool:
        """
        comment
        """
        return True


    #################################################################
    def analyze(self) -> bool:
        """
        comment
        """
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

        # --- LOOP OVER ENDPOINTS -------------------------------------
        for ep in endpoints:
            print("Endpoint: ", ep)
            wfset_ep = waffles.WaveformSet.from_filtered_WaveformSet(wfset_run, nf.allow_ep_wfs, ep)

            ep_ch_dict = wfset_ep.get_run_collapsed_available_channels()
            channels = list(ep_ch_dict[ep])

            # --- LOOP OVER CHANNELS ----------------------------------
            for ch in channels:
                print("Channel: ", ch)
                wfset_ch = waffles.WaveformSet.from_filtered_WaveformSet(wfset_ep, nf.allow_channel_wfs, ch)
                # check if the channel is in the daphne_to_offline dictionary
                channel = np.uint16(np.uint16(ep)*100+np.uint16(ch))
                if channel not in daphne_to_offline:
                    print(f"Channel {channel} not in the daphne_to_offline dictionary")
                    continue
                offline_ch = daphne_to_offline[channel]
        
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
                vgain = run_vgain_dict[run]
                
                # print run, vgain, ep, ch, offline_ch, rms in a csv file
                my_csv_file.write(f"{run},{vgain},{ep},{ch},{offline_ch},{rms}\n")
                # print the FFT in a txt file
                np.savetxt(out_path+"/FFT_txt/fft_run_"+str(run)+"_vgain_"
                           +str(vgain)+"_ch_"+str(channel)+"_offlinech_"
                           +str(offline_ch)+".txt", fft2_avg[0:513])
