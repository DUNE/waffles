from pandas.core import base
from imports import *


# --- MAIN ----------------------------------------------------------
if __name__ == "__main__":
    # --- SETUP -----------------------------------------------------
    # Setup variables according to the configs/time_resolution_config.yml file
    with open("./configs/time_resolution_configs.yml", 'r') as config_stream:
        config_variables = yaml.safe_load(config_stream)

    data_folder = config_variables.get("data_folder")
    ana_folder = config_variables.get("ana_folder")
    raw_ana_folder = ana_folder+config_variables.get("raw_ana_folder")
    # channel_map_file = config_variables.get("channel_map_file")
    new_channel_map_file = config_variables.get("new_channel_map_file")
    mapping_df = pd.read_csv("configs/"+new_channel_map_file, sep=",")
    new_daphne_channels = mapping_df['daphne_ch'].values + 100*mapping_df['endpoint'].values
    new_daphne_to_offline = dict(zip(new_daphne_channels, mapping_df['offline_ch']))
    
    # Setup variables according to the params.yml file
    with open("./params.yml", 'r') as params_stream:
        params_variables = yaml.safe_load(params_stream)

    runs = params_variables.get("runs")
    noise_results_file = params_variables.get("noise_results_file")
    noise_df = pd.read_csv(noise_results_file, sep=",")
    print(noise_df.head(5))

    calibration_file = params_variables.get("calibration_file")
    calibration_df = pd.read_csv(calibration_file, sep=",")
    print(calibration_df.head(5))


    channels = params_variables.get("channels")    
    global_prepulse_ticks = params_variables.get("prepulse_ticks")
    global_int_low = params_variables.get("int_low")
    global_int_up = params_variables.get("int_up")
    global_postpulse_ticks = params_variables.get("postpulse_ticks")
    min_pes = params_variables.get("min_pes")
    global_baseline_rms = params_variables.get("baseline_rms")
    methods = params_variables.get("methods")
    relative_thrs = params_variables.get("relative_thrs")
    filt_levels = params_variables.get("filt_levels")
    h2_nbins = params_variables.get("h2_nbins")
    stat_lost = params_variables.get("stat_lost")

    # --- EXTRA VARIABLES -------------------------------------------
    files = [data_folder+"processed_merged_run_"+str(run)+"_structured.hdf5" for run in runs]
    min_min_pe = []
    max_max_pe = []
    min_min_t0 = []
    max_max_t0 = []
    hp_t0_pes = []
    os.makedirs(ana_folder, exist_ok=True)
    os.makedirs(raw_ana_folder, exist_ok=True)
    
    # --- LOOP OVER RUNS --------------------------------------------
    for file, run in zip(files, runs):
        print("Reading run ", run)
        try:
            wfset_run = reader.load_structured_waveformset(file)
        except FileNotFoundError:
            print(f"File {file} not found. Skipping.")
            continue
            
        a = tr.TimeResolution(wf_set=wfset_run) 
           
        # --- LOOP OVER CHANNELS ------------------------------------------
        if (channels == []):
            print("No channels specified. Using all channels.")
        for daphne_ch in channels:
            if daphne_ch not in new_daphne_to_offline:
                print(f"Channel {daphne_ch} not in new channel map")
                continue

            offline_ch = new_daphne_to_offline[daphne_ch]

            if global_prepulse_ticks != 0:
                prepulse_ticks = global_prepulse_ticks
            else:
                prepulse_ticks = int(calibration_df.loc[calibration_df['DaphneCh'] == daphne_ch, 'prepulse_ticks'].values[0])

            if global_int_low != 0:
                int_low = global_int_low
            else:
                int_low = int(calibration_df.loc[calibration_df['DaphneCh'] == daphne_ch, 'int_low'].values[0])

            if global_int_up != 0:
                int_up = global_int_up
            else:
                int_up = int(calibration_df.loc[calibration_df['DaphneCh'] == daphne_ch, 'int_up'].values[0])

            if global_postpulse_ticks != 0:
                postpulse_ticks = global_postpulse_ticks
            else:
                postpulse_ticks = int(calibration_df.loc[calibration_df['DaphneCh'] == daphne_ch, 'postpulse_ticks'].values[0])

                
            spe_charge = float(calibration_df.loc[calibration_df['DaphneCh'] == daphne_ch, 'Gain'].values[0])
            spe_ampl = float(calibration_df.loc[calibration_df['DaphneCh'] == daphne_ch, 'SpeAmpl'].values[0])
            baseline_rms = float(noise_df.loc[noise_df['OfflineCh'] == offline_ch, 'RMS'].values[0])

           
            try:
                a.set_analysis_parameters(ch=daphne_ch, prepulse_ticks=prepulse_ticks,
                                          postpulse_ticks=postpulse_ticks, int_low=int_low,
                                          int_up=int_up, spe_charge=spe_charge,
                                          spe_ampl=spe_ampl, min_pes=min_pes,
                                          baseline_rms=baseline_rms)
            except ValueError as e:
                print(f"Error: {e}")
                continue
            
            a.create_wfs()
            a.select_time_resolution_wfs()
   
            out_root_file_name = raw_ana_folder+f"Run_{run}_DaphneCh_{daphne_ch}_OfflineCh_{offline_ch}_time_resolution.root"
            root_file = TFile(out_root_file_name, "RECREATE")
        
            if a.n_select_wfs > 500:
                n_pwfs = min(9000, a.n_select_wfs)
                all_wfs = np.array([wf.adcs_float for wf in a.wfs[:n_pwfs] if wf.time_resolution_selection]).flatten()
                all_tikcs = np.array([np.arange(len(wf.adcs_float)) for wf in a.wfs[:n_pwfs] if wf.time_resolution_selection]).flatten()
                counts, xedges, yedges = np.histogram2d(all_tikcs, all_wfs, bins=(len(a.wfs[0].adcs_float),h2_nbins),
                                                        range=[[0, len(a.wfs[0].adcs_float)],
                                                               [np.min(all_wfs), np.max(all_wfs)]])
                                                        

                # Histogram 2D of t0 vs pe
                h2_persistence = TH2F("persistence", ";time [ticks]; Amplitude [ADC]",
                                len(a.wfs[0].adcs_float), 0, len(a.wfs[0].adcs_float),
                                h2_nbins, np.min(all_wfs), np.max(all_wfs))

                for i in range(len(a.wfs[0].adcs_float)):
                    for j in range(h2_nbins):
                        h2_persistence.SetBinContent(i+1, j+1, counts[i, j])
                
                h2_persistence.Write()

                for method in methods:
                        if method == "denoise":
                            loop_filt_levels = filt_levels
                        else:
                            loop_filt_levels = [0]

                        for relative_thr in relative_thrs:

                            for filt_level in loop_filt_levels:
                                rel_thr = str(relative_thr).replace(".", "p")
                                root_file.mkdir(f"{method}_filt_{filt_level}_thr_{rel_thr}")
                                root_file.cd(f"{method}_filt_{filt_level}_thr_{rel_thr}")

                                print("Channel ", daphne_ch)
                                
                                if (method == "denoise" and filt_level > 0):
                                    a.create_denoised_wfs(filt_level=filt_level)
                               
                                t0s, pes, tss = a.set_wfs_t0(method=method, relative_thr=relative_thr)
                                
                                t = TTree("time_resolution", "time_resolution")
                                t0 = np.zeros(1, dtype=np.float64)
                                pe = np.zeros(1, dtype=np.float64)
                                ts = np.zeros(1, dtype=np.float64)
                                t.Branch("t0", t0, "t0/D")
                                t.Branch("pe", pe, "pe/D")
                                t.Branch("timestamp", ts, "timestamp/D")

                                for i in range(len(t0s)):
                                    t0[0] = t0s[i]
                                    pe[0] = pes[i]
                                    ts[0] = tss[i]
                                    t.Fill()

                                t.Write()
    
            root_file.Close()
        del a, wfset_run
