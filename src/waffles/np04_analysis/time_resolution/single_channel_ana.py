from imports import *


# --- MAIN ----------------------------------------------------------
if __name__ == "__main__":
    # --- SETUP -----------------------------------------------------
    # Setup variables according to the configs/time_resolution_config.yml file
    with open("./configs/time_resolution_configs.yml", 'r') as config_stream:
        config_variables = yaml.safe_load(config_stream)

    data_folder = config_variables.get("data_folder")
    output_folder = config_variables.get("output_folder")
    
    # Setup variables according to the params.yml file
    with open("./params.yml", 'r') as params_stream:
        params_variables = yaml.safe_load(params_stream)

    runs = params_variables.get("runs")
    methods = params_variables.get("methods")
    channels = params_variables.get("channels")    
    endpoints = params_variables.get("endpoints")
    prepulse_ticks = params_variables.get("prepulse_ticks")
    postpulse_ticks = params_variables.get("postpulse_ticks")
    baseline_rms = params_variables.get("baseline_rms")
    relative_thrs = params_variables.get("relative_thrs")
    filt_levels = params_variables.get("filt_levels")
    min_amplitudes = params_variables.get("min_amplitudes")
    amp_range = params_variables.get("amp_range")
    max_amplitudes = [min_amplitude + amp_range for min_amplitude in min_amplitudes]

    # --- EXTRA VARIABLES -------------------------------------------
    files = [data_folder+"wfset_run_"+str(run)+".pkl" for run in runs]
    out_root_file_name = output_folder+"time_resolution.root"
    root_file = ROOT.TFile(out_root_file_name, "RECREATE")
    nbins = 200
    
    # --- LOOP OVER RUNS --------------------------------------------
    for file, run in zip(files, runs):
        print("Reading run ", run)
        with open(f'{file}', 'rb') as f:
            wfset_run = pickle.load(f)

        a = tr.TimeResolution(wf_set=wfset_run, 
                              prepulse_ticks=prepulse_ticks,
                              postpulse_ticks=postpulse_ticks,
                              min_amplitude=min_amplitudes[0],
                              max_amplitude=max_amplitudes[0],
                              baseline_rms=baseline_rms)

        endpoints = wfset_run.get_set_of_endpoints()
        print("Endpoints ", endpoints)
        for endpoint in endpoints:
            for channel in channels:
                for method in methods:
                    if method == "denoise":
                        loop_filt_levels = filt_levels
                    else:
                        loop_filt_levels = [0]

                    for relative_thr in relative_thrs:

                        for filt_level in loop_filt_levels:
                            root_file.mkdir(f"{method}_filt_{filt_level}/thr_{relative_thr}")
                            root_file.cd(f"{method}_filt_{filt_level}/thr_{relative_thr}")


                            print("Channel ", channel)
                            a.ref_ep = endpoint
                            a.ref_ch = channel
                            a.create_wfs(tag="ref")
                            
                            if (method == "denoise" and filt_level > 0):
                                a.create_denoised_wfs(filt_level=filt_level)
                            
                            out_file =  output_folder+"Ch_"+str(channel)+"_raw_results.csv"
       
                            for min_amplitude, max_amplitude in zip(min_amplitudes, max_amplitudes):
                                print("Setting min ", min_amplitude, " max ", max_amplitude)
                                a.min_amplitude=min_amplitude
                                a.max_amplitude=max_amplitude

                                a.select_time_resolution_wfs(tag="ref")

                                a.set_wfs_t0(tag="ref", method=method, relative_thr=relative_thr, filt_level=filt_level)
                                
                                if a.ref_n_select_wfs > 100:
                                    print("Save this ")
                                    hist_range = (np.min(a.ref_t0s), np.max(a.ref_t0s))
                                    hist = ROOT.TH1F(f"MinAmp_{a.min_amplitude}_MaxAmp_{a.max_amplitude}",
                                                     "Example Histogram",
                                                     nbins, *hist_range)

                                    counts, bin_edges = np.histogram(a.ref_t0s, bins=nbins, range=hist_range)
                                    for i in range(nbins):
                                        hist.SetBinContent(i+1, counts[i])

                                    hist.Write()

                                    file_exists = os.path.isfile(out_file)
                                    with open(out_file, mode='a', newline='') as file:
                                        writer = csv.writer(file)
                                        
                                        #Write the header only if new file
                                        if not file_exists:
                                            writer.writerow(['Run', 'method',
                                                             'thr', 'filt', 'Ch',
                                                             'min', 'max',
                                                             't0', 't0_std',
                                                             'n_wfs,'])

                                        writer.writerow([run, method,
                                                         relative_thr, filt_level,
                                                         endpoint*100+channel,
                                                         min_amplitude, max_amplitude,
                                                         a.ref_t0, a.ref_t0_std,
                                                         a.ref_n_select_wfs])
    root_file.Close()
