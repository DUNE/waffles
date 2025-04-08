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
    endpoints = params_variables.get("endpoints")
    channels = params_variables.get("channels")    
    prepulse_ticks = params_variables.get("prepulse_ticks")
    int_low = params_variables.get("int_low")
    int_up = params_variables.get("int_up")
    postpulse_ticks = params_variables.get("postpulse_ticks")
    spe_charge = params_variables.get("spe_charge")
    spe_ampl = params_variables.get("spe_ampl")
    min_pes = params_variables.get("min_pes")
    baseline_rms = params_variables.get("baseline_rms")
    methods = params_variables.get("methods")
    relative_thrs = params_variables.get("relative_thrs")
    filt_levels = params_variables.get("filt_levels")
    h2_nbins = params_variables.get("h2_nbins")
    stat_lost = params_variables.get("stat_lost")

    # --- EXTRA VARIABLES -------------------------------------------
    files = [data_folder+"wfset_run_"+str(run)+".pkl" for run in runs]
    min_min_pe = []
    max_max_pe = []
    min_min_t0 = []
    max_max_t0 = []
    hp_t0_pes = []
    os.makedirs(output_folder, exist_ok=True)
    
    # --- LOOP OVER RUNS --------------------------------------------
    for file, run in zip(files, runs):
        print("Reading run ", run)
        with open(f'{file}', 'rb') as f:
            wfset_run = pickle.load(f)

        a = tr.TimeResolution(wf_set=wfset_run, 
                              prepulse_ticks=prepulse_ticks,
                              int_low=int_low,
                              int_up=int_up,
                              postpulse_ticks=postpulse_ticks,
                              spe_charge = spe_charge,
                              spe_ampl = spe_ampl,
                              min_pes = min_pes,
                              baseline_rms=baseline_rms)

        endpoints = wfset_run.get_set_of_endpoints()
        print("Endpoints ", endpoints)
        for endpoint in endpoints:
            a.ep = endpoint
            ch_iter = 0
            
            for channel in channels:
                a.ch = channel
                a.create_wfs()
                a.select_time_resolution_wfs()
   
                ch = endpoint*100 + channel
                out_root_file_name = output_folder+f"ch_{ch}_time_resolution.root"
                root_file = TFile(out_root_file_name, "RECREATE")
            
                if a.n_select_wfs > 500:
                    root_file.mkdir(f"run_{run}")
                    root_file.cd(f"run_{run}")
                    n_pwfs = min(9000, a.n_select_wfs)
                    all_wfs = np.array([wf.adcs_float for wf in a.wfs[:n_pwfs] if wf.time_resolution_selection]).flatten()
                    all_tikcs = np.array([np.arange(len(wf.adcs_float)) for wf in a.wfs[:n_pwfs] if wf.time_resolution_selection]).flatten()
                    counts, xedges, yedges = np.histogram2d(all_tikcs, all_wfs, bins=(len(a.wfs[0].adcs_float),h2_nbins),
                                                            range=[[0, len(a.wfs[0].adcs_float)],
                                                                   [np.min(all_wfs), np.max(all_wfs)]])
                                                            

                    # Histogram 2D of t0 vs pe
                    h2_persistence = TH2F("persistence", ";#p.e;t0 [ticks]",
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
                                root_file.mkdir(f"run_{run}_{method}_filt_{filt_level}/thr_{rel_thr}")
                                root_file.cd(f"run_{run}_{method}_filt_{filt_level}/thr_{rel_thr}")

                                print("Channel ", channel)
                                
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

