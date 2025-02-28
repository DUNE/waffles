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
    out_root_file_name = output_folder+"time_resolution.root"
    root_file = TFile(out_root_file_name, "RECREATE")
    min_min_pe = []
    max_max_pe = []
    min_min_t0 = []
    max_max_t0 = []
    hp_t0_pes = []
    
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
            
            for channel in channels:
                a.ch = channel
                a.create_wfs()
                a.select_time_resolution_wfs()
            
                if a.n_select_wfs > 500:
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
                                
                                a.set_wfs_t0(method=method, relative_thr=relative_thr)
                                t0s, pes = np.array(a.t0s), np.array(a.pes)

                                sorted_indices = np.argsort(pes)
                                t0s, pes = t0s[sorted_indices], pes[sorted_indices]
                                n = len(pes)
                                low, high = int(stat_lost * n), int((1-stat_lost) * n)
                                t0s, pes = t0s[low:high], pes[low:high]
                                
                                sorted_indices = np.argsort(t0s)
                                t0s, pes = t0s[sorted_indices], pes[sorted_indices]
                                n = len(t0s)
                                low, high = int(stat_lost * n), int((1-stat_lost) * n)
                                t0s, pes = t0s[low:high], pes[low:high]

                                min_min_pe.append(np.min(pes))
                                max_max_pe.append(np.max(pes))
                                min_min_t0.append(np.min(t0s))
                                max_max_t0.append(np.max(t0s))

                                counts, xedges, yedges = np.histogram2d(pes, t0s, bins=(h2_nbins,h2_nbins),
                                                                        range=[[np.min(pes), np.max(pes)],
                                                                              [np.min(t0s), np.max(t0s)]])

                                h2_t0_pe = TH2F("hist", "2D Histogram;#p.e;t0 [ticks]",
                                                 h2_nbins, np.min(pes), np.max(pes),
                                                 h2_nbins, np.min(t0s), np.max(t0s))

                                for i in range(h2_nbins):
                                    for j in range(h2_nbins):
                                        h2_t0_pe.SetBinContent(i+1, j+1, counts[i, j]) 
                                
                                corr = h2_t0_pe.GetCorrelationFactor()
                                h2_t0_pe.SetTitle(f"Correlation: {corr}")
                                h2_t0_pe.Write()

                                g_res_pe = TGraph()
                                # Segment the h2_t0_pe in 20 bins of pes, do the projection
                                # and retriev the sigma of the t0 distribution
                                for i in range(20):
                                    min_pe = h2_t0_pe.GetYaxis().GetBinLowEdge(int(h2_nbins/20)*i+1)
                                    max_pe = h2_t0_pe.GetYaxis().GetBinUpEdge(int(h2_nbins/20)*i+1)
                                    h1_t0 = h2_t0_pe.ProjectionX(f"proj_{i}", int(h2_nbins/20)*i+1, int(h2_nbins/20)*(i+1))
                                    sigma = h1_t0.GetRMS()
                                    g_res_pe.SetPoint(i, (min_pe+max_pe)/2, sigma)

                                g_res_pe.SetTitle("Resolution vs p.e.")
                                g_res_pe.Write()

                                if relative_thr == 0.5:
                                    # Create a TProfile from the 2D histogram
                                    hp_t0_pe = h2_t0_pe.ProfileX()
                                    hp_name = f"run_{run}"
                                    hp_t0_pe.SetTitle(hp_name)
                                    hp_t0_pe.SetName(hp_name)
                                    root_file.cd()
                                    hp_t0_pe.Write()
                                    hp_t0_pes.append(hp_t0_pe)

    # Draw all the TProfile in the same canvas
    # print(min_min_pe, max_max_pe)
    # print(np.min(min_min_pe), np.max(max_max_pe))
    # root_file.cd()
    # c = TCanvas("c", "c", 800, 800)
    # for hp, run in zip(hp_t0_pes, runs):
    #     hp.SetLineColor(run%9+1)
    #     hp.GetXaxis().SetRangeUser(np.min(min_min_pe), np.max(max_max_pe))
    #     if run == runs[0]:
    #         hp.Draw("HIST")
    #     else:
    #         hp.Draw("same HIST")
    # 
    # c.Write()
    root_file.Close()
