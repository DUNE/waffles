import yaml
import numpy as np

from ROOT import TFile, TH2F, TGraph, TTree
import uproot




# --- MAIN ----------------------------------------------------------
if __name__ == "__main__":
    # --- SETUP -----------------------------------------------------
    # Setup variables according to the configs/time_resolution_config.yml file
    with open("../configs/time_resolution_configs.yml", 'r') as config_stream:
        config_variables = yaml.safe_load(config_stream)

    output_folder = config_variables.get("output_folder")
    
    # Setup variables according to the params.yml file
    with open("../params.yml", 'r') as params_stream:
        params_variables = yaml.safe_load(params_stream)

    runs = params_variables.get("runs")
    endpoints = params_variables.get("endpoints")
    channels = params_variables.get("channels")    
    methods = params_variables.get("methods")
    relative_thrs = params_variables.get("relative_thrs")
    filt_levels = params_variables.get("filt_levels")
    h2_nbins = params_variables.get("h2_nbins")
    stat_lost = params_variables.get("stat_lost")

    # --- EXTRA VARIABLES -------------------------------------------
    
    # --- LOOP OVER RUNS --------------------------------------------
    for endpoint in endpoints:
        for channel in channels:
            ch = endpoint*100 + channel
            in_root_file_name = output_folder+f"ch_{ch}_time_resolution.root"
            root_file = uproot.open(in_root_file_name)

            out_root_file_name = output_folder+f"ch_{ch}_time_resolution_plots.root"
            print("opening ", out_root_file_name)
            out_root_file = TFile(out_root_file_name, "RECREATE")
            
            for run in runs:
                for method in methods:
                    if method == "denoise":
                        loop_filt_levels = filt_levels
                    else:
                        loop_filt_levels = [0]

                    for relative_thr in relative_thrs:

                        for filt_level in loop_filt_levels:
                            rel_thr = str(relative_thr).replace(".", "p")

                            # Get a root TTRee from the file
                            try:
                                directory = root_file[f"run_{run}_{method}_filt_{filt_level}/thr_{rel_thr}"]
                            except:
                                continue
                            tree = directory["time_resolution"]
                            branches = tree.keys()
                            arrays = tree.arrays(branches, library="np")
                            
                            t0s = arrays["t0"]
                            pes = arrays["pe"]
                            tss = arrays["timestamp"]

                            # Copy pes and tss in a new array
                            pes_aux = np.array(pes, dtype='d')
                            tss_aux = np.array(tss, dtype='d')

                            print(len(t0s), len(pes), len(tss))
                                
                            out_root_file.mkdir(f"run_{run}_{method}_filt_{filt_level}/thr_{rel_thr}")
                            out_root_file.cd(f"run_{run}_{method}_filt_{filt_level}/thr_{rel_thr}")

                            sorted_indices = np.argsort(pes)
                            t0s, pes, tss = t0s[sorted_indices], pes[sorted_indices], tss[sorted_indices]
                            n = len(pes)
                            low, high = int(stat_lost * n), int((1-stat_lost) * n)
                            t0s, pes, tss = t0s[low:high], pes[low:high], tss[low:high]
                            
                            sorted_indices = np.argsort(t0s)
                            t0s, pes, tss = t0s[sorted_indices], pes[sorted_indices], tss[sorted_indices]
                            n = len(t0s)
                            low, high = int(stat_lost * n), int((1-stat_lost) * n)
                            t0s, pes, tss = t0s[low:high], pes[low:high], tss[low:high]

                            counts, xedges, yedges = np.histogram2d(pes, t0s, bins=(h2_nbins,h2_nbins),
                                                                    range=[[np.min(pes), np.max(pes)],
                                                                          [np.min(t0s), np.max(t0s)]])
                            
                            # Histogram 2D of t0 vs pe
                            h2_t0_pe = TH2F("t0_vs_pe", "2D Histogram;#p.e;t0 [ticks]",
                                             h2_nbins, np.min(pes), np.max(pes),
                                             h2_nbins, np.min(t0s), np.max(t0s))

                            for i in range(h2_nbins):
                                for j in range(h2_nbins):
                                    h2_t0_pe.SetBinContent(i+1, j+1, counts[i, j]) 
                            
                            corr = h2_t0_pe.GetCorrelationFactor()
                            h2_t0_pe.SetTitle(f"Correlation: {corr}")
                            h2_t0_pe.Write()

                            # Graph of time resolution vs pe
                            g_res_pe = TGraph()
                            n_points = 30
                            for i in range(n_points):
                                min_pe = h2_t0_pe.GetXaxis().GetBinLowEdge(int(h2_nbins/n_points)*i+1)
                                max_pe = h2_t0_pe.GetXaxis().GetBinUpEdge(int(h2_nbins/n_points)*i+1)
                                h1_t0 = h2_t0_pe.ProjectionY(f"proj_{i}", int(h2_nbins/n_points)*i+1, int(h2_nbins/n_points)*(i+1))
                                sigma = h1_t0.GetRMS()
                                g_res_pe.SetPoint(i, (min_pe+max_pe)/2, sigma*16)

                            g_res_pe.SetName("res_vs_pe")
                            g_res_pe.SetTitle("Resolution vs p.e.;p.e.;#sigma_{t0} [ns]")
                            g_res_pe.Write()

                            # Create a TProfile from the 2D histogram
                            hp_t0_pe = h2_t0_pe.ProfileX()
                            hp_name = "profile_t0_vs_pe;#p.e.;t0 [ticks]"
                            hp_t0_pe.SetTitle(hp_name)
                            hp_t0_pe.SetName(hp_name)
                            hp_t0_pe.Write()

                            # TGraph p.e. vs timestamp
                            sorted_indices = np.argsort(tss_aux)
                            pes_aux = pes_aux[sorted_indices]
                            g_pe_ts = TGraph(pes.size, np.array([pes_aux], dtype='d'))
                            g_pe_ts.SetName("pe_vs_ts")
                            g_pe_ts.SetTitle("p.e. vs timestamp;timestamp [a.u.];p.e.")
                            g_pe_ts.Write()
                            



            out_root_file.Close()
