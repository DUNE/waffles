import yaml
import os
import numpy as np

from ROOT import TFile, TH2F, TGraph, TGraphErrors
import uproot




# --- MAIN ----------------------------------------------------------
if __name__ == "__main__":
    # --- SETUP -----------------------------------------------------
    # Setup variables according to the configs/time_resolution_config.yml file
    with open("../configs/time_resolution_configs.yml", 'r') as config_stream:
        config_variables = yaml.safe_load(config_stream)

    ana_folder = config_variables.get("ana_folder")
    raw_ana_folder = ana_folder+config_variables.get("raw_ana_folder")
    single_ana_folder = ana_folder+config_variables.get("single_ana_folder")
    
    # Setup variables according to the params.yml file
    with open("../params.yml", 'r') as params_stream:
        params_variables = yaml.safe_load(params_stream)

    endpoints = params_variables.get("endpoints")
    channels = params_variables.get("channels")    
    methods = params_variables.get("methods")
    relative_thrs = params_variables.get("relative_thrs")
    filt_levels = params_variables.get("filt_levels")
    h2_nbins = params_variables.get("h2_nbins")
    stat_lost = params_variables.get("stat_lost")

    # --- EXTRA VARIABLES -------------------------------------------
    os.makedirs(single_ana_folder, exist_ok=True)
    
    # --- LOOP OVER RUNS --------------------------------------------
    files = [raw_ana_folder+f for f in os.listdir(raw_ana_folder) if f.endswith("time_resolution.root")]
    for file in files:
        in_root_file_name = file
        root_file = uproot.open(in_root_file_name)
        root_dirs = root_file.keys()

        out_root_file_name = in_root_file_name.replace(raw_ana_folder, single_ana_folder).replace(".root", "_plots.root")
        print("opening ", out_root_file_name)
        out_root_file = TFile(out_root_file_name, "RECREATE")

        for root_dir in root_dirs:
            try:
                if root_dir == "persistence;1" or "/" in root_dir:
                    continue
                directory = root_file[root_dir]
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
                
            out_root_file.mkdir(root_dir.replace(";1", ""))
            out_root_file.cd(root_dir.replace(";1", ""))

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
            g_res_pe = TGraphErrors()
            n_points = 30
            rebin = int(h2_nbins/n_points)
            h2_t0_pe.RebinX(rebin)
            for i in range(h2_t0_pe.GetNbinsX()):
                h1_t0 = h2_t0_pe.ProjectionY(f"proj_{i}", i+1, i+1)
                if h1_t0.GetEntries() > 50:
                    sigma = h1_t0.GetRMS()
                    err_sigma = sigma/np.sqrt(float(h1_t0.GetEntries()))
                    g_res_pe.SetPoint(i, h2_t0_pe.GetXaxis().GetBinCenter(i+1), sigma*16)
                    g_res_pe.SetPointError(i, h2_t0_pe.GetXaxis().GetBinWidth(i+1)/2, err_sigma*16)

            g_res_pe.SetName("res_vs_pe")
            g_res_pe.SetTitle("Resolution vs p.e.;p.e.;#sigma_{t0} [ns]")
            g_res_pe.Write()

            # Create a TProfile from the 2D histogram
            hp_t0_pe = h2_t0_pe.ProfileX()
            hp_t0_pe.SetName("profile_t0_vs_pe")
            hp_t0_pe.SetTitle("profile_t0_vs_pe;#p.e.;t0 [ticks]")
            hp_t0_pe.Write()

            # TGraph p.e. vs timestamp
            sorted_indices = np.argsort(tss_aux)
            pes_aux = pes_aux[sorted_indices]
            g_pe_ts = TGraph(pes.size, np.array([pes_aux], dtype='d'))
            g_pe_ts.SetName("pe_vs_ts")
            g_pe_ts.SetTitle("p.e. vs timestamp;timestamp [a.u.];p.e.")
            g_pe_ts.Write()
            

        out_root_file.Close()
