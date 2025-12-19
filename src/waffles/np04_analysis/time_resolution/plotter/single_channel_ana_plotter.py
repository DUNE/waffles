import yaml
import os
import numpy as np
import array

import ROOT
from ROOT import TFile, TH2F, TGraph, TGraphErrors




# --- MAIN ----------------------------------------------------------
if __name__ == "__main__":
    # --- SETUP -----------------------------------------------------
    # Setup variables according to the configs/time_resolution_config.yml file
    with open("../configs/time_resolution_configs.yml", 'r') as config_stream:
        cfg = yaml.safe_load(config_stream)

    ana_folder = cfg["ana_folder"]
    raw_ana_folder = ana_folder+cfg["raw_ana_folder"]
    single_ana_folder = ana_folder+cfg["single_ana_folder"]
    
    # Setup variables according to the params.yml file
    with open("../params.yml", 'r') as params_stream:
        params_variables = yaml.safe_load(params_stream)

    channels = params_variables["channels"]
    methods = params_variables["methods"]
    relative_thrs = params_variables["relative_thrs"]
    filt_levels = params_variables["filt_levels"]
    h2_nbins = params_variables["h2_nbins"]
    stat_lost = params_variables["stat_lost"]

    # --- EXTRA VARIABLES -------------------------------------------
    os.makedirs(single_ana_folder, exist_ok=True)
    
    # --- LOOP OVER RUNS --------------------------------------------
    files = [raw_ana_folder+f for f in os.listdir(raw_ana_folder) if f.endswith("time_resolution.root")]
    for file in files:
        int_root_file = TFile(file, "READ")

        out_root_file_name = file.replace(raw_ana_folder, single_ana_folder).replace(".root", "_plots.root")
        print("opening ", out_root_file_name)
        out_root_file = TFile(out_root_file_name, "RECREATE")

        for key in int_root_file.GetListOfKeys():
            root_dir = key.GetName()
            if root_dir == "persistence" or "/" in root_dir:
                continue
            dir = int_root_file.GetDirectory(root_dir)
            if not dir:
                continue
            tree = dir.Get("time_resolution")
            if not tree:
                continue

            rdf = ROOT.RDataFrame(tree)

            out_root_file.mkdir(root_dir.replace(";1", ""))
            out_root_file.cd(root_dir.replace(";1", ""))

            #Computing pe quantiles to define histogram range
            h_pe = rdf.Histo1D("pe").GetValue()
            probs = array.array("d", [stat_lost, 1.0 - stat_lost])
            qs    = array.array("d", [0.0, 0.0])
            h_pe.GetQuantiles(2, qs, probs)
            pe_low, pe_up = qs[0], qs[1]

            x_low  = float(int(pe_low))-0.5
            x_up   = float(int(pe_up)) +0.5
            x_bins = int(x_up - x_low)

            h_t0 = rdf.Histo1D("t0").GetValue()
            h_t0.GetQuantiles(2, qs, probs)
            t0_low, t0_up = qs[0], qs[1]

            # Histogram 2D of t0 vs pe
            h2_model = ROOT.RDF.TH2DModel(
                "t0_vs_pe",
                "2D Histogram;#p.e;t0 [ticks]",
                x_bins,   x_low,  x_up,
                h2_nbins, t0_low, t0_up
            )

            h2_t0_pe = rdf.Histo2D(h2_model, "pe", "t0")
            
            corr = h2_t0_pe.GetCorrelationFactor()
            h2_t0_pe.SetTitle(f"Correlation: {corr:.3f};#p.e;t0 [ticks]")
            h2_t0_pe.Write()

            # Graph of time resolution vs pe
            g_res_pe = TGraphErrors()
            ip = 0
            for i in range(h2_t0_pe.GetNbinsX()+1):
                h1_t0 = h2_t0_pe.ProjectionY(f"proj_{i}", i, i)
                if h1_t0.GetEntries() > 50:
                    sigma = h1_t0.GetRMS()
                    err_sigma = sigma / np.sqrt(float(h1_t0.GetEntries()))
                    g_res_pe.SetPoint(ip, h2_t0_pe.GetXaxis().GetBinCenter(i+1), sigma*16)
                    g_res_pe.SetPointError(ip, h2_t0_pe.GetXaxis().GetBinWidth(i+1)/2, err_sigma*16)
                    ip += 1

            g_res_pe.SetName("res_vs_pe")
            g_res_pe.SetTitle("Resolution vs p.e.;p.e.;#sigma_{t0} [ns]")
            g_res_pe.Write()

            # Create a TProfile from the 2D histogram
            hp_t0_pe = h2_t0_pe.ProfileX()
            hp_t0_pe.SetName("profile_t0_vs_pe")
            hp_t0_pe.SetTitle("profile_t0_vs_pe;#p.e.;t0 [ticks]")
            hp_t0_pe.Write()

            # TGraph p.e. vs timestamp
            g_pe_ts = rdf.Graph("timestamp", "pe")
            g_pe_ts.SetName("pe_vs_ts")
            g_pe_ts.SetTitle("p.e. vs timestamp;timestamp [a.u.];p.e.")
            g_pe_ts.Write()

        out_root_file.Close()
