# --- IMPORTS -------------------------------------------------------
import yaml
import os
import pandas as pd
import numpy as np
from ROOT import TH1F, TH2F, TFile, TGraphErrors
import uproot
import time_alignment as ta

# --- VARIABLES -----------------------------------------------------
ref_chs = [11223, 11221, 11223, 11221, 11223, 11221, 11223, 11221, 11223, 11221, 11223, 11221] 
com_chs = [11225, 11223, 11231, 11231, 11146, 11146, 11247, 11247, 11144, 11144, 11147, 11147]
# ref_chs = [11130, 11144, 11143, 11112, 11145, 11113, 11120, 11223, 11221, 11231, 11221, 11146, 11146] 
# com_chs = [11132, 11146, 11147, 11114, 11147, 11114, 11125, 11225, 11223, 11235, 11231, 11225, 11221]
# root_directory_name = "run_34081_half_amplitude_filt_0/thr_0p5/"
h2_nbins = 200

# --- MAIN ----------------------------------------------------------
if __name__ == "__main__":
    # --- SETUP -----------------------------------------------------
    # Setup variables according to the configs/time_resolution_config.yml file
    with open("./configs/time_resolution_configs.yml", 'r') as config_stream:
        config_variables = yaml.safe_load(config_stream)

    run_info_file = config_variables.get("run_info_file")
    ana_folder = config_variables.get("ana_folder")
    raw_ana_folder = ana_folder+config_variables.get("raw_ana_folder")
    out_folder = ana_folder+config_variables.get("pair_ana_folder")
    os.makedirs(out_folder, exist_ok=True)

    run_info_df = pd.read_csv("configs/"+run_info_file, sep=",")
    



    for ref_ch, com_ch in zip(ref_chs, com_chs):
        files = [raw_ana_folder+f for f in os.listdir(raw_ana_folder) if f.endswith("time_resolution.root") and ((str(ref_ch) in f) or (str(com_ch) in f))]
        runs = set()
        for file in files:
            run = file.split("Run_")[-1].split("_")[0]
            runs.add(run)
        runs = sorted(runs, key=lambda x: int(x))
        pdes = set(run_info_df["PDE"].values )
        
        out_root_file_name = out_folder+f"DaphneCh_{ref_ch}_vs_{com_ch}_time_alignment.root"
        print("opening ", out_root_file_name)
        out_root_file = TFile(out_root_file_name, "RECREATE")

        # create as many tgraphs as there are PDEs
        g_dt0_led = {}
        for pde in pdes:
            g_dt0_led[pde] = TGraphErrors(f"dt0_led_{pde}", "t0 diff vs run")
            g_dt0_led[pde].SetTitle(f"t0 diff vs led;LED;#Delta t0 [ticks=16ns]")
            g_dt0_led[pde].SetName(f"g_dt0_led_{pde}pde")
            g_dt0_led[pde].SetMarkerStyle(20)

        # --- LOOP OVER RUNS --------------------------------------------
        for run in runs:
            ref_file = [f for f in files if str(ref_ch) in f and run in f][0]
            com_file = [f for f in files if str(com_ch) in f and run in f][0]
            if not (os.path.exists(ref_file) and os.path.exists(com_file)):
                continue

            run_dir_name = f"Run_{run}"
            out_root_file.mkdir(run_dir_name)
            out_root_file.cd(run_dir_name)


            in_root_file_name = ref_file
            root_file = uproot.open(in_root_file_name)
            root_dirs = [root_dir for root_dir in root_file.keys() if root_dir != "persistence;1" and "/" not in root_dir ]

            pde = run_info_df.loc[run_info_df["Run"] == int(run), "PDE"].values[0]
            led = run_info_df.loc[run_info_df["Run"] == int(run), "LEDIntensity"].values[0]


            time_alligner = ta.TimeAligner(ref_ch, com_ch)

            for root_dir in root_dirs:
            
                time_alligner.set_quantities(ref_file, com_file, root_dir)
                subdir_name = run_dir_name + "/" + root_dir.replace(";1", "")
                time_alligner.allign_events()
                if len(time_alligner.ref_ch.t0s) < 1000:
                    print(f"Not enough events to plot for {root_dir} in {run}")
                    continue
                
                out_root_file.mkdir(subdir_name)
                out_root_file.cd(subdir_name)

                t0_diff = time_alligner.ref_ch.t0s - time_alligner.com_ch.t0s
                print("Histogram of t0 differences")

                # --- PLOTTING ---------------------------------------------------
                # t0 differences distribution ------------------------------------
                h_t0_diff = TH1F("h_t0_diff", "Reference-Comparison time difference;t0 [ticks=16ns];Counts",
                                 200, np.min(t0_diff), np.max(t0_diff))
                for diff in t0_diff:
                    h_t0_diff.Fill(diff)

                # Com vs Ref pes -------------------------------------------------
                counts, xedges, yedges = np.histogram2d(time_alligner.com_ch.pes, time_alligner.ref_ch.pes,
                                                        bins=(h2_nbins,h2_nbins),
                                                        # range=[[500,1000],[500,1000]])
                                                        range=[[np.min(time_alligner.com_ch.pes), np.max(time_alligner.com_ch.pes)],
                                                               [np.min(time_alligner.ref_ch.pes), np.max(time_alligner.ref_ch.pes)]])
                    
                h2_pes = TH2F("h2_pes", "Comparison vs Reference Channel p.e.s;#pe_{ref};#pe_{com}",
                              # h2_nbins, 500, 1000, h2_nbins, 500, 1000)
                              h2_nbins, np.min(time_alligner.ref_ch.pes), np.max(time_alligner.ref_ch.pes),
                              h2_nbins, np.min(time_alligner.com_ch.pes), np.max(time_alligner.com_ch.pes))

                for i in range(h2_nbins):
                    for j in range(h2_nbins):
                        h2_pes.SetBinContent(i+1, j+1, counts[i, j])

                # Com vs Ref t0 --------------------------------------------------
                counts, xedges, yedges = np.histogram2d(time_alligner.ref_ch.t0s, time_alligner.com_ch.t0s,
                                                        bins=(h2_nbins,h2_nbins),
                                                        range=[[np.min(time_alligner.ref_ch.t0s), np.max(time_alligner.ref_ch.t0s)],
                                                               [np.min(time_alligner.com_ch.t0s), np.max(time_alligner.com_ch.t0s)]])

                h2_t0 = TH2F("h2_t0", "Comparison vs Reference Channel t0;t0_{ref} [ticks=16ns];t0_{com} [ticks=16ns]",
                             h2_nbins, np.min(time_alligner.ref_ch.t0s), np.max(time_alligner.ref_ch.t0s),
                             h2_nbins, np.min(time_alligner.com_ch.t0s), np.max(time_alligner.com_ch.t0s))

                for i in range(h2_nbins):
                    for j in range(h2_nbins):
                        h2_t0.SetBinContent(i+1, j+1, counts[i, j])

                # Add point to overall graphs ------------------------------------
                if "integral" in root_dir:
                    g_dt0_led[pde].SetPoint(g_dt0_led[pde].GetN(), int(led), np.mean(t0_diff))
                    g_dt0_led[pde].SetPointError(g_dt0_led[pde].GetN()-1, 0, np.std(t0_diff)/2.)
        

                # --- WRITING ----------------------------------------------------
                h_t0_diff.Write()
                h2_pes.Write()
                h2_t0.Write()
        
        out_root_file.cd()
        for pde in pdes:
            if g_dt0_led[pde].GetN() > 0:
                g_dt0_led[pde].Write()

        out_root_file.Close()
