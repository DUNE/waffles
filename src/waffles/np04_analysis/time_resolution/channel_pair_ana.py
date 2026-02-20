# --- IMPORTS -------------------------------------------------------
import yaml
import os
import pandas as pd
import numpy as np
from ROOT import TH1F, TH2F, TFile, TGraphErrors
import uproot
import waffles.np04_analysis.time_resolution.time_alignment as ta
from waffles.np04_utils.utils import get_np04_daphne_to_offline_channel_dict
# -------------------------------------------------------------------

# --- MAIN ----------------------------------------------------------
if __name__ == "__main__":
    # --- SETUP -----------------------------------------------------
    with open("steering.yml", 'r') as stream:
        steering_config = yaml.safe_load(stream)
   
    h2_nbins   = steering_config.get("h2_nbins", 100)
    event_type = steering_config.get("event_type", "led")
    if event_type != "led" and event_type != "cosmic":
        raise ValueError("event_type must be either 'led' or 'cosmic'")

    ref_chs = steering_config.get("reference_channels", [])
    com_chs = steering_config.get("comparison_channels", [])
    if len(ref_chs) == 1:
        ref_chs = ref_chs*len(com_chs)
    elif len(ref_chs) != len(com_chs):
        raise ValueError("Reference and Comparison channel lists must have the same length or only one reference channel.")

    # Setup variables according to the configs/time_resolution_config.yml file
    with open("./configs/time_resolution_configs.yml", 'r') as config_stream:
        config_variables = yaml.safe_load(config_stream)

    run_info_file  = config_variables.get("run_info_file")
    ana_folder     = config_variables.get("ana_folder")
    raw_ana_folder = ana_folder+config_variables.get("raw_ana_folder")
    out_folder     = ana_folder+config_variables.get("pair_ana_folder")
    os.makedirs(out_folder, exist_ok=True)

    run_info_df       = pd.read_csv("configs/"+run_info_file, sep=",")
    daphne_to_offline = get_np04_daphne_to_offline_channel_dict(version="new")
    if event_type == "cosmic":
        daphne_to_offline = get_np04_daphne_to_offline_channel_dict(version="old")


    out_df_rows = []
    for ref_ch, com_ch in zip(ref_chs, com_chs):
        files = [raw_ana_folder+f for f in os.listdir(raw_ana_folder) if f.endswith("time_resolution.root") and ((str(ref_ch) in f) or (str(com_ch) in f))]
        runs  = set()

        for file in files:
            run = file.split("Run_")[-1].split("_")[0]
            runs.add(run)
        
        runs = sorted(runs, key=lambda x: int(x))
        pdes = set(run_info_df["PDE"].values )

        out_root_file_name = out_folder+f"DaphneCh_{ref_ch}_vs_{com_ch}_time_alignment.root"
        out_root_file = TFile(out_root_file_name, "RECREATE")

        # create as many tgraphs as there are PDEs
        g_dt0_led = {}
        if event_type == "led":
            for pde in pdes:
                g_dt0_led[pde] = TGraphErrors(f"dt0_led_{pde}", "t0 diff vs run")
                g_dt0_led[pde].SetTitle(f"t0 diff vs led;LED;#Delta t0 [ticks=16ns]")
                g_dt0_led[pde].SetName(f"g_dt0_led_{pde}pde")
                g_dt0_led[pde].SetMarkerStyle(20)

        # --- LOOP OVER RUNS --------------------------------------------
        for run in runs:
            try:
                ref_file = [f for f in files if str(ref_ch) in f and run in f][0]
                com_file = [f for f in files if str(com_ch) in f and run in f][0]
            except IndexError:
                print(f"Missing file for run {run}, skipping...")
                continue
            if not (os.path.exists(ref_file) and os.path.exists(com_file)):
                continue

            run_dir_name = f"Run_{run}"
            out_root_file.mkdir(run_dir_name)
            out_root_file.cd(run_dir_name)


            in_root_file_name = ref_file
            root_file = uproot.open(in_root_file_name)
            root_dirs = [root_dir for root_dir in root_file.keys() if root_dir != "persistence;1" and "/" not in root_dir ]

            if event_type == "led":
                pde = run_info_df.loc[run_info_df["Run"] == int(run), "PDE"].values[0]
                led = run_info_df.loc[run_info_df["Run"] == int(run), "LEDIntensity"].values[0]


            time_alligner = ta.TimeAligner(ref_ch, com_ch)

            for root_dir in root_dirs:
            
                time_alligner.set_quantities(ref_file, com_file, root_dir)
                subdir_name = run_dir_name + "/" + root_dir.replace(";1", "")
                if event_type == "led":
                    time_alligner.allign_events()
                    if len(time_alligner.ref_ch.t0s) < 1000:
                        print(f"Not enough events to plot for {root_dir} in {run}")
                        continue
                
                out_root_file.mkdir(subdir_name)
                out_root_file.cd(subdir_name)

                t0_diff = (time_alligner.com_ch.t0s - time_alligner.ref_ch.t0s)*16.0 # convert to ns

                # --- PLOTTING ---------------------------------------------------
                # t0 differences distribution ------------------------------------
                x_min = np.percentile(t0_diff, 0.5)
                x_max = np.percentile(t0_diff, 99.5)
                h_t0_diff = TH1F("h_t0_diff", "Comparison-Reference time difference;#Deltat [ns];Counts",
                                 200, x_min, x_max)
                for diff in t0_diff:
                    h_t0_diff.Fill(diff)

                t0_offset = h_t0_diff.GetMean()
                out_df_rows.append({
                    "Run": int(run),
                    "ReferenceChannel": ref_ch,
                    "ComparisonChannel": com_ch,
                    "OfflineRefChannel": daphne_to_offline[ref_ch],
                    "OfflineComChannel": daphne_to_offline[com_ch],
                    "Method": root_dir,
                    "PDE": pde if event_type == "led" else np.nan,
                    "LEDIntensity": led if event_type == "led" else np.nan,
                    "T0Offset [ticks]": t0_offset,
                    "T0Offset [ns]": t0_offset*16.0,
                    "T0Offset_StdDev [ns]": h_t0_diff.GetStdDev()*16.0
                    })

                # Com vs Ref pes -------------------------------------------------
                x_min = np.percentile(time_alligner.ref_ch.pes, 0.5)
                x_max = np.percentile(time_alligner.ref_ch.pes, 99.5)
                y_min = np.percentile(time_alligner.com_ch.pes, 0.5)
                y_max = np.percentile(time_alligner.com_ch.pes, 99.5)
                counts, xedges, yedges = np.histogram2d(time_alligner.ref_ch.pes, time_alligner.com_ch.pes,
                                                        bins=(h2_nbins,h2_nbins),
                                                        # range=[[500,1000],[500,1000]])
                                                        range=[[x_min, x_max], [y_min, y_max]])
                    
                h2_pes = TH2F("h2_pes", "Comparison vs Reference Channel p.e.s;#pe_{ref};#pe_{com}",
                              # h2_nbins, 500, 1000, h2_nbins, 500, 1000)
                              h2_nbins, x_min, x_max, h2_nbins, y_min, y_max)

                for i in range(h2_nbins):
                    for j in range(h2_nbins):
                        h2_pes.SetBinContent(i+1, j+1, counts[i, j])

                # Com vs Ref t0 --------------------------------------------------
                x_min = np.percentile(time_alligner.ref_ch.t0s, 0.5)
                x_max = np.percentile(time_alligner.ref_ch.t0s, 99.5)
                y_min = np.percentile(time_alligner.com_ch.t0s, 0.5)
                y_max = np.percentile(time_alligner.com_ch.t0s, 99.5)
                counts, xedges, yedges = np.histogram2d(time_alligner.ref_ch.t0s, time_alligner.com_ch.t0s,
                                                        bins=(h2_nbins,h2_nbins),
                                                        range=[[x_min, x_max],[y_min, y_max]])

                h2_t0 = TH2F("h2_t0", "Comparison vs Reference Channel t0;t0_{ref} [ticks=16ns];t0_{com} [ticks=16ns]",
                             h2_nbins, x_min, x_max, h2_nbins, y_min, y_max)

                for i in range(h2_nbins):
                    for j in range(h2_nbins):
                        h2_t0.SetBinContent(i+1, j+1, counts[i, j])

                # Add point to overall graphs ------------------------------------
                if "integral" in root_dir and event_type == "led":
                    g_dt0_led[pde].SetPoint(g_dt0_led[pde].GetN(), int(led), np.mean(t0_diff))
                    g_dt0_led[pde].SetPointError(g_dt0_led[pde].GetN()-1, 0, np.std(t0_diff)/2.)
        

                # --- WRITING ----------------------------------------------------
                h_t0_diff.Write()
                h2_pes.Write()
                h2_t0.Write()
        
        out_root_file.cd()
        if event_type == "led":
            for pde in pdes:
                if g_dt0_led[pde].GetN() > 0:
                    g_dt0_led[pde].Write()

        out_root_file.Close()

    out_df = pd.DataFrame(out_df_rows)
    out_df = out_df.sort_values(by="OfflineComChannel")
    out_csv_name = out_folder + event_type+"_channel_time_offsets.csv"
    out_df.to_csv(out_csv_name, index=False)
