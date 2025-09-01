# --- IMPORTS -------------------------------------------------------
from pandas._libs.hashtable import mode
import waffles
import waffles.Exceptions as exceptions
import os
import yaml
import numpy as np
import pandas as pd
import noisy_function as nf

# --- MAIN ----------------------------------------------------------
if __name__ == "__main__":
    print("Imports done")
    # --- SETUP -----------------------------------------------------
    # Setup variables according to the noise_run_info.yaml file
    with open("./configs/noise_run_info.yml", 'r') as stream:
        run_info = yaml.safe_load(stream)

    filepath_folder  = run_info.get("filepath_folder")
    fft_folder = run_info.get("fft_folder")
    run_vgain_dict   = run_info.get("run_vgain_dict", {})
    channel_map_file = run_info.get("channel_map_file")
    new_channel_map_file = run_info.get("new_channel_map_file")
    all_noise_runs   = list(run_vgain_dict.keys())
    integratorsON_runs = run_info.get("integratorsON_runs", [])
    fullstreaming_runs = run_info.get("fullstreaming_runs", []) 
    ignore_ch_dict = run_info.get("ignore_ch_dict", {})

    # Setup variables according to the user_config.yaml file
    with open("params.yml", 'r') as stream:
        user_config = yaml.safe_load(stream)
    
    custom_filepath_folder = user_config.get("custom_filepath_folder")
    if (custom_filepath_folder != ""):
        filepath_folder = custom_filepath_folder
    debug_mode = user_config.get("debug_mode")
    ana_path = user_config.get("ana_path")
    out_writing_mode = user_config.get("out_writing_mode")
    full_stat = user_config.get("full_stat")
    runs      = user_config.get("user_runs", [])
    if (len(runs) == 0):
        print("Analyzing all noise runs")
        runs = all_noise_runs
        print("All noise runs: ", runs)
        if (len(runs) == 0):
            print("No runs to analyze")
            exit()

    # Read the channel map file (daphne ch <-> offline ch)
    df = pd.read_csv("configs/"+channel_map_file, sep=",")
    daphne_channels = df['daphne_ch'].values + 100*df['endpoint'].values
    daphne_to_offline = dict(zip(daphne_channels, df['offline_ch']))
    offline_to_sipm_dict = dict(zip(df['offline_ch'], df['sipm']))
    df = pd.read_csv("configs/"+new_channel_map_file, sep=",")
    new_daphne_channels = df['daphne_ch'].values + 100*df['endpoint'].values
    new_daphne_to_offline = dict(zip(new_daphne_channels, df['offline_ch']))


    # Prepare the output directory and the output dataframe
    out_df_rows = []
    os.makedirs(ana_path+fft_folder, exist_ok=True)


    # --- LOOP OVER RUNS ----------------------------------------------
    for run in runs:
        print("Reading run: ", run)
        try:
            wfset_run = nf.read_waveformset(filepath_folder,
                                            run,
                                            full_stat=full_stat)

            if (run in fullstreaming_runs):
                wfset_fullstreaming = nf.read_waveformset(filepath_folder,
                                                          run,
                                                          full_stat=full_stat,
                                                          fullstreaming = True)
        except FileNotFoundError:
            print(f"File for run {run} not found")
            continue
        except exceptions.WafflesBaseException:
            print(f"Error reading file for run {run}")
            continue

        endpoints = wfset_run.get_set_of_endpoints()
        fullstraming_endpoints = []
        if (run in fullstreaming_runs):
            fullstraming_endpoints = list(wfset_fullstreaming.get_set_of_endpoints())
            endpoints = list(endpoints) + list(fullstraming_endpoints)
        
        integrator_ON = False
        if run in integratorsON_runs:
            integrator_ON = True
            daphne_to_offline_dict = daphne_to_offline
        else:
            daphne_to_offline_dict = new_daphne_to_offline

        print(endpoints)
        # exit()


        # --- LOOP OVER ENDPOINTS -------------------------------------
        for ep in endpoints:
            print("Endpoint: ", ep)
            if (ep in fullstraming_endpoints):
                wfset_ep = waffles.WaveformSet.from_filtered_WaveformSet(wfset_fullstreaming, nf.allow_ep_wfs, ep)
            else:
                wfset_ep = waffles.WaveformSet.from_filtered_WaveformSet(wfset_run, nf.allow_ep_wfs, ep)

            ep_ch_dict = wfset_ep.get_run_collapsed_available_channels()
            channels = list(ep_ch_dict[ep])

            # --- LOOP OVER CHANNELS ----------------------------------
            for ch in channels:
                print("Channel: ", ch)
                wfset_ch = waffles.WaveformSet.from_filtered_WaveformSet(wfset_ep, nf.allow_channel_wfs, ch)
                # check if the channel is in the daphne_to_offline dictionary
                channel = np.uint16(np.uint16(ep)*100+np.uint16(ch))
                vgain = run_vgain_dict[run]
    
                if run in ignore_ch_dict:
                    if channel in ignore_ch_dict[run]:
                        print(f"Ignoring channel {channel}")
                        continue

                if channel not in daphne_to_offline_dict:
                    print(f"Channel {channel} not in the daphne_to_offline dictionary")
                    continue
                offline_ch = daphne_to_offline_dict[channel]
                sipm = str(offline_to_sipm_dict[offline_ch])
                
                if debug_mode:
                    nf.plot_heatmaps(wfset_ch, "raw", run, vgain, channel, offline_ch)
                    print("done")

                nf.create_float_waveforms(wfset_ch)
                rms = nf.get_average_rms(wfset_ch)
                wfset_ch = waffles.WaveformSet.from_filtered_WaveformSet(wfset_ch, nf.noise_wf_selection, rms)
                nf.sub_baseline_to_wfs(wfset_ch, 1024)

                norm = 1./len(wfset_ch.waveforms)
                fft2_avg = np.zeros(1024)
                rms = 0.

                # Compute the average FFT of the wfs.adcs_float
                for wf in wfset_ch.waveforms:
                    rms += np.std(wf.adcs_float)
                    fft  = np.fft.fft(wf.adcs_float)
                    fft2 = np.abs(fft)
                    fft2_avg += fft2

                fft2_avg = fft2_avg*norm
                rms = rms*norm
                
                # print run, vgain, ep, ch, offline_ch, rms in a csv file
                integrators = "OFF"
                if integrator_ON:
                    integrators = "ON"
                out_df_rows.append({"Run": run, "SiPM": sipm, "Integrators": integrators,
                                    "VGain": vgain, "DaphneCh": channel, "OfflineCh": offline_ch,
                                    "RMS": rms})


                # print the FFT in a txt file
                print("Writing FFT to txt file")
                integrators = "OFF"
                if integrator_ON:
                    integrators = "ON"
                np.savetxt(ana_path+fft_folder+"/FFT_PDHD_Noise"
                           +"_Run_"+str(run)
                           +"_SiPM_"+str(sipm)
                           +"_Integrators_"+integrators
                           +"_VGain_"+str(vgain)
                           +"_DaphneCh_"+str(channel)
                           +"_OfflineCh_"+str(offline_ch)+".txt", fft2_avg[0:513])

               
                if debug_mode:
                    nf.plot_heatmaps(wfset_ch, "baseline_removed", run, vgain, channel, offline_ch)
                    print("done")

                del wfset_ch

            del wfset_ep

        del wfset_run
    
    # Save the results in a csv file
    out_df = pd.DataFrame(out_df_rows)
    out_df.to_csv(ana_path+"Noise_Studies_Results.csv", index=False, mode=out_writing_mode)
