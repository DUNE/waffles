# --- IMPORTS -------------------------------------------------------
import waffles
import os
import yaml
import numpy as np
import pandas as pd
import noisy_function as nf

# --- MAIN ----------------------------------------------------------
if __name__ == "__main__":

    # Setup variables according to the noise_run_info.yaml file
    with open("./configs/noise_run_info.yml", 'r') as stream:
        run_info = yaml.safe_load(stream)

    filepath_folder  = run_info.get("filepath_folder")
    run_vgain_dict   = run_info.get("run_vgain_dict", {})
    channel_map_file = run_info.get("channel_map_file")
    new_channel_map_file = run_info.get("new_channel_map_file")
    all_noise_runs   = list(run_vgain_dict.keys())
    integratorsON_runs = run_info.get("integratorsON_runs", [])

    # Setup variables according to the user_config.yaml file
    with open("params.yml", 'r') as stream:
        user_config = yaml.safe_load(stream)
    
    debug_mode = user_config.get("debug_mode")
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
    df = pd.read_csv(channel_map_file, sep=",")
    daphne_channels = df['daphne_ch'].values + 100*df['endpoint'].values
    daphne_to_offline = dict(zip(daphne_channels, df['offline_ch']))
    df = pd.read_csv(new_channel_map_file, sep=",")
    new_daphne_channels = df['daphne_ch'].values + 100*df['endpoint'].values
    new_daphne_to_offline = dict(zip(new_daphne_channels, df['offline_ch']))


    # File where the results will be printed (run, vgain, endpoint, channel, offline_channel, rms)
    my_csv_file = open("output/Noise_Studies_Results.csv", out_writing_mode)


    # --- LOOP OVER RUNS ----------------------------------------------
    for run in runs:
        print("Reading run: ", run)
        wfset_run = nf.read_waveformset(filepath_folder,
                                        run,
                                        full_stat=full_stat)

        endpoints = wfset_run.get_set_of_endpoints()
        
        integrator_ON = False
        if run in integratorsON_runs:
            integrator_ON = True
            daphne_to_offline_dict = daphne_to_offline
        else:
            daphne_to_offline_dict = new_daphne_to_offline


        # --- LOOP OVER ENDPOINTS -------------------------------------
        for ep in endpoints:
            print("Endpoint: ", ep)
            wfset_ep = waffles.WaveformSet.from_filtered_WaveformSet(wfset_run, nf.allow_ep_wfs, ep)

            ep_ch_dict = wfset_ep.get_run_collapsed_available_channels()
            channels = list(ep_ch_dict[ep])

            # --- LOOP OVER CHANNELS ----------------------------------
            for ch in channels:
                print("Channel: ", ch)
                wfset_ch = waffles.WaveformSet.from_filtered_WaveformSet(wfset_ep, nf.allow_channel_wfs, ch)
                # check if the channel is in the daphne_to_offline dictionary
                channel = np.uint16(np.uint16(ep)*100+np.uint16(ch))
                if channel not in daphne_to_offline_dict:
                    print(f"Channel {channel} not in the daphne_to_offline dictionary")
                    continue
                offline_ch = daphne_to_offline_dict[channel]
        
                wfs = wfset_ch.waveforms
                del wfset_ch
                nf.create_float_waveforms(wfs)
                nf.sub_baseline_to_wfs(wfs, 1024)

                norm = 1./len(wfs)
                fft2_avg = np.zeros(1024)
                rms = 0.

                # Compute the average FFT of the wfs.adcs_float
                for wf in wfs:
                    rms += np.std(wf.adcs_float)
                    fft  = np.fft.fft(wf.adcs_float)
                    fft2 = np.abs(fft)
                    fft2_avg += fft2

                fft2_avg = fft2_avg*norm
                rms = rms*norm
                vgain = run_vgain_dict[run]
                
                # print run, vgain, ep, ch, offline_ch, rms in a csv file
                my_csv_file.write(f"{run},{vgain},{ep},{ch},{offline_ch},{rms}\n")

                # Check wheter the folder FFT_txt exists in the "output" folder
                if not os.path.exists("output/FFT_txt"):
                    os.makedirs("output/FFT_txt")

                # print the FFT in a txt file
                print("Writing FFT to txt file")
                integrator = "OFF"
                if integrator_ON:
                    integrator = "ON"
                np.savetxt("output/FFT_txt/fft_run_"+str(run)
                           +"_int_"+integrator
                           +"_vgain_"+str(vgain)
                           +"_ch_"+str(channel)
                           +"_offlinech_"+str(offline_ch)+".txt", fft2_avg[0:513])

               
                if debug_mode:
                    nf.plot_heatmaps(wfs, "raw", run, vgain, channel, offline_ch)
                    nf.plot_heatmaps(wfs, "baseline_removed", run, vgain, channel, offline_ch)
                    print("done")

            del wfset_ep
