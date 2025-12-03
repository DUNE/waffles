# --- IMPORTS -------------------------------------------------------
from pandas._libs.hashtable import mode
import waffles
import waffles.Exceptions as exceptions
import os
import yaml
import numpy as np
import pandas as pd
import waffles.np04_analysis.noise_studies.noisy_function as nf
from waffles.np04_utils.utils import get_np04_channel_mapping
from waffles.coldboxVD.utils.utils import get_cb25_channel_mapping
from waffles.coldboxVD.utils.spybuffer_reader import create_waveform_set_from_spybuffer


# --- MAIN ----------------------------------------------------------
if __name__ == "__main__":
    print("Imports done")
    # --- SETUP -----------------------------------------------------
    # Setup variables according to the noise_run_info.yaml file
    with open("./configs/noise_run_info.yml", 'r') as stream:
        run_info = yaml.safe_load(stream)

    filepath_folder  = run_info.get("filepath_folder")
    fft_folder = run_info.get("fft_folder")
    # ignore_ch_dict = run_info.get("ignore_ch_dict", {})

    # Setup variables according to the user_config.yaml file
    with open("params.yml", 'r') as stream:
        user_config = yaml.safe_load(stream)
    
    memorydepth = user_config.get("memorydepth")
    custom_filepath_folder = user_config.get("custom_filepath_folder")
    if (custom_filepath_folder != ""):
        filepath_folder = custom_filepath_folder
    debug_mode = user_config.get("debug_mode")
    ana_path = user_config.get("ana_path")
    out_writing_mode = user_config.get("out_writing_mode")
    vgains      = user_config.get("user_vgains", [])
    if (len(vgains) == 0):
        print("Analyzing all noise runs")
        # list all the directories in filepath_folder
        vgain_dirs = [d for d in os.listdir(filepath_folder) if os.path.isdir(os.path.join(filepath_folder, d))]
        vgains = [int(d.replace("vgain_", "")) for d in vgain_dirs if d.startswith("vgain_")]
        vgain_dirs = [filepath_folder+d for d in os.listdir(filepath_folder) if os.path.isdir(os.path.join(filepath_folder, d))]
        if (len(vgains) == 0):
            print("No runs to analyze")
            exit()
    else:
        vgain_dirs = [filepath_folder+"vgain_"+str(vg) for vg in vgains]

    # Read the channel map file (daphne ch <-> offline ch)
    df = get_cb25_channel_mapping()
    configch_to_sipm = dict(zip(df['ConfigCh'], df['SiPM']))
    configch_to_module = dict(zip(df['ConfigCh'], df['Module']))


    # Prepare the output directory and the output dataframe
    out_df_rows = []
    os.makedirs(ana_path+fft_folder, exist_ok=True)


    # --- LOOP OVER RUNS ----------------------------------------------
    for vgain, vgain_dir in zip(vgains, vgain_dirs):
        ch_dir = vgain_dir+"/compressed_channels/"
        files_in_dir = os.listdir(ch_dir)
        channels = [int (f.replace("channel_", "").replace(".dat", "")) \
                    for f in files_in_dir if f.startswith("channel_") and f.endswith(".dat")]


        # --- LOOP OVER CHANNELS ----------------------------------
        for ch in channels:
            print("  Analyzing Channel: ", ch)
            wfset_ch = create_waveform_set_from_spybuffer(ch_dir+f"/channel_{ch}.dat",
                                                       WFs=-1,
                                                       length=memorydepth,
                                                       config_channel=ch)
            print("red")


            if debug_mode:
                nf.plot_heatmaps(wfset_ch, "raw", 0, vgain, int(ch), 0)
                print("done")

            nf.create_float_waveforms(wfset_ch)
            print("processing")
            rms = nf.get_average_rms(wfset_ch)
            nf.sub_baseline_to_wfs(wfset_ch, memorydepth)
            print("processing")

            norm = 1./len(wfset_ch.waveforms)
            fft2_avg = np.zeros(memorydepth)
            rms = 0.

            # Compute the average FFT of the wfs.adcs_float
            for wf in wfset_ch.waveforms:
                rms += np.std(wf.adcs_float)
                fft  = np.fft.fft(wf.adcs_float)
                fft2 = np.abs(fft)
                fft2_avg += fft2

            fft2_avg = fft2_avg*norm
            rms = rms*norm
           
            sipm = configch_to_sipm[ch]
            out_df_rows.append({"VGain": vgain,
                                "Module": configch_to_module[ch],
                                "ConfigCh": ch,
                                "SiPM": sipm,
                                "Integrators": "OFF",
                                "RMS": rms})
            print("Writing FFT to txt file")


            # print the FFT in a txt file
            np.savetxt(ana_path+fft_folder+"/FFT_CB25_Noise"
                       +"_VGain_"+str(vgain)
                       +"_ConfigCh_"+str(ch)
                       +"_SiPM_"+str(sipm)+".txt", fft2_avg[0:wfset_ch.waveforms[0].adcs_float.size//2+1])
           
            if debug_mode:
                nf.plot_heatmaps(wfset_ch, "baseline_removed", 0, vgain, int(ch), 0)
                print("done")

            del wfset_ch
    
    # Save the results in a csv file
    out_df = pd.DataFrame(out_df_rows)
    out_df.to_csv(ana_path+"Noise_Studies_Results.csv", index=False, mode=out_writing_mode)
