# The macro reads .txt files with the FFT of a given channels and estimates the FFT at a vgain
# required by the user. The files to read are the ones with "channel" in the name right before
# the .txt extension. The vgain used in the file follows the pattern "vgain_XXX" where XXX is
# the vgain value. The FFTs are stored in a numpy array and the vgain values are stored in a list.
# The estimated FFT comes from an interpolation of the FFTs of the channels.

# read the txt files and store the FFTs in numpy arrays
# the txts have only a column
# code here

# files = all the files in the directory /eos/home-f/fegalizz/ProtoDUNE_HD/Noise_Studies/analysis/FFT_txt/
# code to create files list here
import os
import sys
import yaml
import numpy as np
import pandas as pd
import noisy_function as nf


# --- MAIN --------------------------------------------------------------------
if __name__ == "__main__":
    # --- SETUP ---------------------------------------------------------------
    # Setup variables according to the noise_run_info.yaml file ---------------
    with open("./configs/noise_run_info.yml", 'r') as stream:
        run_info = yaml.safe_load(stream)

    fft_folder = run_info.get("fft_folder")
    fft_set_folder = run_info.get("fft_set_folder")
    golden_ch_dict = run_info.get("golden_ch_dict", {})

    # Setup variables according to the user_config.yaml file ------------------
    with open("params.yml", 'r') as stream:
        user_config = yaml.safe_load(stream)

    daphneAFE_vgain_dict = user_config.get("daphneAFE_vgain_dict", {})
    ana_path  = user_config.get("ana_path")
    integrators = user_config.get("integrators")
    if integrators == "OFF":
        channel_map_file = run_info.get("new_channel_map_file")
    elif integrators == "ON":
        channel_map_file = run_info.get("channel_map_file")
    else:
        print("Integrators must be either ON or OFF")
        exit()


    # Read the channel map file (daphne ch <-> offline ch) --------------------
    df = pd.read_csv("configs/"+channel_map_file, sep=",")
    daphne_channels = df['daphne_ch'].values + 100*df['endpoint'].values
    daphne_to_offline_dict = dict(zip(daphne_channels, df['offline_ch']))
    offline_to_daphne_dict = dict(zip(df['offline_ch'], daphne_channels))
    offline_to_sipm_dict = dict(zip(df['offline_ch'], df['sipm']))
    del df

    # Read the results file containing the RMS of the channels ----------------
    df = pd.read_csv(ana_path+"Noise_Studies_Results.csv")


    # Store all the filenames in a list ---------------------------------------
    input_path = ana_path + fft_folder + "/"
    files = [input_path+f for f in os.listdir(input_path) if f.endswith(".txt") and integrators in f]
    if len(files) == 0:
        print("No files found")
        exit()

    # Create set of offline channels ------------------------------------------
    offline_channels = set()
    for file in files:
        if integrators in file:
            offline_channels.add(int(file.split("OfflineCh_")[-1].split(".")[0]))
    print("Found ", len(offline_channels), " offline channels", " and ", len(files), " files")

    # Create output directories and variables ---------------------------------
    out_path = ana_path + fft_set_folder + "/"
    os.makedirs(out_path, exist_ok=True)
    out_ffts_folder = out_path + "FFTs_Config/"
    os.makedirs(out_ffts_folder, exist_ok=True)
    out_df_rows = []
    # out_df = pd.DataFrame(columns=["OfflineCh", "Integrators", "VGain", "RMS"])

    # --- LOOP OVER OFFLINE CHANNELS ------------------------------------------
    missing_channels = []
    missing_fft = []
    for offline_ch in range(160):
        print("Offline channel: ", offline_ch, "/", 159)
        sys.stdout.flush()
        # Prepare variables ---------------------------------------------------
        sipm = str(offline_to_sipm_dict[offline_ch])
        daphne_ch = offline_to_daphne_dict[offline_ch]
        desired_vgain = daphneAFE_vgain_dict[daphne_ch//10]
        expected_rms = np.float64(0.)
        out_fft_file = (out_ffts_folder+"FFT_PDHD_Noise_Template"
                        +"_SiPM_"+str(sipm)
                        +"_Integrators_"+integrators
                        +"_VGain_"+str(desired_vgain)
                        +"_DaphneCh_"+str(daphne_ch)
                        +"_OfflineCh_"+str(offline_ch)+".txt")
        df_ch = df[(df["OfflineCh"] == offline_ch) & (df["Integrators"] == integrators)]
        df_ch = df_ch.sort_values(by=["VGain"])

        if offline_ch not in offline_channels:
            print("Using golden channel: ", golden_ch_dict[sipm])
            missing_channels.append(offline_ch)
            golden_fft = nf.create_golden_fft(golden_ch_dict[sipm], desired_vgain, files)
            df_ch = df[(df["OfflineCh"] == golden_ch_dict[sipm]) & (df["Integrators"] == integrators)]
            df_ch = df_ch.sort_values(by=["VGain"])
            expected_rms = np.interp(desired_vgain, np.array(df_ch["VGain"]), np.array(df_ch["RMS"]))
            out_df_rows.append({"OfflineCh": offline_ch, "Integrators": integrators,
                                              "VGain": desired_vgain, "RMS": expected_rms})
            np.savetxt(out_fft_file, golden_fft)
            continue


        vgains = []
        ffts = []
        found = False
        search_string = f"_DaphneCh_{daphne_ch}_OfflineCh_{offline_ch}" 
        for file in files:
            if search_string in file:
                vgain = int(file.split("VGain_")[-1].split("_")[0])
                
                if vgain == desired_vgain:
                    # Average the RMS of this channel with this vgain in df ---
                    expected_rms = np.average(np.float64(df_ch[df_ch["VGain"] == vgain]["RMS"]))
                    out_df_rows.append({"OfflineCh": offline_ch, "Integrators": integrators,
                                                      "VGain": vgain, "RMS": expected_rms})
                    fft = np.loadtxt(file)
                    np.savetxt(out_fft_file, fft)
                    print("Found the desired vgain: ", vgain, "in file: ", file)
                    found = True
                    break

                else:
                    vgains.append(vgain)
                    fft = np.loadtxt(file)
                    ffts.append(fft)
        
        if found:
            continue

        if len(vgains) <= 1 or desired_vgain > max(vgains) or desired_vgain < min(vgains):
            missing_channels.append(offline_ch)
            golden_fft = nf.create_golden_fft(golden_ch_dict[sipm], desired_vgain, files)
            print("Using golden channel: ", golden_ch_dict[sipm])
            df_ch = df[(df["OfflineCh"] == golden_ch_dict[sipm]) & (df["Integrators"] == integrators)]
            df_ch = df_ch.sort_values(by=["VGain"])
            expected_rms = np.interp(desired_vgain, df_ch["VGain"], df_ch["RMS"])
            out_df_rows.append({"OfflineCh": offline_ch, "Integrators": integrators,
                                              "VGain": desired_vgain, "RMS": expected_rms})
            np.savetxt(out_fft_file, golden_fft)
            missing_fft.append(daphne_ch)
            continue

        # sort the vgains and the ffts
        vgains, ffts = zip(*sorted(zip(vgains, ffts), key=lambda x: x[0]))
        vgains = np.array(vgains)
        ffts = np.array(ffts)
        vgain_fft_dict = dict(zip(vgains, ffts))

        # estimate the FFT at desired_vgain
        estimated_fft = np.zeros(ffts.shape[1])

        for i in range(ffts.shape[1]):
            estimated_fft[i] = np.interp(desired_vgain, vgains, ffts[:,i])

        print("Saving the estimated FFT at vgain: ", desired_vgain)
        expected_rms = np.interp(desired_vgain, df_ch["VGain"], df_ch["RMS"])
        out_df_rows.append({"OfflineCh": offline_ch, "Integrators": integrators,
                                          "VGain": desired_vgain, "RMS": expected_rms})
        np.savetxt(out_fft_file, estimated_fft)

    out_df = pd.DataFrame(out_df_rows)
    out_df.to_csv(out_path+"OfflineCh_RMS_Config.csv", index=False)
    print("\n\n-----------------------------------------------------------------")
    print("Missing channels are saved according to the golden channels (HPK/FBK)")
    print("Missing channels: ", missing_channels)
    print("Missing FFTs: ", missing_fft)
    print("-----------------------------------------------------------------")
