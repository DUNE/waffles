from imports import *

# --- MAIN ----------------------------------------------------------
if __name__ == "__main__":
    # --- SETUP -----------------------------------------------------
    # Setup variables according to the user_config.yaml file
    with open("user_setting.yaml", 'r') as stream:
        user_config = yaml.safe_load(stream)

    path  = user_config.get("out_path")
    files = [path+"FFT_txt/"+f for f in os.listdir(path+"FFT_txt/") if f.endswith(".txt")]
    if len(files) == 0:
        print("No FFT files found in the directory")
        exit()

    # Setup variables according to the noise_run_info.yaml file
    with open("./noise_run_info.yaml", 'r') as stream:
        run_info = yaml.safe_load(stream)

    # filepath_folder  = run_info.get("filepath_folder")
    # run_vgain_dict   = run_info.get("run_vgain_dict", {})
    channel_map_file = run_info.get("channel_map_file")
    
    # Read the channel map file (daphne ch <-> offline ch)
    df = pd.read_csv(channel_map_file, sep=",")
    daphne_channels = df['daphne_ch'].values + 100*df['endpoint'].values
    daphne_to_offline_dict = dict(zip(daphne_channels, df['offline_ch']))
    offline_to_daphne_dict = dict(zip(df['offline_ch'], daphne_channels))

    with open("./channel_vgain.yaml", 'r') as stream:
        channel_vgain = yaml.safe_load(stream)
    daphneAFE_vgain_dict = channel_vgain.get("daphneAFE_vgain_dict", {})


    # --- LOOP OVER OFFLINE CHANNELS --------------------------------
    missing_channels = []
    missing_fft = []
    for offline_ch, daphne_ch in offline_to_daphne_dict.items():
        if daphne_ch not in daphneAFE_vgain_dict:
            missing_channels.append(daphne_ch)
            continue

        vgains = []
        ffts = []
        desired_vgain = daphneAFE_vgain_dict[daphne_ch//10]

        out_path = path+"Config_FFTs/"
        out_fft_file = (out_path+"FFT_Noise_Template_PDHD_daphneAFE_"+str(daphne_ch)
                        +"_OfflineCH_"+str(offline_ch)
                        +"_VGain_"+str(desired_vgain)+".txt")

        for file in files:
            if f"ch_{daphne_ch}" in file:
                vgain = int(file.split("vgain_")[-1].split("_")[0])
                
                if vgain == desired_vgain:
                    fft = np.loadtxt(file)
                    np.savetxt(out_fft_file, fft)
                    continue

                else:
                    vgains.append(vgain)
                    fft = np.loadtxt(file)
                    ffts.append(fft)

        if len(vgains) == 0:
            missing_fft.append(daphne_ch)
            continue

        # sort the vgains and the ffts
        vgains, ffts = zip(*sorted(zip(vgains, ffts)))
        vgains = np.array(vgains)
        ffts = np.array(ffts)
        vgain_fft_dict = dict(zip(vgains, ffts))

        # estimate the FFT at vgain = 1700
        estimated_fft = np.zeros(ffts.shape[1])

        for i in range(ffts.shape[1]):
            estimated_fft[i] = np.interp(desired_vgain, vgains, ffts[:,i])

        np.savetxt(out_fft_file, estimated_fft)

    print("Missing channels: ", missing_channels)
    print("Missing FFTs: ", missing_fft)
