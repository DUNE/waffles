import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- MAIN ----------------------------------------------------------
if __name__ == "__main__":
    # --- SETUP -----------------------------------------------------
    # Setup variables according to the noise_run_info.yaml file
    with open("./configs/noise_run_info.yml", 'r') as stream:
        run_info = yaml.safe_load(stream)

    filepath_folder  = run_info.get("filepath_folder")
    fft_folder = run_info.get("fft_folder")


    with open("params.yml", 'r') as stream:
        user_config = yaml.safe_load(stream)
    
    memorydepth = user_config.get("memorydepth")
    ana_path = user_config.get("ana_path")

    input_path = ana_path + fft_folder + "/"
    output_path = ana_path + "all_ffts_per_offlinech/"
    

    # Create the frequencies array to plot x-axis
    frequencies = np.fft.fftfreq(memorydepth, d=16*1e-9*1e+6)[:memorydepth//2+1]
    frequencies[-1] = -frequencies[-1]
        
    # Store all the filenames in a list
    files = [input_path+f for f in os.listdir(input_path) if f.endswith(".txt")]
    if len(files) == 0:
        print("No files found")
        exit()

    # Create set of offline channels
    channels = set()
    for file in files:
        channels.add(int(file.split("ConfigCh_")[-1].split("_SiPM")[0]))

    os.makedirs(output_path, exist_ok=True)

    for channel, i in zip(sorted(channels), range(len(channels))):
        print(f"Channel {channel} ({i+1}/{len(channels)})")
        vgains = []
        ffts = []

        for file in files:
            if f"ConfigCh_{channel}" in file:
                vgain = int(file.split("VGain_")[-1].split("_")[0])
                vgains.append(vgain)
                fft = np.loadtxt(file)
                ffts.append(fft)

        
        if len(vgains) == 0:
            print(f"No files found for channel {channel}")
            continue

        # sort the vgains and the ffts
        vgains, ffts = zip(*sorted(zip(vgains, ffts), key=lambda x: x[0]))
        vgains = np.array(vgains)
        ffts = np.array(ffts)
        vgain_fft_dict = dict(zip(vgains, ffts))

        # estimate the FFT at vgain = 1700
        # estimated_fft = np.zeros(ffts.shape[1])
        # for i in range(ffts.shape[1]):
        #     estimated_fft[i] = np.interp(estrapolated_vgain, vgains, ffts[:,i])

        # plot the FFTs labelling according the vgain

        plt.figure(figsize=(10,6), dpi=300)
        for vgain, fft in vgain_fft_dict.items():
            print(f"  Plotting VGain {vgain}")
            plt.plot(frequencies, fft, label=f"VGain {vgain} - Ch {channel}")  

            # plt.plot(frequencies, estimated_fft, label="Estimated "+str(estrapolated_vgain), linestyle="--")
            # plt.legend()
            plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
            plt.xscale("log")
            plt.yscale("log")

            plt.xlabel("Frequency [MHz]")
            plt.ylabel("Amplitude")
   
            plt.tight_layout()
            # save the plot in the same directory
            plt.savefig(output_path+f"ConfigCh_{channel}.png")

        # plt.show()
        plt.close()
