import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- HARD CODED ----------------------------------------------------
input_path = "/eos/home-f/fegalizz/ProtoDUNE_HD/Noise_Studies/analysis/FFT_txt/"
output_path = "/eos/home-f/fegalizz/ProtoDUNE_HD/Noise_Studies/analysis/plots/"
channel = 90
estrapolated_vgain = 2190
time_window = 1024 # ticks * ns/tick

# --- MAIN ----------------------------------------------------------
if __name__ == "__main__":
    # Read the channel map file (daphne ch <-> offline ch)
    channel_map_file = "../PDHD_PDS_ChannelMap.csv"
    df = pd.read_csv(channel_map_file, sep=",")
    daphne_channels = df['daphne_ch'].values + 100*df['endpoint'].values
    offline_to_daphne = dict(zip(df['offline_ch'], daphne_channels))
    daphne_to_offline = dict(zip(daphne_channels, df['offline_ch'])) 
        
    files = [input_path+f for f in os.listdir(input_path) if f.endswith(".txt")]
    if len(files) == 0:
        print("No files found")
        exit()

    for key, channel in daphne_to_offline.items():    

        frequencies = np.fft.fftfreq(time_window, d=16*1e-9*1e+6)[:time_window//2+1]
        frequencies[-1] = -frequencies[-1]

        vgains = []
        ffts = []
        runs = []

        for file in files:
            if f"offlinech_{channel}." in file:
                print(file)
                vgain = int(file.split("vgain_")[-1].split("_")[0])
                vgains.append(vgain)
                fft = np.loadtxt(file)
                ffts.append(fft)
                runs.append(int(file.split("run_")[-1].split("_")[0]))

        
        if len(vgains) == 0:
            print(f"No files found for channel {channel}")
            continue

        # sort the vgains and the ffts
        print(len(vgains), len(ffts))
        # vgains, ffts = zip(*sorted(zip(vgains, ffts), key=lambda x: x[0]))
        vgains = np.array(vgains)
        ffts = np.array(ffts)
        vgain_fft_dict = dict(zip(vgains, ffts))

        # estimate the FFT at vgain = 1700
        estimated_fft = np.zeros(ffts.shape[1])
        for i in range(ffts.shape[1]):
            estimated_fft[i] = np.interp(estrapolated_vgain, vgains, ffts[:,i])

        # plot the FFTs labelling according the vgain

        plt.figure(figsize=(10,6), dpi=300)
        for vgain, fft, run in zip(vgains, ffts, runs):
            plt.plot(frequencies, fft, label=f"VGain {vgain} - Run {run}")  

        plt.plot(frequencies, estimated_fft, label="Estimated "+str(estrapolated_vgain), linestyle="--")
        # plt.legend()
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.xscale("log")
        plt.yscale("log")

        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Amplitude")
   
        plt.tight_layout()
        # save the plot in the same directory
        plt.savefig(output_path+f"daphnecg_{offline_to_daphne[channel]}_offlinech_{channel}_vgain_{estrapolated_vgain}.png")

        # plt.show()
