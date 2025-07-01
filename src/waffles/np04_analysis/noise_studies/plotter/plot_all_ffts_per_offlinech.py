import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- HARD CODED ----------------------------------------------------
input_path = "/eos/home-f/fegalizz/ProtoDUNE_HD/Noise_Studies/analysis/FFTs_txt/"
output_path = "/eos/home-f/fegalizz/ProtoDUNE_HD/Noise_Studies/analysis/all_ffts_per_offlinech/"
estrapolated_vgain = 2190
time_window = 1024 # ticks * ns/tick

# --- MAIN ----------------------------------------------------------
if __name__ == "__main__":
    # Create the frequencies array to plot x-axis
    frequencies = np.fft.fftfreq(time_window, d=16*1e-9*1e+6)[:time_window//2+1]
    frequencies[-1] = -frequencies[-1]
        
    # Store all the filenames in a list
    files = [input_path+f for f in os.listdir(input_path) if f.endswith(".txt")]
    if len(files) == 0:
        print("No files found")
        exit()

    # Create set of offline channels
    offline_channels = set()
    for file in files:
        offline_channels.add(int(file.split("OfflineCh_")[-1].split(".")[0]))

    print(len(offline_channels), offline_channels)
    if len(offline_channels) < 160:
        print("Not all channels are present")
        exit()
    
    os.makedirs(output_path, exist_ok=True)

    for offline_channel in offline_channels:
        vgains = []
        ffts = []
        runs = []
        daphne_channels = []

        for file in files:
            if f"OfflineCh_{offline_channel}." in file:
                print(file)
                vgain = int(file.split("VGain_")[-1].split("_")[0])
                vgains.append(vgain)
                fft = np.loadtxt(file)
                ffts.append(fft)
                runs.append(int(file.split("Run_")[-1].split("_")[0]))
                daphne_channels.append(int(file.split("_DaphneCh_")[-1].split("_")[0]))

        
        if len(vgains) == 0:
            print(f"No files found for offline channel {offline_channel}")
            continue

        # sort the vgains and the ffts
        print(len(vgains), len(ffts))
        # vgains, ffts = zip(*sorted(zip(vgains, ffts), key=lambda x: x[0]))
        vgains = np.array(vgains)
        ffts = np.array(ffts)
        vgain_fft_dict = dict(zip(vgains, ffts))

        # estimate the FFT at vgain = 1700
        # estimated_fft = np.zeros(ffts.shape[1])
        # for i in range(ffts.shape[1]):
        #     estimated_fft[i] = np.interp(estrapolated_vgain, vgains, ffts[:,i])

        # plot the FFTs labelling according the vgain

        plt.figure(figsize=(10,6), dpi=300)
        for vgain, fft, run, daphne_channel in zip(vgains, ffts, runs, daphne_channels):
            plt.plot(frequencies, fft, label=f"VGain {vgain} - DaphneCh {daphne_channel} -  Run {run}")  

        # plt.plot(frequencies, estimated_fft, label="Estimated "+str(estrapolated_vgain), linestyle="--")
        # plt.legend()
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.xscale("log")
        plt.yscale("log")

        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Amplitude")
   
        plt.tight_layout()
        # save the plot in the same directory
        plt.savefig(output_path+f"offlinech_{offline_channel}.png")

        # plt.show()
        plt.close()
