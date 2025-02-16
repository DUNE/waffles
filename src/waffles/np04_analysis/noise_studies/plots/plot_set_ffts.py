import os
import numpy as np
import matplotlib.pyplot as plt

# --- HARD CODED ----------------------------------------------------
path = "/eos/home-f/fegalizz/ProtoDUNE_HD/Noise_Studies/analysis/Config_FFTs/"
time_window = 1024 # ticks * ns/tick

# --- MAIN ----------------------------------------------------------
if __name__ == "__main__":
    files = [path+f for f in os.listdir(path) if f.endswith(".txt")]

    frequencies = np.fft.fftfreq(time_window, d=16*1e-9*1e+6)[:time_window//2+1]
    frequencies[-1] = -frequencies[-1]

    dict_channels_vgains = {}
    channels = []
    ffts = []
    for file in files:
        print(file)
        channel = int(file.split("OfflineCH_")[-1].split("_")[0])
        vgain   = int(file.split("VGain_")[-1].split(".")[0])
        dict_channels_vgains[channel] = vgain
        channels.append(channel)

        fft = np.loadtxt(file)
        ffts.append(fft)

    # sort the vgains and the ffts
    channels, ffts = zip(*sorted(zip(channels, ffts), key=lambda x: x[0]))
    channels = np.array(channels)
    ffts = np.array(ffts)
    channel_fft_dict = dict(zip(channels, ffts))


    # plot the FFTs labelling according the channel
    plt.figure(figsize=(10,6), dpi=300)
    nplots = 0
    for channel, fft, in channel_fft_dict.items():
        nplots += 1
        if (nplots<10):
            plt.plot(frequencies, fft, label=f"Channel {channel} - VGain {dict_channels_vgains[channel]}")

    # plt.legend()
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Amplitude")
   
    plt.tight_layout()
    # save the plot in the same directory
    plt.savefig(path+f"VGain_{dict_channels_vgains[channels[0]]}.png")

    plt.show()
