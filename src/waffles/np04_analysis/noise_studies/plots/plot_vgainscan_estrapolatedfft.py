import os
import numpy as np
import matplotlib.pyplot as plt

# --- HARD CODED ----------------------------------------------------
path = "/eos/home-f/fegalizz/ProtoDUNE_HD/Noise_Studies/analysis/FFT_txt/"
channel = 99
estrapolated_vgain = 2190
time_window = 1024 # ticks * ns/tick

# --- MAIN ----------------------------------------------------------
if __name__ == "__main__":
    files = [path+f for f in os.listdir(path) if f.endswith(".txt")]

    frequencies = np.fft.fftfreq(time_window, d=16*1e-9*1e+6)[:time_window//2+1]
    frequencies[-1] = -frequencies[-1]

    vgains = []
    ffts = []
    for file in files:
        if f"offlinech_{channel}." in file:
            print(file)
            vgain = int(file.split("vgain_")[-1].split("_")[0])
            vgains.append(vgain)
            print(vgain)
            fft = np.loadtxt(file)
            ffts.append(fft)

    # sort the vgains and the ffts
    print(len(vgains), len(ffts))
    vgains, ffts = zip(*sorted(zip(vgains, ffts), key=lambda x: x[0]))
    vgains = np.array(vgains)
    ffts = np.array(ffts)
    vgain_fft_dict = dict(zip(vgains, ffts))

    # estimate the FFT at vgain = 1700
    estimated_fft = np.zeros(ffts.shape[1])
    for i in range(ffts.shape[1]):
        estimated_fft[i] = np.interp(estrapolated_vgain, vgains, ffts[:,i])

    # plot the FFTs labelling according the vgain

    plt.figure(figsize=(10,6), dpi=300)
    for vgain, fft in vgain_fft_dict.items():
        plt.plot(frequencies, fft, label=f"VGain {vgain}")

    plt.plot(frequencies, estimated_fft, label="Estimated "+str(estrapolated_vgain), linestyle="--")
    # plt.legend()
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Amplitude")
   
    plt.tight_layout()
    # save the plot in the same directory
    # plt.savefig(path+f"channel_{channel}_vgain_{estrapolated_vgain}.png")

    plt.show()
