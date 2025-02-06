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
import numpy as np
import matplotlib.pyplot as plt

path = "/eos/home-f/fegalizz/ProtoDUNE_HD/Noise_Studies/analysis/FFT_txt/"
files = [path+f for f in os.listdir(path) if f.endswith(".txt")]

channel = 99

vgains = []
ffts = []
for file in files:
    if f"ch_{channel}" in file:
        print(file)
        vgain = int(file.split("vgain_")[-1].split("_")[0])
        vgains.append(vgain)
        print(vgain)
        fft = np.loadtxt(file)
        ffts.append(fft)

# sort the vgains and the ffts
vgains, ffts = zip(*sorted(zip(vgains, ffts)))
vgains = np.array(vgains)
ffts = np.array(ffts)
vgain_fft_dict = dict(zip(vgains, ffts))

# estimate the FFT at vgain = 1700
estimated_fft = np.zeros(ffts.shape[1])
for i in range(ffts.shape[1]):
    estimated_fft[i] = np.interp(2500, vgains, ffts[:,i])

# plot the FFTs labelling according the vgain

plt.figure()
for vgain, fft in vgain_fft_dict.items():
    plt.plot(fft, label=f"vgain {vgain}")

plt.plot(estimated_fft, label="estimated 2500")
plt.legend()
plt.xscale("log")
plt.yscale("log")

plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")

plt.show()
