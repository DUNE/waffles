# -------- Plot a function in an APA grid --------

import waffles.plotting.drawing_tools as draw
import numpy as np
from waffles.data_classes.Waveform import Waveform


#wset = draw.read("../data/wfset_10_30201.hdf5",0,1)
wset = draw.read("../data/wfset_chs_5_30201.hdf5",0,1)

'''

# -------- Time offset histograms --------
draw.plot_function_grid(wset, apa=2,tmin=-1000,tmax=1000,  plot_function=draw.plot_to_function)

input()
def func(wf: Waveform): return wf._Waveform__timestamp - wf._Waveform__daq_window_timestamp;
draw.plot_grid_histogram(wset,func,apa=2,tmin=-1000,tmax=1000)


# ----------- Sigma vs TS plots -----------
draw.plot_function_grid(wset, apa=2,tmin=-1000,tmax=1000,  plot_function=draw.plot_sigma_vs_ts_function)
input()

# ----------- Sigma  -----------
draw.plot_function_grid(wset, apa=2,tmin=-1000,tmax=1000,  plot_function=draw.plot_sigma_function)
input()

# ------------ Mean FFT  ------------
draw.plot_function_grid(wset, apa=2, plot_function=draw.plot_meanfft_function)



input()
def func(wf: Waveform): return np.std(wf.adcs);
draw.plot_grid_histogram(wset,func,apa=2,tmin=-1000,tmax=1000)


# ----------- Amplitude  -----------

input()
def func(wf: Waveform): return min(wf.adcs);
draw.plot_grid_histogram(wset,func,apa=2,tmin=-1000,tmax=1000,xmin=1,xmax=10000)


# ----------- number of bins with 0 adcs  -----------

input()
def func(wf: Waveform):
    nbins0=0
    for adc in wf.adcs:
        if adc==0:
            nbins0 = nbins0 + 1
    return nbins0

draw.plot_grid_histogram(wset,func,apa=1,tmin=-1000,tmax=1000,xmin=0,xmax=30)


# ----------- number of bins with 0 histograms -----------

input()
def func(wf: Waveform):
    nbins0=0
    for adc in wf.adcs:
        if adc==0:
            nbins0 = nbins0 + 1
    return nbins0

draw.plot_grid_histogram(wset,func,apa=1,tmin=-1000,tmax=1000,xmin=0,xmax=30)

'''