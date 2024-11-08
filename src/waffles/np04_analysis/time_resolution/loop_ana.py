import importlib
import sys
sys.path.append("/eos/home-f/fegalizz/ProtoDUNE_HD/waffles/src/waffles/np04_analysis/time_resolution")
import TimeResolution as tr

import numpy as np
import waffles
import pickle


path = "/eos/home-f/fegalizz/ProtoDUNE_HD/TimeResolution/files/wfset_"
runs = [30030, 30031, 30032, 30033, 30035]

files = [path+str(run)+".pkl" for run in runs]

prepulse_ticks = 125
postpulse_ticks = 160
baseline_rms = 5
endpoint = 112
com_ch = 25
ref_ch = 27

min_amplitudes = [100, 200, 300, 400, 500, 700, 900, 1100]
max_amplitudes = [amp+200 for amp in min_amplitudes]

for file in files:
    with open(f'{file}', 'rb') as f:
        wfset_run = pickle.load(f)

    a = tr.TimeResolution(wf_set=wfset_run, 
                          ref_ep=endpoint, ref_ch=ref_ch,
                          com_ep=endpoint, com_ch=com_ch,
                          prepulse_ticks=prepulse_ticks,
                          postpulse_ticks=postpulse_ticks,
                          min_amplitude=min_amplitudes[0],
                          max_amplitude=max_amplitudes[0],
                          baseline_rms=baseline_rms)

    a.create_wfs(tag="ref")
    a.create_wfs(tag="com")
   
    for min_amplitude, max_amplitude in zip(min_amplitudes, max_amplitudes)
        a.min_amplitude=min_amplitude
        a.max_amplitude=max_amplitude

        a.select_time_resolution_wfs(tag="ref")
        a.select_time_resolution_wfs(tag="com")

        a.set_wfs_t0(tag="ref")
        a.set_wfs_t0(tag="com")

        t0_diff = a.calculate_t0_differences()
        if len(t0_diff) > 100:

