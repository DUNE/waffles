import importlib
import os
import csv

import sys
sys.path.append("/eos/home-f/fegalizz/ProtoDUNE_HD/waffles/src/waffles/np04_analysis/time_resolution")
import TimeResolution as tr

import numpy as np
import waffles
import pickle

######################################################################
############### HARDCODE #############################################
path = "/eos/home-f/fegalizz/ProtoDUNE_HD/TimeResolution/files/wfset_"
runs = [30030, 30031, 30032, 30033, 30035]
files = [path+str(run)+".pkl" for run in runs]

channel = [25, 27]
endpoint = 112

prepulse_ticks = 125
postpulse_ticks = 160
baseline_rms = 5

min_amplitudes = [100, 200, 300, 400, 500, 700, 900, 1100]
max_amplitudes = [amp+300 for amp in min_amplitudes]


######################################################################


for file, run in zip(files, runs):
    print("Reading run ", run)
    with open(f'{file}', 'rb') as f:
        wfset_run = pickle.load(f)

    a = tr.TimeResolution(wf_set=wfset_run, 
                          prepulse_ticks=prepulse_ticks,
                          postpulse_ticks=postpulse_ticks,
                          min_amplitude=min_amplitudes[0],
                          max_amplitude=max_amplitudes[0],
                          baseline_rms=baseline_rms)

    for channel in channels:
        print("Channel ", channel)
        a.ref_ep = endpoint
        a.ref_ch = channel
        a.create_wfs(tag="ref")
        out_file =  "Ch_"+str(channel)+"_results.csv"
   
        for min_amplitude, max_amplitude in zip(min_amplitudes, max_amplitudes):
            print("Setting min ", min_amplitude, " max ", max_amplitude)
            a.min_amplitude=min_amplitude
            a.max_amplitude=max_amplitude

            a.select_time_resolution_wfs(tag="ref")

            a.set_wfs_t0(tag="ref")
            
            if len(a.ref_n_selected_wfs) > 100:
                print("Save this ")
                file_exists = os.path.isfile(out_file)
                with open(out_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    
                #Write the header only if new file
                if not file_exists:
                    writer.writerow(['Run', 'min', 'max',
                                     'Ch ref', 't0 ref', 'std ref',
                                     ])

                writer.writerow([run, min_amplitude, max_amplitude,
                                 endpoint*100+ref_ch, a.ref_t0, a.ref_t0_std,
                                 ])

