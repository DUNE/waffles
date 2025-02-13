from tools_analysis import *

import matplotlib.pyplot as plt

'''
import waffles.input.raw_root_reader as reader
wfset = reader.WaveformSet_from_root_file(
    '/afs/cern.ch/user/a/anbalbon/waffles/src/waffles/np04_analysis/channel_saturation/run_27898_0000_dataflow0_datawriter_0.root',                               # path to the root file
    'pyroot',                               # library to read (if ROOT, use 'pyroot', if not `uproot`)
    read_full_streaming_data = False,       # if False, read the self-triggered data
    truncate_wfs_to_minimum = False,        # truncate the waveforms to the minimum size
    start_fraction = 0.0,                   # starting fraction for reading
    stop_fraction = 1.0,                    # stoping fraction for reading
    subsample = 1,                          # subsample the data reading (read every other entry)
    verbose = True)
'''

with open('/afs/cern.ch/user/a/anbalbon/waffles/src/waffles/np04_analysis/channel_saturation/wfset.pkl', 'rb') as f:
    wfset= pickle.load(f)  


# Looking for waveform info
i=0
print(f'Info for waveform: {i}')
print(f"Timestamp = {wfset.waveforms[i].timestamp}")
print(f"DAQ timestamp = {wfset.waveforms[i].daq_window_timestamp}")
print(f"DAQ timestamp - timestamp = {wfset.waveforms[i].daq_window_timestamp - wfset.waveforms[i].timestamp}")
print(f"Time_offset = {wfset.waveforms[i].time_offset}")

# Selecting a given channel
#ch_wfset = wfset.from_filtered_WaveformSet(wfset, channel_filter, end = 109, ch = 35)

# Histogram of the time offest 
timeoffset_list_DAQ = []
for wf in wfset.waveforms: 
    timeoffset_list_DAQ.append(wf.timestamp - wf.daq_window_timestamp)
 
    
fig, ax = plt.subplots(1, 1, figsize=(12, 10))   
ax.hist(timeoffset_list_DAQ, bins=10000, color='blue', edgecolor='black')
#x.set_xlim(-500,500)
fig.savefig('/afs/cern.ch/user/a/anbalbon/waffles/src/waffles/np04_analysis/channel_saturation/imm_DAQ.jpg')
plt.close()