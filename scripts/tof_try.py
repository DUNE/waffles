import numpy as np
from waffles.input_output.hdf5_structured import load_structured_waveformset

# This fuction should extract tof and tof_timestamp into a dictionary 
def create_fake_tof_info(daq_timestamps_list):
    tof_timestamps = daq_timestamps_list - 60  # adding a fake 60 offset
    tofdict = {}
    for tofts in tof_timestamps:
        tofdict[tofts] = int(str(tofts)[-1])
    return tofdict


nwaforms = 4000
wfset = load_structured_waveformset("/afs/cern.ch/work/a/arochefe/private/repositories/waffles/scripts/processed_np02vd_raw_run039105_0000_df-s05-d0_dw_0_20250824T210850.hdf5.copied_structured_membrane.hdf5", max_to_load=nwaforms)

daq_timestamps = {}

# Gets the unique daq timestamps and cound how many
for wf in wfset.waveforms:
    daq_timestamps[wf.daq_window_timestamp] = daq_timestamps.get(wf.daq_window_timestamp, 0) + 1

# just showing first 20
print("just showing first 20 daq timestamps and how many times we had it")
for ts, counts in list(daq_timestamps.items())[:20]:
    print(ts, counts)

daq_timestamps_list = np.array(sorted( [k for k in daq_timestamps.keys()] ))

tof_timestamp_dict = create_fake_tof_info(daq_timestamps_list)

# Checking if the tof dictionary and daq_timestamps have the same length (THEY NEED TO)
if len(tof_timestamp_dict) != len(daq_timestamps_list):
    raise ValueError("This should never happen")

daq_timestamp_to_tof_dict = {}

for daqtime, tof in zip(daq_timestamps_list, tof_timestamp_dict.values()):
    daq_timestamp_to_tof_dict[daqtime] = tof

# Creating tof as attribute... but you can do anything you wish. The world is yours...
for wf in wfset.waveforms:
    wf.tof = daq_timestamp_to_tof_dict[wf.daq_window_timestamp]


# Showing the first 40 fake tofs..
print("Showing the first 40 fake tofs..")
for wf in wfset.waveforms[:40]:
    print(wf.channel, wf.timestamp, wf.daq_window_timestamp, wf.tof)












