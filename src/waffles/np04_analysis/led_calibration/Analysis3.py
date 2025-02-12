
from waffles.input_output.pickle_file_reader import WaveformSet_from_pickle_file

# Learning waffles

'''
# 1. Read a pkl file and extrar atributes from the first waveform

input_path="wfset.pkl"
wfset = WaveformSet_from_pickle_file(input_path)

nof_wfs=len(wfset.waveforms)
adcs_0=wfset.waveforms[0].adcs
adcs_1=wfset.waveforms[1].adcs
nof_timebinsO=len(adcs_0)
nof_timebins1=len(adcs_1)

print('The number of waveforms is', nof_wfs)
print('The vector of adc values of the first waveform is', adcs_0)
print('The number of timebins in the first waveform is',nof_timebinsO)

print('The vector of adc values of the second waveform is', adcs_1)
print('The number of timebins in the second waveform is',nof_timebins1)

'''

# 2. Read a pkl file and extrar atributes from the first 10 waveforms

input_path="wfset_29964.pkl"
wfset = WaveformSet_from_pickle_file(input_path)

nof_wfs=len(wfset.waveforms)
print('Number of waveforms:', nof_wfs)
print('\n')

runs=wfset.runs
print('Run number:', runs)
print('\n')

for i in range(10):  # Del índice 0 al 9 (primeras 10 waveforms)
    waveform = wfset.waveforms[i]
    print(f'Waveform number {i}')
    adcs=wfset.waveforms[i].adcs
    print(f'Vector of adc values:', adcs)
    nof_timebins=len(adcs)
    print(f'Number of timebins:',nof_timebins)
    print('\n')

# Nuevo Analysis3

from waffles.input_output.pickle_file_reader import WaveformSet_from_pickle_file
from waffles.data_classes.WfPeak import WfPeak
from waffles.data_classes.PeakFindingWfAna import PeakFindingWfAna
# Learning waffles

'''
# 1. Read a pkl file and extrar atributes from the first waveform

input_path="wfset.pkl"
wfset = WaveformSet_from_pickle_file(input_path)

nof_wfs=len(wfset.waveforms)
adcs_0=wfset.waveforms[0].adcs
adcs_1=wfset.waveforms[1].adcs
nof_timebinsO=len(adcs_0)
nof_timebins1=len(adcs_1)

print('The number of waveforms is', nof_wfs)
print('The vector of adc values of the first waveform is', adcs_0)
print('The number of timebins in the first waveform is',nof_timebinsO)

print('The vector of adc values of the second waveform is', adcs_1)
print('The number of timebins in the second waveform is',nof_timebins1)

'''

# 2. Read a pkl file and extrar atributes from the first 10 waveforms

input_path="wfset_29964.pkl"
wfset = WaveformSet_from_pickle_file(input_path)

nof_wfs=len(wfset.waveforms)
print('\n')
print('Number of waveforms:', nof_wfs)
print('\n')

runs=wfset.runs
print('Run number:', runs)
print('\n')

for i in range(3):  # Del índice 0 al 9 (primeras 10 waveforms)
    waveform = wfset.waveforms[i]
    print(f'Waveform number {i}')
    adcs=wfset.waveforms[i].adcs
    print(f'Vector of adc values:', adcs)
    nof_timebins=len(adcs)
    print(f'Number of timebins:',nof_timebins)
    endpoint=wfset.waveforms[i].endpoint
    print(f'Endpoint:',endpoint)
    channel=wfset.waveforms[i].channel
    print(f'Channel:',channel)
    time_step=wfset.waveforms[i].time_step_ns
    print(f'Time step:',time_step,f'ns')
    record_number=wfset.waveforms[i].record_number
    print('Record number:',record_number)
    
    PeakFindingWfAna(wfset.waveforms[i])
    pos_peak=PeakFindingWfAna
    print('Peak position:',pos_peak)
    print('\n')
    