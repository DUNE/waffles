# New script for Analysis 

from tools_analysis import *

#Positive polarity (1st beam period - collimator +20)
Beam_run_p_DATA = {'+1':'27338', '+2':'27355', '+3':'27361', '+5':'27367', '+7':'27374'}
#Negative polarity (1st beam period - collimator +20)
Beam_run_n_DATA = {'-1':'27347', '-2':'27358', '-5':'27371', '-7':'27378'} # -3 is missing (look at 27351 or 27352) 

data_folder= '/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/2_daq_root'

filepath_1 = '/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/2_daq_root/run_027338/run027338_0000_dataflow0-3_datawriter_0_20240621T111239.root'
filepath_2 = '/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/2_daq_root/run_027338/run027338_0001_dataflow0-3_datawriter_0_20240621T111302.root'
folderpath = '/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/2_daq_root/run_027338'

## Read root file
wfset_1 = reader.WaveformSet_from_root_file(
    filepath_1,                               # path to the root file
    'pyroot',                               # library to read (if ROOT, use 'pyroot', if not `uproot`)
    read_full_streaming_data = False,       # if False, read the self-triggered data
    truncate_wfs_to_minimum = False,        # truncate the waveforms to the minimum size
    start_fraction = 0.0,                   # starting fraction for reading
    stop_fraction = 0.2,                    # stoping fraction for reading
    subsample = 10,                          # subsample the data reading (read every other entry)
    verbose = True)


wfset_2 = reader.WaveformSet_from_root_file(
    filepath_2,                               # path to the root file
    'pyroot',                               # library to read (if ROOT, use 'pyroot', if not `uproot`)
    read_full_streaming_data = False,       # if False, read the self-triggered data
    truncate_wfs_to_minimum = False,        # truncate the waveforms to the minimum size
    start_fraction = 0.0,                   # starting fraction for reading
    stop_fraction = 0.2,                    # stoping fraction for reading
    subsample = 10,                          # subsample the data reading (read every other entry)
    verbose = True)

'''
## Read files in a folder
wfset = reader.WaveformSet_from_root_files(
    'pyroot',                               # library to read (if ROOT, use 'pyroot', if not `uproot`)
    folderpath = folderpath,
    read_full_streaming_data = False,       # if False, read the self-triggered data
    truncate_wfs_to_minimum = False,        # truncate the waveforms to the minimum size
    start_fraction = 0.0,                   # starting fraction for reading
    stop_fraction = 1.0,                    # stoping fraction for reading
    subsample = 1,                          # subsample the data reading (read every other entry)
    verbose = True)

'''
print(len(wfset_1.waveforms))
print(len(wfset_2.waveforms))

wfset_1.merge(wfset_2) #merged
print(len(wfset_1.waveforms))




bl = [0, 100, 900, 1000]
int_ll = 135
int_ul = 165
peak_finding_kwargs = dict( prominence = 20,rel_height=0.5,width=[0,75])
ip = IPDict(baseline_limits=bl,
            int_ll=int_ll,int_ul=int_ul,amp_ll=int_ll,amp_ul=int_ul,
            points_no=10,
            peak_finding_kwargs=peak_finding_kwargs)
analysis_kwargs = dict(  return_peaks_properties = False)
checks_kwargs   = dict( points_no = wfset_1.points_per_wf )
#if wset.Waveforms[0].has_analysis('standard') == False:

# analyse the waveforms
a= wfset_1.analyse(label='prova',analysis_class=my_BasicWfAna,input_parameters=ip,checks_kwargs = checks_kwargs,overwrite=True)
#a=wfset_1.analyse('standard',BasicWfAna,ip,checks_kwargs = checks_kwargs,overwrite=True)  

print(wfset_1.waveforms[10].analyses['prova'].result['baseline'])
     
'''    
wfset_2 = wfset.from_filtered_WaveformSet(wfset, my_filter, 113)
print(len(wfset_2.waveforms))
'''