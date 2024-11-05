import numpy as np
import os
import sys
waffles_dir = '/afs/cern.ch/user/a/anbalbon/waffles'
sys.path.append(waffles_dir+'/src') 
import waffles.plotting.drawing_tools as draw
from waffles.input.raw_ROOT_reader import WaveformSet_from_ROOT_files



#Positive polarity (1st beam period - collimator +20)
Beam_run_p_DATA = {'+1':'27338', '+2':'27355', '+3':'27361', '+5':'27367', '+7':'27374'}

data_folder= '/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/2_daq_root'

# Input information 
Energy = '+7'
Endpoint = 109
Channels_APA1_104 = [7, 5, 2, 0, 6, 4, 3, 1, 17, 15, 12, 10, 16, 14, 13, 11]
Channels_APA1_105 = [7, 5, 2, 0, 6, 4, 3, 1, 26, 24, 23, 21, 17, 15, 12, 10]
Channels_APA1_107 = [17, 15, 12, 10, 7, 5, 2, 0]
Data = {104 : Channels_APA1_104, 105 : Channels_APA1_105, 107: Channels_APA1_107}

my_folderpath = data_folder + '/run_0'+ Beam_run_p_DATA[Energy]

##wfset=draw.read(my_folderpath + '/' + next((f for f in os.listdir(my_folderpath) if f.endswith('.root')), None), 0,1, True, True)

wfset = WaveformSet_from_ROOT_files( library = 'pyroot', 
                                    folderpath = my_folderpath, 
                                    bulk_data_tree_name = 'raw_waveforms',
                                    meta_data_tree_name = 'metadata',
                                    set_offset_wrt_daq_window = True,
                                    read_full_streaming_data = True,
                                    truncate_wfs_to_minimum = True,
                                    start_fraction = 0.0,
                                    stop_fraction = 0.5,
                                    subsample = 1,
                                    verbose = True)


with open('/afs/cern.ch/user/a/anbalbon/waffles/src/waffles/np04_analysis/channel_saturation/prova_result.txt', 'w') as file:
    for Endpoint, Channels in Data.items():
        for Channel in Channels:  
            print(f"\nLet's study endpoint {Endpoint:.0f} - channel {Channel:.0f}")
            wfset_single = draw.get_wfs_in_channel(wfset,Endpoint,Channel)
            print()
            
            min_list = []
            max_list = []
            
            for wf in wfset_single.Waveforms:
                adcs_beam_region = wf.Adcs[15640:15700]
                min_list.append(min(adcs_beam_region))
                max_list.append(max(adcs_beam_region))
                
            min_list = [int(i) for i in min_list]

            print(f'Endpoint {Endpoint} Channel: {Channel}\nNumber of selected waveforms: {len(wfset_single.Waveforms)}\n{min_list}')  
            file.write(f'Endpoint {Endpoint} Channel: {Channel}\nNumber of selected waveforms: {len(wfset_single.Waveforms)}\n{min_list}\n')
            i = 0
            for ampl in min_list:
                if ampl < 500:
                    i+=1
            if i > 0:
                file.write(f'{i} saturated waveforms')
                print(f'{i} saturated waveforms')
            file.write('\n\n')
            print('\n\n')
               