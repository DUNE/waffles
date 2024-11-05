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
Channels_APA2_109 = [1, 3, 4, 6, 0, 2, 5, 7, 11, 13, 14, 16, 10, 12, 15, 17, 21, 23, 24, 26, 20, 22, 25, 27, 31, 33, 34, 36, 30, 32, 35, 37, 41, 43, 44, 46, 40, 42, 45, 47] 
Channels_APA3_111 = [1, 3, 4, 6, 0, 2, 5, 7, 11, 13, 14, 16, 10, 12, 15, 17, 21, 23, 24, 26, 20, 22, 25, 27, 31, 33, 34, 36, 30, 32, 35, 37, 41, 43, 44, 46, 40, 42, 45, 47]
Channels_APA3_112 = [7, 5, 2, 0, 6, 4, 3, 1, 17, 15, 12, 10, 16, 14, 13, 11, 27, 25, 22, 20, 26, 24, 23, 21, 37, 35, 32, 30, 36, 34, 33, 31, 47, 45, 42, 40]
Channels_APA3_113 = [7, 5, 2, 0]

Data_APA2 = {109 : Channels_APA2_109}
Data_APA3 = {111 : Channels_APA3_111} 
Data_APA4 = {112 : Channels_APA3_112, 113 : Channels_APA3_113} 

my_folderpath = data_folder + '/run_0'+ Beam_run_p_DATA[Energy]

#wfset=draw.read('/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/2_daq_root/run_027374/run027374_0000_dataflow0-3_datawriter_0_20240622T072100.root', 0,1)


wfset = WaveformSet_from_ROOT_files( library = 'pyroot', 
                                    folderpath = my_folderpath, 
                                    bulk_data_tree_name = 'raw_waveforms',
                                    meta_data_tree_name = 'metadata',
                                    set_offset_wrt_daq_window = True,
                                    read_full_streaming_data = False,
                                    truncate_wfs_to_minimum = False,
                                    start_fraction = 0.0,
                                    stop_fraction = 1.0,
                                    subsample = 1,
                                    verbose = True)

with open('/afs/cern.ch/user/a/anbalbon/waffles/src/waffles/np04_analysis/channel_saturation/prova_result.txt', 'w') as file:
    for Endpoint, Channels in Data_APA4.items():
        for Channel in Channels:  
          print(f"\nLet's study endpoint {Endpoint:.0f} - channel {Channel:.0f}")
          wfset_single = draw.get_wfs_in_channel(wfset,Endpoint,Channel)

          if len(wfset_single.Waveforms) < 20:
            print(f'Channel: {Channel}\nNumber of selected waveforms: 0\nSkipped\n\n')  
            file.write(f'Channel: {Channel}\n Number of selected waveforms: 0\nSkipped\n\n')
            continue 

          print("\nLet's focus on Time-offset histogram in order to identify beam events")
          draw.plot_to(wfset_single,nbins=1000)
          
          '''
          answer = input('Do you want to zoom again (y/n)? ')
          while answer == 'y' : 
              hist_min = int(input("Minimum limit: "))
              hist_max = int(input("Maximum limit: "))

              draw.plot_to(wfset_single,nbins=1000,xmin=hist_min,xmax=hist_max)

              answer = input('Do you want to zoom again (y/n)? ')
          '''
            
          answer = input('Do you want to skip this channel (y/n)? ')
          if answer == 'y':
            print(f'Channel: {Channel}\nNumber of selected waveforms: 0\nSkipped\n\n')  
            file.write(f'Channel: {Channel}\n Number of selected waveforms: 0\nSkipped\n\n')
            continue

          else:  
            hist_min = 15500
            hist_max = 15600 
            sub_wfset_single = draw.get_wfs_with_timeoffset_in_range(wfset_single,hist_min,hist_max)
            
            min_list = []
            max_list = []
            for wf in sub_wfset_single.Waveforms:
              min_list.append(min(wf.Adcs[50:120]))
              max_list.append(max(wf.Adcs[50:120]))
              
            min_list = [int(i) for i in min_list]
            
            print(f'Endpoint {Endpoint} Channel: {Channel}\nNumber of selected waveforms: {len(sub_wfset_single.Waveforms)}\n{min_list}')  
            file.write(f'Endpoint {Endpoint} Channel: {Channel}\nNumber of selected waveforms: {len(sub_wfset_single.Waveforms)}\n{min_list}\n')
            i = 0
            for ampl in min_list:
                if ampl < 500:
                    i+=1
            if i > 0:
                file.write(f'{i} saturated waveforms')
                print(f'{i} saturated waveforms')
            file.write('\n\n')
            print('\n\n')

