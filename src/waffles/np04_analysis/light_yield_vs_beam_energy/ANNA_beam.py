'''
LIGHT YIELD : ANNA ANALYSIS 

try to implement for full-streming 


To do:
- try do fit the charge histogram 
- try to compute the mean error (dividing by sqrt of n)
- eliminate saturating events
- think about rising edge alignement

'''

### REMEMBER TO CHANGE line 134 in WaveformAdcs.py (equal to input)


# Import the drawing tools
import numpy as np
import os
import sys
waffles_dir = '/afs/cern.ch/user/a/anbalbon/waffles'
sys.path.append(waffles_dir+'/src') 
import waffles.plotting.drawing_tools as draw
from waffles.input.raw_ROOT_reader import WaveformSet_from_ROOT_files

# Open a png plot 
draw.plotting_mode = 'png'
draw.png_file_path = waffles_dir+'/src/waffles/np04_analysis/light_yield_vs_beam_energy/'+'temp_plot.png'


#Positive polarity (1st beam period - collimator +20)
Beam_run_p_DATA = {'+1':'27338', '+2':'27355', '+3':'27361', '+5':'27367', '+7':'27374'}

#Negative polarity (1st beam period - collimator +20)
Beam_run_n_DATA = {'-1':'27347', '-2':'27358', '-5':'27371', '-7':'27378'} # -3 is missing (look at 27351 or 27352) 

data_folder= '/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/2_daq_root'


# Input information 
Energy = '+1'
Endpoint = 104
Channels = [15, 14, 13]

my_folderpath = data_folder + '/run_0'+ Beam_run_p_DATA[Energy]

# Read one root file 
#wfset=draw.read(my_folderpath + '/' + next((f for f in os.listdir(my_folderpath) if f.endswith('.root')), None), 0,1)
##wfset=draw.read('/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/2_daq_root/run_027338/run027338_0003_dataflow0-3_datawriter_0_20240621T111339.root', 0,1)

#wfset=draw.read(my_folderpath + '/' + next((f for f in os.listdir(my_folderpath) if f.endswith('.root')), None), 0,1, True, True)

# Read whole waveformset 

wfset = WaveformSet_from_ROOT_files( library = 'pyroot', 
                                    folderpath = my_folderpath, 
                                    bulk_data_tree_name = 'raw_waveforms',
                                    meta_data_tree_name = 'metadata',
                                    set_offset_wrt_daq_window = True,
                                    read_full_streaming_data = True,
                                    truncate_wfs_to_minimum = True,
                                    start_fraction = 0.0,
                                    stop_fraction = 1.0,
                                    subsample = 1,
                                    verbose = True)


for Channel in Channels:
    # Select an enedpoint and a channel  
    print(f"\nLet's study endpoint {Endpoint:.0f} - channel {Channel:.0f}")
    wfset_single = draw.get_wfs_in_channel(wfset,Endpoint,Channel)
      
    # Select the integration range by plotting somw waveforms 
    print("\nLet's focus on the waveforms in order to identify integration region")
    draw.plot(wfset_single,nwfs=30, offset=True)
    int_min = int(input("Minimum integration limit: "))
    int_max = int(input("Maximum integration limit: "))

    draw.plot(wfset_single,int_min,int_max,nwfs=40, offset=True)

    answer = input('Do you want to change integration limits (y/n)? ')

    while answer == 'y':
        int_min = int(input("Minimum integration limit: "))
        int_max = int(input("Maximum integration limit: "))

        draw.plot(wfset_single,int_min,int_max,nwfs=40, offset=True)

        answer = input('Do you want to change integration limits (y/n)? ')
        
    
    print("\nLet's focus on integration charge histogram")
    draw.plot_charge(wfset_single,Endpoint,Channel,
                        int_min,int_max,          # integration limits
                        100,0,2000000,   # charge histogram (n_bin, min, max)
                        0,40)           # baseline limits

    answer = input('Do you want to change histogram range (y/n)? ')

    while answer == 'y':
        bin_hist = int(input('Number of bins of the histogram: '))
        min_hist = int(input('Minimum charge histogram limit: '))
        max_hist = int(input('Maximum charge histogram limit: '))
        
        draw.plot_charge(wfset_single,Endpoint,Channel,
                        int_min,int_max,          # integration limits
                        bin_hist,min_hist,max_hist,   # charge histogram (n_bin, min, max)
                        0,40)                   # baseline limits
        
        answer = input('Do you want to change histogram range (y/n)? ')

    int_list = draw.plot_charge_3(wfset_single,Endpoint,Channel,
                            int_min,int_max,          # integration limits
                            140,0,2000000,   # charge histogram (n_bin, min, max)
                            0,40)

    float_list = [float(num) for num in int_list]
    n = len(float_list)
    mean = np.mean(int_list)
    std = np.std(int_list)
    e_mean = std / np.sqrt(n)


    # Stampa 
    print()
    print(f'Channel {Channel}')
    print()
    print(f"Energy = '{Energy}'")
    print(f'int_list = {float_list}')
    print(f'n = {n:.0f}')
    print(f'mean = {mean:.0f}')
    print(f'std = {std:.0f}')
    print(f'e_mean = {e_mean:.0f}')
    print(f'int_start = {int_min:.0f}')
    print(f'int_stop = {int_max:.0f}')

    print()
    print('STOP')
    print()

