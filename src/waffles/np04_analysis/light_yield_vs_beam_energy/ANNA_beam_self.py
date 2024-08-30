'''
LIGHT YIELD : ANNA ANALYSIS 

To do:
- try do fit the charge histogram 
- try to compute the mean error (dividing by sqrt of n)
- eliminate saturating events
- think about rising edge alignement

'''

### REMEMBER TO CHANGE line 134 in WaveformAdcs.py (equal to 0)



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
Energy = '+7'
Endpoint = 109
Channels = [37] #[20,21,22,23,31,33,35,37]

my_folderpath = data_folder + '/run_0'+ Beam_run_p_DATA[Energy]

# Read one root file 
#wfset=draw.read(my_folderpath + '/' + next((f for f in os.listdir(my_folderpath) if f.endswith('.root')), None), 0,1)
##wfset=draw.read('/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/2_daq_root/run_027338/run027338_0003_dataflow0-3_datawriter_0_20240621T111339.root', 0,1)


# Read whole waveformset 

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

for Channel in Channels:
  # Select an enedpoint and a channel  
  print(f"\nLet's study endpoint {Endpoint:.0f} - channel {Channel:.0f}")
  wfset_single = draw.get_wfs_in_channel(wfset,Endpoint,Channel)

  # Create the time-offset histogram
  print("\nLet's focus on Time-offset histogram in order to identify beam events")
  draw.plot_to(wfset_single,nbins=1000)

  answer = input('Do you want to zoom on the plot (y/n)? ')

  while answer == 'y' : 
      hist_min = int(input("Minimum limit: "))
      hist_max = int(input("Maximum limit: "))

      draw.plot_to(wfset_single,nbins=1000,xmin=hist_min,xmax=hist_max)

      answer = input('Do you want to zoom again (y/n)? ')
      
  answer = input('Do you want to skip this channel (y/n)? ')
  if answer == 'y':
    float_list = '[]'
    n = 0
    mean = 0
    std = 0
    e_mean = 0
    timeoffset_min = 0
    timeoffset_max = 0
    int_start = 0
    int_stop = 0
    
  else: 
    # Select the beam window, in terms of timeoffset 
    print('\nPlease input the time-offset limits you selected')
    timeoffset_min = int(input("Minimum time-offset: "))
    timeoffset_max = int(input("Maximum time-offset: "))
    sub_wfset_single = draw.get_wfs_with_timeoffset_in_range(wfset_single,timeoffset_min,timeoffset_max)

    ### NB: potrei fare un histogramma con tutti i dati e sovrapposti in rosso quelli selezionati come controllo!!!!

    # Select the integration range by plotting somw waveforms 
    print("\nLet's focus on the waveforms in order to identify integration region")
    draw.plot(sub_wfset_single,nwfs=100)
    int_min = int(input("Minimum integration limit: "))
    int_max = int(input("Maximum integration limit: "))

    draw.plot(sub_wfset_single,int_min,int_max,nwfs=100)

    answer = input('Do you want to change integration limits (y/n)? ')

    while answer == 'y':
        int_min = int(input("Minimum integration limit: "))
        int_max = int(input("Maximum integration limit: "))

        draw.plot(sub_wfset_single,int_min,int_max,nwfs=50)

        answer = input('Do you want to change integration limits (y/n)? ')

    '''
    # Comupute the integral od all waveform in the integration range selected and create the histogram
    print('\nPlease input the integration limits you selected')
    int_min = int(input("Selected minimum integration limit: "))
    int_max = int(input("Selected maximum integration limit: "))
    '''
    
    print("\nLet's focus on integration charge histogram")
    draw.plot_charge(sub_wfset_single,Endpoint,Channel,
                        int_min,int_max,          # integration limits
                        100,0,2000000,   # charge histogram (n_bin, min, max)
                        0,40)           # baseline limits

    answer = input('Do you want to change histogram range (y/n)? ')

    while answer == 'y':
        bin_hist = int(input('Number of bins of the histogram: '))
        min_hist = int(input('Minimum charge histogram limit: '))
        max_hist = int(input('Maximum charge histogram limit: '))
        
        draw.plot_charge(sub_wfset_single,Endpoint,Channel,
                        int_min,int_max,          # integration limits
                        bin_hist,min_hist,max_hist,   # charge histogram (n_bin, min, max)
                        0,40)                   # baseline limits
        
        answer = input('Do you want to change histogram range (y/n)? ')

    int_list = draw.plot_charge_3(sub_wfset_single,Endpoint,Channel,
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
  print(f'time_offset_start = {timeoffset_min:.0f}')
  print(f'time_offset_stop = {timeoffset_max:.0f}')
  print(f'int_start = {int_min:.0f}')
  print(f'int_stop = {int_max:.0f}')

  print()
  print('STOP')
  print()

