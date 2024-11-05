import numpy as np
import pickle
import click
import os

import sys
waffles_dir = '/afs/cern.ch/user/a/anbalbon/waffles'

sys.path.append(waffles_dir) 
from data.ProtoDUNE_HD_APA_maps import APA_map


sys.path.append(waffles_dir+'/src') 
import waffles.plotting.drawing_tools as draw


@click.command()
@click.option("--input_dir", 
              default='/eos/home-f/fegalizz/public/to_Anna',
              help="Folder with the pickels files (default = '/eos/home-f/fegalizz/public/to_Anna' ") 
@click.option("--energy_selected", 
              default= '+2',
              help="Energy of the run to analyze (default = '+2')") 
@click.option("--endpoint", 
              default= 109,
              help="Endpoint to analyze (default = 109)") 
@click.option("--channels", 
              default= '[37]',
              help="List of daq channels to analyze (default = [37])")        
@click.option("--acquisition_mode", 
              default= 'full-streaming',
              help="self-trigger or full-streaming")          

def main(input_dir, energy_selected, endpoint, channels, acquisition_mode):
  adcs_beam_region = wf.Adcs[15640:15700]
  
  # Open a png plot 
  draw.plotting_mode = 'png'
  draw.png_file_path = '/afs/cern.ch/user/a/anbalbon/waffles/src/waffles/np04_analysis/LightYield_vs_EnergyBeam'+'temp_plot.png'

  #1st beam period - collimator +20
  Beam_run_p_DATA = {'+1':'27338', '+2':'27355', '+3':'27361', '+5':'27367', '+7':'27374'}
  Beam_run_n_DATA = {'-1':'27347', '-2':'27358', '-5':'27371', '-7':'27378'} # -3 is missing (look at 27351 or 27352) 


  #Reading pickels files and create wf set
  wfset_dic_path = {'+1':[], '+2':[], '+3':[], '+5':[], '+7':[]}
  wfset_dic_pickles = {'+1':[], '+2':[], '+3':[], '+5':[], '+7':[]}
  wfset_dic = {'+1':'', '+2':'', '+3':'', '+5':'', '+7':''}
  
  if os.path.exists(input_dir):
    for root, dirs, files in os.walk(input_dir):
      for file in files:
        for energy,run in Beam_run_p_DATA.items():
          file_path = os.path.join(root, file)
          if f'_{run}_' in file:
            wfset_dic_path[energy].append(file_path)
            with open(file_path, 'rb') as f:
              wfset_dic_pickles[energy].append(pickle.load(f))     
  else:
      sys.exit("Error:  Directory doesn't exist.") 
  
  
  for energy, pickles_list in wfset_dic_pickles.items():
    if len(pickles_list)>0:
      wfset = pickles_list[0]
      pickles_list.pop(0)
      
      for i_wfset in pickles_list : 
        wfset.merge(i_wfset)
      
      wfset_dic[energy] = wfset
      del wfset
      
  print(wfset_dic)
  
  for APA, apa_info in APA_map.items():
    if acquisition_mode == 'full-streaming' and APA == 1:
      for row in apa_info.Data: # cycle on rows
        for ch_info in row: # cycle on columns elements
          endpoint = ch_info.Endpoint
          channel = ch_info.Channel
          print(f"\n--------------\nLet's study endpoint {endpoint} - channel {channel}")
          
          for energy, wfset in wfset_dic.items():
            print(f"--- Energy: {energy} ---")
            try:
              wfset_single = draw.get_wfs_in_channel(wfset,endpoint,channel)
              print(f"N° wf: {len(wfset_single.Waveforms)}")
            except Exception as e:
              if str(e) == "'str' object has no attribute 'Waveforms'":
                e = 'Missing data'
              print(f'Skipped ({e})')
              continue 
            
            draw.plot(wfset_single,nwfs=30, offset=True)
            
            min_list = []
            max_list = []
            
            for wf in wfset_single.Waveforms:
                min_list.append(min(adcs_beam_region))
                max_list.append(max(adcs_beam_region))

            min_list = [int(i) for i in min_list]
            
            print(f'Endpoint {endpoint} Channel: {channel}\nNumber of selected waveforms: {len(wfset_single.Waveforms)}\n{min_list}')  
            i = 0
            for ampl in min_list:
                if ampl < 500:
                    i+=1
            if i > 0:
                print(f'{i} saturated waveforms')
            print('\n\n')
    else:
      continue 
    
          
          


######################################################################################

if __name__ == "__main__":
    main()

'''

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

'''


  
'''
#To check which channels are in our wfset
for run, data_dic in wfset_dic['+2'].AvailableChannels.items():
  print(f'Run : {run}')
  for endpoint, ch_set in data_dic.items():
    ch_list = {int(v) for v in ch_set}
    print(f'Endpoint: {endpoint} - Channels: {ch_list}')
    del endpoint, ch_set
'''