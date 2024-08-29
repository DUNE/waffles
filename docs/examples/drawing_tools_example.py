# import the drawing tools
import sys
waffles_dir = '/Users/acervera/HEP/DUNE/ProtoDUNE-HD/PDS/data_taking/waffles'
sys.path.append(waffles_dir+'/src') 
import waffles.plotting.drawing_tools as draw

#open a png plot 
draw.plotting_mode = 'png'
draw.png_file_path = waffles_dir+'/temp_plot.png'

# read the root file 
wset=draw.read(waffles_dir+"/../DATA/run26687.root",0,1)

# plot 10 wfs for endpoint 111 and channel 45
draw.plot(wset,111,45,10)

input()
# Same plot but now with time with respect to daq window 
draw.plot(wset,111,45,10,offset=True)

input()

# plot the heat map for that channel. Numbers beyond 45 correspond to bining (nbinsx,xmin,xmax,nbinsy,ymin,ymax)
draw.plot_hm(wset,111,45,40,130,170,100,8000,8200)

input()
# plot the charge histogram with integration limits 135,165
charge_histo = draw.plot_charge(wset,111,45,135,165)

input()
# plot the charge histogram and show the peaks (only two peaks by default)
draw.plot_charge(wset,111,45,135,165,op="peaks")

input()
# if we want to change peaks parameters use this method, which takes as argument 
# the charge histogram produced above

# plot the charge histogram with 3 peaks 
draw.plot_charge_peaks(charge_histo,3)

input()
# get a WaveformSet with only wfs in ep 111 and ch 45
wset_11145 = draw.get_wfs_in_channel(wset,111,45)

# get all wfs in that channel with integral in the 1 p.e. peak [3500,7500]
wset_11145_1pe = draw.get_wfs_with_integral_in_range(wset_11145,3500,7500)

# plot the heat map for that waveform subsample (1 pe waveforms)
draw.plot_hm(wset_11145_1pe,111,45,40,130,170,100,8000,8200)

input()
# get all wfs in that channel with integral in the 2 p.e. peak [10000,14000]
wset_11145_2pe = draw.get_wfs_with_integral_in_range(wset_11145,10000,14000)

# plot the heat map for that waveform subsample (2 pe waveforms)
draw.plot_hm(wset_11145_2pe,111,45,40,130,170,100,8000,8200)

input()
# plot time offset for all waveforms in channel 111 - 45
draw.plot_to(wset, 111, 45)

input()
# get all wfs with time offset between 14000 and 16000
wset_11145_to_range = draw.get_wfs_with_timeoffset_in_range(wset_11145,10000,20000)

# plot those time offsets
draw.plot_to(wset_11145_to_range)

input()

from waffles.data_classes.Waveform import Waveform 

# example of general filtering method
def filter_example(waveform: Waveform, allowed_channels) -> bool:
    # This condition could be whatever (use all Waveforms data members)
    if waveform.Endpoint == 111 and waveform.Channel in allowed_channels:
        return True
    else:
        return False
    
# collect all waveforms in chs 40 and 45
wset_40_45 = wset.from_filtered_WaveformSet(wset, filter_example,[40,45])

# print the endpoint of the first 10 waveforms
a=[print( i, wset_40_45.Waveforms[i].Channel) for i in range(10)]