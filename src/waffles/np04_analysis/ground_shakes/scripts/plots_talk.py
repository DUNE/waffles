import waffles.plotting.drawing_tools as draw
import numpy as np
from waffles.data_classes.Waveform import Waveform


wset = draw.read("../data/wfset_5_apas_30201.hdf5",0,1)


#--------- Precursors -----------

# in APA1
draw.plot_grid(wset,1,rec=range(1,40),xmin=-10000,xmax=-1500,ymin=7500,ymax=9000,offset=True)

input()
# in APA2
draw.plot_grid(wset,2,rec=range(1,40),xmin=-10000,xmax=-1500,ymin=7500,ymax=9000,offset=True)


#--------- Correlation of reflections -----------

input()
# in APA1
draw.plot_grid(wset,1,rec=range(1,100),tmin=-1500,tmax=5000,xmin=-1000,xmax=5000,offset=True)

input()
# in APA2
draw.plot_grid(wset,2,rec=range(1,100),tmin=-1500,tmax=5000,xmin=-1000,xmax=5000,offset=True)

input()
# in APA3
draw.plot_grid(wset,3,rec=range(1,100),tmin=-1500,tmax=5000,xmin=-1000,xmax=5000,offset=True)

input()
# in APA4
draw.plot_grid(wset,4,rec=range(1,100),tmin=-1500,tmax=5000,xmin=-1000,xmax=5000,offset=True)


#--------- single event in APAs 1 and 2 -----------
draw.line_color='black'
draw.plot(wset,104,rec=[15],xmin=-1000,xmax=3000,offset=True)
draw.line_color='red'
draw.plot(wset,109,rec=[15],xmin=-1000,xmax=3000,offset=True,op="same")

# zoom to see alignment
input()
draw.zoom(-350,-260,7800,8500)


#--------- small partners -----------
input()
# all channels together
draw.zoom(2000,3000,6500,9000)

input()
# same but in grid plot for APA1
draw.plot_grid(wset,1,rec=[15],xmin=2000,xmax=3000,ymin=6500,ymax=9000,offset=True)

input()
# same but in grid plot for APA2
draw.plot_grid(wset,2,rec=[15],xmin=2000,xmax=3000,ymin=7500,ymax=9000,offset=True)


################ Statistical plots (histograms) #####################



def min_adc(wf: Waveform): return min(wf.adcs);
def max_adc(wf: Waveform): return max(wf.adcs);
def std_adc(wf: Waveform): return np.std(wf.adcs);



#--------- Statistics of saturated waveforms -----------

input()
# plot the histogram of the minimum adc value in APA1
draw.plot_grid_histogram(wset,min_adc,apa=1,tmin=-1500,tmax=2000)

input()
# plot the histogram of the minimum adc value in APA2
draw.plot_grid_histogram(wset,min_adc,apa=2,tmin=-1500,tmax=2000)

