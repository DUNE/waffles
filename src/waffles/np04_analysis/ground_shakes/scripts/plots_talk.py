import waffles.plotting.drawing_tools as draw
import numpy as np
from waffles.data_classes.Waveform import Waveform


wset = draw.read("../data/wfset_5_apas_30201.hdf5",0,1)
wset = draw.read("../data/Run_30201_GS_AllAPAs.hdf5",0,1)
#wset2 = draw.read("../data/wfset_5_apas_30202.hdf5",0,1)
wset2 = draw.read("../data/Run_30202_GS_APA2.hdf5",0,1)



#--------- Time signature -----------

# 100 events in APA1
draw.plot_grid(wset,1,-1,rec=range(1,100),tmin=-1500,tmax=500,xmin=-500,xmax=800,offset=True)

# 100 events in APA2
draw.plot_grid(wset,2,-1,rec=range(1,100),tmin=-1500,tmax=500,xmin=-500,xmax=800,offset=True)

# example of single saturation
draw.plot(wset,109,3,rec=[3],tmin=-500,tmax=500,xmin=-500,xmax=800,offset=True)

# example of double saturation
draw.plot(wset,104,41,rec=[14],tmin=-500,tmax=500,xmin=-500,xmax=800,offset=True)

#--------- Precursors -----------

# in APA1
input()
draw.plot_grid(wset,1,rec=range(1,40),xmin=-10000,xmax=-1500,ymin=7500,ymax=9000,offset=True)

input()
# in APA2
draw.plot_grid(wset,2,rec=range(1,40),xmin=-10000,xmax=-1500,ymin=7500,ymax=9000,offset=True)

#--------- Precursors in second run (with no APA1) -----------

input()
# in APA2
draw.plot_grid(wset2,2,rec=range(1,40),xmin=-10000,xmax=-1500,ymin=7500,ymax=9000,offset=True)

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

#--------- Reflections in second run (with no APA1) -----------

input()
# in APA2
draw.plot_grid(wset2,2,rec=range(1,100),tmin=-1500,tmax=5000,xmin=-1000,xmax=5000,offset=True)


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

def min_tick(wf: Waveform): return np.argmin(wf.adcs)+offset(wf);
def max_tick(wf: Waveform): return np.argmax(wf.adcs)+offset(wf);


def min_tick_saturated(wf: Waveform):
    mt = np.argmin(wf.adcs);
    if wf.adcs[mt]<10:  # make sure it is the saturated waveform
        return mt+offset(wf)
    else:
        return -10000000
    
def max_min_tick(wf: Waveform): return np.argmax(wf.adcs)-np.argmin(wf.adcs);

def offset(wf: Waveform): return np.float32(np.int64(wf.timestamp)-np.int64(wf.daq_window_timestamp));


#--------- Statistics of saturated waveforms -----------

input()
# plot the histogram of the minimum adc value in APA1
draw.plot_grid_histogram(wset,min_adc,apa=1,tmin=-1500,tmax=2000)

input()
# plot the histogram of the minimum adc value in APA2
draw.plot_grid_histogram(wset,min_adc,apa=2,tmin=-1500,tmax=2000)

#--------- Width of saturated waveforms -----------

input()
# single waveform in channel 3 and record 3, as example to ilustrate saturated waveform width
draw.plot(wset,109,3,rec=[3],tmin=-1500,tmax=500,xmin=-500,xmax=800,offset=True)

input()
# distance in ticks between min(adc) and max(adc) in APA1
draw.plot_grid_histogram(wset,max_min_tick,apa=1,tmin=-1500,tmax=500,xmin=00,xmax=500)

input()
# distance in ticks between min(adc) and max(adc) in APA2
draw.plot_grid_histogram(wset,max_min_tick,apa=2,tmin=-1500,tmax=500,xmin=00,xmax=500)

input()
# distance in ticks between min(adc) and max(adc) in APA2, for second run
draw.plot_grid_histogram(wset2,max_min_tick,apa=2,tmin=-1500,tmax=500,xmin=00,xmax=500)

#--------- STD of precursors -----------

input()
# for APA1
draw.plot_grid_histogram(wset,std_adc,apa=1,tmin=-5000,tmax=-1500,xmin=0,xmax=200)

input()
# for APA2
draw.plot_grid_histogram(wset,std_adc,apa=2,tmin=-5000,tmax=-1500,xmin=0,xmax=100)

#--------- Tick of min value (max amp) of saturated waveforms -----------

input()
# for APA1
draw.plot_grid_histogram(wset,min_tick_saturated,apa=1,xmin=-600,xmax=0)

input()
# for APA2
draw.plot_grid_histogram(wset,min_tick_saturated,apa=2,xmin=-600,xmax=0)
