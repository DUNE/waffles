import waffles.plotting.drawing_tools as draw
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.Waveform import Waveform
from waffles.np04_data.ProtoDUNE_HD_APA_maps_APA1_104 import APA_map as APA_map_2
from plotly import graph_objects as pgo
from plotly import subplots as psu
import numpy as np

wset = draw.read("../data/wfset_5_apas_30201.hdf5",0,1)
#wset2 = draw.read("wfset_5_apas_30202.hdf5",0,1)
#wset1 = draw.read("../../src/waffles/np04_analysis/beam_example/data/028676_109_30-37_20files_wfset_raw.pkl",0,1)

##################### GS and reflections ###################

draw.plot_grid(wset,1,[-1],rec=range(1,40),tmin=-1000,tmax=3500,xmin=-1000,xmax=4500,offset=True)
input()
draw.plot_grid(wset,2,[-1],rec=range(1,40),tmin=-1000,tmax=3500,xmin=-1000,xmax=4500,offset=True)
input()
draw.plot_grid(wset,3,[-1],rec=range(1,40),tmin=-1000,tmax=1500,xmin=-1000,xmax=1500,offset=True)
input()
draw.plot_grid(wset,4,[-1],rec=range(1,40),tmin=-1000,tmax=1500,xmin=-1000,xmax=1500,offset=True)
input()

##################### Precursor ###################

draw.plot_grid(wset,1,[-1],rec=range(1,40),tmin=-20000,tmax=-1500,xmin=-20000,xmax=-1500,ymin=7500,ymax=9000,offset=True)
input()
draw.plot_grid(wset,2,[-1],rec=range(1,40),tmin=-20000,tmax=-1500,xmin=-20000,xmax=-1500,ymin=7500,ymax=9000,offset=True)

input()
draw.plot_grid_histogram(wset,func,apa=1,tmin=-10000,tmax=-1500,xmin=0,xmax=200)
input()
draw.plot_grid_histogram(wset,func,apa=2,tmin=-10000,tmax=-1500,xmin=0,xmax=200)

# plot few events for APA2
#for i in range (2,10):
#    draw.plot_grid(wset,2,-1,rec=[i],tmin=-1000,tmax=500,offset=True)
#    input()


# plot few events for APA1
#for i in range (2,10):
#    draw.plot_grid(wset,1,-1,rec=[i],tmin=-1000,tmax=500,offset=True)
#    input()
    

# plot many events superimposed for APA2
#draw.plot_grid(wset,2,-1,rec=range(1,60),tmin=t-1000,tmax=500,offset=True)

#draw.plot_grid(wset,1,-1,rec=range(1,20),tmin=-1000,tmax=500,offset=True)
#input()

#draw.plot_grid(wset,2,-1,rec=range(1,20),tmin=-1000,tmax=500,offset=True)
#input()

#draw.plot_grid(wset,3,-1,rec=range(1,20),tmin=-1000,tmax=500,offset=True)
#input()

#draw.plot_grid(wset,4,-1,rec=range(1,20),tmin=-1000,tmax=500,offset=True)
#input()



input()

nwfs=20

draw.nrows=4
draw.ncols=2

draw.current_row=1
draw.current_col=1

draw.line_color='black'
draw.plot(wset,104,24,rec=range(1,nwfs),tmin=-20000,tmax=-1500,offset=True)
draw.line_color='red'
draw.plot(wset,104,2,rec=range(1,nwfs),tmin=-20000,tmax=-1500,offset=True,op='same')
draw.line_color='green'
draw.plot(wset,104,14,rec=range(1,nwfs),tmin=-20000,tmax=-1500,offset=True,op='same')


draw.current_row=1
draw.current_col=2

draw.line_color='black'
draw.plot(wset,104,24,rec=range(1,nwfs),tmin=-1500,tmax=3000,offset=True,op='same')
draw.line_color='red'
draw.plot(wset,104,2,rec=range(1,nwfs),tmin=-1500,tmax=3000,offset=True,op='same')
draw.line_color='green'
draw.plot(wset,104,14,rec=range(1,nwfs),tmin=-1500,tmax=3000,offset=True,op='same')


############
draw.current_row=2
draw.current_col=1


draw.line_color='black'
draw.plot(wset,104,23,rec=range(1,nwfs),tmin=-20000,tmax=-1500,offset=True,op='same')
draw.line_color='red'
draw.plot(wset,104,5,rec=range(1,nwfs),tmin=-20000,tmax=-1500,offset=True,op='same')
draw.line_color='green'
draw.plot(wset,104,13,rec=range(1,nwfs),tmin=-20000,tmax=-1500,offset=True,op='same')



draw.current_row=2
draw.current_col=2


draw.line_color='black'
draw.plot(wset,104,23,rec=range(1,nwfs),tmin=-1500,tmax=3000,offset=True,op='same')
draw.line_color='red'
draw.plot(wset,104,5,rec=range(1,nwfs),tmin=-1500,tmax=3000,offset=True,op='same')
draw.line_color='green'
draw.plot(wset,104,13,rec=range(1,nwfs),tmin=-1500,tmax=3000,offset=True,op='same')


####################

draw.current_row=3
draw.current_col=1

draw.line_color='black'
draw.plot(wset,104,26,rec=range(1,nwfs),tmin=-20000,tmax=-1500,offset=True,op='same')
draw.line_color='red'
draw.plot(wset,104,0,rec=range(1,nwfs),tmin=-20000,tmax=-1500,offset=True,op='same')
draw.line_color='green'
draw.plot(wset,104,16,rec=range(1,nwfs),tmin=-20000,tmax=-1500,offset=True,op='same')



draw.current_row=3
draw.current_col=2

draw.line_color='black'
draw.plot(wset,104,26,rec=range(1,nwfs),tmin=-1500,tmax=3000,offset=True,op='same')
draw.line_color='red'
draw.plot(wset,104,0,rec=range(1,nwfs),tmin=-1500,tmax=3000,offset=True,op='same')
draw.line_color='green'
draw.plot(wset,104,16,rec=range(1,nwfs),tmin=-1500,tmax=3000,offset=True,op='same')



##################################

draw.current_row=4
draw.current_col=1


draw.line_color='black'
draw.plot(wset,104,21,rec=range(1,nwfs),tmin=-20000,tmax=-1500,offset=True,op='same')
draw.line_color='red'
draw.plot(wset,104,7,rec=range(1,nwfs),tmin=-20000,tmax=-1500,offset=True,op='same')
draw.line_color='green'
draw.plot(wset,104,47,rec=range(1,nwfs),tmin=-20000,tmax=-1500,offset=True,op='same')


draw.current_row=4
draw.current_col=2

draw.line_color='black'
draw.plot(wset,104,21,rec=range(1,nwfs),tmin=-1500,tmax=3000,offset=True,op='same')
draw.line_color='red'
draw.plot(wset,104,7,rec=range(1,nwfs),tmin=-1500,tmax=3000,offset=True,op='same')
draw.line_color='green'
draw.plot(wset,104,47,rec=range(1,nwfs),tmin=-1500,tmax=3000,offset=True,op='same')

###############################

draw.line_color='black'

input()

for row in range(0,10):
    channels = []
    for col in range (0,1):
        channels.append(APA_map_2[1].data[row][col].channel)

    print (f"plotting row {row}. Channels: {channels}")        
    draw.plot(wset,104,channels,rec=range(1,40),tmin=-1000,tmax=3500,offset=True)
    input()



for col in range(0,4):
    channels = []
    for row in range (0,10):
        channels.append(APA_map_2[2].data[row][col].channel)

    print (f"plotting column {col}. Channels: {channels}")        
    draw.plot(wset,109,channels,rec=range(1,40),tmin=-1000,tmax=3500,offset=True)
    input()


for col in range(0,4):
    channels = []
    for row in range (0,10):
        channels.append(APA_map_2[1].data[row][col].channel)

    print (f"plotting column {col}. Channels: {channels}")        
    draw.plot(wset,104,channels,rec=range(1,40),tmin=-1000,tmax=3500,offset=True)
    input()
    
