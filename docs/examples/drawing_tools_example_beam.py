# import the drawing tools
import sys
waffles_dir = '/Users/acervera/HEP/DUNE/ProtoDUNE-HD/PDS/data_taking/waffles'
sys.path.append(waffles_dir+'/src') 
import waffles.plotting.drawing_tools as draw
import waffles.data_classes.WaveformSet as WaveformSet

# open a png plot 
draw.plotting_mode = 'png'
draw.png_file_path = waffles_dir+'/temp_plot.png'

wset1 = WaveformSet
wset7 = WaveformSet

########################
def plot_selftrigger():

    global wset1,wset7

    # read files for 1 and 7 GeV
    wset1=draw.read(waffles_dir+"/../DATA/run027338_0000_dataflow0-3_datawriter_0_20240621T111239.root",0,1)
    wset7=draw.read(waffles_dir+"/../DATA/run027374_0000_dataflow0-3_datawriter_0_20240622T072100.root",0,1)

    # get all waveforms in endpoint 109
    wset1_109 = draw.get_wfs_in_channel(wset1,109)
    wset7_109 = draw.get_wfs_in_channel(wset7,109)

    #draw the time offset of all waveforms in ep 109 (APA2)
    draw.plot_to(wset1_109,nbins=1000)

    input()
    # zoom on the beam peak
    draw.plot_to(wset1_109,nbins=1000,xmin=-500,xmax=500)

    input()
    # zoom on the beam peak even more
    draw.plot_to(wset1_109,nbins=1000,xmin=-120,xmax=-80)

    # get all wfs with time offset between 15500 and 15550
    wset1_109_beam = draw.get_wfs_with_timeoffset_in_range(wset1_109,-120,-80)
    wset7_109_beam = draw.get_wfs_with_timeoffset_in_range(wset7_109,-120,-80)

    input()
    # draw the timeoffset of all  beam related waveforms in 109 (no binning specified)
    draw.plot_to(wset1_109_beam)

    # get a subsample in ch 35
    wset1_10935_beam = draw.get_wfs_in_channel(wset1_109_beam,109,35)
    wset7_10935_beam = draw.get_wfs_in_channel(wset7_109_beam,109,35)

    input()
    # Plot some waveforms to decide integration limits
    draw.plot(wset1_10935_beam,nwfs=100)

    input()
    # plot the charge with integration limits between 55 and 85
    draw.plot_charge(wset1_109_beam,109,35,
                        55,85,          # integration limits
                        200,0,200000,   # binning
                        0,40)           # baseline limits


    input()
    # plot the charge with integration limits between 55 and 85
    draw.plot_charge(wset7_109_beam,109,35,55,85,200,0,1500000,0,40)

########################
def plot_fullstreaming():

    global wset1,wset7

    # read files for 1 and 7 GeV. Last two arguments: 
    # 1. Read full streaming
    # 2. Truncate wf size to the smallest waveform. This is because for FS there could be a difference of 
    #    64 ticks in the waveform size  
    wset1=draw.read(waffles_dir+"/../DATA/run027338_0000_dataflow0-3_datawriter_0_20240621T111239.root",0,1,True,True)
    wset7=draw.read(waffles_dir+"/../DATA/run027374_0000_dataflow0-3_datawriter_0_20240622T072100.root",0,1,True,True)

    # Plot 5 wfs in ep 104 ch 15 (APA1)
    draw.plot(wset1,104,15,5)

    input()
    # zoom on the beam peak
    draw.plot(wset1,104,15,200,10000,20000)

    input()
    # zoom on the beam peak even more
    draw.plot(wset1,104,15,200,15500,15800)

    input()
    # Align all waveforms in time applying the offset (timestamp-daq_timestamp)
    draw.plot(wset1,104,15,200,15600,15700,offset=True)

    input()
    # Do the same for 7 GeV
    draw.plot(wset7,104,15,200,15600,15700,offset=True)

    input()
    # plot the charge with integration limits between 15650 and 15680
    draw.plot_charge(wset1,104,15,
                        15650,15680,    # integration limits
                        200,0,3000000,   # binning
                        0,100)           # baseline limits


    input()
    # plot the charge with integration limits between 15650 and 15680
    draw.plot_charge(wset7,104,15,15650,15680,200,0,3000000,0,100)

