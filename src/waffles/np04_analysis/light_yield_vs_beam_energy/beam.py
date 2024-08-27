# import the drawing tools
import sys
waffles_dir = '/Users/acervera/HEP/DUNE/ProtoDUNE-HD/PDS/data_taking/waffles'
sys.path.append(waffles_dir+'/src') 
import waffles.plotting.drawing_tools as draw

# open a png plot 
draw.plotting_mode = 'png'
draw.png_file_path = waffles_dir+'/temp_plot.png'

# read the root file 
#wset=draw.read("../DATA/run26687.root",0.,1)
wset1=draw.read(waffles_dir+"/../DATA/run027338_0000_dataflow0-3_datawriter_0_20240621T111239.root",0,1)
wset7=draw.read(waffles_dir+"/../DATA/run027374_0000_dataflow0-3_datawriter_0_20240622T072100.root",0,1)

detailed=True


# get all waveforms in endpoint 109
wset1_109 = draw.get_wfs_in_channel(wset1,109)
wset7_109 = draw.get_wfs_in_channel(wset7,109)

if detailed:
    input()
    #draw the time offset of all waveforms in ep 109 (APA2)
    draw.plot_to(wset1_109,nbins=1000)

    input()
    # zoom on the beam peak
    draw.plot_to(wset1_109,nbins=1000,xmin=15000,xmax=16000)

    input()
    # zoom on the beam peak even more
    draw.plot_to(wset1_109,nbins=1000,xmin=15500,xmax=15550)

# get all wfs with time offset between 15500 and 15550
wset1_109_beam = draw.get_wfs_with_timeoffset_in_range(wset1_109,15500,15550)
wset7_109_beam = draw.get_wfs_with_timeoffset_in_range(wset7_109,15500,15550)

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
draw.plot_charge(wset1_109_beam,109,35,55,85,200,0,200000)

input()
# plot the charge with integration limits between 55 and 85
draw.plot_charge(wset7_109_beam,109,35,55,85,200,0,2000000)

