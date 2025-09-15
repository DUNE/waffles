# import the drawing tools
import waffles.plotting.drawing_tools as draw

#open a png plot 
draw.plotting_mode = 'html'
draw.html_file_path = 'file.html'

# read a waffles hdf5 file (structured)
wset=draw.read("../DATA/NP02/processed_np02vd_raw_run036576_0000_df-s04-d0_dw_0_20250519T152501.hdf5_structured.hdf5")

# plot all waveforms in record 2, endpoint 107, and channel 0, taking into account the timestamp 
draw.plot(wset,rec=2,ep=107,ch=0,offset=True)

input()
