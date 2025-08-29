# import the drawing tools
import waffles.plotting.drawing_tools as draw

#open a png plot 
draw.plotting_mode = 'html'
draw.html_file_path = 'file.html'

# read a waffles hdf5 file (structured)
wsetc=draw.read("../../../DATA/NP02/processed_np02vd_raw_run038930_0000_df-s05-d0_dw_0_20250820T181308.hdf5_structured_cathode.hdf5",0,1)
wsetm=draw.read("../../../DATA/NP02/processed_np02vd_raw_run038930_0000_df-s05-d0_dw_0_20250820T181308.hdf5_structured_membrane.hdf5",0,1)

# plot all waveforms in record 2, endpoint 107, and channel 0, taking into account the timestamp 
draw.plot(wsetc,rec=2,ep=107,ch=0,offset=True)

