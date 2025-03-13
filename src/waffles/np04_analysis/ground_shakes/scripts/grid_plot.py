import waffles.plotting.drawing_tools as draw
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.np04_data.ProtoDUNE_HD_APA_maps_APA1_104 import APA_map as APA_map_2
from plotly import graph_objects as pgo
from plotly import subplots as psu


wset1 = draw.read("../data/wfset_10_30201.hdf5",0,1)

