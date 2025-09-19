import os
import numpy as np
import pickle
from matplotlib import pyplot as plt
from scipy import signal as spsi
from pydantic import Field, field_validator
from waffles.data_classes.WafflesAnalysis import WafflesAnalysis, BaseInputParams
from waffles.data_classes.WaveformAdcs import WaveformAdcs
import waffles.core.utils as wcu
import csv
import plotly.graph_objs as go
from waffles.input_output.pickle_hdf5_reader import WaveformSet_from_hdf5_pickle
from waffles.input_output.hdf5_structured import load_structured_waveformset

import plotly.graph_objects as pgo
from waffles.plotting.plot import plot_ChannelWsGrid
from waffles.plotting.plot import plot_CustomChannelGrid
from waffles.plotting.plot import plot_Histogram
import plotly.subplots as psu
import gc
import h5py

from waffles.data_classes.BasicWfAna import BasicWfAna
from waffles.data_classes.BasicWfAna2 import BasicWfAna2
from waffles.data_classes.IPDict import IPDict
from waffles.np02_data.ProtoDUNE_VD_maps import mem_geometry_map, cat_geometry_map
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.utils.fit_peaks.fit_peaks import fit_peaks_of_ChannelWsGrid
from waffles.np02_analysis.led_calibration import utils as lc_utils

import waffles.plotting.drawing_tools as draw