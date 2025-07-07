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

from waffles.input_output.pickle_hdf5_reader import WaveformSet_from_hdf5_pickle

import plotly.graph_objects as pgo
from waffles.plotting.plot import plot_ChannelWsGrid
from waffles.plotting.plot import plot_CustomChannelGrid
import plotly.subplots as psu
import gc
import h5py

from waffles.data_classes.BasicWfAna import BasicWfAna
from waffles.data_classes.IPDict import IPDict
from waffles.np02_data.ProtoDUNE_VD_maps import mem_geometry_map, cat_geometry_map
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid


from waffles.np02_analysis.onsite import utils as os_utils

import waffles.plotting.drawing_tools as draw