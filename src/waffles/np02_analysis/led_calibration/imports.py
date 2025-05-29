import os
<<<<<<< HEAD
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
=======
import plotly.subplots as psu
import numpy as np
import pandas as pd
import argparse
import pickle
import plotly.graph_objects as pgo
from pydantic import Field

from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
from waffles.data_classes.IPDict import IPDict
from waffles.data_classes.BasicWfAna import BasicWfAna
from waffles.input_output.raw_root_reader import WaveformSet_from_root_files
from waffles.input_output.pickle_file_reader import WaveformSet_from_pickle_files
from waffles.input_output.hdf5_structured import load_structured_waveformset
from waffles.utils.fit_peaks.fit_peaks import fit_peaks_of_ChannelWsGrid
from waffles.plotting.plot import plot_ChannelWsGrid
from waffles.np04_utils.utils import get_channel_iterator
from waffles.np02_analysis.led_calibration.configs.LED_configuration_to_channel import config_to_channels
from waffles.np02_analysis.led_calibration.configs.configurations import configs
from waffles.np02_analysis.led_calibration.configs.excluded_channels import excluded_channels
from waffles.np02_data.ProtoDUNE_VD_maps import mem_geometry_map, cat_geometry_map
from waffles.np02_analysis.led_calibration import utils as lc_utils
from waffles.data_classes.WafflesAnalysis import WafflesAnalysis, BaseInputParams
from waffles.np02_analysis.led_calibration.configs.metadata import metadata

from waffles.input_output.pickle_hdf5_reader import WaveformSet_from_hdf5_pickle
>>>>>>> ab61a4867ed0a24207a33bc6018db6cce4e18427
