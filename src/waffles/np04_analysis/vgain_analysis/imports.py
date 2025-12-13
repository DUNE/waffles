import os
import json
import numpy as np
import pandas as pd
import copy
import pickle
from pydantic import Field
from plotly import graph_objects as pgo
from plotly import subplots as psu
from collections import defaultdict
from pathlib import Path


from waffles.data_classes.WafflesAnalysis import WafflesAnalysis, BaseInputParams
from waffles.data_classes.IPDict import IPDict
from waffles.data_classes.IODict import IODict
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.ChannelWs import ChannelWs
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
from waffles.data_classes.StoreWfAna import StoreWfAna
from waffles.utils.baseline.WindowBaseliner import WindowBaseliner
from waffles.utils.integral.WindowIntegrator import WindowIntegrator
from waffles.utils.integral.BoxcarIntegrator import BoxcarIntegrator


from waffles.np04_analysis.vgain_analysis import utils as led_utils
from waffles.np04_data.ProtoDUNE_HD_APA_VGAIN_SCAN_map import APA_map
from waffles.input_output.pickle_file_reader import WaveformSet_from_pickle_files
from waffles.utils.fit_peaks.fit_peaks import fit_peaks_of_ChannelWsGrid
from waffles.utils.fit_peaks.fit_peaks import auto_domain_from_grid
from waffles.utils.baseline.baseline_utils import subtract_baseline
from waffles.utils.filtering_utils import fine_selection_for_led_calibration
from waffles.utils.filtering_utils import coarse_selection_for_led_calibration
from waffles.utils.integral.integral_utils import get_pulse_window_limits
from waffles.plotting.plot import plot_ChannelWsGrid
from waffles.utils.numerical_utils import filter_waveform, applyDiscreteFilter