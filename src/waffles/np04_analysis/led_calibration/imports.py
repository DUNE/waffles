import os
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
from waffles.utils.fit_peaks.fit_peaks import fit_peaks_of_ChannelWsGrid
from waffles.plotting.plot import plot_ChannelWsGrid
from waffles.np04_utils.utils import get_channel_iterator
from waffles.np04_analysis.led_calibration.configs.calibration_batches.LED_configuration_to_channel import config_to_channels
from waffles.np04_analysis.led_calibration.configs.calibration_batches.run_number_to_LED_configuration import run_to_config
from waffles.np04_analysis.led_calibration.configs.calibration_batches.excluded_channels import excluded_channels
from waffles.np04_data.ProtoDUNE_HD_APA_maps import APA_map
from waffles.np04_analysis.led_calibration import utils as led_utils
from waffles.data_classes.WafflesAnalysis import WafflesAnalysis, BaseInputParams
from waffles.np04_analysis.led_calibration.configs.calibration_batches.metadata import metadata
