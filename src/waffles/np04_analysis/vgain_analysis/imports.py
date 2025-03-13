from pydantic import *
import pandas as pd
import numpy as np
import os
from waffles.data_classes.WafflesAnalysis import WafflesAnalysis, BaseInputParams
import waffles.input_output.raw_hdf5_reader as reader

from waffles.data_classes.Run import Run
from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.UniqueChannel import UniqueChannel
from waffles.data_classes.ChannelMap import ChannelMap
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
from waffles.data_classes.ChannelWsGrid import ChannelWs
from waffles.data_classes.IPDict import IPDict
from waffles.data_classes.BasicWfAna import BasicWfAna
from waffles.utils.fit_peaks.fit_peaks import fit_peaks_of_ChannelWsGrid
from waffles.plotting.plot import plot_ChannelWsGrid

from waffles.np04_analysis.vgain_analysis.configs.analysis_configuration import *
from waffles.np04_analysis.vgain_analysis.utils import *
from waffles.utils.filtering_utils import *
from waffles.utils.numerical_utils import *
from waffles.utils.baseline.baseline import SBaseline

import plotly.graph_objects as pgo
import matplotlib.pyplot as plt
 