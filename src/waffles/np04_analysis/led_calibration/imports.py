import os
import numpy as np
import pandas as pd
from pydantic import Field


from waffles.data_classes.WafflesAnalysis import WafflesAnalysis, BaseInputParams
from waffles.data_classes.IPDict import IPDict
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.ChannelWs import ChannelWs
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
from waffles.data_classes.StoreWfAna import StoreWfAna
from waffles.utils.baseline.WindowBaseliner import WindowBaseliner
from waffles.utils.integral.WindowIntegrator import WindowIntegrator


from waffles.np04_analysis.led_calibration import utils as led_utils
from waffles.np04_data.ProtoDUNE_HD_APA_maps import APA_map
from waffles.utils.fit_peaks.fit_peaks import fit_peaks_of_ChannelWsGrid
from waffles.utils.baseline.baseline_utils import subtract_baseline
from waffles.utils.filtering_utils import selection_for_led_calibration
from waffles.utils.integral.integral_utils import get_pulse_window_limits
from waffles.plotting.plot import plot_ChannelWsGrid