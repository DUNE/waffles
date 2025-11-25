import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from waffles.data_classes.IPDict import IPDict
from waffles.np04_data.ProtoDUNE_HD_APA_maps import APA_map # apa map
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid # channel grid

from waffles.np04_analysis.led_calibration.utils import compute_average_baseline_std #baseline computation
from waffles.utils.baseline.baseline_utils import subtract_baseline # baseline subtraction

from waffles.data_classes.StoreWfAna import StoreWfAna 
from waffles.utils.integral.integral_utils import get_pulse_window_limits
from waffles.utils.baseline.WindowBaseliner import WindowBaseliner
from waffles.np04_analysis.lightyield_vs_energy.scripts.MY_WindowIntegrator import MY_WindowIntegrator # for integration
