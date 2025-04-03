from pydantic import Field
import pickle
import numpy as np
import matplotlib.pyplot as plt
import yaml
import importlib
from pathlib import Path
from plotly import graph_objects as pgo
from plotly.subplots import make_subplots
import ROOT as root
from scipy.optimize import curve_fit
from scipy.fft import fft, fftshift
from scipy import signal


from waffles.data_classes.WafflesAnalysis import WafflesAnalysis, BaseInputParams
from waffles.np04_data.ProtoDUNE_HD_APA_maps import APA_map
from waffles.np04_analysis.np04_ana import comes_from_channel
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.IPDict import IPDict


from waffles.np04_analysis.light_vs_hv.utils import check_endpoint_and_channel
from waffles.np04_analysis.light_vs_hv.utils import get_ordered_timestamps
from waffles.np04_analysis.light_vs_hv.utils import get_all_double_coincidences, get_all_coincidences, get_level_coincidences
from waffles.np04_analysis.light_vs_hv.utils import filter_not_coindential_wf
from waffles.np04_analysis.light_vs_hv.utils import from_generic
from waffles.np04_analysis.light_vs_hv.utils import start_plot, generic_plot_APA
from waffles.np04_analysis.light_vs_hv.utils import  my_sin, func_tau , calculate_light, birks_law

from waffles.np04_analysis.light_vs_hv.ZeroCrossingAna import ZeroCrossingAna

