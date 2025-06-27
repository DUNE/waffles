from pydantic import Field
from waffles.data_classes.WafflesAnalysis import WafflesAnalysis, BaseInputParams

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit

import waffles
from waffles.input_output.hdf5_structured import load_structured_waveformset
from waffles.data_classes.WaveformSet import WaveformSet