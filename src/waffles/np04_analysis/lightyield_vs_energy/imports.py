import numpy as np
import click
import pickle
import sys
import os
import re
import json
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from tqdm import tqdm 
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages

from waffles.data_classes.IPDict import IPDict
from waffles.data_classes.BasicWfAna import BasicWfAna

from waffles.data_classes.WaveformSet import *
from waffles.data_classes.Waveform import *

from waffles.data_classes.WafflesAnalysis import WafflesAnalysis as WafflesAnalysis, BaseInputParams

#import waffles.input_output.raw_hdf5_reader as hdf5_reader 

 
from waffles.np04_data.ProtoDUNE_HD_APA_maps import APA_map


from pydantic import BaseModel, Field, conlist
from typing import Union, Literal