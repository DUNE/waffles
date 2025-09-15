import numpy as np
import pickle
import os
from pydantic import Field, field_validator

from waffles.data_classes.WaveformSet import WaveformSet
from waffles.input_output.pickle_file_reader import WaveformSet_from_pickle_file
from waffles.np04_analysis.tau_slow_convolution.extractor_waveforms import Extractor
from waffles.np04_data.tau_slow_runs.load_runs_csv import ReaderCSV
from waffles.data_classes.WafflesAnalysis import WafflesAnalysis as WafflesAnalysis, BaseInputParams
<<<<<<< HEAD
from waffles.np04_analysis.tau_slow_convolution.ConvFitter import ConvFitter
=======
from waffles.np04_analysis.tau_slow_convolution.ConvFitterHDWrapper import ConvFitterHDWrapper
>>>>>>> 264bdce2c6b35b5dd071455c3cbe62221217a107

from waffles.utils.baseline.baseline import SBaseline
import waffles.core.utils as wcu


