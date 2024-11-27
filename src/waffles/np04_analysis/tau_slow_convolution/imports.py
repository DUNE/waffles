import numpy as np
import pickle
import os
import argparse

from waffles.data_classes.WaveformSet import WaveformSet
from waffles.input.pickle_file_reader import WaveformSet_from_pickle_file
from waffles.np04_analysis.tau_slow_convolution.extractor_waveforms import Extractor
from waffles.np04_data.tau_slow_runs.load_runs_csv import ReaderCSV
from waffles.data_classes.WafflesAnalysis import WafflesAnalysis as WafflesAnalysis
from waffles.np04_analysis.tau_slow_convolution.ConvFitter import ConvFitter

from waffles.utils.baseline.baseline import SBaseline


