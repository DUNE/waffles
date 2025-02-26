import os
import time
import pickle
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as spsi

from pydantic import Field, field_validator
from waffles.data_classes.WafflesAnalysis import WafflesAnalysis, BaseInputParams
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.WaveformAdcs import WaveformAdcs
from waffles.Exceptions import GenerateExceptionMessage

import waffles.input_output.raw_hdf5_reader as reader
import waffles.np04_analysis.example_analysis.utils as wnu
import waffles.np04_analysis.tp.utils as tp_utils
import waffles.core.utils as wcu
from waffles.input_output.persistence_utils import WaveformSet_to_file,WaveformSet_from_hdf5_pickle