import os
import numpy as np
import pickle
from matplotlib import pyplot as plt
from scipy import signal as spsi
from pydantic import Field, field_validator
from waffles.data_classes.WafflesAnalysis import WafflesAnalysis, BaseInputParams
from waffles.data_classes.WaveformAdcs import WaveformAdcs
import waffles.input_output.raw_hdf5_reader as reader
import waffles.np04_analysis.example_analysis.utils as wnu
import waffles.core.utils as wcu