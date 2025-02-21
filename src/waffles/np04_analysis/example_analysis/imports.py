import os
import numpy as np
import pickle
from matplotlib import pyplot as plt
from scipy import signal as spsi
import waffles.input_output.WaveformAdcs as WaveformAdcs
import waffles.input_output.raw_hdf5_reader as reader
import waffles.np04_analysis.example_analysis.utils as utils