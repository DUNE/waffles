from my_BasicWfAna import my_BasicWfAna # Modified the baseline value!!!

import numpy as np


import sys
waffles_dir = '/afs/cern.ch/user/a/anbalbon/waffles'
sys.path.append(waffles_dir+'/src') 

import waffles.data_classes.Waveform
import waffles.input.raw_root_reader as reader
from waffles.data_classes.IPDict import IPDict
from waffles.data_classes.BasicWfAna import BasicWfAna


def my_filter(waveform : waffles.Waveform, endpoint : int = 112) -> bool:
    if (waveform.endpoint == endpoint) :
        return True
    else:
        return False