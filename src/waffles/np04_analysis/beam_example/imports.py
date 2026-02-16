import numpy as np
import pickle
import os
from pydantic import Field

from waffles.data_classes.WaveformSet import WaveformSet
from waffles.input_output.pickle_file_reader import WaveformSet_from_pickle_file
from waffles.input_output.raw_root_reader import BeamInfo_from_root_file
from waffles.input_output.event_file_reader import events_from_pickle_and_beam_files
from waffles.input_output.event_file_reader import events_from_pickle_file
from waffles.data_classes.WafflesAnalysis import WafflesAnalysis as WafflesAnalysis, BaseInputParams
from waffles.data_classes.IPDict import IPDict
from waffles.data_classes.BasicWfAna import BasicWfAna
