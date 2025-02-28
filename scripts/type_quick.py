import numpy as np
import plotly.graph_objects as pgo
import waffles
import pickle
from waffles.plotting.plot import plot_ChannelWsGrid
import plotly.subplots as psu
import gc
import h5py
from waffles.data_classes.BasicWfAna import BasicWfAna
from waffles.data_classes.IPDict import IPDict

import waffles.np02_data.ProtoDUNE_VD_maps 
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
with h5py.File(f'/afs/cern.ch/work/a/arochefe/private/repositories/waffles/scripts/processed_merged_run_028676.hdf5', 'r')  as f:
    raw_wfset1=f['wfset'][:]
st_wfset1 = pickle.loads(raw_wfset1.tobytes())
print(type(st_wfset1)) 
with h5py.File(f'/afs/cern.ch/work/a/arochefe/private/repositories/waffles/src/waffles/np04_analysis/beam_example/data/wfset_028676_gzip.hdf5', 'r')  as f:
    raw_wfset=f['wfset'][:]
st_wfset = pickle.loads(raw_wfset.tobytes())
print(type(st_wfset)) 
