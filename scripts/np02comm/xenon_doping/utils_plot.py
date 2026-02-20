import os
import numpy as np
from pathlib import Path
from typing import Union
from ConvFitterVDWrapper import ConvFitterVDWrapper
from waffles.data_classes.WaveformSet import WaveformSet
from DeconvFitterVDWrapper import DeconvFitterVDWrapper
from waffles.np02_utils.AutoMap import getModuleName
from waffles.utils.utils import print_colored
import matplotlib.pyplot as plt

from utils_deconv import process_deconvfit
from utils_conv import process_convfit

def plot_fit(wfset:WaveformSet, cfit: dict[int, dict[int,Union[DeconvFitterVDWrapper, ConvFitterVDWrapper]]], templates:dict[int, dict[int, np.ndarray]], responses: dict[int, dict[int, np.ndarray]], dofit=True, verbose=False, scan=8, slice_template:slice = slice(200,240), slice_response:slice = slice(None,54)):
    ep = wfset.waveforms[0].endpoint
    ch = wfset.waveforms[0].channel
    modulename = getModuleName(ep, ch)
    if ep not in cfit.keys() or ch not in cfit[ep].keys():
        return
    cfitch = cfit[ep][ch]

    oneexp = False
    if dofit:
        # isinstance does not work if we are using jupyter and editing the class
        if isinstance(cfitch, DeconvFitterVDWrapper):
            process_deconvfit(ep, ch, responses[ep][ch], templates[ep][ch], cfitch, print_flag=verbose, oneexp=oneexp, slice_template=slice_template, slice_response=slice_response)
        elif isinstance(cfitch, ConvFitterVDWrapper):
            process_convfit(ep, ch, responses[ep][ch], templates[ep][ch], cfitch, scan=scan, print_flag=verbose, oneexp=oneexp, slice_template=slice_template, slice_response=slice_response)
        else:
            print(type(cfitch))
            print_colored(f"Error: cfitch for {ep}-{ch} is not an instance of DeconvFitterVDWrapper or ConvFitterVDWrapper. Cannot fit.", "ERROR")
    elif verbose:
        print(cfitch.fit_results, '...')

    cfitch.plot()
    plt.title(f"{modulename}: {ep}-{ch}")
    plt.legend()
