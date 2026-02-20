import numpy as np
import yaml
from pathlib import Path
from DeconvFitterVDWrapper import DeconvFitterVDWrapper
from waffles.np02_utils.AutoMap import getModuleName
from waffles.utils.time_align_utils import find_threshold_crossing
from dataclasses import dataclass, field, asdict
from waffles.utils.utils import print_colored

from utils import update_values_from_yaml

@dataclass
class DeconvWrapperParams:
    scinttype:str='lar'
    error:float=0.05
    filter_type:str='Gauss'
    cutoff_MHz:float=2.5
    dtime:float=16

class DeconvFitParams:
    """
    Parameters for the convolution fit. Do not change these values, create a yaml instead.
    """
    def __init__(self, response:str, template:str):
        self.response = response
        self.template = template
        self.deconvparams = asdict(DeconvWrapperParams())
        self.print_flag:bool = False 
        self.slice_template:slice = slice(200,240)
        self.slice_response:slice = slice(None,54) 
    
    def update_values_from_yaml(self, fileparams:Path, response:str, template:str):

        allparams, slice_template, slice_response = update_values_from_yaml(fileparams, response, template)
        if slice_template is not None:
            self.slice_template = slice_template
        if slice_response is not None:
            self.slice_response = slice_response
        for k, v in allparams.items():
            if k in self.__dict__.keys():
                setattr(self, k, v)
            else:
                print_colored(f"Warning: {k} is not a valid parameter for ConvFitParams. Ignoring it.", 'WARNING')

def process_deconvfit(ep:int,
                      ch:int,
                      response:np.ndarray,
                      template:np.ndarray,
                      cfitch:DeconvFitterVDWrapper,
                      print_flag:bool=False,
                      oneexp:bool=False,
                      slice_template:slice = slice(200,240),
                      slice_response:slice = slice(None,54)
                      ):

    modulename = getModuleName(ep, ch)
    print(f"Processing {ep}-{ch}: {modulename}")

    if modulename[:2] == "M7" and cfitch.scinttype != "xe":
        oneexp = True

    wvft = (template).astype(np.float32)
    wvfr = (response).astype(np.float32)
    if slice_template.start != 0 and slice_template.stop != 0: # If both are zero, no re-baseline
        wvft = wvft - np.mean(wvft[slice_template])
    if slice_response.start != 0 and slice_response.stop != 0: # if both are zero, no re-baseline
        wvfr = wvfr - np.mean(wvfr[slice_response])

    cfitch.set_template_waveform(wvft)
    cfitch.set_response_waveform(wvfr)
    cfitch.generate_deconvolved_signal()
    
    cfitch.fit(oneexp=oneexp, print_flag=print_flag)

