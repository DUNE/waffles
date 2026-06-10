import numpy as np
import yaml
from pathlib import Path
from DeconvFitterVDWrapper import DeconvFitterVDWrapper
from waffles.np02_utils.LArXeFitUtils import FitInitParams
from waffles.np02_utils.AutoMap import getModuleName
from dataclasses import dataclass, field, asdict
from waffles.utils.utils import print_colored

from utils import update_values_from_yaml, setup_response_template

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
                print_colored(f"Warning: {k} is not a valid parameter for DeconvFitterVDWrapper. Ignoring it.", 'WARNING')

def process_deconvfit(ep:int,
                      ch:int,
                      response:np.ndarray,
                      template:np.ndarray,
                      cfitch:DeconvFitterVDWrapper,
                      print_flag:bool=False,
                      oneexp:bool=False,
                      slice_template:slice = slice(200,240),
                      slice_response:slice = slice(None,54),
                      high_pass_freq_MHz:float = 0.5,
                      dofit:bool = True 
                      ):

    modulename = getModuleName(ep, ch)
    if dofit:
        print(f"Processing {ep}-{ch}: {modulename}")

    # if modulename[:2] == "M7" and cfitch.scinttype != "xe":
    #     oneexp = True
    if ep == 110:
        high_pass_freq_MHz = 0.0 # No high pass filter for PMTs
    setup_response_template(response, template, cfitch, slice_template, slice_response, high_pass_freq_MHz)
    if ep != 110:
        cfitch.generate_deconvolved_signal()
    else:
        cfitch.deconvolved = cfitch.response.copy()

    if not dofit:
        return

    init = FitInitParams.for_larxe() if cfitch.scinttype == 'xe' else FitInitParams.for_lar_oneexp() if oneexp else FitInitParams.for_lar()
    if modulename[:2] == "M7" and cfitch.scinttype == "xe":
        init.t3 = 2800
    
    cfitch.fit(oneexp=oneexp, print_flag=print_flag, init=init)

