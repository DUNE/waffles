import numpy as np
import yaml
from pathlib import Path
from waffles.data_classes.WaveformSet import WaveformSet
from ConvFitterVDWrapper import ConvFitterVDWrapper
from waffles.np02_utils.AutoMap import getModuleName
import matplotlib.pyplot as plt

from dataclasses import dataclass, field, asdict

from utils import print_colored, makeslice

@dataclass
class ConvWrapperParams:
    threshold_align_template:float=0.27
    threshold_align_response:float=0.2
    error:float=10
    dointerpolation:bool=True
    interpolation_factor:int=8
    align_waveforms:bool=True
    dtime:int=16
    convtype:str='fft'
    usemplhep:bool=True
    scinttype:str='lar'

class ConvFitParams:
    """
    Parameters for the convolution fit. Do not change these values, create a yaml instead.
    """
    def __init__(self, response:str, template:str):
        self.response = response
        self.template = template
        self.cfitparams = asdict(ConvWrapperParams())
        self.scan:int = 8
        self.print_flag:bool = False 
        self.slice_template:slice = slice(200,240)
        self.slice_response:slice = slice(None,54) 
    
    def update_values_from_yaml(self, fileparams:Path, response:str, template:str):

        with open(fileparams, 'r') as f:
            allparams = yaml.safe_load(f)
            # Those are the only ones that can cause problems...
            if 'slice_template' in allparams.keys():
                self.slice_template = makeslice(allparams.pop('slice_template'))
            if 'slice_response' in allparams.keys():
                self.slice_response = makeslice(allparams.pop('slice_response'))

            if 'response' in allparams.keys():
                if response != allparams['response']:
                    print_colored(f"Warning: response in {fileparams.as_posix()} is different from the one specified in the arguments. Using the one in the arguments ({response}).", 'WARNING')
            if 'template' in allparams.keys():
                if template != allparams['template']:
                    print_colored(f"Warning: template in {fileparams.as_posix()} is different from the one specified in the arguments. Using the one in the arguments ({template}).", 'WARNING')

            for k, v in allparams.items():
                if k in self.__dict__.keys():
                    setattr(self, k, v)
                else:
                    print_colored(f"Warning: {k} is not a valid parameter for ConvFitParams. Ignoring it.", 'WARNING')


def process_convfit(ep:int,
                    ch:int,
                    response:np.ndarray,
                    template:np.ndarray,
                    cfitch:ConvFitterVDWrapper,
                    scan:int=8,
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
    cfitch.prepare_waveforms()
    cfitch.fit(scan=scan, print_flag=print_flag, oneexp=oneexp)



def plot_convfit(wfset:WaveformSet, cfit: dict[int, dict[int,ConvFitterVDWrapper]], templates:dict[int, dict[int, np.ndarray]], responses: dict[int, dict[int, np.ndarray]], dofit=True, verbose=False, scan=8, slice_template:slice = slice(200,240), slice_response:slice = slice(None,54)):
    ep = wfset.waveforms[0].endpoint
    ch = wfset.waveforms[0].channel
    modulename = getModuleName(ep, ch)
    if ep not in cfit.keys() or ch not in cfit[ep].keys():
        return
    cfitch = cfit[ep][ch]

    oneexp = False
    if dofit:
        process_convfit(ep, ch, responses[ep][ch], templates[ep][ch], cfitch, scan=scan, print_flag=verbose, oneexp=oneexp, slice_template=slice_template, slice_response=slice_response)
    elif verbose:
        print(cfitch.fit_results, '...')

    cfitch.plot()
    plt.title(f"{modulename}: {ep}-{ch}")
    plt.legend()
