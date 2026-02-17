import numpy as np
import yaml
from pathlib import Path
from waffles.data_classes.WaveformSet import WaveformSet
from ConvFitterVDWrapper import ConvFitterVDWrapper
from waffles.np02_utils.AutoMap import dict_uniqch_to_module, strUch
import matplotlib.pyplot as plt

from dataclasses import dataclass, field, asdict

from utils import print_colored, makeslice

@dataclass
class ConvWrapperParams:
    threshold_align_template=0.27,
    threshold_align_response=0.2,
    error=10,
    dointerpolation=True,
    interpolation_factor=8,
    align_waveforms = True,
    dtime=16,
    convtype='fft',
    usemplhep=True,
    scinttype='lar',

class ConvFitParams:
    """
    Parameters for the convolution fit. Do not change these values, create a yaml instead.
    """
    def __init__(self, response:str, template:str):
        self.response = response
        self.template = template
        self.cfitparams:dict = asdict(ConvWrapperParams())
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
                    dofit:bool=True,
                    slice_template:slice = slice(200,240),
                    slice_response:slice = slice(None,54)
                    ):

    modulename = dict_uniqch_to_module[strUch(ep,ch)]

    if modulename[:2] == "M7" and cfitch.scinttype != "xe":
        oneexp = True
    print(slice_template, slice_response)
    wvft = (template).astype(np.float32)
    wvfr = (response).astype(np.float32)
    wvft = wvft - np.mean(wvft[slice_template])
    wvfr = wvfr - np.mean(wvfr[slice_response])

    cfitch.set_template_waveform(wvft)
    cfitch.set_response_waveform(wvfr)
    cfitch.prepare_waveforms()
    if dofit:
        cfitch.fit(scan=scan, print_flag=print_flag, oneexp=oneexp)
    if not dofit and print_flag:
        print(cfitch.fit_results)



def plot_convfit(wfset:WaveformSet, cfit: dict[int, dict[int,ConvFitterVDWrapper]], templates:dict[int, dict[int, np.ndarray]], responses: dict[int, dict[int, np.ndarray]], dofit=True, verbose=False, scan=8, slice_template:slice = slice(200,240), slice_response:slice = slice(None,54)):
    ep = wfset.waveforms[0].endpoint
    ch = wfset.waveforms[0].channel
    modulename = dict_uniqch_to_module[strUch(ep,ch)]
    print(f"Processing {ep}-{ch}: {modulename}")
    cfitch = cfit[ep][ch]

    oneexp = False
    process_convfit(ep, ch, responses[ep][ch], templates[ep][ch], cfitch, scan=scan, print_flag=verbose, oneexp=oneexp, dofit=dofit, slice_template=slice_template, slice_response=slice_response)

    cfitch.plot()
    plt.title(f"{modulename}: {ep}-{ch}")
    plt.legend()
