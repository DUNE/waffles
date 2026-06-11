import os
import numpy as np
from typing import Union
from pathlib import Path
import yaml
from scipy.signal import lfilter

from waffles.utils.utils import print_colored
from DeconvFitterVDWrapper import DeconvFitterVDWrapper




DEFAULT_RESPONSE:str = "average_waveforms"
DEFAULT_TEMPLATE:str = "templates_large_pulses"
DEFAULT_ANA_NAME:str = "analysis_results"
PATH_XE_AVERAGES:str = "/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/xenon_averages/"
PATH_XE_OUTPUTS:str = "/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/xenon_outputs/"



def make_standard_analysis_name(outputdir: Path, analysisname:str) -> str:

    # Ensures same standard for the analysis name, so it can be easily identified and sorted in the output directory
    if '-' in analysisname:
        check_name = analysisname.split('-')
        expected_user_name = check_name[-1]
        if expected_user_name != os.getlogin():
            if Path( outputdir / analysisname ).exists():
                print_colored(f"Warning: analysis name {analysisname} ends with {expected_user_name}, which does not match the current user {os.getlogin()}. Please, be sure you are not doing any mistake.", "WARNING")
            else:
                print_colored(f"As a python script... I don't know if the argument after '-' is another user or not. So I am just replacing it by underscore and adding your user. Feel free to change the directory name later if you need to.", "WARNING")
                analysisname = analysisname.replace("-", "_")
                analysisname = analysisname + '-' + os.getlogin()
    else:
        analysisname = analysisname + '-' + os.getlogin()
    return analysisname


def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def makeslice(slicevalues:dict) -> slice:
    t0 = slicevalues['t0']
    t0 = int(t0) if isinstance(t0, int) else None
    tf = slicevalues['tf']
    tf = int(tf) if isinstance(tf, int) else None
    return slice(t0,tf)


def high_pass_filter_scipy(raw, frequency_cut_MHz, dt_ns):
    RC = 1.0 / (2 * np.pi * frequency_cut_MHz)
    alpha = RC / (RC + dt_ns*1e-3)
    b = [alpha, -alpha]
    a = [1, -(alpha)]
    return lfilter(b, a, raw)

def setup_response_template(
                    response:np.ndarray,
                    template:np.ndarray,
                    cfitch:DeconvFitterVDWrapper,
                    slice_template:slice = slice(200,240),
                    slice_response:slice = slice(None,54),
                    high_pass_freq_MHz:float = 0,
                    ):

    wvft = (template).astype(np.float32)
    wvfr = (response).astype(np.float32)
    if high_pass_freq_MHz > 0:
        wvft:ndarray = high_pass_filter_scipy(wvft, high_pass_freq_MHz, 16) # type: ignore
        wvfr:ndarray = high_pass_filter_scipy(wvfr, high_pass_freq_MHz, 16) # type: ignore
    if slice_template.start != 0 and slice_template.stop != 0: # If both are zero, no re-baseline
        wvft = wvft - np.mean(wvft[slice_template])
    if slice_response.start != 0 and slice_response.stop != 0: # if both are zero, no re-baseline
        wvfr = wvfr - np.mean(wvfr[slice_response])

    cfitch.set_template_waveform(wvft) 
    cfitch.set_response_waveform(wvfr)


def update_values_from_yaml(fileparams:Path, response:str, template:str):

    with open(fileparams, 'r') as f:
        allparams = yaml.safe_load(f)
        # Those are the only ones that can cause problems...
        slice_template = None
        slice_response = None
        if 'slice_template' in allparams.keys():
            slice_template = makeslice(allparams.pop('slice_template'))
        if 'slice_response' in allparams.keys():
            slice_response = makeslice(allparams.pop('slice_response'))

        if 'response' in allparams.keys():
            if response != allparams['response']:
                print_colored(f"Warning: response in {fileparams.as_posix()} is different from the one specified in the arguments. Using the one in the arguments ({response}).", 'WARNING')
        if 'template' in allparams.keys():
            if template != allparams['template']:
                print_colored(f"Warning: template in {fileparams.as_posix()} is different from the one specified in the arguments. Using the one in the arguments ({template}).", 'WARNING')
        return allparams, slice_template, slice_response

