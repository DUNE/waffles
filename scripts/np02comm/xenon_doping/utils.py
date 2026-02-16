import numpy as np
import copy
from waffles.data_classes.WaveformSet import WaveformSet
from ConvFitterVDWrapper import ConvFitterVDWrapper
from waffles.np02_utils.AutoMap import dict_uniqch_to_module, strUch
import matplotlib.pyplot as plt


DEFAULT_RESPONSE:str = "average_waveforms"
DEFAULT_TEMPLATE:str = "templates_large_pulses"
DEFAULT_CONV_NAME:str = "convfit_results"
PATH_XE_AVERAGES:str = "/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/xenon_averages/"


def process_convfit(response:np.ndarray, template:np.ndarray, cfitch:ConvFitterVDWrapper, scan:int=8, print_flag:bool=False, oneexp:bool=False, dofit:bool=True, verbose:bool=False):
    wvft = (template).astype(np.float32)
    wvfr = (response).astype(np.float32)
    wvft = wvft - np.mean(wvft[200:240])
    wvfr = wvfr - np.mean(wvfr[:54])

    cfitch.set_template_waveform(wvft)
    cfitch.set_response_waveform(wvfr)
    cfitch.prepare_waveforms()
    if dofit:
        cfitch.fit(scan=8, print_flag=verbose, oneexp=oneexp)
    if not dofit and verbose:
        print(cfitch.fit_results)

    cfitch.set_template_waveform(template)
    cfitch.set_response_waveform(response)
    cfitch.prepare_waveforms()
    if dofit:
        cfitch.fit(scan=scan, print_flag=print_flag, oneexp=oneexp)
    if not dofit and print_flag:
        print(cfitch.fit_results)


def plot_convfit(wfset:WaveformSet, cfit: dict[int, dict[int,ConvFitterVDWrapper]], templates:dict[int, dict[int, np.ndarray]], responses: dict[int, dict[int, np.ndarray]], dofit=True, verbose=False):
    ep = wfset.waveforms[0].endpoint
    ch = wfset.waveforms[0].channel
    modulename = dict_uniqch_to_module[strUch(ep,ch)]
    print(f"Processing {ep}-{ch}: {modulename}")
    cfitch = cfit[ep][ch]

    oneexp = False
    if modulename[:2] == "M7" and cfitch.scinttype != "xe":
        oneexp = True

    process_convfit(responses[ep][ch], templates[ep][ch], cfitch, scan=8, print_flag=verbose, oneexp=oneexp, dofit=dofit)

    cfitch.plot()
    plt.title(f"{modulename}: {ep}-{ch}")
    plt.legend()

