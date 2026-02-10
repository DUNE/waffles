import numpy as np
import copy
from glob import glob
from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.UniqueChannel import UniqueChannel
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
from waffles.input_output.hdf5_structured import load_structured_waveformset
from waffles.np02_utils.AutoMap import dict_uniqch_to_module, ordered_channels_membrane
from waffles.np02_utils.load_utils import open_processed

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def remove_weird_channels(waveform: Waveform, validchannels = []) -> bool:
    if waveform.channel not in validchannels:
        return False
    
    return True

def process_by_file(file:str, channels = None, endpoints:int = 106, nwaveforms=None, verbose=True, subwaves=2000, restsub=False, rates_results={}, nrecords={}, allpeaks={}):
    wfset_full = load_structured_waveformset(file, max_to_load=nwaveforms, channels_filter= channels, endpoint_filter=endpoints, verbose=verbose)

    if endpoints == 107:
        print(wfset_full)
        wfset_full= WaveformSet.from_filtered_WaveformSet(wfset_full, remove_weird_channels, ordered_channels_membrane, show_progress=True)
        print(wfset_full)


    wfset_peaks = WaveformSet.from_filtered_WaveformSet(wfset_full, apply_full_peak_search, show_progress=True)
    get_peaks_info(wfset_peaks, nrecords, allpeaks, endpoint=endpoints)
    if restsub:
        wfsetsub = WaveformSet(*wfset_full.waveforms[:subwaves]) # create a subset for quick access...
        return wfsetsub
    else:
        return None


def load_data_search_peaks(run, dettype, datadir, endpoint, nwaveforms=None, subwaves=2000, file_slice=slice(None, None), rates_results={}):
    channels=None

    files = glob(f"{datadir}/processed/run{run:06d}_{dettype}/processed_*_run{run:06d}_*_{dettype}.hdf5")
    files = files[file_slice]
    print("List of files found:")
    print(files)
    

    wfsetsub: WaveformSet = None # type: ignore
    nrecords = {}
    allpeaks = {}
    for i, file in enumerate(files):
        retsub = True
        if i > 0: 
            retsub = False
        tmp = process_by_file(file, channels, endpoint, nwaveforms, verbose=True, subwaves=subwaves, restsub=retsub, rates_results=rates_results, nrecords=nrecords, allpeaks=allpeaks) 
        if retsub and tmp is not None:
            wfsetsub = tmp

    compute_all_peaks(wfsetsub, nrecords, allpeaks, endpoint, binswidth_ticks=1024, rates_results=rates_results)

    return wfsetsub
    
def get_peaks_info(wfset: WaveformSet, nrecords:dict, allpeaks:dict, endpoint=106):
    wfsetch = ChannelWsGrid.clusterize_waveform_set(wfset)[endpoint]
    for ch, wfs in wfsetch.items():
        nrecords[ch] = nrecords.get(ch, 0) + len(set([ wf.record_number for wf in wfs.waveforms ]))
        allpeaks_tmp = [ np.array(wf.peaks)[:,0] for wf in wfs.waveforms[:] if len(wf.peaks) ]
        allpeaks_tmp = [ v for x in allpeaks_tmp for v in x ]
        allpeaks[ch] = allpeaks.get(ch, []) + allpeaks_tmp

def compute_all_peaks(wfset: WaveformSet, nrecords:dict, allpeaks:dict, endpoint=106, binswidth_ticks=1024, rates_results={}):
    wfsetch = ChannelWsGrid.clusterize_waveform_set(wfset)[endpoint]
    
    for ch, wfs in wfsetch.items():
        binswidth_ticks = 1024
        bmin = -1250
        bmax = 93750
        if min(allpeaks[ch]) < -20000 - 1024:
            bmin = -187500
            bmax = 187500
        bins = np.arange(bmin, bmax, binswidth_ticks)
        
        counts, bins = np.histogram(allpeaks[ch], bins=bins, weights=np.ones_like(allpeaks[ch])/nrecords[ch])
        rates_results[ch] = rates_results.get(ch, {})
        rates_results[ch][list(wfs.runs)[0]] = (counts, bins)
    
def plot_counts(wfset: WaveformSet, run:int, rates_results:dict={}):
    
    counts, bins = rates_results[wfset.waveforms[0].channel][run]
    module = f"{dict_uniqch_to_module[str(UniqueChannel(wfset.waveforms[0].endpoint, wfset.waveforms[0].channel))]}"
    plt.hist(bins[:-1], bins=bins, weights=counts, histtype='step')
    plt.ylabel(f'#peaks / event / {bins[1]-bins[0]} ticks', fontsize=16)
    plt.xlabel('Ticks', fontsize=16)
    plt.title(module, fontsize=14)

    
    # plt.yscale('log')
    # plt.ylim(0.03, 0.11)


def plot_ratio(wfset:WaveformSet, rates_results = {}, run_more = 0, run_less= 0):
    counts1, bins1 = rates_results[wfset.waveforms[0].channel][run_more]
    counts2, bins2 = rates_results[wfset.waveforms[0].channel][run_less]

    bin_width = bins1[1] - bins1[0]
    
    # find where bins1 starts and ends inside bins2
    start_idx = int(round((bins1[0] - bins2[0]) / bin_width))
    end_idx   = start_idx + len(counts1)
    
    # slice counts2 to match counts1
    counts2_slice = counts2[start_idx:end_idx]
    
    # safe division
    ratio = np.divide(
        counts1,
        counts2_slice,
        out=np.zeros_like(counts1, dtype=float),
        where=counts2_slice != 0
    )
    ratio = counts1 - counts2_slice
    module = f"{dict_uniqch_to_module[str(UniqueChannel(wfset.waveforms[0].endpoint, wfset.waveforms[0].channel))]}"
    plt.hist(bins1[:-1], bins1, weights=ratio, histtype='step')
    plt.ylabel(f'ratio', fontsize=16)
    plt.xlabel('Ticks', fontsize=16)
    plt.title(module, fontsize=14)

def plot_runs(wfset:WaveformSet, rates_results = {}, runs_to_plot=[]):
    ratesinfo = rates_results.get(wfset.waveforms[0].channel, {})
    if not ratesinfo:
        return
    if not runs_to_plot:
        runs_to_plot = sorted(list(ratesinfo.keys()))

    countsref, binsref = ratesinfo[runs_to_plot[0]]
    
    bin_width = binsref[1] - binsref[0]
    
    
    for run in runs_to_plot:
        counts, bins = ratesinfo[run]

        if bins[0] == binsref[0]:
            counts_plot = counts
        else:
            # find where bins1 starts and ends inside bins2
            start_idx = int(round((binsref[0] - bins[0]) / bin_width))
            end_idx   = start_idx + len(countsref)
            # slice counts2 to match counts1
            counts_plot = counts[start_idx:end_idx]
        
        module = f"{dict_uniqch_to_module[str(UniqueChannel(wfset.waveforms[0].endpoint, wfset.waveforms[0].channel))]}"
        plt.hist(binsref[:-1], binsref, weights=counts_plot, histtype='step', label=run)
        plt.ylabel(f'#peaks / event / {bins[1]-bins[0]} ticks', fontsize=16)
        plt.xlabel('Ticks', fontsize=16)
        plt.title(module, fontsize=14)

        handles, labels = plt.gca().get_legend_handles_labels()
        new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]
        
        plt.legend(handles=new_handles, labels=labels)




# This function makes first derivative, then two moving averages and the second derivative.
# This all done at once because cathode data is too heavy. 
def apply_full_peak_search(waveform: Waveform, mult_factor=10, width=12, deriv_threshold=20, too_big=1000, jump=1000) -> bool:
    data = derivative(waveform.adcs, mult_factor=mult_factor) 
    data = moving_average(data, width=width)
    data = moving_average(data, width=width)
    data_2 = derivative(data, mult_factor=mult_factor)
    peaks = find_peaks_from_derivative_2(
        derivative_1=data,
        derivative_2=data_2,
        timestamp=waveform.timestamp,
        daq_window_timestamp=waveform.daq_window_timestamp,
        deriv_threshold=deriv_threshold,
        too_big=too_big,
        jump=jump,
    )
    waveform.peaks = peaks # type: ignore
    return True

    

def apply_derivativeWf(waveform: Waveform, mult_factor=10, again=False) -> bool:
    if not again:
        waveform.derivative = derivative(waveform.adcs, mult_factor=mult_factor) # type: ignore
    else:
        waveform.derivative_2 = derivative(waveform.derivative, mult_factor=mult_factor) # type: ignore
    return True

def derivative(raw: np.ndarray, mult_factor=12) -> np.ndarray:
    adcs = np.astype(raw, np.float32)
    derivative = np.zeros_like(raw, dtype=np.float32) 
    derivative[1:-1] = mult_factor*(adcs[2:] - adcs[:-2])*0.5 
    return derivative

def apply_moving_averageWf(waveform: Waveform, width=12) -> bool:
    waveform.derivative = moving_average(waveform.derivative, width=width) # type: ignore
    return True

def moving_average(data: np.ndarray, width=10) -> np.ndarray:
    if width % 2 == 0:
        width += 1

    n = len(data)
    midpoint = (width - 1) // 2

    result = np.zeros_like(data, dtype=np.float32)
    kernel = np.ones(width) / width

    conv = np.convolve(data, kernel, mode='same')

    result[midpoint:n - midpoint] = conv[midpoint:n - midpoint]
    return result




def apply_peak_search_from_derivative_2(
    waveform: Waveform,
    deriv_threshold=20,
    too_big=1000,
    jump=1000,
) -> bool:
    peaks = find_peaks_from_derivative_2(
        derivative_1=waveform.derivative, # type: ignore
        derivative_2=waveform.derivative_2, # type: ignore
        timestamp=waveform.timestamp,
        daq_window_timestamp=waveform.daq_window_timestamp,
        deriv_threshold=deriv_threshold,
        too_big=too_big,
        jump=jump,
    )
    waveform.peaks = peaks # type: ignore
    return True
   

def find_peaks_from_derivative_2(
    derivative_1: np.ndarray,
    derivative_2: np.ndarray,
    timestamp: int,
    daq_window_timestamp: int,
    deriv_threshold=20,
    too_big=1000,
    jump=1000,
    fall_time_jump = 100,
) -> np.ndarray:

    derivative_1 = derivative_1

    sign = np.sign(derivative_2)

    pos_to_neg = np.where((sign[:-1] > 0) & (sign[1:] <= 0))[0]
    
    if not len(pos_to_neg):
      return np.array([]) # type: ignore
        
    peaks = []
    skip_until = -1
    skip_until_fall = -1

    for peak_idx in pos_to_neg:
        if peak_idx < skip_until:
            continue
        if peak_idx < skip_until_fall:
            continue
            
        peak_val = derivative_1[peak_idx]
        
        if peak_val <= deriv_threshold:
            continue
        
        # dynamic jump
        if peak_val > too_big:
            skip_until = peak_idx + jump
            continue

        skip_until_fall = peak_idx + fall_time_jump
        peak_idx = peak_idx + (timestamp - daq_window_timestamp)
        peaks.append((peak_idx, peak_val))

    return np.array(peaks)
   














# 3x slower, but uses only first derivative 
def find_peaks_from_derivative(
    waveform: Waveform,
    deriv_threshold=20,
    too_big=1000,
    jump=1000,
) -> bool:
    d = waveform.derivative # type: ignore

    sign = np.sign(d)

    neg_to_pos = np.where((sign[:-1] <= 0) & (sign[1:] > 0))[0]
    pos_to_neg = np.where((sign[:-1] > 0) & (sign[1:] <= 0))[0]
    if not len(neg_to_pos) or not len(pos_to_neg):
        waveform.peaks =  np.array([]) # type: ignore
        return True
    if pos_to_neg[0] < neg_to_pos[0]: # starts going down..
        pos_to_neg = pos_to_neg[1:]

    peaks = []
    skip_until = -1

    for start, end in zip(neg_to_pos, pos_to_neg):
        if start < skip_until:
            continue
        rel = np.argmax(d[start:end+1])
        # peak location in ADC
        peak_idx = start + rel
        peak_val = d[peak_idx]
        
        # derivative strength filter
        try: 
            if peak_val <= deriv_threshold:
                continue
        except:
            print(neg_to_pos)
            print(pos_to_neg)
            print(start, end, len(d))
            print(waveform)
            waveform.peaks =  peaks # type: ignore
            return True
            raise Exception("...")

        
        # dynamic jump
        if peak_val > too_big:
            skip_until = peak_idx + jump

        peak_idx = peak_idx + (waveform.timestamp - waveform.daq_window_timestamp)
        peaks.append((peak_idx, peak_val))


    waveform.peaks =  peaks # type: ignore
    return True
