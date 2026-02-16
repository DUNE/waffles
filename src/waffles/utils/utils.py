from rich import print as print
<<<<<<< HEAD
=======
import numpy as np
from waffles.utils.numerical_utils import average_wf_ch
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid

import logging

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[1;31m', # Bold Red
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)

>>>>>>> main

def print_colored(string, color="white", styles=[]):
    '''
    Print a string in a specific styles

    Args:
        string (str):       string to be printed
        color  (str):       color to be used (default: white)
        styles (list(str)): list of styles to be used (i.e bold, underline, etc)
    '''
    
    from rich import print as print
    
    colors = { "DEBUG": 'magenta', "ERROR": 'red', "SUCCESS": 'green', "WARNING": 'yellow', "INFO": 'cyan' }
    if color in list(colors.keys()): color = colors[color]
    for style in styles: color += f' {style}'
    print(f'[{color}]{string}[/]')

<<<<<<< HEAD
=======
def compute_peaks_rise_fall_ch(wfset: WaveformSet):

    """
    Compute peak characteristics and rise/fall times for each channel in a ChannelWsGrid.

    For each valid channel in the provided grid, this function:
      - Computes the average waveform.
      - Finds the peak value and its index.
      - Calculates the rise time (10% → 90% of peak) and fall time (90% → 10% of peak).
      - Stores intermediate times corresponding to 10% and 90% amplitudes for rise and fall.

    Parameters
    ----------
    g : ChannelWsGrid
        A grid object containing waveform sets for multiple channels. 
        Channels not present in `dict_uniqch_to_module` or missing waveforms are skipped.

    Returns
    -------
    peaks_all : dict
        Dictionary with keys `(endpoint, channel)` and values as another dict containing:
            - "peak_index": int, index of the peak in the averaged waveform
            - "peak_time": int, time tick of the peak
            - "peak_value": float, value of the peak
            - "rise_time": float, duration from 10% to 90% of peak
            - "fall_time": float, duration from 90% to 10% of peak
            - "t_low": int, time of 10% of peak during rise
            - "t_high": int, time of 90% of peak during rise
            - "t_high_fall": int, time of 90% of peak during fall
            - "t_low_fall": int, time of 10% of peak during fall
            - "time": np.ndarray, array of time indices
            - "avg": np.ndarray, averaged waveform

    """

    peaks_all = {}
    for endpoint, wfbych in ChannelWsGrid.clusterize_waveform_set(wfset).items():
        for channel, wfch in wfbych.items(): 

            avg = average_wf_ch(wfch)
            time = np.arange(avg.size)

            peak_idx = np.argmax(avg)
            peak_value = avg[peak_idx]
            peak_time = time[peak_idx]

            amp_10 = 0.1 * peak_value
            amp_90 = 0.9 * peak_value

            t_low = -1
            t_high = -1
            for j in range(peak_idx + 1):
                if t_low < 0 and avg[j] >= amp_10:
                    t_low = time[j]
                if t_high < 0 and avg[j] >= amp_90:
                    t_high = time[j]
                    break
            rise_time = t_high - t_low

            t_high_fall = -1
            t_low_fall = -1
            for j in range(peak_idx, len(avg)):
                if t_high_fall < 0 and avg[j] <= amp_90:
                    t_high_fall = time[j]
                if t_low_fall < 0 and avg[j] <= amp_10:
                    t_low_fall = time[j]
                    break
            fall_time = t_low_fall - t_high_fall

            peaks_all[(endpoint, channel)] = {
                "peak_index": peak_idx,
                "peak_time": peak_time,
                "peak_value": peak_value,
                "rise_time": rise_time,
                "fall_time": fall_time,
                "t_low": t_low,
                "t_high": t_high,
                "t_high_fall": t_high_fall,
                "t_low_fall": t_low_fall,
                "time": time,   
                "avg": avg      
            }

    return peaks_all

>>>>>>> main
