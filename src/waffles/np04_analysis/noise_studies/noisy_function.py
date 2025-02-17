
import waffles.input_output.raw_hdf5_reader as reader
import numpy as np
import os
import waffles

# --- FUNCTIONS -----------------------------------------------------
def read_waveformset(filepath_folder: str,
                     run: int, 
                     full_stat = True,
                     filetxt_exists: bool,
                     filehdf5_exists: bool
                     ) -> waffles.WaveformSet:
    """
    Read the WaveformSet from the hdf5 file
    Parameters:
    - filepath_folder: str, folder where the file with the paths to the hdf5 files is located
    - run: int, run number
    - full_stat: bool, if True, merge all the waveform_set in the run
    """
    filepath_file = filepath_folder + "0" + str(run) + ".txt"
    # check if the file exists
    if not os.path.isfile(filepath_file):
        print(f"File {filepath_file} does not exist")
        filetxt_exists = False
        return waffles.WaveformSet()

    filepath = reader.get_filepaths_from_rucio(filepath_file)

    if (full_stat == True and len(filepath) > 1):
        if not os.path.isfile(filepath[0]):
            print(f"File {filepath[0]} does not exist")
            filehdf5_exists = False
            return waffles.WaveformSet()
        wfset = reader.WaveformSet_from_hdf5_file(filepath[0], erase_filepath = False)
        for fp in filepath[1:]:
            ws = reader.WaveformSet_from_hdf5_file(fp, erase_filepath = False)
            wfset.merge(ws)
    else:
        if not os.path.isfile(filepath[0]):
            print(f"File {filepath[0]} does not exist")
            filehdf5_exists = False
            return waffles.WaveformSet()
        wfset = reader.WaveformSet_from_hdf5_file(filepath[0], erase_filepath = False)

    return wfset


def allow_ep_wfs(waveform: waffles.Waveform, endpoint) -> bool:
    """
    Function to filter the WaveformSet by endpoint
    Parameters:
    - waveform: waffles.Waveform
    - endpoint: int, endpoint number
    """
    return waveform.endpoint == endpoint


def allow_channel_wfs(waveform: waffles.Waveform, channel: int) -> bool:
    """
    Function to filter the WaveformSet by channel
    Parameters:
    - waveform: waffles.Waveform
    - channel: int, channel number
    """
    return waveform.channel == channel


def create_float_waveforms(waveforms: waffles.Waveform) -> None:
    """
    Convert the waveform.adcs from int to np.float64, creating a new attribute adcs_float
    Parameters:
    - waveforms: waffles.Waveform
    """
    for wf in waveforms:
        wf.adcs_float = wf.adcs.astype(np.float64)


def sub_baseline_to_wfs(waveforms: waffles.Waveform, prepulse_ticks: int):
    """
    Subtract the baseline from the waveforms and invert the signal
    Parameters:
    - waveforms: waffles.Waveform
    - prepulse_ticks: int, number of ticks to calculate the baseline
    """
    norm = 1./prepulse_ticks
    for wf in waveforms:
        baseline = np.sum(wf.adcs_float[:prepulse_ticks])*norm
        wf.adcs_float -= baseline
        wf.adcs_float *= -1
