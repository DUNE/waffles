# --- IMPORTS -------------------------------------------------------
import waffles.input.raw_hdf5_reader as reader
import numpy as np
import waffles
import matplotlib.pyplot as plt
import os

# --- FUNCTIONS -----------------------------------------------------
def read_waveformset(filepath_folder: str, run: int, full_stat = True) -> waffles.WaveformSet:
    """
    Read the WaveformSet from the hdf5 file
    Parameters:
    - filepath_folder: str, folder where the file with the paths to the hdf5 files is located
    - run: int, run number
    - full_stat: bool, if True, merge all the waveform_set in the run
    """
    filepath_file = filepath_folder + "0" + str(run) + ".txt"
    filepath = reader.get_filepaths_from_rucio(filepath_file)

    if full_stat:
        wfset = reader.WaveformSet_from_hdf5_file(filepath[0])
        for fp in filepath[1:]:
            ws = reader.WaveformSet_from_hdf5_file(fp)
            wfset.merge(ws)
    else:
        wfset = reader.WaveformSet_from_hdf5_file(filepath[0])

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


def plot_heatmaps(wfs: waffles.Waveform, flag: str, run: int, vgain: int, ch: int, offline_ch: int) -> None:
    # Convert waveform data to numpy array
    if (flag == "baseline_removed"):
        raw_wf_arrays = np.array([wf.adcs_float for wf in wfs])
    else:
        raw_wf_arrays = np.array([wf.adcs for wf in wfs])

    # Create time arrays for plotting
    time_arrays = np.array([np.arange(1024) for _ in range(len(raw_wf_arrays))])

    # Flatten arrays for histogram
    time_flat = time_arrays.flatten()
    adc_flat = raw_wf_arrays.flatten()

    # Compute histogram
    h, xedges, yedges = np.histogram2d(
        time_flat, adc_flat,
        bins=(1024, int(np.max(adc_flat) - np.min(adc_flat))),
        range=[[0, 1023], [np.min(adc_flat), np.max(adc_flat)]]
    )

    # Replace zeros with NaN for better visualization
    h[h == 0] = np.nan

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    # x, y = np.meshgrid(xedges, yedges)
    pcm = ax.pcolormesh(xedges[:], yedges[:], h.T, shading='auto')

    # Labels and colorbar
    ax.set_ylabel("Amplitude [adcs]", fontsize=15)
    ax.set_xlabel("Time [ticks]", fontsize=15)
    cax = ax.inset_axes([1.002, 0., 0.02, 1.])
    plt.colorbar(pcm, ax=ax, cax=cax)
    plt.text(0.85, 0.85, f'Run {run}, Channel {ch}', transform=ax.transAxes, fontsize=15)

    # Save figure instead of displaying it
    output_dir = "heatmaps/{flag}"
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    filename = os.path.join(output_dir, f"heatmap_run_{run}_vgain_{vgain}_ch_{ch}_offline_{offline_ch}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close figure to save memory

    print(f"Saved heatmap: {filename}")
