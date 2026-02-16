import waffles
import numpy as np
import TimeResolution as tr

from ROOT import TH2D, TH1D


def create_persistence_plot(waveforms: waffles.Waveform, tr: tr) -> TH2D:
    h2_persistence = TH2D('persistence', 'persistence', 1024//2, 0., 1024, 250, -tr.max_amplitude*0.4, tr.max_amplitude*1.1)
    
    wf_selected = np.array([ wf.adcs_float[::2] for wf in waveforms if wf.time_resolution_selection==True ])
    times = np.array([ np.linspace(0, 1024, 1024//2) for _ in wf_selected])

    wf_selected_flat = wf_selected.flatten()
    times_flat = times.flatten()

    for t, v in zip(times_flat, wf_selected_flat):
        h2_persistence.Fill(t, v)

    return h2_persistence


def build_max_position_histogram(max_positions: np.array,
                                 h_low: int,
                                 h_up:  int,
                                 multiplicity=1) -> TH1D:
    """
    Args:
    - nbins: Number of bins for the histogram.

    Returns:
    - h
    """
    # Create histogram
    h_pos = TH1D('h_pos', 'h_pos', (h_up-h_low)*multiplicity, h_low, h_up)
    for mp in max_positions:
        h_pos.Fill(mp)
    
    return h_pos
