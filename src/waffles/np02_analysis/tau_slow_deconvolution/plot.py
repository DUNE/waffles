import numpy as np
import plotly.graph_objects as pgo
import plotly.subplots as psu
import h5py

from waffles.input_output.hdf5_structured import load_structured_waveformset
from waffles.data_classes.BasicWfAna import BasicWfAna
from waffles.data_classes.IPDict import IPDict
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
from waffles.np02_data.ProtoDUNE_VD_maps import mem_geometry_map
from waffles.plotting.plot import plot_ChannelWsGrid

def create_channel_grids(wfset, bins=115, domain=(-10000., 50000.)):
    """Creates TCO and non-TCO ChannelWsGrid dictionaries from a WaveformSet."""
    return {
        "TCO": ChannelWsGrid(
            mem_geometry_map[2],
            wfset,
            compute_calib_histo=False,
            bins_number=bins,
            domain=np.array(domain),
            variable='integral',
            analysis_label=''
        ),
        "nTCO": ChannelWsGrid(
            mem_geometry_map[1],
            wfset,
            compute_calib_histo=False,
            bins_number=bins,
            domain=np.array(domain),
            variable='integral',
            analysis_label=''
        ),
    }


def plot_single_grid(grid, title="Grid Plot", save_path=None):
    """Plots a single ChannelWsGrid in overlay mode."""
    figure = psu.make_subplots(rows=4, cols=2)
    plot_ChannelWsGrid(
        figure=figure,
        channel_ws_grid=grid,
        share_x_scale=True,
        share_y_scale=True,
        mode='overlay',
        wfs_per_axes=50
    )
    figure.update_layout(
        title={'text': title, 'font': {'size': 24}},
        width=1000,
        height=800,
        template="plotly_white",
        showlegend=True
    )
    if save_path:
        figure.write_html(str(save_path))
        print(f"Saved: {save_path}")
    else:
        figure.show()


def plot_processed_file(path, max_waveforms=2000, label="standard"):
    """Load, analyze and plot waveform data from a structured HDF5 file."""
    print(f"Loading structured data from: {path}")
    wfset = load_structured_waveformset(path, max_waveforms=max_waveforms)
    grids = create_channel_grids(wfset)
    plot_single_grid(grids["TCO"], title="TCO")
    plot_single_grid(grids["nTCO"], title="nTCO")

"""
plot_processed_file("data/led.hdf5")
plot_processed_file("data/cosmic.hdf5")
plot_processed_file("data/noise.hdf5")
plot_processed_file("data/led.hdf5", max_waveforms=2000, label="standard")
"""
