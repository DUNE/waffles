#!/usr/bin/env python3

import gc
import numpy as np
from pathlib import Path
import plotly.subplots as psu
from waffles.np02_data.ProtoDUNE_VD_maps import mem_geometry_tco, mem_geometry_nontco
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
from waffles.data_classes.BasicWfAna import BasicWfAna
from waffles.data_classes.IPDict import IPDict
from waffles.plotting.plot import plot_ChannelWsGrid
from waffles.input_output.hdf5_structured import load_structured_waveformset

# ------------------------------------------------------------------------------
# SINGLE-RUN UTILS
# ------------------------------------------------------------------------------

def print_waveform_timing_info(wfset):
    """Logs min/max timestamp and the approximate time delta."""
    timestamps = [wf.timestamp for wf in wfset.waveforms]
    if not timestamps:
        print("No waveforms found!")
        return
    a, b = np.min(timestamps), np.max(timestamps)
    print(f"Min timestamp: {a}, Max timestamp: {b}, Δt: {b - a} ticks")
    print(f"Δt in seconds: {(b - a) * 16e-9:.3e} s")
    print(f"Light travels: {(3e5)*(b - a)*16e-9:.2f} Km (approx)")

def analyze_waveforms(wfset, label="standard", starting_tick=50, width=70):
    """Performs a basic waveform analysis."""
    baseline_limits = [0, 50, 900, 1000]
    input_params = IPDict(
        baseline_limits=baseline_limits,
        int_ll=starting_tick,
        int_ul=starting_tick + width,
        amp_ll=starting_tick,
        amp_ul=starting_tick + width
    )
    checks_kwargs = dict(points_no=wfset.points_per_wf)

    print("Running waveform analysis...")
    wfset.analyse(
        label,
        BasicWfAna,
        input_params,
        *[],
        analysis_kwargs={},
        checks_kwargs=checks_kwargs,
        overwrite=True
    )

def create_channel_grids(wfset, bins=115, domain=(-10000., 50000.)):
    """Creates TCO and non-TCO ChannelWsGrid dictionaries from a WaveformSet."""
    return {
        "TCO": ChannelWsGrid(
            mem_geometry_tco,
            wfset,
            compute_calib_histo=False,
            bins_number=bins,
            domain=np.array(domain),
            variable='integral',
            analysis_label=''
        ),
        "nTCO": ChannelWsGrid(
            mem_geometry_nontco,
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

def plot_single_run(filepath, max_waveforms=2000, label="standard"):
    """
    Loads, analyzes, and plots TCO & nTCO from a single structured HDF5 file.
    Modify if you prefer 'average' or 'heatmap' instead of 'overlay'.
    """
    print(f"Loading single run from: {filepath}")
    wfset = load_structured_waveformset(filepath, max_waveforms=max_waveforms)
    print_waveform_timing_info(wfset)
    analyze_waveforms(wfset, label=label)
    grids = create_channel_grids(wfset)

    # Plot TCO
    plot_single_grid(grids["TCO"], title=f" TCO")
    # Plot nTCO
    plot_single_grid(grids["nTCO"], title=f" nTCO")

# ------------------------------------------------------------------------------
# MULTI-RUN UTILS
# ------------------------------------------------------------------------------

def compare_runs_side_by_side(
    filepaths,
    label="standard",
    max_waveforms=2000,
    starting_tick=50,
    width=70,
    bins=115,
    domain=(-10000., 50000.),
    mode='average'
):
    """
    Loads multiple runs (structured HDF5 waveforms), analyzes each, and plots TCO vs. non-TCO
    side by side in a single figure. Each run is on its own row.

    Args:
        filepaths (List[str]): List of structured HDF5 paths to compare.
        label (str): Analysis label for waveform analysis.
        max_waveforms (int): Limit waveforms to load per run for speed.
        starting_tick (int): Baseline and integration start for the analysis.
        width (int): Integration window for the analysis.
        bins (int): Bins for creating ChannelWsGrid.
        domain (tuple): Domain for creating ChannelWsGrid (e.g. integral range).
        mode (str): Plot mode: 'overlay' | 'average' | 'heatmap' | 'calibration'.

    Returns:
        fig (plotly.graph_objects.Figure): A figure with rows=len(filepaths), cols=2.
          Each row: TCO in col=1, nTCO in col=2.
    """
    # Make a figure with 2 columns (TCO, nTCO), row per file
    fig = psu.make_subplots(
        rows=len(filepaths),
        cols=2,
        shared_xaxes=False,
        shared_yaxes=False,
        subplot_titles=[f"Run {i+1} TCO" for i in range(len(filepaths))]
                      + [f"Run {i+1} non-TCO" for i in range(len(filepaths))]
    )

    for i, fp in enumerate(filepaths):
        print(f"=== Loading run {i+1} from: {fp}")
        wfset = load_structured_waveformset(fp, max_waveforms=max_waveforms)
        print_waveform_timing_info(wfset)
        analyze_waveforms(wfset, label=label, starting_tick=starting_tick, width=width)
        grids = create_channel_grids(wfset, bins=bins, domain=domain)

        # Plot TCO in col=1, row=i+1
        plot_ChannelWsGrid(
            channel_ws_grid=grids["TCO"],
            figure=fig,
            mode=mode,
            row=i+1,
            col=1,
            share_x_scale=True,
            share_y_scale=True
        )

        # Plot non-TCO in col=2, row=i+1
        plot_ChannelWsGrid(
            channel_ws_grid=grids["nTCO"],
            figure=fig,
            mode=mode,
            row=i+1,
            col=2,
            share_x_scale=True,
            share_y_scale=True
        )

        # Annotate each row with the filename
        fig.add_annotation(
            x=0.5,
            y=1.1,
            xref="x domain",
            yref="y domain",
            text=f"File: {Path(fp).name}",
            showarrow=False,
            row=i+1, col=1,
            font=dict(size=12, color="gray")
        )

    fig.update_layout(
        title="Comparison of Multiple Runs: TCO vs. nTCO",
        height=600 * len(filepaths),  # each row ~600px
        width=1200,
        template="ggplot2"
    )
    return fig

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
if __name__ == "__main__":

    filepaths = ["/nfs/home/marroyav/waffles/scripts/processed_np02vd_raw_run035845_0000_df-s04-d0_dw_0_20250415T202538.hdf5_structured.hdf5",
                 "/nfs/home/marroyav/hdf5_update/scripts/processed_np02vd_raw_run035804_0000_df-s04-d0_dw_0_20250410T175432.hdf5_structured.hdf5",
        "/nfs/home/marroyav/hdf5_update/scripts/processed_np02vd_raw_run035799_0000_df-s04-d0_dw_0_20250410T171546.hdf5_structured.hdf5",
        "/nfs/home/marroyav/hdf5_update/scripts/processed_np02vd_raw_run035797_0000_df-s04-d0_dw_0_20250410T165146.hdf5_structured.hdf5",
        "/nfs/home/marroyav/hdf5_update/scripts/output_07/processed_np02vd_raw_run035730_0001_df-s04-d0_dw_0_20250404T145714.hdf5_structured.hdf5",
        "/nfs/home/marroyav/hdf5_update/scripts/output_07/processed_np02vd_raw_run035734_0000_df-s04-d0_dw_0_20250404T151840.hdf5_structured.hdf5",
        "/nfs/home/marroyav/hdf5_update/scripts/output_07/processed_np02vd_raw_run035737_0000_df-s04-d0_dw_0_20250405T205029.hdf5_structured.hdf5",
        "/nfs/home/marroyav/hdf5_update/scripts/output_07/processed_np02vd_raw_run035738_0000_df-s04-d0_dw_0_20250405T210441.hdf5_structured.hdf5",
        "/nfs/home/marroyav/hdf5_update/scripts/output_07/processed_np02vd_raw_run035739_0000_df-s04-d0_dw_0_20250405T211434.hdf5_structured.hdf5",
        "/nfs/home/marroyav/hdf5_update/scripts/output_07/processed_np02vd_raw_run035740_0000_df-s04-d0_dw_0_20250405T214417.hdf5_structured.hdf5",
        "/nfs/home/marroyav/hdf5_update/scripts/output_07/processed_np02vd_raw_run035744_0000_df-s04-d0_dw_0_20250405T223721.hdf5_structured.hdf5",
        "/nfs/home/marroyav/hdf5_update/scripts/output_07/processed_np02vd_raw_run035745_0000_df-s04-d0_dw_0_20250405T224204.hdf5_structured.hdf5",
        "/nfs/home/marroyav/hdf5_update/scripts/output_07/processed_np02vd_raw_run035746_0000_df-s04-d0_dw_0_20250405T224650.hdf5_structured.hdf5",
        "/nfs/home/marroyav/hdf5_update/scripts/output_07/processed_np02vd_raw_run035747_0000_df-s04-d0_dw_0_20250405T225146.hdf5_structured.hdf5",
        "/nfs/home/marroyav/hdf5_update/scripts/output_07/processed_np02vd_raw_run035750_0000_df-s04-d0_dw_0_20250407T101151.hdf5_structured.hdf5",
        "/nfs/home/marroyav/hdf5_update/scripts/output_07/processed_np02vd_raw_run035751_0000_df-s04-d0_dw_0_20250407T101757.hdf5_structured.hdf5",

    ]
    # EXAMPLE usage: single run
    single_path = filepaths[0]
    plot_single_run(single_path)

    # EXAMPLE usage: multiple runs

    # fig = compare_runs_side_by_side(filepaths, mode='average')
    # fig.show()

    gc.collect()
