import numpy as np
from pathlib import Path
from plotly import graph_objects as go
import plotly.subplots as psu
import logging
from typing import List, Union
from typing import Optional, Callable
import yaml
from importlib import resources

from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
from waffles.data_classes.UniqueChannel import UniqueChannel
from waffles.data_classes.BasicWfAna import BasicWfAna
from waffles.data_classes.CalibrationHistogram import CalibrationHistogram
from waffles.data_classes.IPDict import IPDict
from waffles.plotting.plot import plot_ChannelWsGrid, plot_CustomChannelGrid
from waffles.plotting.plot import plot_CalibrationHistogram
from waffles.utils.fit_peaks.fit_peaks import fit_peaks_of_CalibrationHistogram
from waffles.utils.baseline.baseline import SBaseline
from waffles.np02_data.ProtoDUNE_VD_maps import mem_geometry_map
from waffles.np02_data.ProtoDUNE_VD_maps import cat_geometry_map
from waffles.np02_utils.AutoMap import generate_ChannelMap, dict_uniqch_to_module

def np02_resolve_detectors(wfset, detectors: Union[List[str], List[UniqueChannel], List[Union[UniqueChannel, str]]], rows=0, cols=1) -> dict[str, ChannelWsGrid]:
    """
    Resolve the detectors and generate grids for the given waveform set.
    Parameters
    ----------
    wfset: WaveformSet
    detectors: List[str] | List[UniqueChannel] | List[UniqueChannel | str]
        List of detectors to resolve.
    rows: int, optional
    cols: int, optional
        Number of rows and columns for the grid.
    Returns
    -------
    dict[str, ChannelWsGrid]
        Dictionary containing the grids for the specified detectors.
    """

    detmap = generate_ChannelMap(channels=detectors, rows=rows, cols=cols)
    return dict(
        Custom=ChannelWsGrid(detmap, wfset)
    )


def np02_gen_grids(wfset, detector: Union[str, List[str], List[UniqueChannel], List[Union[UniqueChannel, str ]]] = "VD_Cathode_PDS", rows=0, cols=0) -> dict[str, ChannelWsGrid]:
    """
    Generate grids for the given waveform set and detector(s).
    Parameters
    ----------
    wfset: WaveformSet
    detector: str | List[str] | List[UniqueChannel] | List[UniqueChannel | str], optional
    Returns
    -------
    dict[str, ChannelWsGrid]
        Dictionary containing the grids for the specified detector(s).
    """

    if isinstance(detector, str):
        if detector == 'VD_Membrane_PDS':
            return dict(
                TCO=ChannelWsGrid(mem_geometry_map[2], wfset,
                                  bins_number=115,
                                  domain=np.array([-1e4, 5e4]),
                                  variable="integral")
                ,
                nTCO=ChannelWsGrid(mem_geometry_map[1], wfset,
                                   bins_number=115,
                                   domain=np.array([-1e4, 5e4]),
                                   variable="integral")

            )
        elif detector == 'VD_Cathode_PDS':
            return dict(
                TCO=ChannelWsGrid(cat_geometry_map[2], wfset,
                                  bins_number=115,
                                  domain=np.array([-1e4, 5e4]),
                                  variable="integral")
                ,
                nTCO=ChannelWsGrid(cat_geometry_map[1], wfset,
                                   bins_number=115,
                                   domain=np.array([-1e4, 5e4]),
                                   variable="integral")

            )
        else:
            detectors = [detector]
    else:
        detectors = detector
    if isinstance(detectors, list):
        return np02_resolve_detectors(wfset, detectors, rows, cols)

    raise ValueError(f"Could not resolve detector: {detector} or {detectors}")

def plot_detectors(wfset: WaveformSet, detector:list, plot_function: Optional[Callable] = None, **kwargs):
    for n, g in np02_gen_grids(wfset, detector).items():
        # Keeping standard plotting 
        if n == "nTCO" or n == "TCO":
            if "shared_xaxes" not in kwargs:
                kwargs["shared_xaxes"] = True
            if "shared_yaxes" not in kwargs:
                kwargs["shared_yaxes"] = True

        plot_grid(chgrid=g, title=n, html=kwargs.pop("html", None), detector=detector, plot_function=plot_function, **kwargs)


def plot_grid(chgrid: ChannelWsGrid, title:str = "", html: Union[Path, None] = None, detector: Union[str, List[str]] = "", plot_function: Optional[Callable] = None, **kwargs):

    rows, cols= chgrid.ch_map.rows, chgrid.ch_map.columns

    showplots = kwargs.pop("showplots", False)

    subtitles = chgrid.titles

    fig = psu.make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subtitles,
        shared_xaxes=kwargs.pop("shared_xaxes", False),
        shared_yaxes=kwargs.pop("shared_yaxes", False)
    )

    if plot_function is None:
        plot_ChannelWsGrid(chgrid,
                           figure=fig,
                           share_x_scale=kwargs.pop("share_x_scale", True),
                           share_y_scale=kwargs.pop("share_y_scale", True),
                           mode=kwargs.pop("mode", "overlay"),
                           wfs_per_axes=kwargs.pop("wfs_per_axes", 2000),
                           **kwargs
                           )
    else:
        plot_CustomChannelGrid(chgrid, plot_function, figure=fig, wf_func=kwargs.pop("doprocess", None), **kwargs)

    fig.update_layout(title=title, template="plotly_white",
                      width=1000, height=800, showlegend=True)
    if html:
        fig.write_html(html.as_posix())
        logging.info("💾 %s", html)
        if showplots:
            fig.show()
    else:
        fig.show()
# ╰────────────────────────────────────────────────────────────────────────────╯


def genhist(wfset:WaveformSet, figure:go.Figure, row, col):
    values = [wf.analyses["std"].result['integral'] for wf in wfset.waveforms]
    bins = np.linspace(-50e3, 50e3, 100)

    figure.add_trace(
        go.Histogram(
            x=values,
            xaxis='x',
            xbins=dict(start=bins[0], end=bins[-1], size=(bins[1] - bins[0])),
            autobinx=False,

        ),
        row=row, col=col
    )


def fithist(wfset:WaveformSet, figure:go.Figure, row, col, doprocess=lambda: False):

    params = ch_read_params()
    endpoint = wfset.waveforms[0].endpoint
    channel = wfset.waveforms[0].channel

    if endpoint not in params or channel not in params[endpoint]:
        raise ValueError(f"No parameters found for endpoint {endpoint} and channel {channel} in the configuration file.")

    
    if doprocess():
        runBasicWfAnaNP02(wfset,
                          int_ll=params[endpoint][channel]['fit'].get('int_ll', 254),
                          int_ul=params[endpoint][channel]['fit'].get('int_ul', 270),
                          amp_ll=params[endpoint][channel]['fit'].get('amp_ll', 250),
                          amp_ul=params[endpoint][channel]['fit'].get('amp_ul', 280),
                          show_progress=False
                          )

    bins_int = params[endpoint][channel]['fit'].get('bins_int', 100)
    domain_int_str = params[endpoint][channel]['fit'].get('domain_int', [-10e3, 100e3])
    domain_int = [float(x) for x in domain_int_str]
    domain_int = np.array(domain_int)

    max_peaks = params[endpoint][channel]['fit'].get('max_peaks', 3)
    prominence = params[endpoint][channel]['fit'].get('prominence', 0.15)
    half_point_to_fit = params[endpoint][channel]['fit'].get('half_point_to_fit', 2)
    initial_percentage = params[endpoint][channel]['fit'].get('initial_percentage', 0.15)
    percentage_step = params[endpoint][channel]['fit'].get('percentage_step', 0.05)


    hInt = CalibrationHistogram.from_WaveformSet(
        wfset,
        bins_number=bins_int,
        domain=domain_int,
        variable='integral',
        analysis_label = "std",
    )

    # This method in case histogram should cut average
    # average_hits = hInt.edges[:-1]*hInt.counts
    # average_hits = np.sum(average_hits)/np.sum(hInt.counts)

    # This method in case histogram should not cut average
    integral_sum = np.nansum( np.array([ wf.analyses["std"].result['integral'] for wf in wfset.waveforms ]) )
    n_integrals = np.sum( [1 if wf.analyses["std"].result['integral'] is not np.nan else 0 for wf in wfset.waveforms ] )
    average_hits = integral_sum / n_integrals if n_integrals > 0 else 0

    fit_hist = fit_peaks_of_CalibrationHistogram(
        hInt,
        max_peaks,
        prominence,
        half_point_to_fit,
        initial_percentage,
        percentage_step
    )
    fit_params = hInt.gaussian_fits_parameters

    zero_charge = fit_params['mean'][0][0]
    spe_charge = fit_params['mean'][1][0]
    baseline_stddev = fit_params['std'][0][0]
    spe_stddev = fit_params['std'][1][0]

    gain = spe_charge - zero_charge
    snr = gain / baseline_stddev

    plot_CalibrationHistogram(
        hInt,
        figure=figure,
        row=row, col=col,
        plot_fits=True,
        name=f"{dict_uniqch_to_module[str(UniqueChannel(wfset.waveforms[0].endpoint, wfset.waveforms[0].channel))]}; snr={snr:.2f}",
    )

    print(
        f"{list(wfset.runs)[0]},",
        f"{dict_uniqch_to_module[str(UniqueChannel(wfset.waveforms[0].endpoint, wfset.waveforms[0].channel))]},",
        f"{snr:.2f},",
        f"{gain:.2f},",
        f"{baseline_stddev:.2f},",
        f"{spe_stddev:.2f},",
        f"{average_hits/gain:.2f}",
    )

def runBasicWfAnaNP02(wfset: WaveformSet,
                      int_ll: int = 254,
                      int_ul: int = 270,
                      amp_ll: int = 250,
                      amp_ul: int = 280,
                      threshold: float = 25,
                      baselinefinish:int = 240,
                      minimumfrac: float = 0.67,
                      onlyoptimal: bool = True,
                      show_progress: bool = True
                      ):
    params = ch_read_params()
    baseline = SBaseline(threshold=threshold, baselinefinish=baselinefinish, default_filtering=2, minimumfrac=minimumfrac, data_base=params)

    ip = IPDict(
        baseline_method="SBaseline",      # ← NEW (or "Mean", "Fit", …)
        int_ll=int_ll, int_ul=int_ul,
        amp_ll=amp_ll, amp_ul=amp_ul,
        baseliner=baseline,
        baseline_limits=[0,240],
        onlyoptimal=onlyoptimal,
    )
    if show_progress: print("Processing waveform set with BasicWfAna")
    _ = wfset.analyse("std", BasicWfAna, ip,
                      analysis_kwargs={},
                      checks_kwargs=dict(points_no=wfset.points_per_wf),
                      overwrite=True,
                      show_progress=show_progress
                     )

def ch_read_params():
    try:
        with resources.files('waffles.np02_utils.data').joinpath('ch_snr_parameters.yaml').open('r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(e)
        print("\n\n")
        raise FileNotFoundError(
            "Could not find the ch_snr_parameters.json file in the waffles.np02_utils.PlotUtils.data package.\nWaffles should be installed with -e option to access this file.\n"
        )

