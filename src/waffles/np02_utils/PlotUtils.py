import numpy as np
from pathlib import Path
from plotly import graph_objects as go
import plotly.subplots as psu
import logging
from typing import List, Union
from typing import Optional, Callable
import yaml
from importlib import resources
import matplotlib.pyplot as plt

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
from waffles.np02_utils.AutoMap import generate_ChannelMap, dict_uniqch_to_module, dict_module_to_uniqch, ordered_modules_cathode, ordered_modules_membrane, strUch


tol_colors = [
    "#E41A1C",  # red
    "#377EB8",  # blue
    "#4DAF4A",  # green
    "#984EA3",  # purple
    "#FF7F00",  # orange
    "#FFFF33",  # yellow
    "#A65628",  # brown
    "#F781BF"   # pink
]
modules_colormap = {module: tol_colors[int(module[1:-3])-1] for module in ordered_modules_cathode + ordered_modules_membrane}
endpoint_channel_colormap = {}
for module, uniqch in dict_module_to_uniqch.items():
    endpoint_channel_colormap[uniqch.endpoint] = endpoint_channel_colormap.get(uniqch.endpoint, {})
    endpoint_channel_colormap[uniqch.endpoint][uniqch.channel] = modules_colormap[module]

style_map_matplotlib = {
    "1": {"linestyle": "-",  "lw":2, "alpha": 1.0},
    "2": {"linestyle": "--", "lw":2, "alpha": 0.8}
}

def get_style_channel(endpoint: int, channel:int) -> dict:
    modulename = dict_uniqch_to_module[strUch(endpoint,channel)]
    return get_style_module(modulename)

def get_style_module(modulename:str) -> dict:
    ch_num = modulename[3:-1] # "1" or "2"
    return {
        "color": modules_colormap[modulename],
        **style_map_matplotlib[ch_num]
    }


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
    for title, g in np02_gen_grids(wfset, detector, rows=kwargs.pop("rows", 0), cols=kwargs.pop("cols", 0)).items():
        # Keeping standard plotting 
        if title == "nTCO" or title == "TCO":
            if "shared_xaxes" not in kwargs:
                kwargs["shared_xaxes"] = True
            if "shared_yaxes" not in kwargs:
                kwargs["shared_yaxes"] = True

        plot_grid(chgrid=g, title=title, html=kwargs.pop("html", None), detector=detector, plot_function=plot_function, **kwargs)


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

    width = kwargs.pop("width", 1000)
    height = kwargs.pop("height", 800)
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
        plot_CustomChannelGrid(chgrid, plot_function, figure=fig, wf_func=kwargs.pop("wf_func", None), **kwargs)

    title = title if title != "Custom" else ""
    fig.update_layout(title=title, template="plotly_white",
                      width=width, height=height, showlegend=True)
    fig.update_annotations(
        font=dict(size=14),
        align="center",
    )
    if html:
        fig.write_html(html.as_posix())
        logging.info("💾 %s", html)
        if showplots:
            fig.show()
    else:
        fig.show()
# ╰────────────────────────────────────────────────────────────────────────────╯


def genhist(wfset:WaveformSet, figure:go.Figure, row, col, wf_func = None):
    variabletype = wf_func.get('variable', 'integral') if wf_func is not None else 'integral'
    values = [wf.analyses["std"].result[variabletype] for wf in wfset.waveforms]
    bins = np.linspace(-50e3, 50e3, 100)
    if wf_func is not None:
        if 'bins' in wf_func:
            bins = wf_func['bins']
    counts, edges = np.histogram(values, bins=bins)

    figure.add_trace(
        go.Scatter(  
            x=edges,
            y=counts,
            mode='lines',
            line=dict(  
                color='black', 
                width=0.5,
                shape='hv'),
        ),
        row=row, col=col
    )


def runBasicWfAnaNP02Updating(wfset: WaveformSet, updatethreshold:bool, show_progress: bool, params: dict = {}, configyaml = ""):
    endpoint = wfset.waveforms[0].endpoint
    channel = wfset.waveforms[0].channel
    if not params:
        configyaml = configyaml if configyaml else 'ch_snr_parameters.yaml'
        params = ch_read_params(filename=configyaml)

    if endpoint not in params or channel not in params[endpoint]:
        raise ValueError(f"No parameters found for endpoint {endpoint} and channel {channel} in the configuration file.")

    if(len(wfset.available_channels[list(wfset.runs)[0]][endpoint]) > 1):
        raise ValueError(f"Should have only one channel in the waveform set...")

    if updatethreshold:
        run = list(wfset.runs)[0]
        threshold = params['updates'][endpoint][channel].get(run, 0)
        if threshold != 0:
            params[endpoint][channel]['baseline']['threshold'] = threshold
    
    runBasicWfAnaNP02(wfset,
                      int_ll=params[endpoint][channel]['fit'].get('int_ll', 254),
                      int_ul=params[endpoint][channel]['fit'].get('int_ul', 270),
                      amp_ll=params[endpoint][channel]['fit'].get('amp_ll', 254),
                      amp_ul=params[endpoint][channel]['fit'].get('amp_ul', 270),
                      show_progress=show_progress,
                      configyaml=params
                      )

def fithist(wfset:WaveformSet, figure:go.Figure, row, col, wf_func = {}):
    doprocess = wf_func.get("doprocess", True)
    dofit = wf_func.get("dofit", True)
    normalize_hist = wf_func.get("normalize_hist", False)
    variable = wf_func.get('variable', 'integral')
    show_progress = wf_func.get('show_progress', False)
    params = ch_read_params(filename=wf_func.get('configyaml', 'ch_snr_parameters.yaml'))
    
    endpoint = wfset.waveforms[0].endpoint
    channel = wfset.waveforms[0].channel

    if endpoint not in params or channel not in params[endpoint]:
        raise ValueError(f"No parameters found for endpoint {endpoint} and channel {channel} in the configuration file.")

    if(len(wfset.available_channels[list(wfset.runs)[0]][endpoint]) > 1):
        raise ValueError(f"Should have only one channel in the waveform set...")

    update_threshold = wf_func.get("update_threshold", False)
    
    if doprocess:
        # params get updated inide here
        runBasicWfAnaNP02Updating(
            wfset,
            updatethreshold=update_threshold,
            show_progress=show_progress,
            params=params
        )

    bins_int = params[endpoint][channel]['fit'].get('bins_int', 100)
    domain_int_str = params[endpoint][channel]['fit'].get('domain_int', [-10e3, 100e3])


    domain_int = [float(x) for x in domain_int_str]
    domain_int = np.array(domain_int)
    if 'bins' in wf_func:
        tbins = wf_func['bins']
        bins_int = len(tbins)
        domain_int = np.array([tbins[0], tbins[-1]])

    max_peaks = params[endpoint][channel]['fit'].get('max_peaks', 3)
    prominence = params[endpoint][channel]['fit'].get('prominence', 0.15)
    half_point_to_fit = params[endpoint][channel]['fit'].get('half_point_to_fit', 2)
    initial_percentage = params[endpoint][channel]['fit'].get('initial_percentage', 0.15)
    percentage_step = params[endpoint][channel]['fit'].get('percentage_step', 0.05)

    hInt = CalibrationHistogram.from_WaveformSet(
        wfset,
        bins_number=bins_int,
        domain=domain_int,
        variable=variable,
        analysis_label = "std",
        normalize_histogram=normalize_hist
    )

    if not dofit:
        plot_CalibrationHistogram(
            hInt,
            figure=figure,
            row=row, col=col,
            plot_fits=False,
            name=f"{dict_uniqch_to_module[str(UniqueChannel(wfset.waveforms[0].endpoint, wfset.waveforms[0].channel))]}",
        )
        return

    # This method in case histogram should cut average
    # average_hits = hInt.edges[:-1]*hInt.counts
    # average_hits = np.sum(average_hits)/np.sum(hInt.counts)

    # This method in case histogram should not cut average
    integral_sum = np.nansum( np.array([ wf.analyses["std"].result[variable] for wf in wfset.waveforms ]) )
    n_integrals = np.sum( [1 if wf.analyses["std"].result[variable] is not np.nan else 0 for wf in wfset.waveforms ] )
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

    zero_charge = 0
    spe_charge = 0
    baseline_stddev = 0
    spe_stddev = 0

    gain = 0
    snr = 0
    errgain=0
    try:
        zero_charge = fit_params['mean'][0][0]
        zero_charge_err = fit_params['mean'][0][1]
        spe_charge = fit_params['mean'][1][0]
        spe_charge_err = fit_params['mean'][1][1]
        baseline_stddev = abs(fit_params['std'][0][0])
        spe_stddev = fit_params['std'][1][0]

        gain = spe_charge - zero_charge
        errgain = np.sqrt( zero_charge_err**2 + spe_charge_err**2 )
        snr = gain / baseline_stddev
    except:
        print(f"Could not fit for {dict_uniqch_to_module[str(UniqueChannel(wfset.waveforms[0].endpoint, wfset.waveforms[0].channel))]}")

    plot_CalibrationHistogram(
        hInt,
        figure=figure,
        row=row, col=col,
        plot_fits=True,
        name=f"{dict_uniqch_to_module[str(UniqueChannel(wfset.waveforms[0].endpoint, wfset.waveforms[0].channel))]}; snr={snr:.2f}",
        showfitlabels=False,
    )

    if snr != 0:
        for wf in wfset.waveforms:
            wf.analyses["std"].result['snr'] = snr
            wf.analyses["std"].result['normalization'] = hInt.normalization
            wf.analyses["std"].result['gain'] = gain
            wf.analyses["std"].result['errgain'] = errgain
        print(
            f"{list(wfset.runs)[0]},",
            # f"{dict_uniqch_to_module[str(UniqueChannel(wfset.waveforms[0].endpoint, wfset.waveforms[0].channel))]},",
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
                      show_progress: bool = True,
                      configyaml:Union[str,dict] = 'ch_snr_parameters.yaml'
                      ):
    
    params = {}
    if isinstance(configyaml, dict):
        params = configyaml
    elif isinstance(configyaml, str):
        if configyaml is not None and configyaml != "":
            params = ch_read_params(configyaml)
    baseline = SBaseline(threshold=threshold, baselinefinish=baselinefinish, default_filtering=2, minimumfrac=minimumfrac, data_base=params)

    ip = IPDict(
        baseline_method="SBaseline",      # ← NEW (or "Mean", "Fit", …)
        int_ll=int_ll, int_ul=int_ul,
        amp_ll=amp_ll, amp_ul=amp_ul,
        baseliner=baseline,
        baseline_limits=[0,240],
        onlyoptimal=onlyoptimal,
        amplitude_method="max_minus_baseline"
    )
    if show_progress: print("Processing waveform set with BasicWfAna")
    _ = wfset.analyse("std", BasicWfAna, ip,
                      analysis_kwargs={},
                      checks_kwargs=dict(points_no=wfset.points_per_wf),
                      overwrite=True,
                      show_progress=show_progress
                     )

def ch_read_params(filename:str = 'ch_snr_parameters.yaml') -> dict:
    try:
        with resources.files('waffles.np02_utils.data').joinpath(filename).open('r') as f:
            return yaml.safe_load(f)
    except Exception as error:
        print(error)
        print("\n\n")
        raise FileNotFoundError(
            f"Could not find the {filename} file in the waffles.np02_utils.PlotUtils.data package.\nWaffles should be installed with -e option to access this file.\n"
        )

