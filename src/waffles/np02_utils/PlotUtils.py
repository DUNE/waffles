import numpy as np
from pathlib import Path
from plotly import graph_objects as go
import plotly.subplots as psu
import logging
from typing import List, Union
from typing import Optional, Callable
import matplotlib.pyplot as plt
from typing import Optional
import os

from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
from waffles.data_classes.UniqueChannel import UniqueChannel
from waffles.data_classes.BasicWfAna import BasicWfAna
from waffles.data_classes.CalibrationHistogram import CalibrationHistogram
from waffles.data_classes.IPDict import IPDict
from waffles.plotting.plot import plot_ChannelWsGrid, plot_CustomChannelGrid
from waffles.plotting.plot import plot_CalibrationHistogram
from waffles.utils.fit_peaks.fit_peaks import fit_peaks_of_CalibrationHistogram
from waffles.utils.baseline.baseline import SBaseline
from waffles.utils.numerical_utils import average_wf_ch, compute_peaks_rise_fall_ch
from waffles.np02_data.ProtoDUNE_VD_maps import mem_geometry_map
from waffles.np02_data.ProtoDUNE_VD_maps import cat_geometry_map
from waffles.np02_utils.AutoMap import generate_ChannelMap, dict_uniqch_to_module, dict_module_to_uniqch, ordered_modules_cathode, ordered_modules_membrane, strUch
from waffles.np02_utils.load_utils import ch_read_params

import waffles.Exceptions as we
import matplotlib.pyplot as plt


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
    figs = []
    return_figs = kwargs.get("return_fig", False)
    if return_figs:
        kwargs["showplots"] = False
        kwargs["html"] = kwargs.get("html", None)
        
    for title, g in np02_gen_grids(wfset, detector, rows=kwargs.pop("rows", 0), cols=kwargs.pop("cols", 0)).items():
        # Keeping standard plotting 
        if title == "nTCO" or title == "TCO":
            if "shared_xaxes" not in kwargs:
                kwargs["shared_xaxes"] = True
            if "shared_yaxes" not in kwargs:
                kwargs["shared_yaxes"] = True
 
        
        ret = plot_grid(chgrid=g, title=title, html=kwargs.pop("html", None), detector=detector, plot_function=plot_function, **kwargs)
        if return_figs:
            fig, rows, cols = ret
            figs.append((fig, rows, cols, title, g))

    if return_figs:
        return figs
    return None


def plot_grid(chgrid: ChannelWsGrid, title:str = "", html: Union[Path, None] = None, detector: Union[str, List[str]] = "", plot_function: Optional[Callable] = None, return_fig: bool = False, **kwargs):

    rows, cols= chgrid.ch_map.rows, chgrid.ch_map.columns

    showplots = kwargs.pop("showplots", True)

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
        if showplots:
            fig.show()

    if return_fig:
        return fig, rows, cols
    return None
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


def __update_dict(dictd, dictup, key):
        dictd[key] = dictup.get(key, dictd[key])

def runBasicWfAnaNP02Updating(wfset: WaveformSet, updatethreshold:bool, show_progress: bool, params: dict = {}, configyaml = "", doprocess:bool = True, onlyoptimal=True, onlyonerun=True):
    endpoint = wfset.waveforms[0].endpoint
    channel = wfset.waveforms[0].channel
    if not params:
        configyaml = configyaml if configyaml else 'ch_snr_parameters.yaml'
        params = ch_read_params(filename=configyaml)

    if endpoint not in params or channel not in params[endpoint]:
        print(f"No parameters found for endpoint {endpoint} and channel {channel} in the configuration file.")

    if onlyonerun and (len(wfset.available_channels[list(wfset.runs)[0]][endpoint]) > 1):
        raise ValueError(f"Should have only one channel in the waveform set...")

    if updatethreshold:
        run = list(wfset.runs)[0]
        dictbaseline = params[endpoint][channel]['baseline']
        dictfit = params[endpoint][channel]['fit']
        if run in params['updates'][endpoint][channel]:
            dictupdate = params['updates'][endpoint][channel][run]
            __update_dict(dictbaseline, dictupdate, 'threshold')
            __update_dict(dictbaseline, dictupdate, 'default_filtering')
            __update_dict(dictfit, dictupdate, 'bins_int')
            __update_dict(dictfit, dictupdate, 'domain_int')
            __update_dict(dictfit, dictupdate, 'max_peaks')
            __update_dict(dictfit, dictupdate, 'prominence')
            __update_dict(dictfit, dictupdate, 'half_point_to_fit')
    
    if doprocess:
        runBasicWfAnaNP02(wfset,
                          int_ll=params[endpoint][channel]['fit'].get('int_ll', 254),
                          int_ul=params[endpoint][channel]['fit'].get('int_ul', 270),
                          amp_ll=params[endpoint][channel]['fit'].get('amp_ll', 254),
                          amp_ul=params[endpoint][channel]['fit'].get('amp_ul', 270),
                          show_progress=show_progress,
                          onlyoptimal=onlyoptimal,
                          configyaml=params
                          )

def process_by_channel(
        wfset: WaveformSet,
        configyaml: str = 'ch_snr_parameters.yaml',
        updatethreshold: bool = False,
        show_progress: bool = True,
        onlyoptimal: bool = True
    ):

    params = ch_read_params(filename=configyaml)

    wfsetch = ChannelWsGrid.clusterize_waveform_set(wfset)

    for ep, ch_dict in wfsetch.items():
        for ch, wfset_channel in ch_dict.items():
            runBasicWfAnaNP02Updating(
                wfset_channel,
                updatethreshold=updatethreshold,
                show_progress=show_progress,
                doprocess=True,
                onlyoptimal=onlyoptimal,
                params=params
            )

def fithist(wfset:WaveformSet, figure:go.Figure, row, col, wf_func = {}):

    if len(list(wfset.runs)) > 1:
        raise ValueError(f"Should have only one run in the waveform set...")

    doprocess = wf_func.get("doprocess", True)
    dofit = wf_func.get("dofit", True)
    normalize_hist = wf_func.get("normalize_hist", False)
    variable = wf_func.get('variable', 'integral')
    show_progress = wf_func.get('show_progress', False)
    fitmultigauss = wf_func.get('fitmultigauss', False)
    fit_type:str = 'independent_gaussians'
    if fitmultigauss:
        fit_type = wf_func.get('fit_type', 'multigauss_iminuit')
    verbosemultigauss = wf_func.get('verbosemultigauss', False)
    params = ch_read_params(filename=wf_func.get('configyaml', 'ch_snr_parameters.yaml'))
    update_threshold = wf_func.get("update_threshold", False)
    onlyoptimal=wf_func.get("onlyoptimal", True)
    histautorange = wf_func.get('histautorange', False)

    onlyonerun = wf_func.get("onlyonerun", True) # A safe check that there is only one run per ch.
                                                 # This can be done otherwise too, but carefully.
    
    endpoint = wfset.waveforms[0].endpoint
    channel = wfset.waveforms[0].channel

    if endpoint not in params or channel not in params[endpoint]:
        params[endpoint] = {channel: {'fit': {}, 'baseline': {}}}
        print(f"No parameters found for endpoint {endpoint} and channel {channel} in the configuration file.")

    if(len(wfset.available_channels[list(wfset.runs)[0]][endpoint]) > 1):
        raise ValueError(f"Should have only one channel in the waveform set...")

    
    # params get updated inide here
    runBasicWfAnaNP02Updating(
        wfset,
        updatethreshold=update_threshold,
        show_progress=show_progress,
        params=params,
        doprocess=doprocess,
        onlyoptimal=onlyoptimal,
        onlyonerun=onlyonerun
    )

    bins_int = params[endpoint][channel]['fit'].get('bins_int', 100)
    domain_int_str = params[endpoint][channel]['fit'].get('domain_int', [-10e3, 100e3])
    force_max_peaks = params[endpoint][channel]['fit'].get('force_max_peaks', False)
    fit_limits = params[endpoint][channel]['fit'].get('fit_limits',[None, None])
    fit_limits = [ None if x == 'None' else x for x in fit_limits ]


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
    
    if histautorange:
        chargevalues = np.array([wf.analyses["std"].result[variable] for wf in wfset.waveforms if wf.analyses["std"].result[variable] is not np.nan])
        if chargevalues.size == 0:
            print(f"No valid charge values for endpoint {endpoint} and channel {channel}. Skipping histogram generation.")
            return
        domain_int=np.quantile(chargevalues, [0.02, 0.98])
        if len(chargevalues) < 5:
            domain_int = np.array([np.min(chargevalues), np.max(chargevalues)])
        bins_int = 200
    try:
        hInt = CalibrationHistogram.from_WaveformSet(
            wfset,
            bins_number=bins_int,
            domain=domain_int,
            variable=variable,
            analysis_label = "std",
            normalize_histogram=normalize_hist
        )
    except we.EmptyCalibrationHistogram as ehe:
        print(f"EmptyCalibrationHistogram for endpoint {endpoint} and channel {channel}: {ehe}")
        print("Changing range and binning..")
        # repating it.... not the best
        chargevalues = np.array([wf.analyses["std"].result[variable] for wf in wfset.waveforms if wf.analyses["std"].result[variable] is not np.nan])
        if len(chargevalues) == 0:
            print(f"No valid charge values for endpoint {endpoint} and channel {channel}. Skipping histogram generation.")
            return
        domain_int=np.quantile(chargevalues, [0.02, 0.98])
        if len(chargevalues) < 5:
            domain_int = np.array([np.min(chargevalues), np.max(chargevalues)])
        if len(chargevalues[(chargevalues >= domain_int[0]) & (chargevalues <= domain_int[1])]) < 1:
            print(f"Not enough entries in the charge values for endpoint {endpoint} and channel {channel}. Skipping histogram generation.")
            return
        hInt = CalibrationHistogram.from_WaveformSet(
            wfset,
            bins_number=200,
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
            name=f"{dict_uniqch_to_module.get(str(UniqueChannel(wfset.waveforms[0].endpoint, wfset.waveforms[0].channel)), '')}",
        )
        if wf_func.get("log_y", False):
            figure.update_yaxes(type="log")
        return

    # This method in case histogram should cut average
    # average_hits = hInt.edges[:-1]*hInt.counts
    # average_hits = np.sum(average_hits)/np.sum(hInt.counts)

    # This method in case histogram should not cut average
    charnges = np.array([ wf.analyses["std"].result[variable] for wf in wfset.waveforms if wf.analyses["std"].result[variable] is not np.nan ]) 
    #getting only 96% of the distribution to avoid huge outliers
    charnges = charnges[(charnges >= np.quantile(charnges, 0.02)) & (charnges <= np.quantile(charnges, 0.98))]
    integral_sum = np.sum(charnges)
    n_integrals = len(charnges)
    average_hits = integral_sum / n_integrals if n_integrals > 0 else 0

    fit_hist = fit_peaks_of_CalibrationHistogram(
        hInt,
        max_peaks          = max_peaks,
        prominence         = prominence,
        initial_percentage = initial_percentage,
        half_points_to_fit = half_point_to_fit,
        percentage_step    = percentage_step,
        fit_type           = fit_type,
        force_max_peaks    = force_max_peaks,
        fit_limits         = fit_limits,
    )
    if verbosemultigauss:
        print(getattr(hInt, "iminuit", None))
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
        mm = getattr(hInt, "iminuit", None)
        if mm is not None:
            errgain = hInt.iminuit.params[3].error
        snr = gain / baseline_stddev
    except:
        print(f"Could not fit for {dict_uniqch_to_module.get(str(UniqueChannel(wfset.waveforms[0].endpoint, wfset.waveforms[0].channel)),'')}")

    plot_CalibrationHistogram(
        hInt,
        figure=figure,
        row=row, col=col,
        plot_fits=True,
        plot_sum_of_gaussians=True,
        name=f"{dict_uniqch_to_module.get(str(UniqueChannel(wfset.waveforms[0].endpoint, wfset.waveforms[0].channel)),'')}; snr={snr:.2f}",
        showfitlabels=False,
    )
    if wf_func.get("log_y", False):
        figure.update_yaxes(type="log", range=[-1, np.log10(np.max(hInt.counts)*2) ], row=row, col=col)

    if snr != 0:
        for wf in wfset.waveforms:
            wf.analyses["std"].result['snr'] = snr
            wf.analyses["std"].result['normalization'] = hInt.normalization
            wf.analyses["std"].result['gain'] = gain
            wf.analyses["std"].result['errgain'] = errgain
            wf.analyses["std"].result['spe_charge'] = spe_charge
            wf.analyses["std"].result['spe_stddev'] = spe_stddev
        print(
            f"{list(wfset.runs)[0]},",
            # f"{dict_uniqch_to_module[str(UniqueChannel(wfset.waveforms[0].endpoint, wfset.waveforms[0].channel))]},",
            f"{endpoint},",
            f"{channel},",
            f"{dict_uniqch_to_module.get(strUch(endpoint,channel), 'None')},",
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
                      baselinestart:int = 0,
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
    baseline = SBaseline(threshold=threshold, baselinestart=baselinestart, baselinefinish=baselinefinish, default_filtering=2, minimumfrac=minimumfrac, data_base=params)

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


def select_spe(waveform:Waveform, pretrigger=200, posttrigger=400, threshold=80) -> bool:
    if 'spe_charge' not in waveform.analyses['std'].result:
        return False
    spe_mean = waveform.analyses['std'].result['spe_charge']
    spe_std = waveform.analyses['std'].result['spe_stddev']
    waveform_charge = waveform.analyses['std'].result['integral']
    baseline = waveform.analyses['std'].result['baseline']
    
    if abs(waveform_charge - spe_mean) <= spe_std:
        if np.all(waveform.adcs[:pretrigger] - baseline < threshold):
            if np.all(waveform.adcs[posttrigger:] - baseline < threshold):
                return True
    return False

def plot_average_spe(wfset: WaveformSet,
                     pretrigger=200,
                     posttrigger=400,
                     threshold=80,
                     show_progress=True,
                     maximum_begin = 250,
                     maximum_end = 300
                     ):


    try:
        wfset_spe = WaveformSet.from_filtered_WaveformSet(wfset, select_spe, pretrigger=pretrigger, posttrigger=posttrigger, threshold=threshold,  show_progress=show_progress)
    except:
        # print(f"No SPE waveforms found for channel module {dict_uniqch_to_module[str(UniqueChannel(wfset.waveforms[0].endpoint, wfset.waveforms[0].channel))]}")
        print("NaN")
        return

    spewf = np.array([ wf.adcs - wf.analyses['std'].result['baseline'] for wf in wfset_spe.waveforms ])
    spewf = np.mean(spewf, axis=0)
    plt.plot(spewf, label=f"{dict_uniqch_to_module[str(UniqueChannel(wfset_spe.waveforms[0].endpoint, wfset_spe.waveforms[0].channel))]}")
    plt.xlabel('Time [ticks]')
    plt.ylabel('Amplitude [ADCs]')
    plt.legend()
    maximum = np.max(spewf[maximum_begin:maximum_end])
    print(f"{maximum:.2f}")


def matplotlib_plot_WaveformSetGrid(wfset: WaveformSet, detector: Union[List[UniqueChannel], List[str], List[Union[UniqueChannel, str]]], plot_function:Callable, func_params:dict={}, figsize=(16,8), rows=0, cols=0):

    detChMap = generate_ChannelMap(detector, rows=rows, cols=cols)
    gridWs = ChannelWsGrid(detChMap, wfset)

    fig, axs = plt.subplots(gridWs.ch_map.rows, gridWs.ch_map.columns, figsize=figsize)
    axs = np.atleast_2d(axs)

    for (i, j), ax in np.ndenumerate(axs):
        plt.sca(ax)
        wfs = gridWs.get_channel_ws_by_ij_position_in_map(i,j)
        if not wfs: continue
        plot_function(wfs, **func_params)


def plot_averages(fig:go.Figure, g:ChannelWsGrid):
    """
    Plot average waveforms for all valid channels in a channel grid.

    This function iterates over the channel map stored in a `ChannelWsGrid`,
    computes the average waveform for each available channel using
    `average_wf_ch`, and adds it as a line trace to the corresponding subplot
    in a Plotly figure.

    Only channels that are present in `dict_uniqch_to_module` and for which
    waveform data are available are plotted.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure with a predefined subplot layout. 
    g : ChannelWsGrid
        Channel grid object containing the channel mapping and the associated
        waveform sets (`ch_wf_sets`).

    Returns
    -------
    None
        The function modifies the input figure in place by adding traces.


    Example
    --------
    fig = make_subplots(rows=1, cols=2)
    gt = np02_gen_grids(mywfset, detector=["M3(1)", "M3(2)"])
    plot_averages(fig, gt["Custom"])
    fig.show()
    """

    for (row, col), uch in np.ndenumerate(g.ch_map.data):
        row += 1
        col += 1
        
        if str(uch) not in dict_uniqch_to_module:
            continue
        if uch.channel not in g.ch_wf_sets[uch.endpoint]:
            continue
        wfch = g.ch_wf_sets[uch.endpoint][uch.channel]
        avg = average_wf_ch(wfch)
        time = np.arange(avg.size)

        fig.add_trace(
            go.Scatter(
                x = time,
                y = avg,
                mode = "lines",
            ),
            row=row, col=col
        )

def plot_averages_w_peaks_rise_fall(peaks_all, fig:go.Figure, g:ChannelWsGrid, x_range=None, rise_fall:bool = True):

    """
    Plot average waveforms for each channel in a ChannelWsGrid with peak and rise/fall indicators.

    For each valid channel in the grid, this function:
      - Plots the averaged waveform in the corresponding subplot.
      - Optionally restricts the x-axis to a specified range.
      - Optionally adds vertical dashed lines at key points: t10 and t90 of rise, t90 and t10 of fall.
      - Annotates each subplot with channel info, peak amplitude, and optionally rise/fall times.

    Parameters
    ----------
    peaks_all : dict
        Dictionary returned by `compute_peaks_rise_fall_ch`. Keys are (endpoint, channel),
        values contain peak, rise/fall times, and averaged waveform data.
    fig : go.Figure
        Plotly Figure object containing the subplots where waveforms will be plotted.
    g : ChannelWsGrid
        Grid object containing the channel waveform sets.
    x_range : tuple of two floats, optional
        (min, max) time range to display on the x-axis for all subplots. 
        If None, the full waveform is shown.
    rise_fall : bool, True by default.
        If True the rise and fall time dashed lines are shown and the values are added in the legend

    Returns
    -------
    go.Figure
        The updated Plotly Figure with all average waveforms plotted, vertical lines for 
        rise/fall times, and annotations with peak and timing information.
    """

    ncols = len(g.ch_map.data[0])    

    fig.data = []
    fig.layout.annotations = []

    for (row, col), uch in np.ndenumerate(g.ch_map.data):
        row += 1
        col += 1

        subplot_idx = (row - 1) * ncols + col
        
        if str(uch) not in dict_uniqch_to_module:
            continue
        if uch.channel not in g.ch_wf_sets[uch.endpoint]:
            continue

        vals = peaks_all[(uch.endpoint, uch.channel)]
        time = vals["time"]
        avg = vals["avg"]
        peak_time = vals["peak_time"]
        peak_value = vals["peak_value"]

        if x_range is not None:
            x_min, x_max = x_range
            mask = (time >= x_min) & (time <= x_max)
            time = time[mask]
            avg = avg[mask]
        
        fig.add_trace(
            go.Scatter(
                x=time,
                y=avg,
                mode="lines",
                name=f"{uch.endpoint}-{uch.channel}"
            ),
            row=row, col=col
        )

        if rise_fall:

            for t, color, label in [
                (vals["t_low"], "green", "t10 rise"),
                (vals["t_high"], "blue", "t90 rise"),
                (vals["t_high_fall"], "orange", "t90 fall"),
                (vals["t_low_fall"], "purple", "t10 fall"),
            ]:
                fig.add_trace(
                    go.Scatter(
                        x=[t, t],
                        y=[0, peak_value], 
                        mode="lines",
                        line=dict(color=color, dash="dash"),
                        showlegend=False,  
                    ),
                    row=row, col=col
                ) 
                    
        if subplot_idx == 1:
            xref = "x domain"
            yref = "y domain"
        else:
            xref = f"x{subplot_idx} domain"
            yref = f"y{subplot_idx} domain"

        key = f"{uch.endpoint}-{uch.channel}"
        module_name = dict_uniqch_to_module.get(key, None)

        if rise_fall:
            fig.add_annotation(
                x=0.98,
                y=0.95,
                xref=xref,
                yref=yref,
                text=(
                    f"{module_name}<br>"
                    f"Peak = {peak_value:.1f} ADC<br>"
                    f"Rise time = {vals['rise_time']:.0f} ticks<br>"
                    f"Fall time = {vals['fall_time']:.0f} ticks"
                ),
                showarrow=False,
                align="left",
                font=dict(size=11),
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="black",
                borderwidth=1
            )

        else:
            fig.add_annotation(
                x=0.98,
                y=0.95,
                xref=xref,
                yref=yref,
                text=(
                    f"{module_name}<br>"
                    f"Peak = {peak_value:.1f} ADC<br>"
                ),
                showarrow=False,
                align="left",
                font=dict(size=11),
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="black",
                borderwidth=1
            )

        if x_range is not None:
            fig.update_xaxes(range=[x_min, x_max])
        
    return fig

def plot_averages_normalized(fig:go.Figure, g:ChannelWsGrid, spe_by_channel, save: bool = False, save_dir: Optional[str] = None):

    """
    Plot normalized average waveforms for each channel in a ChannelWsGrid.

    For each valid channel in the grid, this function:
      - Computes the average waveform.
      - Normalizes it to a single photoelectron (SPE) amplitude.
      - Plots the normalized waveform in the corresponding subplot.
      - Optionally saves the normalized waveform to a text file for each channel.

    Parameters
    ----------
    fig : go.Figure
        Plotly Figure object containing subplots for each channel.
    g : ChannelWsGrid
        Grid object containing waveform sets for multiple channels.
    spe_by_channel : dict
        Dictionary mapping channel numbers to single photoelectron (SPE) amplitudes.
        Used for normalizing the waveform.
    save : bool, optional
        If True, normalized waveforms are saved to text files. Default is False.
    save_dir : str, optional
        Directory where normalized waveform files will be saved. Must be specified if `save` is True.

    Returns
    -------
    None
        The function updates the provided Plotly Figure with normalized waveforms.

    """

    if save:
        if save_dir is None:
            raise ValueError("save_dir must be specified")
        os.makedirs(save_dir, exist_ok=True)

    for (row, col), uch in np.ndenumerate(g.ch_map.data):
        row += 1
        col += 1
        
        if str(uch) not in dict_uniqch_to_module:
            continue
        if uch.channel not in g.ch_wf_sets[uch.endpoint]:
            continue
            
        wfch = g.ch_wf_sets[uch.endpoint][uch.channel]
        avg = average_wf_ch(wfch)
        time = np.arange(avg.size)

        ch = uch.channel
        if ch not in spe_by_channel:
            print(f"Missing calibration for channel {ch}")
            continue

        spe_amp = spe_by_channel[ch]
        peak_avg = np.max(avg)

        if peak_avg == 0:
            print(f"Zero peak for channel {ch}")
            continue

        avg_norm = avg * (spe_amp / peak_avg )

        if save:
            key = f"{uch.endpoint}-{uch.channel}"
            module_name = dict_uniqch_to_module.get(key, None)

            if module_name is None:
                print(f"No module mapping found for endpoint {uch.endpoint}, channel {uch.channel}")
                continue

            module_for_title = module_name[:2]
            channel_for_title = module_name[3]

            filename = f"template_{module_for_title}_{channel_for_title}.txt"
            filepath = os.path.join(save_dir, filename)

            np.savetxt(
                filepath,
                avg_norm,
                fmt="%.9e"
            )

            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                print(f"File saved: {filepath}")
            else:
                print(f"Failed to save: {filepath}")
        
        fig.add_trace(
            go.Scatter(
                x = time,
                y = avg_norm,
                mode = "lines",
            ),
            row=row, col=col
        )
