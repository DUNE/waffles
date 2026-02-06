from waffles.coldboxVD.november_25.ab_coldbox.imports import *



def plotting_overlap_wf(wfset, n_wf = None, index_list = None, show : bool = True, save : bool = False, 
                       x_min=None, x_max=None, y_min=None, y_max=None, int_ll=None, int_ul=None, 
                       baseline=None, output_folder : str = 'output'):
    """
    Plot overlapping waveforms from waveform set using Plotly.

    Plots either first N waveforms or specific indices. Supports custom axis limits, 
    integration limit lines (vertical), baseline line (horizontal), interactive display, 
    and PNG export.

    Parameters
    ----------
    wfset : object
    n_wf : int, optional
        Number of waveforms to plot from start (ignored if index_list provided). Default: None.
    index_list : list[int], optional
        Specific waveform indices to plot. Default: None.
    show : bool, default=True
        Display plot interactively with fig.show().
    save : bool, default=False
        Save plot as PNG to `{output_folder}/waveform_plot.png`.
    x_min, x_max, y_min, y_max : float, optional
        Custom axis range limits.
    int_ll, int_ul : float, optional
        Vertical dashed lines for lower/upper integration limits.
    baseline : float, optional
        Horizontal dashed baseline line.
    output_folder : str, default='output'
        Directory for saved PNG file.

    Returns
    -------
    None
        Displays plot if show=True; saves PNG if save=True.

    Raises
    ------
    ValueError
        If both n_wf and index_list are None.

    """

    if index_list is None and n_wf is None:
        raise ValueError("You must provide either n_wf or index_list")

    if index_list is not None:
        indices = [i for i in index_list if 0 <= i < len(wfset.waveforms)]
    else:
        indices = range(min(n_wf, len(wfset.waveforms)))
    
    fig = go.Figure()

    for i in indices:
        
        y = wfset.waveforms[i].adcs
            
        fig.add_trace(go.Scatter(
            x=np.arange(len(y)) + wfset.waveforms[i].time_offset,
            y=y,
            mode='lines',
            line=dict(width=0.5),
            showlegend=True, 
            name = f"Waveform {i}"))

    xaxis_range = dict(range=[x_min, x_max]) if x_min is not None and x_max is not None else {}
    yaxis_range = dict(range=[y_min, y_max]) if y_min is not None and y_max is not None else {}

    fig.update_layout(
        xaxis_title="Time ticks (AU)",
        yaxis_title="Amplitude (Adcs)",
        xaxis=xaxis_range,  
        yaxis=yaxis_range,  
        margin=dict(l=50, r=50, t=20, b=50),
        template="plotly_white",
        legend=dict(
            x=1,  
            y=1,  
            xanchor="right",
            yanchor="top",
            orientation="v", 
            bgcolor="rgba(255, 255, 255, 0.8)" ))

    if int_ll is not None:
        fig.add_shape(
            type="line",
            x0=int_ll,
            x1=int_ll,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="coral", width=2, dash="dash"),
            name=f"Lower integral limit \n(x = {int_ll})"
        )

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color="coral", width=2, dash="dash"),
            showlegend=True,
            name=f"Lower integral limit \n(x = {int_ll})"
        ))

    if int_ul is not None:
        fig.add_shape(
            type="line",
            x0=int_ul,
            x1=int_ul,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="chocolate", width=2, dash="dash"),
            name=f"Upper integral limit \n(x = {int_ul})"
        )

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color="chocolate", width=2, dash="dash"),
            showlegend=True,
            name=f"Upper integral limit \n(x = {int_ul})"
        ))

    if baseline is not None:
        fig.add_shape(
            type="line",
            x0=0,
            x1=1,
            y0=baseline,
            y1=baseline,
            xref="paper",
            yref="y",
            line=dict(color="red", width=1.5, dash="dash"),
            name=f"Baseline \n(y = {baseline})"
        )

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color="red", width=1.5, dash="dash"),
            showlegend=True,
            name=f"Baseline \n(y = {baseline})"
        ))

    if save:
        fig.write_image(f"{output_folder}/waveform_plot.png", scale=2)
    
    if show:
        fig.show()

#########################################################################################################

def coldbox_single_channel_grid(wfset, config_channel):
    """
    Generate a waveform grid for a single specified detector channel.

    This function creates a ChannelWsGrid for one specific channel, 
    based on the waveform dataset provided.

    Parameters
    ----------
    wfset : object
        Waveform dataset containing multiple channels and associated metadata.
    config_channel : int or str
        Identifier of the channel to generate the grid for.

    Returns
    -------
    grid : ChannelWsGrid
        A waveform grid object corresponding to the specified channel.
    """
    
    kwargs = {}
    detector = [UniqueChannel(10, x) for x in [config_channel]]
    grid = np02_gen_grids(wfset, detector, rows=kwargs.pop("rows", 0), cols=kwargs.pop("cols", 0))
    return grid['Custom']


#########################################################################################################

def baseline_std_selection(waveform: Waveform, baseline_analysis_label: str, mean_std: float, n_std: float = 1.0) -> bool:
    """
    Check if a waveform passes a baseline standard deviation threshold, defined as
    mean_std multiplied by n_std.

    Parameters
    ----------
    waveform : object
        The waveform object containing analysis results.
    baseline_analysis_label : str
        Label identifying which baseline analysis to use.
    mean_std : float
        Reference standard deviation used as a threshold.
    n_std : float, optional
        Multiplicative factor for the threshold (default is 1.0).

    Returns
    -------
    bool
        True if the waveform's baseline_std is less than mean_std * n_std,
        False otherwise.
    """

    if waveform.analyses[baseline_analysis_label].result['baseline_std'] < mean_std*n_std:
        return True
    else:
        return False
    

#########################################################################################################

# Similar to coarse_selection_for_led_calibration from Julio's code, but with possibility to select time range and without using baseline
def adcs_threshold_filter(
    waveform: Waveform,
    time_range: Optional[Tuple[int, int]] = None,
    adcs_minimum_threshold: Optional[float] = None,
    adcs_maximum_threshold: Optional[float] = None
) -> bool:
    """
    Check whether a waveform's ADC values lie within optional minimum and/or
    maximum thresholds over an optional time range.

    Parameters
    ----------
    waveform : Waveform
        The waveform object containing the ADC samples in the `adcs` attribute.
    time_range : tuple of 2 int, optional
        Time range as (start, stop) where start is inclusive and stop is exclusive.
        If None (default), checks all samples.
    adcs_minimum_threshold : float, optional
        Lower bound for the ADC values. If None, no minimum cut is applied.
    adcs_maximum_threshold : float, optional
        Upper bound for the ADC values. If None, no maximum cut is applied.

    Returns
    -------
    bool
        True if all ADC values in the time_range are above the minimum threshold 
        (when provided) and below the maximum threshold (when provided). 
        False otherwise.
    """

    if time_range is None:
        values = waveform.adcs
    else:
        start, stop = time_range
        values = waveform.adcs[start:stop]

    if adcs_minimum_threshold is not None and np.min(values) <= adcs_minimum_threshold:
        return False

    if adcs_maximum_threshold is not None and np.max(values) >= adcs_maximum_threshold:
        return False

    return True


#########################################################################################################

def persistance_plot_helper(
    wfset: WaveformSet, 
    channel: int, 
    null_baseline_analysis_label: str = 'null_baseliner', 
    ymin: Optional[float] = None, 
    ymax: Optional[float] = None, 
    adc_bins: int = 1000,
    show: bool = False,
    color_bar: bool = False
) -> go.Figure:
    """
    Create persistence heatmap for single channel waveforms with baseline correction.

    Generates 2D occupancy heatmap (time vs ADC) via grid transformation. Applies 
    optional baseline correction and y-axis ADC limits. Controls colorbar visibility.

    Parameters
    ----------
    wfset : WaveformSet
        Waveform set containing all channel waveforms.
    channel : int
        Target channel number (0-based indexing).
    null_baseline_analysis_label : str, default='null_baseliner'
        Analysis label for baseline subtraction in grid transformation.
    ymin, ymax : Optional[float], default=None
        Y-axis ADC range limits (below/above baseline). Auto-scales if None.
    adc_bins : int, default=1000
        Number of ADC value bins in heatmap.
    show : bool, default=False
        Call fig.show() for immediate display (Jupyter-friendly).
    color_bar : bool, default=False
        Enable colorbar with "Counts" title on heatmap traces.

    Returns
    -------
    go.Figure
        Plotly figure with heatmap(s) ready for display/export.

    """
    
    fig = plot_ChannelWsGrid(
        coldbox_single_channel_grid(wfset, config_channel=channel),
        mode='heatmap',
        wfs_per_axes=None,
        analysis_label=null_baseline_analysis_label,
        detailed_label=False,
        adc_range_above_baseline = ymax,
        adc_range_below_baseline = ymin,
        adc_bins = adc_bins
    )
    
    # Color bar settings
    if color_bar:
        for trace in fig.data:
            if isinstance(trace, go.Heatmap):
                trace.showscale = True
                trace.colorbar.title = "Counts"  

    if show:  # in jupyter notebooks, returning fig automatically shows it
        fig.show()

    return fig


##########################################################################################################

def auto_histogram(wfset: WaveformSet,
                   analysis_label: str, 
                   result_key: str = 'integral', 
                   range_method: str = 'percentile',  # 'full', 'percentile'
                   lower_percentile: float = 0.2,
                   upper_percentile: float = 99,
                   bin_method: str = 'number',  # 'number' or 'width'
                   bin_width: int = 50, 
                   bin_number: int = 275,
                   show_results: bool = False
                   ):
    """
    Compute optimal histogram parameters from waveform analysis results.

    Extracts numeric results, determines data range (full or percentile-trimmed), 
    and sets bins (fixed count or width). Filters NaN values automatically.

    Parameters
    ----------
    wfset : WaveformSet
        Input waveforms with pre-computed analysis results.
    analysis_label : str
        Analysis identifier (e.g., 'integral', 'amplitude').
    result_key : str, default='integral'
        Key for numeric value in analysis result dict.
    range_method : str, default='percentile'
        - 'full': min-max of data.
        - 'percentile': trim via lower/upper percentiles.
    lower_percentile : float, default=0.2
        Lower cutoff percentile (ignored if range_method='full').
    upper_percentile : float, default=99
        Upper cutoff percentile (ignored if range_method='full').
    bin_method : str, default='number'
        - 'number': fixed bin count.
        - 'width': fixed bin width.
    bin_width : int, default=50
        Bin width in data units (used if bin_method='width').
    bin_number : int, default=275
        Total bins (used if bin_method='number').
    show_results : bool, default=False
        Print computed range, n_bins, bin_width.

    Returns
    -------
    tuple: (np.ndarray, int, float)
        - data_range: [min, max] array for histogram x-limits.
        - n_bins: Computed number of bins.
        - bin_width: Computed/used bin width.

    Raises
    ------
    ValueError
        Invalid range_method or bin_method.
        No valid (non-NaN) data found.

    """

    data = [wfset.waveforms[idx].get_analysis(analysis_label).result[result_key]  for idx in range(len(wfset.waveforms))
            if wfset.waveforms[idx].get_analysis(analysis_label).result[result_key] is not np.nan]

    if range_method == 'full':
        data_range = [data.min(), data.max()]
    elif range_method == 'percentile':
        data_range = [np.percentile(data, lower_percentile), np.percentile(data, upper_percentile)]
    else:
        raise ValueError("range_method deve essere 'full' or 'percentile'")
    
    if bin_method == 'number':
        n_bins = bin_number
        bin_width = float((data_range[1] - data_range[0]))/n_bins
    elif bin_method == 'width':
        n_bins = int(np.ceil((data_range[1] - data_range[0]) / bin_width))
    else:
        raise ValueError("bin_method deve essere 'number' or 'width'")

    if show_results: print(f"Histogram domain: {data_range}\nHistogram number of bins: {n_bins}\nHistogram bin width: {bin_width}")

    return np.array(data_range), n_bins, bin_width


########################################################################################################## 

def print_correlated_gaussians_fit_parameters(my_grid, federico_conversion: bool = False, show: bool = False):
    """
    Extract and compute Gaussian fit parameters for calibration from grid data.

    Calculates pedestal (μ₀, σ₀), gain (G = μ₁ - μ₀), SNR (G/σ₀), and std increment (Δσ = σ₁² - σ₀²) 
    with propagation of uncertainties. Supports optional /16 scaling.

    Parameters
    ----------
    my_grid : object
        Grid with `ch_map`, `ch_wf_sets`; accesses first channel's `calib_histo.gaussian_fits_parameters`.
    federico_conversion : bool, default=False
        Divide mean/std/gain by 16 if True.
    show : bool, default=False
        Print raw params (means/stds/scales) and formatted results.

    Returns
    -------
    dict
        Keys: 'num_peaks', 'params' (raw dict), 'mean_0'/'e_mean_0', 'std_0'/'e_std_0', 
              'gain'/'e_gain', 'snr'/'e_snr', 'delta_std'/'e_delta_std'.

    Notes
    -----
    - Assumes ≥2 peaks; uses peak 0 (pedestal), peak 1 (signal).
    - Uncertainties: √(σ₁² + σ₀²) for gain; standard propagation for others.
    """

    ch_id = my_grid.ch_map.data[0][0]
    channel_ws = my_grid.ch_wf_sets[ch_id.endpoint][ch_id.channel]
    params = channel_ws.calib_histo.gaussian_fits_parameters
    
    mean_0 = params['mean'][0][0]
    e_mean_0 = params['mean'][0][1]
    std_0 = params['std'][0][0]
    e_std_0 = params['std'][0][1]

    gain = params['mean'][1][0] - mean_0
    e_gain = np.sqrt(params['mean'][1][1]**2 + e_mean_0**2)

    delta_std = (params['std'][1][0]**2 - std_0**2)
    e_delta_std = np.sqrt((2 * params['std'][1][0] * params['std'][1][1])**2 + (2 * std_0 * e_mean_0)**2)

    snr = gain / std_0
    e_snr = snr * np.sqrt((e_gain / gain)**2 + (e_std_0 / std_0)**2)

    fg = 16 if federico_conversion else 1

    results = {
        "num_peaks": len(params["scale"]),
        "params": params,
        "mean_0": mean_0 / fg,
        "e_mean_0": e_mean_0 / fg,
        "std_0": std_0 / fg,
        "e_std_0": e_std_0 / fg,
        "gain": gain / fg,
        "e_gain": e_gain / fg,
        "snr": snr,
        "e_snr": e_snr,
        "delta_std": delta_std,
        "e_delta_std": e_delta_std
    }

    if show:
        print(f"Peaks found: {results['num_peaks']}")
        print(f"Gaussian Mean: {params['mean']}")
        print(f"Gaussian Std: {params['std']}")
        print(f"Gaussian Amplitude: {params['scale']}\n")

        print(f"mu 0: {results['mean_0']:.2f} +/- {results['e_mean_0']:.2f}")
        print(f"std 0: {results['std_0']:.2f} +/- {results['e_std_0']:.2f}")
        print(f"Gain: {results['gain']:.2f} +/- {results['e_gain']:.2f}")
        print(f"SNR: {results['snr']:.2f} +/- {results['e_snr']:.2f}")
        print(f"Std increment: {results['delta_std']:.2f} +/- {results['e_delta_std']:.2f}")

    return results

##########################################################################################################

def add_fig_to_subplot(source_fig, target_fig, row, col):
    """ For PDF CREATION """
    for trace in source_fig.data:
        target_fig.add_trace(trace, row=row, col=col)

##########################################################################################################

"""Function to round results with error, and by using scientific notation"""

def round_to_significant(value, error):
    if not isinstance(error, (int, float, np.number)) or not np.isfinite(error) or error == 0:
        return "N/A", "N/A"  # Restituisce una stringa se l'errore non è valido

    error_order = int(np.log10(error))
    significant_digits = 2  
    rounded_error = round(error, -error_order + (significant_digits - 1))
    rounded_value = round(value, -error_order + (significant_digits - 1))
    return rounded_value, rounded_error

def to_scientific_notation(value, error):
    # Controllo se il valore non è un numero valido
    if not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
        return "N/A"  # Se il valore è NaN o infinito, restituisce "N/A"

    if value == 0:
        return "0"  # Caso speciale: zero non ha notazione scientifica

    exponent = int(np.floor(np.log10(abs(value)))) if value != 0 else 0  # Evita log10(0)
    mantissa_value = value / 10**exponent
    mantissa_error = error / 10**exponent if error and np.isfinite(error) else 0  # Evita divisioni errate

    mantissa_value, mantissa_error = round_to_significant(mantissa_value, mantissa_error)
    
    return f"({mantissa_value} ± {mantissa_error}) × 10^{exponent}" if mantissa_value != "N/A" else "N/A"


def fmt(val, err):
        if not isinstance(err, (int, float, np.number)) or not np.isfinite(err) or err == 0:
            return "N/A"
        # usa notazione scientifica solo se necessario
        if abs(val) >= 1e3 or abs(val) <= 1e-2:
            return to_scientific_notation(val, err)
        v, e = round_to_significant(val, err)
        return f"{v} ± {e}"

##########################################################################################################

def build_calibration_annotation(
    output_parameters,
    hist_domain,
    hist_nbins,
    hist_bins_width
    ):
    """
    Function per build_calibration_annotation for the PDF
    """

    lines = []
    lines.append(f"N peaks = {output_parameters['num_peaks']}")
    lines.append(f"μ₀ = {fmt(output_parameters['mean_0'], output_parameters['e_mean_0'])}")
    lines.append(f"σ₀ = {fmt(output_parameters['std_0'], output_parameters['e_std_0'])}")
    lines.append(f"G  = {fmt(output_parameters['gain'], output_parameters['e_gain'])}")
    lines.append(f"SNR = {fmt(output_parameters['snr'], output_parameters['e_snr'])}")
    lines.append(f"Δσ = {fmt(output_parameters['delta_std'], output_parameters['e_delta_std'])}")

    lines.append("")  # separatore visivo piccolo
    lines.append(f"domain = [{hist_domain[0]:.0f}, {hist_domain[1]:.0f}]")
    lines.append(f"n bins = {hist_nbins:.0f}")
    lines.append(f"bin width = {hist_bins_width:.0f}")

    return "<br>".join(lines)


##########################################################################################################


def single_vgain_analysis(membrane: str, 
                          channel: int, 
                          bias: str, 
                          vgain: int,
                          dict_params,
                          show: bool = False,
                          save_pdf: bool = False,
                          federico_limits: bool = True,
                          federico_conversion: bool = True,
                          coldbox_folder: str = "/eos/experiment/neutplatform/protodune/experiments/ColdBoxVD/November2025run/spy_buffer/VGAIN_SCAN",
                          output_folder: str = "/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/coldboxVD/november_25/ab_coldbox/output"):
    """
    Full analysis pipeline for single Vgain: load → filter → integrate → fit → plot calibration.

    Processes spybuffer data: baseline subtraction/filtering, pulse window detection, 
    histogram fitting (multi-Gaussian), and 4-panel summary plot (PDF optional).

    Parameters
    ----------
    membrane : str
        Membrane name for file path.
    channel : int
        Target channel.
    bias : str
        bias voltage.
    vgain : int
        Vgain.
    dict_params : dict
        Analysis params from vgain_analysis_parameters() or external dict.
    show : bool, default=False
        Display persistance and calibration plots.
    save_pdf : bool, default=False
        Save 4-panel PDF to output_folder/membrane/channel/bias/vgain_vgain.pdf.
    federico_limits : bool, default=True
        use federico integration limits.
    federico_conversion : bool, default=True
        /16 scaling in Gaussian fits.
    coldbox_folder : str, default=...
        Spybuffer input root.
    output_folder : str, default=...
        Output plots path.

    Returns
    -------
    None
    """

    wfset_original = create_waveform_set_from_spybuffer(filename=f"{coldbox_folder}/{membrane}/vgain_scan_{membrane}_DVbias_{bias}/vgain_{vgain}/channel_{channel}.dat", WFs=40000, length=1024, config_channel=channel)

    # Baseline removing
    baseliner_input_parameters = IPDict({'baseline_limits': (0,dict_params['baseline_timeticks_limit']), 'std_cut': 1., 'type': 'mean'})
    checks_kwargs = IPDict({'points_no': wfset_original.points_per_wf})
    baseline_analysis_label = 'baseline'
    _ = wfset_original.analyse(baseline_analysis_label, WindowBaseliner, baseliner_input_parameters, checks_kwargs=checks_kwargs, overwrite=True)
    wfset_original.apply(subtract_baseline, baseline_analysis_label, show_progress=False)

    null_baseline_analysis_label = 'null_baseliner'
    _ = wfset_original.analyse(null_baseline_analysis_label, StoreWfAna, {'baseline': 0.}, overwrite=True)

    wfset_1 = WaveformSet.from_filtered_WaveformSet(wfset_original, adcs_threshold_filter, time_range = [0,dict_params['baseline_timeticks_limit']], adcs_minimum_threshold=-dict_params['adcs_threshold'], adcs_maximum_threshold=dict_params['adcs_threshold'])

    average_baseline_std = compute_average_baseline_std(wfset_1, baseline_analysis_label)
    wfset_2 = WaveformSet.from_filtered_WaveformSet(wfset_1, baseline_std_selection, baseline_analysis_label, average_baseline_std, dict_params['n_std_baseline'])

    wfset_filtered = wfset_2

    mean_wf = wfset_filtered.compute_mean_waveform()

    aux_limits = get_pulse_window_limits(adcs_array = -mean_wf.adcs, baseline = 0, deviation_from_baseline = dict_params['deviation_from_baseline'], get_zero_crossing_upper_limit = False)
    if federico_limits:
        aux_limits = (382, 406)

    integration_analysis_label = 'integration_analysis'
    integrator_input_parameters = IPDict({'baseline_analysis': null_baseline_analysis_label, 'inversion': False, 'int_ll': aux_limits[0], 'int_ul': aux_limits[1], 'amp_ll': aux_limits[0], 'amp_ul': aux_limits[1]})
    checks_kwargs = IPDict({ 'points_no': wfset_filtered.points_per_wf})
    _ = wfset_filtered.analyse( integration_analysis_label, WindowIntegrator, integrator_input_parameters, checks_kwargs=checks_kwargs, overwrite=True)

    hist_domain, hist_nbins, hist_bins_width = auto_histogram(wfset_filtered, integration_analysis_label, show_results=False)

    my_grid = coldbox_single_channel_grid(wfset_filtered, config_channel=channel)

    my_grid.compute_calib_histos(
        bins_number=hist_nbins, 
        domain=hist_domain, 
        variable='integral',
        analysis_label=integration_analysis_label
    )

    fit_peaks_of_ChannelWsGrid(
        my_grid,
        max_peaks=dict_params['max_peaks'], 
        prominence=float(dict_params['prominence']), 
        initial_percentage=dict_params['initial_percentage'],
        percentage_step=dict_params['percentage_step'],
        return_last_addition_if_fail=True,
        fit_type='multigauss_iminuit',
        weigh_fit_by_poisson_sigmas=True,
        ch_span_fraction_around_peaks=dict_params['ch_span_fraction_around_peaks']
    )

    ch_id = my_grid.ch_map.data[0][0]
    channel_ws = my_grid.ch_wf_sets[ch_id.endpoint][ch_id.channel]
    params = channel_ws.calib_histo.gaussian_fits_parameters
    if len(params['scale']) > 10:
        print(f"Warning: More than 10 peaks found for membrane {membrane}, channel {channel}, bias {bias}, vgain {vgain}.")

    output_parameters = print_correlated_gaussians_fit_parameters(my_grid, federico_conversion=federico_conversion)

    if show or save_pdf:
        fig1 = persistance_plot_helper(wfset_original, channel, ymin = dict_params['heatmap_min'], ymax = dict_params['heatmap_max'], adc_bins = 1000)
        fig2 = persistance_plot_helper(wfset_1, channel, ymin = dict_params['heatmap_min'], ymax = dict_params['heatmap_max'], adc_bins = 1000)
        fig3 = persistance_plot_helper(wfset_2, channel, ymin = dict_params['heatmap_min'], ymax = dict_params['heatmap_max'], adc_bins = 1000)

        fig4 = plot_ChannelWsGrid(
            my_grid, 
            mode='calibration',
            plot_peaks_fits=True,           
            plot_sum_of_gaussians=True      
            )
        
        big_fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=[
                f"Persistence – Original: {len(wfset_original.waveforms)} wfs",
                f"Persistence – Baseline adcs cut: {len(wfset_1.waveforms)} wfs",
                f"Persistence – Baseline std cut: {len(wfset_2.waveforms)} wfs",
                f"Calibration plot"
            ]
        )

        add_fig_to_subplot(fig1, big_fig, 1, 1)
        add_fig_to_subplot(fig2, big_fig, 2, 1)
        add_fig_to_subplot(fig3, big_fig, 3, 1)
        add_fig_to_subplot(fig4, big_fig, 4, 1)

        annotation_text = build_calibration_annotation(
            output_parameters,
            hist_domain,
            hist_nbins,
            hist_bins_width
        )

        big_fig.add_annotation(
            text=annotation_text,
            xref="x4 domain",
            yref="y4 domain",
            x=0.98,
            y=1,
            showarrow=False,
            align="left",
            font=dict(size=9),  
            bgcolor="rgba(255,255,255,0.85)",
            borderwidth=1
        )

        big_fig.update_layout(
            title=dict(
                text=f"{membrane} CH{channel} - Bias {bias} - {vgain} Vgain",
                x=0.5,                 
                xanchor="center",
                y=0.97,                
                yanchor="top",
                font=dict(size=18)
            )
        )

        big_fig.update_layout(
        height=1000,
        width=1000,
        showlegend=False
        )

        if save_pdf:
            output_path = Path(output_folder) / membrane / str(channel) / bias
            output_path.mkdir(parents=True, exist_ok=True)  
            big_fig.write_image(output_path / f"{vgain}_vgain.pdf")

        if show:
            big_fig.show()


##########################################################################################################

def vgain_analysis_parameters(vgain):
    """
    Compute Vgain-dependent parameters for waveform analysis pipeline.

    Returns config dict for analysis parameters.

    Parameters
    ----------
    vgain : int or float

    Returns
    -------
    dict[str, float|int]
        Keys:
        - 'baseline_timeticks_limit'
        - 'deviation_from_baseline' 
        - 'heatmap_min'
        - 'heatmap_max'
        - 'adcs_threshold'
        - 'n_std_baseline'
        - 'max_peaks'
        - 'prominence'
        - 'initial_percentage'
        - 'percentage_step'
        - 'ch_span_fraction_around_peaks'

    Notes
    -----
    - No input validation; assumes valid vgain ≥500.
    - Used for dynamic thresholding in persistence/peak detection.
    """

    v_index = (vgain - 500) // 100
    heatmap_min = (-200 + v_index * 10)
    heatmap_max =( 700 - v_index * 50 if vgain <= 1600 else 100)

    adcs_threshold = (
        90 - v_index * 10 if vgain <= 900 else
        50 - (v_index-4) * 5  if vgain <= 1300 else
        30 - (v_index-8) * 3  if vgain <= 1900 else
        11
    )
    
    n_std_baseline = (1 if vgain <= 1800 else 1.1)

    max_peaks=  7
    prominence= 0.5
    initial_percentage=0.1
    percentage_step=0.02
    ch_span_fraction_around_peaks=0.05 

    baseline_timeticks_limit = 380
    deviation_from_baseline = 0.6

    return {
        'baseline_timeticks_limit': baseline_timeticks_limit,
        'deviation_from_baseline': deviation_from_baseline,
        'heatmap_min': heatmap_min,
        'heatmap_max': heatmap_max,
        'adcs_threshold': adcs_threshold,
        'n_std_baseline': n_std_baseline,
        'max_peaks': max_peaks, 
        'prominence': prominence, 
        'initial_percentage': initial_percentage,
        'percentage_step': percentage_step,
        'ch_span_fraction_around_peaks': ch_span_fraction_around_peaks
    }


##########################################################################################################

def channel_vgain_scan_analysis(
        membrane: str, 
        channel: int, 
        bias: str, 
        vgain_list,
        external_dict_paramas: Optional[dict] = None,
        federico_limits: bool = True,
        federico_conversion: bool = True,
        show: bool = False,
        save_pdf: bool = True,
        coldbox_folder: str = "/eos/experiment/neutplatform/protodune/experiments/ColdBoxVD/November2025run/spy_buffer/VGAIN_SCAN",
        output_folder: str = "/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/coldboxVD/november_25/ab_coldbox/output"):
    """
    Sequential Vgain scan analysis for fixed membrane/channel/bias across vgain_list.

    Loops over vgain_list, calling single_vgain_analysis() with auto/override params. 
    Prints progress; supports Federico scaling/limits and PDF export.

    Parameters
    ----------
    membrane : str
        Membrane identifier.
    channel : int
        Channel number.
    bias : str
        Bias configuration.
    vgain_list : list[int|float]
        Vgain values to analyze sequentially.
    external_dict_paramas : Optional[dict[int|float, dict]], default=None
        Per-vgain param overrides; else uses vgain_analysis_parameters(vgain).
    federico_limits : bool, default=True
        Apply Federico-specific limits in single_vgain_analysis.
    federico_conversion : bool, default=True
        Apply /16 scaling in fits.
    show : bool, default=False
        Enable intermediate plot displays.
    save_pdf : bool, default=True
        Save analysis outputs as PDF.
    coldbox_folder : str, default=...
        Input data path (ColdBox VGAIN_SCAN).
    output_folder : str, default=...
        Output directory for plots/PDFs.

    Returns
    -------
    None
    """

    for vgain in vgain_list:
        print(f"\n\n------------------------ \n------ Vgain: {vgain} ------\n")

        single_vgain_analysis(
            membrane=membrane,
            channel=channel,
            bias=bias,
            vgain=vgain,
            dict_params = (external_dict_paramas[vgain] if external_dict_paramas is not None else vgain_analysis_parameters(vgain)),
            federico_limits = federico_limits,
            federico_conversion = federico_conversion,
            coldbox_folder = coldbox_folder,
            output_folder = output_folder,
            show = show, 
            save_pdf = save_pdf
        )
        print("\ndone")
