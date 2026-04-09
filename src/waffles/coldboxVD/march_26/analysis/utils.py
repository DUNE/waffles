from waffles.coldboxVD.march_26.analysis.imports import *


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

def print_correlated_gaussians_fit_parameters(my_grid, parameters_conversion: bool = False, show: bool = False):
    """
    Extract and compute Gaussian fit parameters for calibration from grid data.

    Calculates pedestal (μ₀, σ₀), gain (G = μ₁ - μ₀), SNR (G/σ₀), and std increment (Δσ = σ₁² - σ₀²) 
    with propagation of uncertainties. Supports optional /16 scaling.

    Parameters
    ----------
    my_grid : object
        Grid with `ch_map`, `ch_wf_sets`; accesses first channel's `calib_histo.gaussian_fits_parameters`.
    parameters_conversion : bool, default=False
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

    fg = 16 if parameters_conversion else 1

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

def single_vgain_analysis(
    membrane: str,
    channel: int,
    bias: str,
    vgain: int,
    led: int,
    dict_params,
    input_file: str = "",
    show: bool = False,
    save_pdf: bool = False,
    parameters_conversion: bool = True,
    output_folder: str = "output",
    output_files_comment : str = ''
):
    """
    Analisi per un singolo valore di VGAIN.
    """

    # Check input parameters
    validate_analysis_params(dict_params)

    # Check existance of input file
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File non trovato: {input_file}")

    print(f"\n\n------------------------------------------------------------------------ \n\n------ Membrane: {membrane} - Channel: {channel} - Led: {led} - Bias: {bias} - Vgain: {vgain} ------\n")


    wfset_original = create_waveform_set_from_spybuffer(filename=input_file, WFs=40000, length=1024, config_channel=channel)

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

    if dict_params['fixed_integral_limits'] is not None: 
        aux_limits = dict_params['fixed_integral_limits'] #(382, 406)
    else: 
        wfset_filtered_meanwf = WaveformSet.from_filtered_WaveformSet(wfset_filtered, adcs_threshold_filter, time_range = dict_params["time_range_afterpulse_for_meanwf"], adcs_minimum_threshold=-dict_params['adcs_threshold'], adcs_maximum_threshold=dict_params['adcs_threshold'])
        mean_wf = wfset_filtered_meanwf.compute_mean_waveform()
        aux_limits = my_integration_window(adcs_array = mean_wf.adcs, deviation_upper_limit=dict_params['deviation_upper_limit'])
    

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

    output_parameters = print_correlated_gaussians_fit_parameters(my_grid, parameters_conversion=parameters_conversion)

    spe_result = spe_amplitude_computation(wfset_filtered, channel, output_parameters, show_persistance = False, show_spe_hist = False)
    dynamic_range = dynamic_range_computation(spe_result['mean'][0])

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
                text=f"{membrane} CH{channel} - Led {led} - Bias {bias} - {vgain} Vgain",
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
            if output_files_comment != '':
                comment = '_' + output_files_comment
            else: 
                comment = output_files_comment
            output_filename = f"{output_folder}/membrane{membrane}_channel{channel}_led{led}_bias{bias}_vgain{vgain}{comment}.pdf"
            big_fig.write_image(output_filename)
            print(f"Pdf saved : {output_filename}\n")

        if show:
            big_fig.show()
    
    return {   
        "vgain": vgain,
        "channel": channel,                   
        "led": led,
        "bias": bias,                   
        "mean_0": output_parameters["mean_0"],
        "e_mean_0": output_parameters["e_mean_0"],
        "std_0": output_parameters["std_0"],
        "e_std_0": output_parameters["e_std_0"],
        "gain": output_parameters["gain"],
        "e_gain": output_parameters["e_gain"],
        "snr": output_parameters["snr"],
        "e_snr": output_parameters["e_snr"],
        "delta_std": output_parameters["delta_std"],
        "e_delta_std": output_parameters["e_delta_std"],
        "spe_amplitude": spe_result["mean"][0],
        "e_spe_amplitude": spe_result["mean"][1],
        "dynamic range" : dynamic_range
    }


##########################################################################################################

# NO MORE WORKING 26/03/2026 --> starting modifing it on 30/03/2026 to have possibility to use auto parameters or external ones passed as argument in a dict with vgain as key

def vgain_analysis_parameters(vgain):

    v_index = (vgain - 500) // 100
    heatmap_min = (-200 + v_index * 10)
    heatmap_max =( 700 - v_index * 50 if vgain <= 1600 else 100)

    adcs_threshold = (
        100 - v_index * 10 if vgain < 900 else
        80 - (v_index-4) * 5  if vgain <= 1300 else
        30 - (v_index-8) * 3  if vgain <= 1900 else
        11
    )
    
    n_std_baseline = (1 if vgain <= 800 else 1.5)

    max_peaks=  7
    prominence= 0.5
    initial_percentage=0.1
    percentage_step=0.02
    ch_span_fraction_around_peaks=0.05 

    baseline_timeticks_limit = 120
    deviation = None

    return {
        'baseline_timeticks_limit': baseline_timeticks_limit,
        'deviation_upper_limit': None,
        "fixed_integral_limits":[126,150],
        'heatmap_min': heatmap_min,
        'heatmap_max': heatmap_max,
        'adcs_threshold': adcs_threshold,
        'n_std_baseline': n_std_baseline,
        'max_peaks': max_peaks, 
        'prominence': prominence, 
        'initial_percentage': initial_percentage,
        'percentage_step': percentage_step,
        'ch_span_fraction_around_peaks': ch_span_fraction_around_peaks,
        "time_range_afterpulse_for_meanwf": [300,-1]
    }




##########################################################################################################


def channel_vgain_scan_analysis(
        membrane: str, 
        channel: int, 
        bias: str, 
        vgain_list,
        led: int, 
        input_filename_path_style: str, # Example of complete .dat filepath where between {} you use the exact name of the variables you need: = "/eos/experiment/neutplatform/protodune/experiments/ColdBoxVD/2026March/20260324_sof_ch18to21_vgain_scan_led270_i830_800_930_920_biasmap_remote_r1/vgain_{vgain:04d}/afe2bias_0000/led_{led}_ch{channel}/channel_{channel}.dat"
        input_filename_variable: List[str], # Example: ['vgain', 'led', 'channel'] i.e. list of the variable name to use in the filename path 
        external_dict_paramas : Optional[dict] = None,
        show: bool = False,
        save_pdf: bool = True,
        output_folder: str = "output",
        output_files_comment: str = '', #additional str to add to the output filename 
        parameters_conversion: bool = True
        ):
    """
    Sequential Vgain scan analysis for fixed membrane/channel/bias across vgain_list.

    Loops over vgain_list, calling single_vgain_analysis() with auto/override params. 
    """

    rows = []
    output_folder_analysis = f"output/membrane_{membrane}/channel_{channel}/bias_{bias:04d}"
    if not os.path.exists(output_folder_analysis):
        os.makedirs(output_folder_analysis)

    for vgain in vgain_list:
        input_file = _build_input_filepath(
            input_filename_path_style,
            bias=bias,
            vgain=vgain,
            led=led,
            channel=channel,
        )
        row = single_vgain_analysis(
            membrane=membrane,
            channel=channel,
            bias=bias,
            vgain=vgain,
            led = led,
            dict_params = (external_dict_paramas[vgain] if external_dict_paramas is not None else vgain_analysis_parameters(vgain)),
            input_file= input_file, 
            parameters_conversion = parameters_conversion,
            output_folder = output_folder_analysis,
            output_files_comment = output_files_comment,
            show = show, 
            save_pdf = save_pdf
        )

        rows.append(row)

    out_df = pd.DataFrame(rows)

            
    if output_files_comment != '':
        comment = '_' + output_files_comment
    else:
        comment = output_files_comment

    csv_filename = f"{output_folder_analysis}/membrane{membrane}_channel{channel}_led{led}_bias{bias}_vgain_scan{comment}.csv"
    out_df.to_csv(csv_filename, index=False)
    print(f"Results in {output_folder_analysis}")



##########################################################################################################


def my_integration_window(
    adcs_array: np.ndarray,
    deviation_upper_limit: float = 0.75,
    deviation_lower_limit: float = 0.2,
    lower_limit_window: int = 10
) -> Tuple[int, int]:
    """This function takes an unidimensional numpy array
    representing the pulse signal, in ADCs, coming from an SiPM.
    It returns a tuple of two integers representing the lower
    and upper limits, respectively, of the SiPM pulse in the
    given array. The lower and upper limits are computed looking
    at the points at which adcs reach the threshold = deviation*peak

    Parameters
    ----------
    adcs_array: numpy.ndarray
        The input signal to analyze
    deviation: float
        It must be greater than 0.0 and smaller than 1.0. 

    lower_limit_window: int = 10
        Window before the peak where search for lower limit
        
    Returns
    -------
    Tuple[int, int]
        A tuple containing the lower and upper limits of the
        pulse window.
    """
    
    idx_max = np.argmax(adcs_array)
    peak = adcs_array[idx_max]
    lower_threshold = deviation_lower_limit * peak
    upper_threshold = deviation_upper_limit * peak

    start = idx_max - lower_limit_window
    lower_window = adcs_array[start:idx_max]

    candidates = np.where(lower_window >= lower_threshold)[0]
    if len(candidates) > 0:
        lower_limit = start + candidates[0]
    else:
        lower_limit = start  

    upper_window = adcs_array[idx_max:]
    candidates = np.where(upper_window <= upper_threshold)[0]
    if len(candidates) > 0:
        upper_limit = idx_max + candidates[0]
    else:
        upper_limit = len(adcs_array) - 1  

    return (lower_limit, upper_limit)



##########################################################################################################


def search_integration_window(       
        wfset,
        dict_params,
        channel: int,
        deviation_range = (0.4,0.8),
        null_baseline_analysis_label:str =  'null_baseliner',
        show_meanwf : bool = True,
        parameters_conversion : bool = True,
        deviation_step = 0.05): 
    
    deviation_list = np.arange(deviation_range[0], deviation_range[1], deviation_step)
    
    wfset_meanwf = WaveformSet.from_filtered_WaveformSet(wfset, adcs_threshold_filter, time_range = dict_params["time_range_afterpulse_for_meanwf"], adcs_minimum_threshold=-dict_params['adcs_threshold'], adcs_maximum_threshold=dict_params['adcs_threshold'])
    
    mean_wf = wfset_meanwf.compute_mean_waveform()

    if show_meanwf:
        fig, axes = plt.subplots(len(deviation_list), 1, figsize=(12, 3*len(deviation_list)), sharex=True)

    
    subplot_titles = []
    for deviation in deviation_list: 
        subplot_titles.append(f"Calibration plot: integral threshold {deviation*100:.0f}% peak")
    big_fig_calib = make_subplots(rows= len(deviation_list), cols=1, subplot_titles=subplot_titles)            

    i=1
    for deviation in deviation_list:
        
        aux_limits = my_integration_window(adcs_array = mean_wf.adcs, deviation_upper_limit = deviation)

        integration_analysis_label = 'integration_analysis'
        integrator_input_parameters = IPDict({'baseline_analysis': null_baseline_analysis_label, 'inversion': False, 'int_ll': aux_limits[0], 'int_ul': aux_limits[1], 'amp_ll': aux_limits[0], 'amp_ul': aux_limits[1]})
        checks_kwargs = IPDict({ 'points_no': wfset.points_per_wf})
        _ = wfset.analyse( integration_analysis_label, WindowIntegrator, integrator_input_parameters, checks_kwargs=checks_kwargs, overwrite=True)

        hist_domain, hist_nbins, hist_bins_width = auto_histogram(wfset, integration_analysis_label, show_results=False)

        my_grid = coldbox_single_channel_grid(wfset, config_channel=channel)

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
        # if len(params['scale']) > 10:
        #     print(f"Warning: More than 10 peaks found for membrane {membrane}, channel {channel}, bias {bias}, vgain {vgain}.")

        try: 
            output_parameters = print_correlated_gaussians_fit_parameters(my_grid, parameters_conversion=parameters_conversion)

            fig_calib = plot_ChannelWsGrid(
                my_grid, 
                mode='calibration',
                plot_peaks_fits=True,           
                plot_sum_of_gaussians=True      
                )
            add_fig_to_subplot(fig_calib, big_fig_calib, i, 1)


            annotation_text = build_calibration_annotation(
                output_parameters,
                hist_domain,
                hist_nbins,
                hist_bins_width
            )

            big_fig_calib.add_annotation(
                text=annotation_text,
                xref = f"x{i} domain" if i > 1 else "x domain",
                yref = f"y{i} domain" if i > 1 else "y domain",
                x=0.98,
                y=1,
                showarrow=False,
                align="left",
                font=dict(size=9),  
                bgcolor="rgba(255,255,255,0.85)",
                borderwidth=1
            )
        except:
            print('something went wrong with output_parameters computation')

        if show_meanwf:
            axes[i-1].plot(np.arange(1024), mean_wf.adcs, label="Mean wf")
            
            axes[i-1].axvline(aux_limits[0], linestyle="--", color="red",
                    linewidth=1, label=f"Lower limit ({aux_limits[0]:.0f})")
            axes[i-1].axvline(aux_limits[1], linestyle="--", color="blue",
                    linewidth=1, label=f"Upper limit ({aux_limits[1]:.0f})")
            
            # soglia orizzontale basata su deviation
            threshold = deviation * np.max(mean_wf.adcs)
            axes[i-1].axhline(threshold, linestyle='--', color='orange',
                    linewidth=1, label=f"Threshold ({deviation*100:.0f}% peak)")
            
            # baseline
            axes[i-1].axhline(0, linestyle='--', color='yellow', linewidth=1, label="Baseline")
            
            axes[i-1].set_ylabel("Adcs")
            axes[i-1].set_title(f"Mean waveform - Deviation = {deviation:.2f}")
            axes[i-1].legend(fontsize=8)

        i+=1

   
    big_fig_calib.update_layout(title=dict(
        text=f"Integral window study", #{membrane} CH{channel} - Bias {bias} - {vgain} Vgain
        x=0.5,                 
        xanchor="center",
        y=0.97,                
        yanchor="top",
        font=dict(size=18))
    )

    big_fig_calib.update_layout(
    height=1000,
    width=1000,
    showlegend=False)

    big_fig_calib.show()


    if show_meanwf:
        axes[-1].set_xlabel("Time ticks (AU)")
        plt.tight_layout()
        fig.show()


##########################################################################################################


def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def chi2_func(y, y_fit, yerr, n_params):
    """
    Calcola chi-quadro e chi-quadro ridotto.
    """
    residuals = (y - y_fit) / yerr
    chi2_val = np.sum(residuals**2)
    ndf = len(y) - n_params
    chi2_red = chi2_val / ndf
    return chi2_val, chi2_red, ndf

def r2_score(y, y_fit):
    """
    Calcola il coefficiente di determinazione R².
    """
    ss_res = np.sum((y - y_fit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot
    return r2


##########################################################################################################

def spe_filter_integral(
    waveform: Waveform,
    spe_integral_mean : float,
    spe_integral_sigma : float, 
    integration_analysis_label : str = 'integration_analysis',
    n_sigma : float = 1
    ) -> bool:
    
    if waveform.analyses[integration_analysis_label].result['integral'] > (spe_integral_mean-n_sigma*spe_integral_sigma) and waveform.analyses[integration_analysis_label].result['integral'] < (spe_integral_mean+n_sigma*spe_integral_sigma):
        return True
    else:
        return False


def spe_amplitude_computation(wfset_filtered, channel, output_parameters, n_sigma_spe_selection : float = 1., integration_analysis_label : str = 'integration_analysis', show_persistance: bool = False, show_spe_hist: bool = True):
    wfset_spe = WaveformSet.from_filtered_WaveformSet(wfset_filtered, spe_filter_integral, spe_integral_mean = output_parameters['params']['mean'][1][0], spe_integral_sigma = output_parameters['params']['std'][1][0], integration_analysis_label = integration_analysis_label, n_sigma = n_sigma_spe_selection)

    if show_persistance: 
        persistance_plot_helper(wfset_spe, channel, ymin = -100, ymax = 100, adc_bins = 1000, show=show_persistance)

    amplitude_list = []
    for wf in wfset_spe.waveforms:
        amplitude_list.append(wf.analyses[integration_analysis_label].result['amplitude'])

    data = np.array(amplitude_list)

    counts, bin_edges = np.histogram(data, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    A0 = np.max(counts)
    mu0 = np.mean(data)
    sigma0 = np.std(data)

    # Fit
    popt, pcov = curve_fit(gauss, bin_centers, counts, p0=[A0, mu0, sigma0])

    A_fit, mu_fit, sigma_fit = popt

    perr = np.sqrt(np.diag(pcov))
    A_err, mu_err, sigma_err = perr

    if show_spe_hist: 
        plt.figure()
        plt.hist(data, bins=50, histtype='step', linewidth=1.5,
                label=f"Data (entries = {len(data)})")


        x = np.linspace(min(data), max(data), 1000)
        y = gauss(x, *popt)

        plt.plot(
            x, y, linewidth=2,
            label=(
                "Gaussian fit\n"
                f"A = {fmt(A_fit, A_err)}\n"
                f"μ = {fmt(mu_fit, mu_err)}\n"
                f"σ = {fmt(sigma_fit, sigma_err)}"
            )
        )

        plt.xlabel("Amplitude (ADCs)")
        plt.ylabel("Counts")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.show()

    # print(("SPE amplitude evalutation: \n"
    #     f"A = {fmt(A_fit, A_err)}\n"
    #     f"μ = {fmt(mu_fit, mu_err)}\n"
    #     f"σ = {fmt(sigma_fit, sigma_err)}"))


    return {'mean': (mu_fit, mu_err), 'sigma': (sigma_fit,sigma_err)}



def dynamic_range_computation(spe_adc, daphne_range: int = 2**14): # no cold electronic saturation
    return daphne_range/spe_adc

##########################################################################################################

def intersection_numeric(A1, mu1, sigma1, A2, mu2, sigma2):
    def f(x):
        return gauss(x, A1, mu1, sigma1) - gauss(x, A2, mu2, sigma2)
    
    try:
        return brentq(f, mu1, mu2)  # cerca tra i due picchi
    except ValueError:
        return None

def gaussian_intersections(output_parameters):
    params = output_parameters["params"]

    scales = [s[0] for s in params["scale"]]
    means  = [m[0] for m in params["mean"]]
    stds   = [s[0] for s in params["std"]]

    intersections = []

    for i in range(len(means) - 1):
        x = intersection_numeric(
            scales[i], means[i], stds[i],
            scales[i+1], means[i+1], stds[i+1]
        )
        intersections.append(x)

    return intersections

def calibration_histogram_peak_counts(wfset_filtered, output_parameters, integration_analysis_label :str= 'integration_analysis'):
    
    bounds = [-10000]
    bounds.extend(gaussian_intersections(output_parameters))
    bounds.append(100000)

    num_peaks = output_parameters['num_peaks']

    for i_peak in range(num_peaks):
        count = 0

        if bounds[i_peak] is None:
            continue
        elif bounds[i_peak + 1] is None:
            bounds[i_peak + 1] = 100000
            break

        for wf in wfset_filtered.waveforms:
            val = wf.analyses[integration_analysis_label].result['integral']
            
            if bounds[i_peak] < val < bounds[i_peak + 1]:
                count += 1

        print(f"{i_peak} peak: {count} counts")


##########################################################################################################

def _build_input_filepath(template: str, **kwargs) -> str:
    return template.format(**kwargs)


##########################################################################################################

def validate_analysis_params(dict_params: Dict) -> None:
    """
    Validate the content of dict_params for channel_vgain_scan_analysis.
        - "baseline_timeticks_limit" (int): Baseline is computed from timetick 0 to this limit.
        - "deviation_upper_limit" (float or null/None): Percentage of the mean waveform peak amplitude used for integral computation. Must be set if `fixed_integral_limits` is null/None.
        - "fixed_integral_limits" (tuple of 2 floats or null/None): Lower and upper limits for integral computation. Must be set if `deviation_upper_limit` is null/None.
        - "heatmap_min" (float): Minimum amplitude (ADCs) for hitmap.
        - "heatmap_max" (float): Maximum amplitude (ADCs) for hitmap.
        - "adcs_threshold" (float): Max/min ADC acceptable in baseline range for waveform selection.
        - "n_std_baseline" (float): Number of baseline standard deviations accepted in baseline range.
        - "max_peaks" (int): Maximum peaks to consider in calibration histogram fit.
        - "prominence" (float): Peak prominence in histogram fit.
        - "initial_percentage" (float): Initial percentage used in histogram fit.
        - "percentage_step" (float): Step percentage used in histogram fit.
        - "ch_span_fraction_around_peaks" (float): Fraction of channel span around peaks used in fit.
        - "time_range_afterpulse_for_meanwf" (list of 2 ints): Time region used to cut afterpulses when `deviation_upper_limit` is not None.

    Raises
    ------
    ValueError
        If any rule is violated.
    """

    required_keys = [
        "baseline_timeticks_limit",
        "deviation_upper_limit",
        "fixed_integral_limits",
        "heatmap_min",
        "heatmap_max",
        "adcs_threshold",
        "n_std_baseline",
        "max_peaks",
        "prominence",
        "initial_percentage",
        "percentage_step",
        "ch_span_fraction_around_peaks",
        "time_range_afterpulse_for_meanwf",
    ]

    # ---- Check all keys exist
    for key in required_keys:
        if key not in dict_params:
            raise ValueError(f"Missing required parameter: '{key}'")

    # ---- Type checks
    if not isinstance(dict_params["baseline_timeticks_limit"], int):
        raise ValueError("baseline_timeticks_limit must be int")

    if dict_params["deviation_upper_limit"] is not None:
        if not isinstance(dict_params["deviation_upper_limit"], (int, float)):
            raise ValueError("deviation_upper_limit must be float or None")

    fil = dict_params["fixed_integral_limits"]
    if fil is not None:
        if (
            not isinstance(fil, list)
            or len(fil) != 2
            or not all(isinstance(x, (int, float)) for x in fil)
        ):
            raise ValueError(
                "fixed_integral_limits must be a list of two numbers (low, high) or None"
            )

    if not isinstance(dict_params["heatmap_min"], (int, float)):
        raise ValueError("heatmap_min must be a number")

    if not isinstance(dict_params["heatmap_max"], (int, float)):
        raise ValueError("heatmap_max must be a number")

    if not isinstance(dict_params["adcs_threshold"], (int, float)):
        raise ValueError("adcs_threshold must be a number")

    if not isinstance(dict_params["n_std_baseline"], (int, float)):
        raise ValueError("n_std_baseline must be a number")

    if not isinstance(dict_params["max_peaks"], int):
        raise ValueError("max_peaks must be int")

    for k in ["prominence", "initial_percentage", "percentage_step", "ch_span_fraction_around_peaks"]:
        if not isinstance(dict_params[k], (int, float)):
            raise ValueError(f"{k} must be a number")

    tr = dict_params["time_range_afterpulse_for_meanwf"]

    if (
        not isinstance(tr, list)
        or len(tr) != 2
        or not isinstance(tr[0], int)
        or not (isinstance(tr[1], int) and (tr[1] >= 0 or tr[1] == -1))
    ):
        raise ValueError(
            "time_range_afterpulse_for_meanwf must be a tuple of two integers, "
            "where the second can be -1"
        )

    # ---- Logical rules

    # One of the two integration methods must be provided
    if dict_params["deviation_upper_limit"] is None and fil is None:
        raise ValueError(
            "You must provide either 'deviation_upper_limit' or 'fixed_integral_limits'"
        )

    # They cannot be both set
    if dict_params["deviation_upper_limit"] is not None and fil is not None:
        raise ValueError(
            "'deviation_upper_limit' and 'fixed_integral_limits' are mutually exclusive"
        )

    # Heatmap limits
    if dict_params["heatmap_min"] >= dict_params["heatmap_max"]:
        raise ValueError("heatmap_min must be smaller than heatmap_max")
    
