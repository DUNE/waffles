import numpy as np
from plotly import graph_objects as pgo
from typing import List, Optional

from waffles.data_classes.WaveformSet import WaveformSet
from waffles.utils.numerical_utils import histogram2d
from waffles.Exceptions import generate_exception_message

def check_dimensions_of_suplots_figure( figure : pgo.Figure,
                                        nrows : int,
                                        ncols : int) -> None:
    
    """
    This function checks that the given figure has
    the given number of rows and columns. If not,
    it raises an exception.

    Parameters
    ----------
    figure : plotly.graph_objects.Figure
        The figure to be checked. It must have been
        generated using plotly.subplots.make_subplots()
        with a 'rows' and 'cols' parameters matching
        the given nrows and ncols parameters.
    nrows (resp. ncols) : int
        The number of rows (resp. columns) that the
        given figure must have

    Returns
    ----------
    None
    """

    try:
        fig_rows, fig_cols = figure._get_subplot_rows_columns() # Returns two range objects
        fig_rows, fig_cols = list(fig_rows)[-1], list(fig_cols)[-1]

    except Exception:   # Happens if figure was not created using plotly.subplots.make_subplots

        raise Exception(generate_exception_message( 1,
                                                    'check_dimensions_of_suplots_figure()',
                                                    'The given figure is not a subplot grid.'))
    if fig_rows != nrows or fig_cols != ncols:
        
        raise Exception(generate_exception_message( 2,
                                                    'check_dimensions_of_suplots_figure()',
                                                    f"The number of rows and columns in the given figure ({fig_rows}, {fig_cols}) must match the nrows ({nrows}) and ncols ({ncols}) parameters."))
    return

def update_shared_axes_status(  figure : pgo.Figure,
                                share_x : bool = False,
                                share_y : bool = True) -> pgo.Figure:
    
    """
    If share_x (resp. share_y) is True, then this
    function makes the x-axis (resp. y-axis) scale 
    of every subplot in the given figure shared.
    If share_x (resp. share_y) is False, then this
    function will reset the shared-status of the 
    x-axis (resp. y-axis) so that they are not 
    shared anymore. Finally, it returns the figure 
    with the shared y-axes.

    Parameters
    ----------
    figure : plotly.graph_objects.Figure
        The figure whose subplots will share the
        selected axes scale.
    share_x (resp. share_y): bool
        If True, the x-axis (resp. y-axis) scale will be
        shared among all the subplots. If False, the
        x-axis (resp. y-axis) scale will not be shared
        anymore.
    
    Returns
    ----------
    figure : plotly.graph_objects.Figure
    """
    
    try:
        fig_rows, fig_cols = figure._get_subplot_rows_columns() # Returns two range objects
    except Exception:   # Happens if figure was not created using plotly.subplots.make_subplots
        raise Exception(generate_exception_message( 1,
                                                    'update_shared_axes_status()',
                                                    'The given figure is not a subplot grid.'))
    
    fig_rows, fig_cols = list(fig_rows)[-1], list(fig_cols)[-1]

    aux_x = None if not share_x else 'x'
    aux_y = None if not share_y else 'y'
    
    for i in range(fig_rows):
        for j in range(fig_cols):
            figure.update_xaxes(matches=aux_x, row=i+1, col=j+1)
            figure.update_yaxes(matches=aux_y, row=i+1, col=j+1)

    return figure

def __add_no_data_annotation(   figure : pgo.Figure,
                                row : int,
                                col : int) -> pgo.Figure:
    
    """
    It is the caller's responsibility to ensure
    that the well-formedness of the input
    parameters, such as the fact that the
    'figure' parameter is, indeed, a pgo.Figure
    object which was created with the
    psu.make_subplots() method and contains
    at least 'row' rows and 'col' columns.
    No checks are performed in this function. This 
    function adds an empty trace and a centered 
    annotation displaying 'No data' to the given 
    figure at the given row and column. Finally, 
    this function returns the figure.

    Parameters
    ----------
    figure : pgo.Figure
        The figure where the annotation will be
        added
    row (resp. col) : int
        The row (resp. column) where the annotation
        will be added. These values are expected 
        to be 1-indexed, so they are directly passed 
        to the 'row' and 'col' parameters of the 
        plotly.graph_objects.Figure.add_trace() and 
        plotly.graph_objects.Figure.add_annotation()
        methods.

    Returns
    ----------
    figure_ : plotly.graph_objects.Figure
        The figure with the annotation added
    """

    figure_ = figure

    figure_.add_trace(  pgo.Scatter(x = [], 
                                    y = []), 
                        row = row, 
                        col = col)

    figure_.add_annotation( text = "No data",
                            xref = 'x domain',
                            yref = 'y domain',
                            x = 0.5,
                            y = 0.5,
                            showarrow = False,
                            font = dict(size = 14, 
                                        color='black'),
                            row = row,
                            col = col)
    return figure_

def get_string_of_first_n_integers_if_available(input_list : List[int],
                                                queried_no : int = 3) -> str:

    """
    This function returns an string with the first
    comma-separated n integers of the given list
    where n is the minimum between queried_no and 
    the length of the given list, input_list. If 
    n is 0, then the output is an empty string. 
    If n equals queried_no, (i.e. if queried_no
    is smaller than the length of the input list) 
    then the ',...' string is appended to the 
    output.

    Parameters
    ----------
    input_list : list of int
    queried_no : int
        It must be a positive integer

    Returns
    ----------
    output : str
    """

    if queried_no < 1:
        raise Exception(generate_exception_message( 1,
                                                    'get_string_of_first_n_integers_if_available()',
                                                    f"The given queried_no ({queried_no}) must be positive."))
    actual_no = queried_no
    fAppend = True

    if queried_no >= len(input_list):
        actual_no = len(input_list)
        fAppend = False

    output = ''

    for i in range(actual_no):
        output += (str(input_list[i])+',')

    output = output[:-1] if not fAppend else (output[:-1] + ',...')

    return output

def __subplot_heatmap(  waveform_set : WaveformSet, 
                        figure : pgo.Figure,
                        name : str,
                        row : int,
                        col : int,
                        wf_idcs : List[int],
                        analysis_label : str,
                        time_bins : int,
                        adc_bins : int,
                        ranges : np.ndarray,
                        show_color_bar : bool = False) -> pgo.Figure:

    """
    This is a helper function for the 
    plot_WaveformSet() function. It should only
    be called by that one, where the 
    data-availability and the well-formedness 
    checks of the input have already been 
    performed. No checks are performed in
    this function. For each subplot in the grid 
    plot generated by the plot_WaveformSet()
    function when its 'mode' parameter is
    set to 'heatmap', such function delegates
    plotting the heatmap to the current function.
    This function takes the given figure, and 
    plots on it the heatmap of the union of 
    the waveforms whose indices are contained 
    within the given 'wf_idcs' list. The 
    position of the subplot where this heatmap 
    is plotted is given by the 'row' and 'col' 
    parameters. Finally, this function returns 
    the figure.

    Parameters
    ----------
    waveform_set : WaveformSet
        The WaveformSet object whose waveforms
        will be plotted in the heatmap
    figure : pgo.Figure
        The figure where the heatmap will be
        plotted
    name : str
        The name of the heatmap. It is given
        to the 'name' parameter of 
        plotly.graph_objects.Heatmap().
    row (resp. col) : int
        The row (resp. column) where the 
        heatmap will be plotted. These values
        are expected to be 1-indexed, so they
        are directly passed to the 'row' and
        'col' parameters of the figure.add_trace()
        method.
    wf_idcs : list of int
        Indices of the waveforms, with respect
        to the waveform_set.Waveforms list, 
        which will be added to the heatmap.
    analysis_label : str
        For each considered waveform, it is the
        key for its Analyses attribute which gives
        the WfAna object whose computed baseline
        is subtracted from the waveform prior to
        addition to the heatmap. This function does
        not check that an analysis for such label
        exists.
    time_bins : int
        The number of bins for the horizontal axis
        of the heatmap
    adc_bins : int
        The number of bins for the vertical axis
        of the heatmap
    ranges : np.ndarray
        A 2x2 integer numpy array where ranges[0,0]
        (resp. ranges[0,1]) gives the lower (resp.
        upper) bound of the horizontal axis of the
        heatmap, and ranges[1,0] (resp. ranges[1,1])
        gives the lower (resp. upper) bound of the
        vertical axis of the heatmap.
    show_color_bar : bool
        It is given to the 'showscale' parameter of
        plotly.graph_objects.Heatmap(). If True, a
        bar with the color scale of the plotted 
        heatmap is shown. If False, it is not.
    
    Returns
    ----------
    figure_ : plotly.graph_objects.Figure
        The figure whose subplot at position 
        (row, col) has been filled with the heatmap
    """

    figure_ = figure

    time_step   = (ranges[0,1] - ranges[0,0]) / time_bins
    adc_step    = (ranges[1,1] - ranges[1,0]) / adc_bins
    
    aux_x = np.hstack([np.arange(   0,
                                    waveform_set.PointsPerWf,
                                    dtype = np.float32) + waveform_set.Waveforms[idx].TimeOffset for idx in wf_idcs])

    aux_y = np.hstack([waveform_set.Waveforms[idx].Adcs - waveform_set.Waveforms[idx].Analyses[analysis_label].Result.Baseline for idx in wf_idcs])

    aux = histogram2d(  np.vstack((aux_x, aux_y)), 
                        np.array((time_bins, adc_bins)),
                        ranges)
    
    heatmap =   pgo.Heatmap(z = aux,
                            x0 = ranges[0,0],
                            dx = time_step,
                            y0 = ranges[1,0],
                            dy = adc_step,
                            name = name,
                            transpose = True,
                            showscale = show_color_bar)

    figure_.add_trace(  heatmap,
                        row = row,
                        col = col)
    return figure_

def arrange_time_vs_ADC_ranges( waveform_set : WaveformSet,
                                time_range_lower_limit : Optional[int] = None,
                                time_range_upper_limit : Optional[int] = None,
                                adc_range_above_baseline : int = 100,
                                adc_range_below_baseline : int = 200) -> np.ndarray:
    
    """
    This function arranges a 2x2 numpy array with a time and 
    ADC range which is constrained to the number of points 
    in the waveforms of the given waveform_set, i.e. 
    waveform_set.PointsPerWf.
    
    Parameters
    ----------
    waveform_set : WaveformSet
        The WaveformSet object for which the time and ADC
        ranges will be built.
    time_range_lower_limit (resp. time_range_upper_limit) : int
        If it is defined, then it gives the lower (resp. upper) 
        limit of the time range, in time ticks. If it is not
        defined, then the lower (resp. upper) will be set to 
        0 (resp. waveform_set.PointsPerWf - 1). It must be 
        smaller (resp. greater) than time_range_upper_limit 
        (resp. time_range_lower_limit).
    adc_range_above_baseline (resp. adc_range_below_baseline) : int
        Its absolute value times one (resp. minus one) gives
        the upper (resp. lower) limit of the ADC range.

    Returns
    ----------
    np.ndarray
        It is a 2x2 numpy array, say output, where the time (resp. 
        ADC) range is given by [output[0, 0], output[0, 1]] (resp. 
        [output[1, 0], output[1, 1]]).
    """

    time_range_lower_limit_ = 0
    if time_range_lower_limit is not None:
        time_range_lower_limit_ = time_range_lower_limit

    time_range_upper_limit_ = waveform_set.PointsPerWf - 1
    if time_range_upper_limit is not None:
        time_range_upper_limit_ = time_range_upper_limit

    if time_range_lower_limit_ >= time_range_upper_limit_:
        raise Exception(generate_exception_message( 1,
                                                    'arrange_time_vs_ADC_ranges()',
                                                    f"The time range limits ({time_range_lower_limit_}, {time_range_upper_limit_}) are not well-formed."))
    
    return np.array([   [time_range_lower_limit_,           time_range_upper_limit_         ],
                        [-1*abs(adc_range_below_baseline),  abs(adc_range_above_baseline)   ]])