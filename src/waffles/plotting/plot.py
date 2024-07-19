from typing import Optional
from plotly import graph_objects as pgo
from plotly import subplots as psu

from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.Map import Map
from waffles.Exceptions import generate_exception_message

from waffles.plotting.plot_utils import check_dimensions_of_suplots_figure
from waffles.plotting.plot_utils import update_shared_axes_status
from waffles.plotting.plot_utils import get_string_of_first_n_integers_if_available
from waffles.plotting.plot_utils import __add_no_data_annotation
from waffles.plotting.plot_utils import __subplot_heatmap
from waffles.plotting.plot_utils import arrange_time_vs_ADC_ranges


def plot_WaveformSet(   waveform_set : WaveformSet,  
                        *args,
                        nrows : int = 1,
                        ncols : int = 1,
                        figure : Optional[pgo.Figure] = None,
                        wfs_per_axes : Optional[int] = 1,
                        map_of_wf_idcs : Optional[Map] = None,
                        share_x_scale : bool = False,
                        share_y_scale : bool = False,
                        mode : str = 'overlay',
                        analysis_label : Optional[str] = None,
                        plot_analysis_markers : bool = False,
                        show_baseline_limits : bool = False, 
                        show_baseline : bool = True,
                        show_general_integration_limits : bool = False,
                        show_general_amplitude_limits : bool = False,
                        show_spotted_peaks : bool = True,
                        show_peaks_integration_limits : bool = False,
                        time_bins : int = 512,
                        adc_bins : int = 100,
                        time_range_lower_limit : Optional[int] = None,
                        time_range_upper_limit : Optional[int] = None,
                        adc_range_above_baseline : int = 100,
                        adc_range_below_baseline : int = 200,
                        detailed_label : bool = True,
                        **kwargs) -> pgo.Figure: 

    """ 
    This function returns a plotly.graph_objects.Figure 
    with a nrows x ncols grid of axes, with plots of
    some of the waveforms in the given WaveformSet object.

    Parameters
    ----------
    waveform_set : WaveformSet
        The WaveformSet object which contains the 
        waveforms to be plotted.
    *args
        These arguments only make a difference if the
        'mode' parameter is set to 'average' and the
        'analysis_label' parameter is not None. In such
        case, these are the positional arguments handled 
        to the WaveformAdcs.analyse() instance method of 
        the computed mean waveform. I.e. for the mean 
        waveform wf, the call to its analyse() method
        is wf.analyse(analysis_label, *args, **kwargs).
        The WaveformAdcs.analyse() method does not 
        perform any well-formedness checks, so it is 
        the caller's responsibility to ensure so for 
        these parameters.
    nrows (resp. ncols) : int
        Number of rows (resp. columns) of the returned 
        grid of axes.
    figure : plotly.graph_objects.Figure
        If it is not None, then it must have been
        generated using plotly.subplots.make_subplots()
        (even if nrows and ncols equal 1). It is the
        caller's responsibility to ensure this.
        If that's the case, then this function adds the
        plots to this figure and eventually returns 
        it. In such case, the number of rows (resp. 
        columns) in such figure must match the 'nrows' 
        (resp. 'ncols') parameter.
    wfs_per_axes : int
        If it is not None, then the argument given to 
        'map_of_wf_idcs' will be ignored. In this case,
        the number of waveforms considered for each
        axes is wfs_per_axes. P.e. for wfs_per_axes 
        equal to 2, the axes at the first row and first
        column contains information about the first
        two waveforms in the set. The axes in the first 
        row and second column will consider the 
        following two, and so on.
    map_of_wf_idcs : Map of lists of integers
        This Map must contain lists of integers.
        map_of_wf_idcs.Data[i][j] gives the indices of the 
        waveforms, with respect to the given WaveformSet, 
        waveform_set, which should be considered for 
        plotting in the axes which are located at the i-th 
        row and j-th column.
    share_x_scale (resp. share_y_scale) : bool
        If True, the x-axis (resp. y-axis) scale will be 
        shared among all the subplots.
    mode : str
        This parameter should be set to 'overlay', 'average',
        or 'heatmap'. If any other input is given, an
        exception will be raised. The default setting is 
        'overlay', which means that all of the considered 
        waveforms will be plotted. If it set to 'average', 
        instead of plotting every waveform, only the 
        averaged waveform of the considered waveforms will 
        be plotted. If it is set to 'heatmap', then a 
        2D-histogram, whose entries are the union of all 
        of the points of every considered waveform, will 
        be plotted. In the 'heatmap' mode, the baseline 
        of each waveform is subtracted from each waveform 
        before plotting. Note that to perform such a 
        correction, the waveforms should have been 
        previously analysed, so that at least one baseline
        value is available. The analysis which gave the 
        baseline value which should be used is specified
        via the 'analysis_label' parameter. Check its
        documentation for more information.
    analysis_label : str
        The meaning of this parameter varies slightly
        depending on the value given to the 'mode'
        parameter. 
            If mode is set to 'overlay', then this 
        parameter is optional and it only makes a 
        difference if the 'plot_analysis_markers' 
        parameter is set to True. In such case, this 
        parameter is given to the 'analysis_label'
        parameter of the Waveform.plot() (actually 
        WaveformAdcs.plot()) method for each WaveformAdcs 
        object(s) which will be plotted. This parameter 
        gives the key for the WfAna object within the 
        Analyses attribute of each plotted waveform from 
        where to take the information for the analysis 
        markers plot. In this case, if 'analysis_label' 
        is None, then the last analysis added to 
        the Analyses attribute will be the used one. 
            If mode is set to 'average' and this 
        parameter is defined, then this function will 
        analyse the newly computed average waveform, 
        say wf, by calling 
        wf.analyse(analysis_label, *args, **kwargs).
        Additionally, if the 'plot_analysis_markers'
        parameter is set to True and this parameter
        is defined, then this parameter is given to 
        the 'analysis_label' parameter of the wf.plot() 
        method, i.e. the analysis markers for the 
        plotted average waveform are those of the 
        newly computed analysis. This parameter gives 
        the key for the WfAna object within the 
        Analyses attribute of the average waveform 
        where to take the information for the analysis 
        markers plot.
            If 'mode' is set to 'heatmap', this 
        parameter is not optional, i.e. it must be 
        defined, and gives the analysis whose baseline 
        will be subtracted from each waveform before 
        plotting. In this case, it will not be checked 
        that, for each waveform, the analysis with the 
        given label is available. It is the caller's 
        responsibility to ensure so.
    plot_analysis_markers : bool
        This parameter only makes a difference if the
        'mode' parameter is set to 'overlay' or 'average'.
            If mode is set to 'overlay', then this 
        parameter is given to the 
        'plot_analysis_markers' argument of the 
        WaveformAdcs.plot() method for each waveform in 
        which will be plotted. 
            If mode is set to 'average' and the
        'analysis_label' parameter is defined, then this
        parameter is given to the 'plot_analysis_markers'
        argument of the WaveformAdcs.plot() method for
        the newly computed average waveform. If the
        'analysis_label' parameter is not defined, then
        this parameter will be automatically interpreted
        as False.
            In both cases, if True, analysis markers 
        for the plotted WaveformAdcs objects will 
        potentially be plotted together with each 
        waveform. For more information, check the 
        'plot_analysis_markers' parameter documentation 
        in the WaveformAdcs.plot() method. If False, no 
        analysis markers will be plot.
    show_baseline_limits : bool
        This parameter only makes a difference if the
        'mode' parameter is set to 'overlay' or 'average',
        and the 'plot_analysis_markers' parameter is set 
        to True. In that case, this parameter means 
        whether to plot vertical lines framing the 
        intervals which were used to compute the baseline.
    show_baseline : bool
        This parameter only makes a difference if the
        'mode' parameter is set to 'overlay' or 'average',
        and the 'plot_analysis_markers' parameter is set 
        to True. In that case, this parameter means whether 
        to plot an horizontal line matching the computed 
        baseline.
    show_general_integration_limits : bool
        This parameter only makes a difference if the
        'mode' parameter is set to 'overlay' or 'average',
        and the 'plot_analysis_markers' parameter is set 
        to True. In that case, this parameter means whether 
        to plot vertical lines framing the general 
        integration interval.
    show_general_amplitude_limits : bool
        This parameter only makes a difference if the
        'mode' parameter is set to 'overlay' or 'average',
        and the 'plot_analysis_markers' parameter is set 
        to True. In that case, this parameter means whether 
        to plot vertical lines framing the general 
        amplitude interval.
    show_spotted_peaks : bool
        This parameter only makes a difference if the
        'mode' parameter is set to 'overlay' or 'average',
        and the 'plot_analysis_markers' parameter is set 
        to True. In that case, this parameter means whether 
        to plot a triangle marker over each spotted peak.
    show_peaks_integration_limits : bool
        This parameter only makes a difference if the
        'mode' parameter is set to 'overlay' or 'average',
        and the 'plot_analysis_markers' parameter is set 
        to True. In that case, this parameter means whether 
        to plot two vertical lines framing the integration 
        interval for each spotted peak.
    time_bins (resp. adc_bins) : int
        This parameter only makes a difference if the 'mode'
        parameter is set to 'heatmap'. In that case, it is
        the number of bins along the horizontal (resp. 
        vertical) axis, i.e. the time (resp. ADCs) axis.
    time_range_lower_limit (resp. time_range_upper_limit) : int
        This parameter only makes a difference if the
        'mode' parameter is set to 'heatmap'. In such case,
        it gives the inclusive lower (resp. upper) limit of 
        the time range, in time ticks, which will be considered 
        for the heatmap plot. If it is not defined, then it 
        is assumed to be 0 (resp. waveform_set.PointsPerWf - 1).
        It must be smaller (resp. greater) than
        time_range_upper_limit (resp. time_range_lower_limit).
    adc_range_above_baseline (resp. adc_range_below_baseline) : int
        This parameter only makes a difference if the
        'mode' parameter is set to 'heatmap'. In that case,
        its absolute value times one (resp. minus one) is 
        the upper (resp. lower) limit of the ADCs range 
        which will be considered for the heatmap plot. 
        Note that, in this case, each waveform is 
        corrected by its own baseline.
    detailed_label : bool
        This parameter only makes a difference if
        the 'mode' parameter is set to 'average' or
        'heatmap', respectively. If the 'mode' parameter
        is set to 'average', then this parameter means
        whether to show the iterator values of the two
        first available waveforms (which were used to
        compute the mean waveform) in the label of the
        mean waveform plot. If the 'mode' parameter is 
        set to 'heatmap', then this parameter means 
        whether to show the iterator values of the two 
        first available waveforms (which were used to 
        compute the 2D-histogram) in the top annotation 
        of each subplot.
    **kwargs
        These arguments only make a difference if the
        'mode' parameter is set to 'average' and the
        'analysis_label' parameter is not None. In such
        case, these are the keyword arguments handled 
        to the WaveformAdcs.analyse() instance method of 
        the computed mean waveform. I.e. for the mean 
        waveform wf, the call to its analyse() method
        is wf.analyse(analysis_label, *args, **kwargs).
        The WaveformAdcs.analyse() method does not 
        perform any well-formedness checks, so it is 
        the caller's responsibility to ensure so for 
        these parameters.
            
    Returns
    ----------
    figure : plotly.graph_objects.Figure
        The figure with the grid plot of the waveforms
    """

    if nrows < 1 or ncols < 1:
        raise Exception(generate_exception_message( 1,
                                                    'plot_WaveformSet()',
                                                    'The number of rows and columns must be positive.'))
    if figure is not None:
        check_dimensions_of_suplots_figure( figure,
                                            nrows,
                                            ncols)
        figure_ = figure
    else:
        figure_ = psu.make_subplots(    rows = nrows, 
                                        cols = ncols)

    data_of_map_of_wf_idcs = None         # Logically useless

    if wfs_per_axes is not None:    # wfs_per_axes is defined, so ignore map_of_wf_idcs

        if wfs_per_axes < 1:
            raise Exception(generate_exception_message( 2,
                                                        'plot_WaveformSet()',
                                                        'The number of waveforms per axes must be positive.'))

        data_of_map_of_wf_idcs = waveform_set.get_map_of_wf_idcs(   nrows,
                                                                    ncols,
                                                                    wfs_per_axes = wfs_per_axes).Data

    elif map_of_wf_idcs is None:    # Nor wf_per_axes, nor 
                                    # map_of_wf_idcs are defined

        raise Exception(generate_exception_message( 3,
                                                    'plot_WaveformSet()',
                                                    "The 'map_of_wf_idcs' parameter must be defined if wfs_per_axes is not."))
    
    elif not Map.list_of_lists_is_well_formed(  map_of_wf_idcs.Data,    # wf_per_axes is not defined, 
                                                nrows,                  # but map_of_wf_idcs is, but 
                                                ncols):                 # it is not well-formed
        
        raise Exception(generate_exception_message( 4,
                                                    'plot_WaveformSet()',
                                                    f"The given map_of_wf_idcs is not well-formed according to nrows ({nrows}) and ncols ({ncols})."))
    else:   # wf_per_axes is not defined,
            # but map_of_wf_idcs is,
            # and it is well-formed

        data_of_map_of_wf_idcs = map_of_wf_idcs.Data

    update_shared_axes_status(  figure_,                    # An alternative way is to specify 
                                share_x = share_x_scale,    # shared_xaxes=True (or share_yaxes=True)
                                share_y = share_y_scale)    # in psu.make_subplots(), but, for us, 
                                                            # that alternative is only doable for 
                                                            # the case where the given 'figure'
                                                            # parameter is None.
    if mode == 'overlay':
        for i in range(nrows):
            for j in range(ncols):
                if len(data_of_map_of_wf_idcs[i][j]) > 0:
                    for k in data_of_map_of_wf_idcs[i][j]:

                        aux_name = f"({i+1},{j+1}) - Wf {k}, Ch {waveform_set.Waveforms[k].Channel}, Ep {waveform_set.Waveforms[k].Endpoint}"

                        waveform_set.Waveforms[k].plot( figure = figure_,
                                                        name = aux_name,
                                                        row = i + 1,  # Plotly uses 1-based indexing
                                                        col = j + 1,
                                                        plot_analysis_markers = plot_analysis_markers,
                                                        show_baseline_limits = show_baseline_limits,
                                                        show_baseline = show_baseline,
                                                        show_general_integration_limits = show_general_integration_limits,
                                                        show_general_amplitude_limits = show_general_amplitude_limits,
                                                        show_spotted_peaks = show_spotted_peaks,
                                                        show_peaks_integration_limits = show_peaks_integration_limits,
                                                        analysis_label = analysis_label)
                else:
                    __add_no_data_annotation(   figure_,
                                                i + 1,
                                                j + 1)
    elif mode == 'average':
        for i in range(nrows):
            for j in range(ncols):

                try: 
                    aux = waveform_set.compute_mean_waveform(wf_idcs = data_of_map_of_wf_idcs[i][j])    # WaveformSet.compute_mean_waveform() will raise an
                                                                                                        # exception if data_of_map_of_wf_idcs[i][j] is empty

                except Exception:       ## At some point we should implement a number of exceptions which are self-explanatory,
                                        ## so that we can handle in parallel exceptions due to different reasons if we need it
                    
                    __add_no_data_annotation(   figure_,
                                                i + 1,
                                                j + 1)
                    continue

                fAnalyzed = False
                if analysis_label is not None:
                    
                    _ = aux.analyse(    analysis_label,
                                        *args,
                                        **kwargs)
                    fAnalyzed = True

                aux_name = f"{len(data_of_map_of_wf_idcs[i][j])} Wf(s)"
                if detailed_label:
                    aux_name += f": [{get_string_of_first_n_integers_if_available(data_of_map_of_wf_idcs[i][j], queried_no = 2)}]"

                aux.plot(   figure = figure_,
                            name = f"({i+1},{j+1}) - Mean of " + aux_name,
                            row = i + 1,
                            col = j + 1,
                            plot_analysis_markers = plot_analysis_markers if fAnalyzed else False,
                            show_baseline_limits = show_baseline_limits,
                            show_baseline = show_baseline,
                            show_general_integration_limits = show_general_integration_limits,
                            show_general_amplitude_limits = show_general_amplitude_limits,
                            show_spotted_peaks = show_spotted_peaks,
                            show_peaks_integration_limits = show_peaks_integration_limits,
                            analysis_label = analysis_label if (plot_analysis_markers and fAnalyzed) else None)
    elif mode == 'heatmap':

        if analysis_label is None:  # In the 'heatmap' mode, the 'analysis_label' parameter must be defined
            raise Exception(generate_exception_message( 5,
                                                        'plot_WaveformSet()',
                                                        "The 'analysis_label' parameter must be defined if the 'mode' parameter is set to 'heatmap'."))

        aux_ranges = arrange_time_vs_ADC_ranges(waveform_set,
                                                time_range_lower_limit = time_range_lower_limit,
                                                time_range_upper_limit = time_range_upper_limit,
                                                adc_range_above_baseline = adc_range_above_baseline,
                                                adc_range_below_baseline = adc_range_below_baseline)
        for i in range(nrows):
            for j in range(ncols):
                if len(data_of_map_of_wf_idcs[i][j]) > 0:

                    aux_name = f"Heatmap of {len(data_of_map_of_wf_idcs[i][j])} Wf(s)"
                    if detailed_label:
                        aux_name += f": [{get_string_of_first_n_integers_if_available(data_of_map_of_wf_idcs[i][j], queried_no = 2)}]"

                    figure_ = __subplot_heatmap(waveform_set,
                                                figure_,
                                                aux_name,
                                                i + 1,
                                                j + 1,
                                                data_of_map_of_wf_idcs[i][j],
                                                analysis_label,
                                                time_bins,
                                                adc_bins,
                                                aux_ranges,
                                                show_color_bar = False)     # The color scale is not shown          ## There is a way to make the color scale match for     # https://community.plotly.com/t/trying-to-make-a-uniform-colorscale-for-each-of-the-subplots/32346
                                                                            # since it may differ from one plot     ## every plot in the grid, though, but comes at the
                                                                            # to another.                           ## cost of finding the max and min values of the 
                                                                                                                    ## union of all of the histograms. Such feature may 
                                                                                                                    ## be enabled in the future, using a boolean input
                                                                                                                    ## parameter.
                    figure_.add_annotation( xref = "x domain", 
                                            yref = "y domain",      
                                            x = 0.,             # The annotation is left-aligned
                                            y = 1.25,           # and on top of each subplot
                                            showarrow = False,
                                            text = aux_name,
                                            row = i + 1,
                                            col = j + 1)
                else:

                    __add_no_data_annotation(   figure_,
                                                i + 1,
                                                j + 1)
    else:                                                                                                           
        raise Exception(generate_exception_message( 6,
                                                    'plot_WaveformSet()',
                                                    f"The given mode ({mode}) must match either 'overlay', 'average', or 'heatmap'."))
    return figure_