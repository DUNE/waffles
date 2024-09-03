import numpy as np
from typing import Optional
from plotly import graph_objects as pgo
from plotly import subplots as psu

from waffles.data_classes.WaveformAdcs import WaveformAdcs
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.ChannelWSGrid import ChannelWsGrid
from waffles.data_classes.CalibrationHistogram import CalibrationHistogram
from waffles.data_classes.Map import map_

import waffles.plotting.plot_utils as wpu
import waffles.utils.numerical_utils as wun

from waffles.Exceptions import GenerateExceptionMessage


def plot_waveform_adcs(
        WaveformAdcs: WaveformAdcs,
        figure: pgo.Figure,
        name: Optional[str] = None,
        row: Optional[int] = None,
        col: Optional[int] = None,
        plot_analysis_markers: bool = False,
        show_baseline_limits: bool = False,
        show_baseline: bool = True,
        show_general_integration_limits: bool = False,
        show_general_amplitude_limits: bool = False,
        show_spotted_peaks: bool = True,
        show_peaks_integration_limits: bool = False,
        analysis_label: Optional[str] = None,
        verbose: bool = False) -> None:
    """
    This function plots the given WaveformAdcs object
    in the given figure.

    Parameters
    ----------
    figure : plotly.graph_objects.Figure
        The figure in which the Waveform will be plotted
    name : str
        The name for the Waveform trace which will be added
        to the given figure.
    row (resp. col) : int
        The row (resp. column) in which the Waveform will
        be plotted. This parameter is directly handled to
        the 'row' (resp. 'col') parameter of
        plotly.graph_objects.Figure.add_trace() and
        plotly.graph_objects.Figure.add_shape(). It is the
        caller's responsibility to ensure two things:

            - if the given 'figure' parameter does not contain
                a subplot grid (p.e. it was not created by
                plotly.subplots.make_subplots()) then 'row' and
                'col' must be None.

            - if the given 'figure' parameter contains a subplot
                grid, then 'row' and 'col' must be valid 1-indexed
                integers.
    plot_analysis_markers : bool
        If True, this function will potentially plot the
        analysis markers for the given Waveform in the given
        figure. If False, it will not. By analysis markers
        we mean those which are available among:

            - Vertical lines for the baseline limits
            - An horizontal line for the computed baseline
            - Two vertical lines for the integration limits
            - A triangle marker over each spotted peak                  ## This is not true yet. Change the 
            - Two vertical lines framing each spotted peak              ## vertical lines for triangle markers.
                marking the integration limits for each peak.
    show_baseline_limits : bool
        This parameter only makes a difference if
        'plot_analysis_markers' is set to True. In that case,
        this parameter means whether to plot vertical lines
        framing the intervals which were used to compute
        the baseline. The positions of these lines are
        grabbed from the 'baseline_limits' key in the
        InputParameters attribute of the specified WfAna 
        object, up to the analysis_label input parameter.
    show_baseline : bool
        This parameter only makes a difference if
        'plot_analysis_markers' is set to True. In that case,
        this parameter means whether to plot an horizontal
        line matching the computed baseline. The baseline
        value is grabbed from the 'baseline' key in the 
        Result attribute of the specified WfAna object, 
        up to the analysis_label input parameter.
    show_general_integration_limits : bool
        This parameter only makes a difference if
        'plot_analysis_markers' is set to True. In that case,
        this parameter means whether to plot vertical lines
        framing the general integration interval. The
        positions of these lines are grabbed from the
        'int_ll' and 'int_ul' keys in the InputParameters
        attribute of the specified WfAna object, up to the
        analysis_label input parameter.
    show_general_amplitude_limits : bool
        This parameter only makes a difference if
        'plot_analysis_markers' is set to True. In that case,
        this parameter means whether to plot vertical lines
        framing the general amplitude interval. The
        positions of these lines are grabbed from the
        'amp_ll' and 'amp_ul' keys in the InputParameters
        attribute of the specified WfAna object, up to the
        analysis_label input parameter.
    show_spotted_peaks : bool
        This parameter only makes a difference if
        'plot_analysis_markers' is set to True. In that case,
        this parameter means whether to plot a triangle
        marker over each spotted peak. The positions of 
        these markers are grabbed from the 'peaks' key 
        in the Result attribute of the specified WfAna object, 
        up to the analysis_label input parameter.
    show_peaks_integration_limits : bool
        This parameter only makes a difference if
        'plot_analysis_markers' is set to True. In that case,
        this parameter means whether to plot two vertical
        lines framing the integration interval for each
        spotted peak.
    analysis_label : str
        This parameter only makes a difference if 
        'plot_analysis_markers' is set to True. In that case, 
        'analysis_label' is the key for the WfAna object within 
        WaveformAdcs.Analyses from where to take the information 
        for the analysis markers plot. If 'analysis_label' is 
        None, then the last analysis added to WaveformAdcs.Analyses 
        will be the used one.
    verbose : bool
        Whether to print functioning-related messages
    """

    x = np.arange(
        len(WaveformAdcs.adcs),
        dtype=np.float32)

    wf_trace = pgo.Scatter(
        x=x + WaveformAdcs.time_offset,
        # If at some point we think x might match for
        # every Waveform, in a certain WaveformSet
        # object, it might be more efficient to let
        # the caller define it, so as not to recompute
        # this array for each Waveform.
        y=WaveformAdcs.adcs,
        mode='lines',
        line=dict(
            color='black',
            width=0.5),
        name=name)

    figure.add_trace(
        wf_trace,
        row=row,
        col=col)

    if plot_analysis_markers:

        ana = WaveformAdcs.get_analysis(analysis_label)

        if show_baseline_limits:    # Plot the markers for the baseline limits

            try:
                aux = ana.InputParameters['baseline_limits']

            except KeyError:
                if verbose:
                    print(
                        "In function plot_waveform_adcs(): No baseline-limits were found in the specified analysis.")
            else:

                for i in range(len(aux)//2):

                    figure.add_shape(
                        type='line',
                        # If you are wondering why only the trace of
                        x0=x[aux[2*i]], y0=0,
                        # the Waveform is offset according to WaveformAdcs.time_offset,
                        x1=x[aux[2*i]], y1=1,
                        # but not the markers, note that this is, indeed,
                        # the reason why a time offset is useful. The WfAna
                        # class moves the analysis ranges (p.e. baseline or
                        # integral ranges) according to the time_offset. Then
                        # to show a consistent plot, either the markers are
                        # displaced or the Waveform is displaced, but not both.
                        # For the sake of having a grid-plot where the analysis
                        # ranges are aligned among different subplots, I chose
                        # to displace the Waveform.

                        line=dict(
                            color='grey',         # Properties for
                            width=1,              # the beginning of
                            dash='dash'),         # a baseline chunk
                        xref='x',
                        yref='y domain',
                        row=row,
                        col=col)

                    figure.add_shape(
                        type='line',
                        x0=x[aux[(2*i) + 1]], y0=0,
                        x1=x[aux[(2*i) + 1]], y1=1,
                        line=dict(
                            color='grey',         # Properties for
                            width=1,              # the end of a
                            dash='dashdot'),      # baseline chunk
                        xref='x',
                        yref='y domain',
                        row=row,
                        col=col)

        if show_baseline:       # Plot the baseline

            try:
                aux = ana.Result['baseline']

            except KeyError:
                if verbose:
                    print(
                        "In function plot_waveform_adcs(): No baseline was found in the specified analysis.")
            else:

                figure.add_shape(
                    type="line",
                    x0=0, y0=aux,
                    x1=1, y1=aux,
                    line=dict(
                        color='grey',             # Properties for
                        width=1,                  # the computed
                        dash='dot'),              # baseline
                    xref='x domain',
                    yref='y',
                    row=row,
                    col=col)

        if show_general_integration_limits:     # Plot the markers for the general integration limits

            try:
                aux_1 = ana.InputParameters['int_ll']
                aux_2 = ana.InputParameters['int_ul']

            except KeyError:
                if verbose:
                    print(
                        "In function plot_waveform_adcs(): No general-integration-limits were found in the specified analysis.")
            else:

                figure.add_shape(
                    type='line',
                    x0=x[aux_1], y0=0,
                    x1=x[aux_1], y1=1,
                    line=dict(
                        color='black',
                        width=1,
                        dash='solid'),
                    xref='x',
                    yref='y domain',
                    row=row,
                    col=col)

                figure.add_shape(
                    type='line',
                    x0=x[aux_2], y0=0,
                    x1=x[aux_2], y1=1,
                    line=dict(color='black',
                              width=1,
                              dash='solid'),
                    xref='x',
                    yref='y domain',
                    row=row,
                    col=col)

        if show_general_amplitude_limits:       # Plot the markers for the general amplitude limits

            try:
                aux_1 = ana.InputParameters['amp_ll']
                aux_2 = ana.InputParameters['amp_ul']

            except KeyError:
                if verbose:
                    print(
                        "In function plot_waveform_adcs(): No general-amplitude-limits were found in the specified analysis.")
            else:

                figure.add_shape(
                    type='line',
                    x0=x[aux_1], y0=0,
                    x1=x[aux_1], y1=1,
                    line=dict(color='green',
                              width=1,
                              dash='solid'),
                    xref='x',
                    yref='y domain',
                    row=row,
                    col=col)

                figure.add_shape(
                    type='line',
                    x0=x[aux_2], y0=0,
                    x1=x[aux_2], y1=1,
                    line=dict(color='green',
                              width=1,
                              dash='solid'),
                    xref='x',
                    yref='y domain',
                    row=row,
                    col=col)

        if show_spotted_peaks:      # Plot the markers for the spotted peaks

            try:
                peaks = ana.Result['peaks']

            except KeyError:
                if verbose:
                    print(
                        "In function plot_waveform_adcs(): No peaks were found in the specified analysis.")
            else:

                for peak in peaks:

                    aux = x[peak.Position]

                    figure.add_shape(
                        type='line',
                        x0=aux, y0=0,
                        x1=aux, y1=1,
                        line=dict(
                            color='red',      # Properties for
                            width=1,          # the peaks markers
                            dash='dot'),
                        xref='x',
                        yref='y domain',
                        row=row,
                        col=col)

        if show_peaks_integration_limits:   # Plot the markers for the peaks integration limits
            raise NotImplementedError(GenerateExceptionMessage(
                1,
                'plot_waveform_adcs()',
                "The 'show_peaks_integration_limits' parameter is not implemented yet."))
    return


def plot_WaveformSet(
        WaveformSet: WaveformSet,
        *args,
        nrows: int = 1,
        ncols: int = 1,
        figure: Optional[pgo.Figure] = None,
        wfs_per_axes: Optional[int] = 1,
        map_of_wf_idcs: Optional[map_] = None,
        share_x_scale: bool = False,
        share_y_scale: bool = False,
        mode: str = 'overlay',
        analysis_label: Optional[str] = None,
        plot_analysis_markers: bool = False,
        show_baseline_limits: bool = False,
        show_baseline: bool = True,
        show_general_integration_limits: bool = False,
        show_general_amplitude_limits: bool = False,
        show_spotted_peaks: bool = True,
        show_peaks_integration_limits: bool = False,
        time_bins: int = 512,
        adc_bins: int = 100,
        time_range_lower_limit: Optional[int] = None,
        time_range_upper_limit: Optional[int] = None,
        adc_range_above_baseline: int = 100,
        adc_range_below_baseline: int = 200,
        detailed_label: bool = True,
        verbose: bool = False,
        **kwargs) -> pgo.Figure:
    """ 
    This function returns a plotly.graph_objects.Figure 
    with a nrows x ncols grid of axes, with plots of
    some of the waveforms in the given WaveformSet object.

    Parameters
    ----------
    WaveformSet : WaveformSet
        The WaveformSet object which contains the 
        waveforms to be plotted.
    *args
        These arguments only make a difference if the
        'mode' parameter is set to 'average' and the
        'analysis_label' parameter is not None. In such
        case, these are the positional arguments handled 
        to the WaveformAdcs.analyse() instance method of 
        the computed mean Waveform. I.e. for the mean 
        Waveform wf, the call to its analyse() method
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
    map_of_wf_idcs : map_ of lists of integers
        This map_ must contain lists of integers.
        map_of_wf_idcs.data[i][j] gives the indices of the 
        waveforms, with respect to the given WaveformSet, 
        WaveformSet, which should be considered for 
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
        instead of plotting every Waveform, only the 
        averaged Waveform of the considered waveforms will 
        be plotted. If it is set to 'heatmap', then a 
        2D-histogram, whose entries are the union of all 
        of the points of every considered Waveform, will 
        be plotted. In the 'heatmap' mode, the baseline 
        of each Waveform is subtracted from each Waveform 
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
        parameter of the plot_waveform_adcs() function 
        for each WaveformAdcs object(s) which will be 
        plotted. This parameter gives the key for the 
        WfAna object within the Analyses attribute of 
        each plotted Waveform from where to take the 
        information for the analysis markers plot. In 
        this case, if 'analysis_label' is None, then 
        the last analysis added to the Analyses attribute 
        will be the used one. 
            If mode is set to 'average' and this 
        parameter is defined, then this function will 
        analyse the newly computed average Waveform, 
        say wf, by calling 
        wf.analyse(analysis_label, *args, **kwargs).
        Additionally, if the 'plot_analysis_markers'
        parameter is set to True and this parameter
        is defined, then this parameter is given to 
        the 'analysis_label' parameter of the 
        plot_waveform_adcs(wf, ...) function for the 
        newly computed average Waveform, i.e. the 
        analysis markers for the plotted average 
        Waveform are those of the newly computed analysis. 
        This parameter gives the key for the WfAna 
        object within the Analyses attribute of the 
        average Waveform where to take the information 
        for the analysis markers plot.
            If 'mode' is set to 'heatmap', this 
        parameter is not optional, i.e. it must be 
        defined, and gives the analysis whose baseline 
        will be subtracted from each Waveform before 
        plotting. Namely, the baseline will be grabbed
        from the 'baseline' key of the specified 
        analysis. In this case, it will not be checked 
        that, for each Waveform, the analysis with the 
        given label is available. It is the caller's 
        responsibility to ensure so.
    plot_analysis_markers : bool
        This parameter only makes a difference if the
        'mode' parameter is set to 'overlay' or 'average'.
            If mode is set to 'overlay', then this 
        parameter is given to the 
        'plot_analysis_markers' argument of the 
        plot_waveform_adcs() function for each Waveform 
        which will be plotted. 
            If mode is set to 'average' and the
        'analysis_label' parameter is defined, then this
        parameter is given to the 'plot_analysis_markers'
        argument of the plot_waveform_adcs() function for
        the newly computed average Waveform. If the
        'analysis_label' parameter is not defined, then
        this parameter will be automatically interpreted
        as False.
            In both cases, if True, analysis markers 
        for the plotted WaveformAdcs objects will 
        potentially be plotted together with each 
        Waveform. For more information, check the 
        'plot_analysis_markers' parameter documentation 
        in the plot_waveform_adcs() function. If False, 
        no analysis markers will be plot.
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
        is assumed to be 0 (resp. WaveformSet.PointsPerWf - 1).
        It must be smaller (resp. greater) than
        time_range_upper_limit (resp. time_range_lower_limit).
    adc_range_above_baseline (resp. adc_range_below_baseline) : int
        This parameter only makes a difference if the
        'mode' parameter is set to 'heatmap'. In that case,
        its absolute value times one (resp. minus one) is 
        the upper (resp. lower) limit of the ADCs range 
        which will be considered for the heatmap plot. 
        Note that, in this case, each Waveform is 
        corrected by its own baseline.
    detailed_label : bool
        This parameter only makes a difference if
        the 'mode' parameter is set to 'average' or
        'heatmap', respectively. If the 'mode' parameter
        is set to 'average', then this parameter means
        whether to show the iterator values of the two
        first available waveforms (which were used to
        compute the mean Waveform) in the label of the
        mean Waveform plot. If the 'mode' parameter is 
        set to 'heatmap', then this parameter means 
        whether to show the iterator values of the two 
        first available waveforms (which were used to 
        compute the 2D-histogram) in the top annotation 
        of each subplot.
    verbose : bool
        Whether to print functioning-related messages
    **kwargs
        These arguments only make a difference if the
        'mode' parameter is set to 'average' and the
        'analysis_label' parameter is not None. In such
        case, these are the keyword arguments handled 
        to the WaveformAdcs.analyse() instance method of 
        the computed mean Waveform. I.e. for the mean 
        Waveform wf, the call to its analyse() method
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
        raise Exception(GenerateExceptionMessage(
            1,
            'plot_WaveformSet()',
            'The number of rows and columns must be positive.'))
    if figure is not None:
        wpu.check_dimensions_of_suplots_figure(
            figure,
            nrows,
            ncols)
        figure_ = figure
    else:
        figure_ = psu.make_subplots(rows=nrows,
                                    cols=ncols)

    data_of_map_of_wf_idcs = None         # Logically useless

    if wfs_per_axes is not None:    # wfs_per_axes is defined, so ignore map_of_wf_idcs

        if wfs_per_axes < 1:
            raise Exception(GenerateExceptionMessage(
                2,
                'plot_WaveformSet()',
                'The number of waveforms per axes must be positive.'))

        data_of_map_of_wf_idcs = WaveformSet.get_map_of_wf_idcs(
            nrows,
            ncols,
            wfs_per_axes=wfs_per_axes).data

    elif map_of_wf_idcs is None:    # Nor wf_per_axes, nor
        # map_of_wf_idcs are defined

        raise Exception(GenerateExceptionMessage(
            3,
            'plot_WaveformSet()',
            "The 'map_of_wf_idcs' parameter must be defined if wfs_per_axes is not."))

    elif not map_.list_of_lists_is_well_formed(
            map_of_wf_idcs.data,    # wf_per_axes is not defined,
            nrows,                  # but map_of_wf_idcs is, but
            ncols):                 # it is not well-formed

        raise Exception(GenerateExceptionMessage(
            4,
            'plot_WaveformSet()',
            f"The given map_of_wf_idcs is not well-formed according to nrows ({nrows}) and ncols ({ncols})."))
    else:   # wf_per_axes is not defined,
        # but map_of_wf_idcs is,
        # and it is well-formed

        data_of_map_of_wf_idcs = map_of_wf_idcs.data

    wpu.update_shared_axes_status(
        figure_,                    # An alternative way is to specify
        # shared_xaxes=True (or share_yaxes=True)
        share_x=share_x_scale,
        # in psu.make_subplots(), but, for us,
        share_y=share_y_scale)
    # that alternative is only doable for
    # the case where the given 'figure'
    # parameter is None.
    if mode == 'overlay':
        for i in range(nrows):
            for j in range(ncols):
                if len(data_of_map_of_wf_idcs[i][j]) > 0:
                    for k in data_of_map_of_wf_idcs[i][j]:

                        aux_name = f"({i+1}, {j+1}) - Wf {k}, Ch {WaveformSet.waveforms[k].channel}, Ep {
                            WaveformSet.waveforms[k].endpoint}"

                        plot_waveform_adcs(
                            WaveformSet.waveforms[k],
                            figure=figure_,
                            name=aux_name,
                            row=i + 1,  # Plotly uses 1-based indexing
                            col=j + 1,
                            plot_analysis_markers=plot_analysis_markers,
                            show_baseline_limits=show_baseline_limits,
                            show_baseline=show_baseline,
                            show_general_integration_limits=show_general_integration_limits,
                            show_general_amplitude_limits=show_general_amplitude_limits,
                            show_spotted_peaks=show_spotted_peaks,
                            show_peaks_integration_limits=show_peaks_integration_limits,
                            analysis_label=analysis_label,
                            verbose=verbose)
                else:
                    wpu.__add_no_data_annotation(
                        figure_,
                        i + 1,
                        j + 1)
    elif mode == 'average':
        for i in range(nrows):
            for j in range(ncols):

                try:
                    # WaveformSet.compute_mean_waveform() will raise an
                    aux = WaveformSet.compute_mean_waveform(
                        wf_idcs=data_of_map_of_wf_idcs[i][j])
                    # exception if data_of_map_of_wf_idcs[i][j] is empty

                except Exception:  # At some point we should implement a number of exceptions which are self-explanatory,
                    # so that we can handle in parallel exceptions due to different reasons if we need it

                    wpu.__add_no_data_annotation(
                        figure_,
                        i + 1,
                        j + 1)
                    continue

                fAnalyzed = False
                if analysis_label is not None:

                    _ = aux.analyse(analysis_label,
                                    *args,
                                    **kwargs)
                    fAnalyzed = True

                aux_name = f"{len(data_of_map_of_wf_idcs[i][j])} Wf(s)"
                if detailed_label:
                    aux_name += f": [{wpu.get_string_of_first_n_integers_if_available(
                        data_of_map_of_wf_idcs[i][j], queried_no=2)}]"

                plot_waveform_adcs(
                    aux,
                    figure=figure_,
                    name=f"({i+1},{j+1}) - Mean of " + aux_name,
                    row=i + 1,
                    col=j + 1,
                    plot_analysis_markers=plot_analysis_markers if fAnalyzed else False,
                    show_baseline_limits=show_baseline_limits,
                    show_baseline=show_baseline,
                    show_general_integration_limits=show_general_integration_limits,
                    show_general_amplitude_limits=show_general_amplitude_limits,
                    show_spotted_peaks=show_spotted_peaks,
                    show_peaks_integration_limits=show_peaks_integration_limits,
                    analysis_label=analysis_label if (
                        plot_analysis_markers and fAnalyzed) else None,
                    verbose=verbose)
    elif mode == 'heatmap':

        if analysis_label is None:  # In the 'heatmap' mode, the 'analysis_label' parameter must be defined
            raise Exception(GenerateExceptionMessage(
                5,
                'plot_WaveformSet()',
                "The 'analysis_label' parameter must be defined if the 'mode' parameter is set to 'heatmap'."))

        aux_ranges = wpu.arrange_time_vs_adc_ranges(WaveformSet,
                                                    time_range_lower_limit=time_range_lower_limit,
                                                    time_range_upper_limit=time_range_upper_limit,
                                                    adc_range_above_baseline=adc_range_above_baseline,
                                                    adc_range_below_baseline=adc_range_below_baseline)
        for i in range(nrows):
            for j in range(ncols):
                if len(data_of_map_of_wf_idcs[i][j]) > 0:

                    aux_name = f"Heatmap of {
                        len(data_of_map_of_wf_idcs[i][j])} Wf(s)"
                    if detailed_label:
                        aux_name += f": [{wpu.get_string_of_first_n_integers_if_available(
                            data_of_map_of_wf_idcs[i][j], queried_no=2)}]"

                    figure_ = wpu.__subplot_heatmap(WaveformSet,
                                                    figure_,
                                                    aux_name,
                                                    i + 1,
                                                    j + 1,
                                                    data_of_map_of_wf_idcs[i][j],
                                                    analysis_label,
                                                    time_bins,
                                                    adc_bins,
                                                    aux_ranges,
                                                    show_color_bar=False)     # The color scale is not shown          ## There is a way to make the color scale match for     # https://community.plotly.com/t/trying-to-make-a-uniform-colorscale-for-each-of-the-subplots/32346
                    # since it may differ from one plot     ## every plot in the grid, though, but comes at the
                    # to another.                           ## cost of finding the max and min values of the
                    # union of all of the histograms. Such feature may
                    # be enabled in the future, using a boolean input
                    # parameter.
                    figure_.add_annotation(xref="x domain",
                                           yref="y domain",
                                           x=0.,             # The annotation is left-aligned
                                           y=1.25,           # and on top of each subplot
                                           showarrow=False,
                                           text=aux_name,
                                           row=i + 1,
                                           col=j + 1)
                else:

                    wpu.__add_no_data_annotation(figure_,
                                                 i + 1,
                                                 j + 1)
    else:
        raise Exception(GenerateExceptionMessage(6,
                                                 'plot_WaveformSet()',
                                                 f"The given mode ({mode}) must match either 'overlay', 'average', or 'heatmap'."))
    return figure_


def plot_calibration_histogram(CalibrationHistogram: CalibrationHistogram,
                               figure: pgo.Figure,
                               name: Optional[str] = None,
                               row: Optional[int] = None,
                               col: Optional[int] = None,
                               plot_fits: bool = False,
                               fit_npoints: int = 200) -> bool:
    """
    This function plots the given calibration histogram in 
    the given figure and returns a boolean which is True if 
    at least one gaussian fit has been plotted, and False 
    otherwise. Note that, if the 'plot_fits' parameter is
    set to False, then this output is automatically False.

    Parameters
    ----------
    CalibrationHistogram : CalibrationHistogram
        The CalibrationHistogram object to be plotted
    figure : plotly.graph_objects.Figure
        The figure in which the calibration histogram (CH) 
        will be plotted
    name : str
        The name for the CH trace which will be added to 
        the given figure.
    row (resp. col) : int
        The row (resp. column) in which the CH will be 
        plotted. This parameter is directly handled to
        the 'row' (resp. 'col') parameter of
        plotly.graph_objects.Figure.add_trace(). It is the
        caller's responsibility to ensure two things:

            -   if the given 'figure' parameter does not contain
                a subplot grid (p.e. it was not created by
                plotly.subplots.make_subplots()) then 'row' and
                'col' must be None.

            -   if the given 'figure' parameter contains a subplot
                grid, then 'row' and 'col' must be valid 1-indexed
                integers.
    plot_fits : bool
        If True, then the gaussian fits of the peaks, if any, 
        will be plotted over the CH. If False, then only the 
        CH will be plotted. Note that if no fit has been performed
        yet, then the CalibrationHistogram.GaussianFitsParameters 
        attribute will be empty and no fit will be plotted.
    fit_npoints : int
        This parameter only makes a difference if 'plot_fits'
        is set to True. In that case, it gives the number of
        points to use to plot each gaussian fit. Note that
        the plot range of the fit will be the same as the
        range of the CH. It must be greater than 1. It is
        the caller's responsibility to ensure this.

    Returns
    ----------
    fPlottedOneFit : bool
        If 'plot_fits' is set to False, then this function
        returns False. If 'plot_fits' is set to True, then
        this function returns True if at least one fit has
        been plotted, and False otherwise.
    """

    histogram_trace = pgo.Scatter(
        x=CalibrationHistogram.Edges,
        y=CalibrationHistogram.Counts,
        mode='lines',
        line=dict(
            color='black',
            width=0.5,
            shape='hv'),
        name=name)

    figure.add_trace(
        histogram_trace,
        row=row,
        col=col)

    fPlottedOneFit = False

    if plot_fits:

        for i in range(len(CalibrationHistogram.GaussianFitsParameters['scale'])):

            fPlottedOneFit = True

            fit_x = np.linspace(
                CalibrationHistogram.Edges[0],
                CalibrationHistogram.Edges[-1],
                num=fit_npoints)

            fit_y = wun.gaussian(
                fit_x,
                CalibrationHistogram.GaussianFitsParameters['scale'][i][0],
                CalibrationHistogram.GaussianFitsParameters['mean'][i][0],
                CalibrationHistogram.GaussianFitsParameters['std'][i][0])

            fit_trace = pgo.Scatter(
                x=fit_x,
                y=fit_y,
                mode='lines',
                line=dict(
                    color='red',
                    width=0.5),
                name=f"{name} (Fit {i})")

            figure.add_trace(
                fit_trace,
                row=row,
                col=col)
    return fPlottedOneFit


def plot_channel_ws_grid(
        channel_ws_grid: ChannelWsGrid,
        *args,
        figure: Optional[pgo.Figure] = None,
        share_x_scale: bool = False,
        share_y_scale: bool = False,
        mode: str = 'overlay',
        wfs_per_axes: Optional[int] = 1,
        analysis_label: Optional[str] = None,
        plot_analysis_markers: bool = False,
        show_baseline_limits: bool = False,
        show_baseline: bool = True,
        show_general_integration_limits: bool = False,
        show_general_amplitude_limits: bool = False,
        show_spotted_peaks: bool = True,
        show_peaks_integration_limits: bool = False,
        time_bins: int = 512,
        adc_bins: int = 100,
        time_range_lower_limit: Optional[int] = None,
        time_range_upper_limit: Optional[int] = None,
        adc_range_above_baseline: int = 100,
        adc_range_below_baseline: int = 200,
        plot_peaks_fits: bool = False,
        detailed_label: bool = True,
        verbose: bool = True,
        **kwargs) -> pgo.Figure:
    """
    This function returns a plotly.graph_objects.Figure 
    with a grid of subplots which are arranged according
    to the ChannelWsGrid.ch_map attribute. The subplot at 
    position i,j may be empty if there is no ChannelWS object 
    in ChannelWsGrid.ch_wf_sets which matches the UniqueChannel 
    object at position i,j in the ChannelWsGrid.ch_map 
    attribute. If it is not empty, a subplot may contain a 
    Waveform representation (either overlayed, averaged or 
    heatmapped), or a calibration histogram. The type of 
    representation is determined by the 'mode' parameter.

    Parameters
    ----------
    ChannelWsGrid : ChannelWsGrid
        The ChannelWsGrid object which contains the 
        ChannelWS objects to be plotted.
    *args
        These arguments only make a difference if the
        'mode' parameter is set to 'average' and the
        'analysis_label' parameter is not None. In such
        case, these are the positional arguments handled 
        to the WaveformAdcs.analyse() instance method of 
        the computed mean Waveform. I.e. for the mean 
        Waveform wf, the call to its analyse() method
        is wf.analyse(analysis_label, *args, **kwargs).
        The WaveformAdcs.analyse() method does not 
        perform any well-formedness checks, so it is 
        the caller's responsibility to ensure so for 
        these parameters.
    figure : plotly.graph_objects.Figure
        If it is not None, then it must have been
        generated using plotly.subplots.make_subplots()
        with a 'rows' and 'cols' parameters matching
        the rows and columns attribute of 
        ChannelWsGrid.ch_map. If that's the case, then 
        this function adds the plots to this figure and 
        eventually returns it. If it is None, then this 
        function generates a new figure and returns it.
    share_x_scale (resp. share_y_scale) : bool
        If True, the x-axis (resp. y-axis) scale will be 
        shared among all the subplots.
    mode : str
        This parameter should be set to 'overlay', 
        'average', 'heatmap' or 'calibration'. If any
        other input is given, an exception will be raised. 
            The default setting is 'overlay', which means 
        that all of the considered waveforms, up to the 
        'wfs_per_axes' parameter, will be plotted. 
            If it set to 'average', instead of plotting 
        every considered Waveform, only the averaged 
        Waveform of the considered waveforms will be plotted. 
            If it is set to 'heatmap', then a 2D-histogram, 
        whose entries are the union of all of the points 
        of every considered Waveform, will be plotted. In 
        the 'heatmap' mode, the baseline of each Waveform 
        is subtracted from each Waveform before plotting. 
        Note that to perform such a correction, the waveforms 
        should have been previously analysed, so that at 
        least one baseline value is available. The analysis 
        which gave the baseline value which should be used 
        is specified via the 'analysis_label' parameter. 
        Check its documentation for more information.
            If it is set to 'calibration', then the
        calibration histogram of each ChannelWS object
        will be plotted. In this case, the CalibHisto
        attribute of each ChannelWS object must be
        defined, i.e. it must be different to None.
        If it is not, then an exception will be raised.
    wfs_per_axes : int
        If it is None, then every Waveform in each
        ChannelWS object will be considered. Otherwise,
        only the first wfs_per_axes waveforms of each
        ChannelWS object will be considered. If 
        wfs_per_axes is greater than the number of 
        waveforms in a certain ChannelWS object, then 
        all of its waveforms will be considered.
    analysis_label : str
        The meaning of this parameter varies slightly
        depending on the value given to the 'mode'
        parameter. 
            If mode is set to 'overlay', then this 
        parameter is optional and it only makes a 
        difference if the 'plot_analysis_markers' 
        parameter is set to True. In such case, this 
        parameter is given to the 'analysis_label'
        parameter of the plot_waveform_adcs() function 
        for each WaveformAdcs object(s) which will be 
        plotted. This parameter gives the key for the 
        WfAna object within the Analyses attribute of 
        each plotted Waveform from where to take the 
        information for the analysis markers plot. In 
        this case, if 'analysis_label' is None, then 
        the last analysis added to the Analyses attribute 
        will be the used one. 
            If mode is set to 'average' and this 
        parameter is defined, then this function will 
        analyse the newly computed average Waveform, 
        say wf, by calling 
        wf.analyse(analysis_label, *args, **kwargs).
        Additionally, if the 'plot_analysis_markers'
        parameter is set to True and this parameter
        is defined, then this parameter is given to 
        the 'analysis_label' parameter of the 
        plot_waveform_adcs(wf, ...) function, i.e. the 
        analysis markers for the plotted average Waveform 
        are those of the newly computed analysis. This 
        parameter gives the key for the WfAna object 
        within the Analyses attribute of the average 
        Waveform where to take the information for the 
        analysis markers plot.
            If 'mode' is set to 'heatmap', this 
        parameter is not optional, i.e. it must be 
        defined, and gives the analysis whose baseline 
        will be subtracted from each Waveform before 
        plotting. Namely, the baseline will be grabbed
        from the 'baseline' key of the specified 
        analysis. In this case, it will not be checked 
        that, for each Waveform, the analysis with the 
        given label is available. It is the caller's 
        responsibility to ensure so.
            If 'mode' is set to 'calibration', then
        this parameter is not used.
    plot_analysis_markers : bool
        This parameter only makes a difference if the
        'mode' parameter is set to 'overlay' or 'average'.
            If mode is set to 'overlay', then this 
        parameter is given to the 
        'plot_analysis_markers' argument of the 
        plot_waveform_adcs() function for each Waveform 
        which will be plotted. 
            If mode is set to 'average' and the
        'analysis_label' parameter is defined, then this
        parameter is given to the 'plot_analysis_markers'
        argument of the plot_waveform_adcs() function for
        the newly computed average Waveform. If the
        'analysis_label' parameter is not defined, then
        this parameter will be automatically interpreted
        as False.
            In both cases, if True, analysis markers 
        for the plotted WaveformAdcs objects will 
        potentially be plotted together with each 
        Waveform. For more information, check the 
        'plot_analysis_markers' parameter documentation 
        in the plot_waveform_adcs() function. If False, no 
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
        and the 'plot_analysis_markers' parameter is 
        set to True. In that case, this parameter means 
        whether to plot vertical lines framing the general 
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
        is assumed to be 0 (resp. ChannelWs.PointsPerWf - 1, 
        where ChannelWs is the ChannelWS object to be plotted 
        in each subplot). It must be smaller (resp. greater) 
        than time_range_upper_limit (resp. time_range_lower_limit).
    adc_range_above_baseline (resp. adc_range_below_baseline) : int
        This parameter only makes a difference if the
        'mode' parameter is set to 'heatmap'. In that case,
        its absolute value times one (resp. minus one) is 
        the upper (resp. lower) limit of the ADCs range 
        which will be considered for the heatmap plot. 
        Note that, in this case, each Waveform is 
        corrected by its own baseline.
    plot_peaks_fits : bool
        This parameter only makes a difference if the
        'mode' parameter is set to 'calibration'. In that
        case, then for the calibration histogram of each 
        subplot, this parameter is given to the 'plot_fits' 
        parameter of the call to plot_CalibrationHistogram().
        It means whether to plot the fits of the peaks, if
        available, over the histogram.
    detailed_label : bool
        This parameter only makes a difference if
        the 'mode' parameter is set to 'average' or
        'heatmap', respectively. If the 'mode' parameter
        is set to 'average', then this parameter means
        whether to show the iterator values of the two
        first available waveforms (which were used to
        compute the mean Waveform) in the label of the
        mean Waveform plot. If the 'mode' parameter is 
        set to 'heatmap', then this parameter means 
        whether to show the iterator values of the two 
        first available waveforms (which were used to 
        compute the 2D-histogram) in the top annotation 
        of each subplot.
    verbose : bool
        Whether to print functioning-related messages
    **kwargs
        These arguments only make a difference if the
        'mode' parameter is set to 'average' and the
        'analysis_label' parameter is not None. In such
        case, these are the keyword arguments handled 
        to the WaveformAdcs.analyse() instance method of 
        the computed mean Waveform. I.e. for the mean 
        Waveform wf, the call to its analyse() method
        is wf.analyse(analysis_label, *args, **kwargs).
        The WaveformAdcs.analyse() method does not 
        perform any well-formedness checks, so it is 
        the caller's responsibility to ensure so for 
        these parameters.

    Returns
    ----------
    figure : plotly.graph_objects.Figure
        This function returns a plotly.graph_objects.Figure 
        with a grid of subplots which are arranged 
        according to the ChannelWsGrid.ch_map attribute.
    """

    if figure is not None:
        wpu.check_dimensions_of_suplots_figure(
            figure,
            channel_ws_grid.ch_map.rows,
            channel_ws_grid.ch_map.columns)
        figure_ = figure
    else:
        figure_ = psu.make_subplots(rows=channel_ws_grid.ch_map.rows,
                                    cols=channel_ws_grid.ch_map.columns)
    fPlotAll = True
    if wfs_per_axes is not None:

        if wfs_per_axes < 1:
            raise Exception(GenerateExceptionMessage(
                1,
                'plot_channel_ws_grid()',
                'If defined, the number of waveforms per axes must be positive.'))
        fPlotAll = False

    wpu.__add_unique_channels_top_annotations(
        channel_ws_grid,
        figure_,
        # If mode is 'heatmap', then
        also_add_run_info=True if mode != 'heatmap' else False)
    # there is already a right-aligned
    # top annotation which shows the
    # iterator values of the first
    # waveforms. We are not adding a
    # new one so that they don't collide.
    wpu.update_shared_axes_status(
        figure_,                    # An alternative way is to specify
        # shared_xaxes=True (or share_yaxes=True)
        share_x=share_x_scale,
        # in psu.make_subplots(), but, for us,
        share_y=share_y_scale)
    # that alternative is only doable for
    # the case where the given 'figure'
    # parameter is None.
    if mode == 'overlay':
        for i in range(channel_ws_grid.ch_map.rows):
            for j in range(channel_ws_grid.ch_map.columns):

                try:
                    ChannelWs = channel_ws_grid.ch_wf_sets[channel_ws_grid.ch_map.data[i]
                                                           [j].endpoint][channel_ws_grid.ch_map.data[i][j].channel]

                except KeyError:
                    wpu.__add_no_data_annotation(
                        figure_,
                        i + 1,
                        j + 1)
                    continue

                if fPlotAll:
                    aux_idcs = range(len(ChannelWs.waveforms))
                else:
                    # If wfs_per_axes is defined, then it has been
                    aux_idcs = range(
                        min(wfs_per_axes, len(ChannelWs.waveforms)))
                    # checked to be >=1. If it is not defined, then
                    # still len(ChannelWs.waveforms) is >=1 (which
                    # is ensured by WaveformSet.__init__), so the
                    # minimum is always >=1.
                for idx in aux_idcs:

                    aux_name = f"({
                        i+1}, {j+1}) - Wf {idx}, Ch {channel_ws_grid.ch_map.data[i][j]}"

                    plot_waveform_adcs(
                        ChannelWs.waveforms[idx],
                        figure=figure_,
                        name=aux_name,
                        row=i + 1,  # Plotly uses 1-based indexing
                        col=j + 1,
                        plot_analysis_markers=plot_analysis_markers,
                        show_baseline_limits=show_baseline_limits,
                        show_baseline=show_baseline,
                        show_general_integration_limits=show_general_integration_limits,
                        show_general_amplitude_limits=show_general_amplitude_limits,
                        show_spotted_peaks=show_spotted_peaks,
                        show_peaks_integration_limits=show_peaks_integration_limits,
                        analysis_label=analysis_label,
                        verbose=verbose)
    elif mode == 'average':
        for i in range(channel_ws_grid.ch_map.rows):
            for j in range(channel_ws_grid.ch_map.columns):

                try:
                    ChannelWs = channel_ws_grid.ch_wf_sets[channel_ws_grid.ch_map.data[i]
                                                           [j].endpoint][channel_ws_grid.ch_map.data[i][j].channel]

                except KeyError:
                    wpu.__add_no_data_annotation(
                        figure_,
                        i + 1,
                        j + 1)
                    continue

                if fPlotAll:
                    aux_idcs = range(len(ChannelWs.waveforms))
                else:
                    aux_idcs = range(
                        min(wfs_per_axes, len(ChannelWs.waveforms)))

                aux = ChannelWs.compute_mean_waveform(wf_idcs=list(
                    aux_idcs))    # WaveformSet.compute_mean_waveform()
                # will raise an exception if
                # list(aux_idcs) is empty
                fAnalyzed = False
                if analysis_label is not None:

                    _ = aux.analyse(analysis_label,
                                    *args,
                                    **kwargs)
                    fAnalyzed = True

                aux_name = f"{len(aux_idcs)} Wf(s)"
                if detailed_label:
                    aux_name += f": [{wpu.get_string_of_first_n_integers_if_available(
                        list(aux_idcs), queried_no=2)}]"

                plot_waveform_adcs(
                    aux,
                    figure=figure_,
                    name=f"({i+1},{j+1}) - Mean of " + aux_name,
                    row=i + 1,
                    col=j + 1,
                    plot_analysis_markers=plot_analysis_markers if fAnalyzed else False,
                    show_baseline_limits=show_baseline_limits,
                    show_baseline=show_baseline,
                    show_general_integration_limits=show_general_integration_limits,
                    show_general_amplitude_limits=show_general_amplitude_limits,
                    show_spotted_peaks=show_spotted_peaks,
                    show_peaks_integration_limits=show_peaks_integration_limits,
                    analysis_label=analysis_label if (
                        plot_analysis_markers and fAnalyzed) else None,
                    verbose=verbose)
    elif mode == 'heatmap':

        if analysis_label is None:  # In the 'heatmap' mode, the 'analysis_label' parameter must be defined
            raise Exception(GenerateExceptionMessage(
                2,
                'plot_channel_ws_grid()',
                "The 'analysis_label' parameter must be defined if the 'mode' parameter is set to 'heatmap'."))
        for i in range(channel_ws_grid.ch_map.rows):
            for j in range(channel_ws_grid.ch_map.columns):

                try:
                    ChannelWs = channel_ws_grid.ch_wf_sets[channel_ws_grid.ch_map.data[i]
                                                           [j].endpoint][channel_ws_grid.ch_map.data[i][j].channel]

                except KeyError:
                    wpu.__add_no_data_annotation(
                        figure_,
                        i + 1,
                        j + 1)
                    continue

                if fPlotAll:
                    aux_idcs = range(len(ChannelWs.waveforms))
                else:
                    aux_idcs = range(
                        min(wfs_per_axes, len(ChannelWs.waveforms)))

                aux_name = f"{len(aux_idcs)} Wf(s)"
                if detailed_label:
                    aux_name += f": [{wpu.get_string_of_first_n_integers_if_available(
                        list(aux_idcs), queried_no=2)}]"

                aux_ranges = wpu.arrange_time_vs_adc_ranges(
                    ChannelWs,
                    time_range_lower_limit=time_range_lower_limit,
                    time_range_upper_limit=time_range_upper_limit,
                    adc_range_above_baseline=adc_range_above_baseline,
                    adc_range_below_baseline=adc_range_below_baseline)

                figure_ = wpu.__subplot_heatmap(
                    ChannelWs,
                    figure_,
                    aux_name,
                    i + 1,
                    j + 1,
                    list(aux_idcs),
                    analysis_label,
                    time_bins,
                    adc_bins,
                    aux_ranges,
                    show_color_bar=False)  # The color scale is not shown          ## There is a way to make the color scale match for     # https://community.plotly.com/t/trying-to-make-a-uniform-colorscale-for-each-of-the-subplots/32346
                # since it may differ from one plot     ## every plot in the grid, though, but comes at the
                # to another.                           ## cost of finding the max and min values of the
                # union of all of the histograms. Such feature may
                # be enabled in the future, using a boolean input
                # parameter.
                figure_.add_annotation(
                    xref="x domain",
                    yref="y domain",
                    x=1.,             # The annotation is right-aligned,
                    # and placed on top of each subplot.
                    y=1.25,
                    showarrow=False,
                    text=aux_name,
                    row=i + 1,
                    col=j + 1)

    elif mode == 'calibration':

        fPlottedOneFit = False

        for i in range(channel_ws_grid.ch_map.rows):
            for j in range(channel_ws_grid.ch_map.columns):

                try:
                    ChannelWs = channel_ws_grid.ch_wf_sets[channel_ws_grid.ch_map.data[i]
                                                           [j].endpoint][channel_ws_grid.ch_map.data[i][j].channel]

                except KeyError:
                    wpu.__add_no_data_annotation(
                        figure_,
                        i + 1,
                        j + 1)
                    continue

                if ChannelWs.CalibHisto is None:
                    raise Exception(GenerateExceptionMessage(
                        3,
                        'plot_channel_ws_grid()',
                        f"In 'calibration' mode, the CalibHisto attribute of each considered ChannelWS object must be defined."))

                aux_name = f"C.H. of channel {
                    channel_ws_grid.ch_map.data[i][j]}"

                fPlottedOneFit |= plot_CalibrationHistogram(
                    ChannelWs.CalibHisto,
                    figure_,
                    name=aux_name,
                    row=i + 1,
                    col=j + 1,
                    plot_fits=plot_peaks_fits,
                    fit_npoints=200)

        if verbose:
            if plot_peaks_fits and not fPlottedOneFit:
                print("In function plot_channel_ws_grid(): No gaussian fit was found for plotting. You may have forgotten to call the fit_peaks_of_calibration_histograms() method of ChannelWsGrid.")
    else:
        raise Exception(GenerateExceptionMessage(
            4,
            'plot_channel_ws_grid()',
            f"The given mode ({mode}) must match either 'overlay', 'average', 'heatmap' or 'calibration'."))
    return figure_
