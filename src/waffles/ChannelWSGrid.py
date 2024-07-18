import numpy as np
from typing import List, Dict, Optional
from plotly import graph_objects as pgo
from plotly import subplots as psu

from .UniqueChannel import UniqueChannel
from .WaveformSet import WaveformSet
from .ChannelWS import ChannelWS
from .ChannelMap import ChannelMap
from .Exceptions import generate_exception_message

class ChannelWSGrid:

    """
    Stands for Channel Waveform Set Grid. This class 
    implements a set of ChannelWS which are ordered 
    according to some ChannelMap object. 

    Attributes
    ----------
    ChMap : ChannelMap
        A ChannelMap object which is used to physically 
        order the ChannelWS objects
    ChWfSets : dict of dict of ChannelWS
        A dictionary whose keys are endpoint values
        for which there is at least one ChannelWS object
        in this ChannelWSGrid object. The values of such
        dictionary are dictionaries, whose keys are
        channel values for which there is a ChannelWS
        object in this ChannelWSGrid object. The values 
        for the deeper-level dictionaries are ChannelWS 
        objects. Note that there might be a certain
        UniqueChannel object which is present in the
        ChannelMap, but for which there is no ChannelWS
        object in this attribute (ChWfSets). I.e.
        appearance of a certain UniqueChannel object
        in the ChannelMap does not guarantee that there
        will be a ChannelWS object in this attribute
        which comes from such unique channel. Hence, 
        one should always handle a KeyError exceptions
        when trying to subscribe ChWfSets with the endpoint
        and channel coming from an UniqueChannel object
        within the ChMap attribute.

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    def __init__(self,  ch_map : ChannelMap,
                        input_waveformset : WaveformSet,
                        compute_calib_histo : bool = False,
                        bins_number : Optional[int] = None,
                        domain : Optional[np.ndarray] = None,
                        variable : str = 'integral',
                        analysis_label : Optional[str] = None):
                        
        """
        ChannelWSGrid class initializer. This initializer
        takes a WaveformSet object as an input, and creates
        a ChannelWSGrid object by partitioning the given
        WaveformSet object using the Endpoint and Channel
        attributes of the UniqueChannel objects which are
        present in the ChannelMap object given to the 
        'ch_map' input parameter. To do so, this initializer 
        delegates the ChannelWSGrid.clusterize_WaveformSet() 
        static method.
        
        Parameters
        ----------
        ch_map : ChannelMap
            The waveforms, within input_waveformset, which
            come from unique channels (endpoint and channel)
            which do not belong to this ChannelMap will not
            be added to this ChannelWSGrid object.
        input_waveformset : WaveformSet
            The WaveformSet object which will be partitioned
            into ChannelWS objects and ordered according
            to the given ChannelMap object. This parameter
            is given to the 'waveform_set' parameter of the
            'clusterize_WaveformSet' static method.
        compute_calib_histo : bool
            If True, then the calibration histogram for each
            resulting ChannelWS object will be computed. 
            It is given to the 'compute_calib_histo' 
            parameter of the 'clusterize_WaveformSet' static
            method.
        bins_number : int
            This parameter only makes a difference if
            'compute_calib_histo' is set to True.
            If so, this parameter must be defined.
            It is given to the 'bins_number' parameter 
            of the 'clusterize_WaveformSet' static method.
            Check its docstring for more information.
        domain : np.ndarray
            This parameter only makes a difference if
            'compute_calib_histo' is set to True. It 
            is given to the 'domain' parameter of the 
            'clusterize_WaveformSet' static method. 
            Check its docstring for more information.
        variable : str
            This parameter only makes a difference if
            'compute_calib_histo' is set to True.
            It is given to the 'variable' parameter of 
            the 'clusterize_WaveformSet' static method. 
            Check its docstring for more information.
        analysis_label : str
            This parameter only makes a difference if
            'compute_calib_histo' is set to True.
            It is given to the 'analysis_label' parameter 
            of the 'clusterize_WaveformSet' static 
            method. Check its docstring for more 
            information.
        """

        ## Shall we add type checks here?

        self.__ch_map = ch_map

        self.__ch_wf_sets = ChannelWSGrid.clusterize_WaveformSet(   input_waveformset,
                                                                    channel_map = ch_map,
                                                                    compute_calib_histo = compute_calib_histo,
                                                                    bins_number = bins_number,
                                                                    domain = domain,
                                                                    variable = variable,
                                                                    analysis_label = analysis_label)
                
    #Getters
    @property
    def ChMap(self):
        return self.__ch_map
    
    @property
    def ChWfSets(self):
        return self.__ch_wf_sets
    
    def get_ChannelWS_by_ij_position_in_map(self,   i : int, 
                                                    j : int) -> Optional[ChannelWS]:
        
        """
        This method returns the ChannelWS object whose
        Endpoint (resp. Channel) attribute matches the
        Endpoint (resp. Channel) attribute of the UniqueChannel
        object which is placed the i-th row and j-th column 
        of the self.__ch_map ChannelMap, if any. If there is
        no such ChannelWS object, then this method returns
        None.
        """

        try:
            output = self.__ch_wf_sets[self.__ch_map.Data[i][j].Endpoint][self.__ch_map.Data[i][j].Channel]
        except KeyError:
            output = None

        return output
    
    @staticmethod
    def clusterize_WaveformSet( waveform_set : WaveformSet,
                                channel_map : Optional[ChannelMap] = None,
                                compute_calib_histo : bool = False,
                                bins_number : Optional[int] = None,
                                domain : Optional[np.ndarray] = None,
                                variable : str = 'integral',
                                analysis_label : Optional[str] = None) -> Dict[int, Dict[int, ChannelWS]]:

        """
        This function returns a dictionary, say output, 
        whose keys are endpoint values. The values of
        of such dictionary are dictionaries, whose keys
        are channel values. The values for the deeper-level
        dictionaries are ChannelWS objects, which are
        initialized by this static method, in a way that
        output[i][j] is the ChannelWS object which contains
        all of the Waveform objects within the given
        WaveformSet object which come from endpoint i and
        channel j.

        This method is useful to partition the given 
        WaveformSet object into WaveformSet objects 
        (actually ChannelWS objects, which inherit from 
        the WaveformSet class but require the Endpoint 
        and the Channel attribute of its constituent 
        Waveform objects to be homogeneous) which are 
        subsets of the given WaveformSet object, and 
        whose Waveform objects have homogeneous endpoint 
        and channel values.

        Parameters
        ----------
        waveform_set : WaveformSet
            The WaveformSet object which will be partitioned
            into ChannelWS objects.
        channel_map : ChannelMap
            If it is not given, then all of the waveforms
            in this WaveformSet object will be considered
            for partitioning. If it is given, then only
            the waveforms which come from channels which
            are present in this ChannelMap object will be
            considered for partitioning.
        compute_calib_histo : bool
            If True, then the calibration histogram for each
            ChannelWS object will be computed. It is given
            to the 'compute_calib_histo' parameter of the
            ChannelWS initializer. Check its docstring for
            more information.
        bins_number : int
            This parameter only makes a difference if
            'compute_calib_histo' is set to True.
            If so, this parameter must be defined.
            It is the number of bins that the calibration 
            histogram will have.
        domain : np.ndarray
            This parameter only makes a difference if
            'compute_calib_histo' is set to True. It 
            is given to the 'domain' parameter of the
            ChannelWS initializer. Check its docstring 
            for more information.
        variable : str
            This parameter only makes a difference if
            'compute_calib_histo' is set to True.
            It is given to the 'variable' parameter of 
            the ChannelWS initializer. Check its docstring 
            for more information.
        analysis_label : str
            This parameter only makes a difference if
            'compute_calib_histo' is set to True.
            It is given to the 'analysis_label' parameter 
            of the ChannelWS initializer. Check its 
            docstring for more information.

        Returns
        ----------
        output : dict of dict of ChannelWS
        """

        if channel_map is None:
            idcs = {}

            for idx in range(len(waveform_set.Waveforms)):
                try:
                    aux = idcs[waveform_set.Waveforms[idx].Endpoint]

                except KeyError:
                    idcs[waveform_set.Waveforms[idx].Endpoint] = {}
                    aux = idcs[waveform_set.Waveforms[idx].Endpoint]

                try:
                    aux[waveform_set.Waveforms[idx].Channel].append(idx)

                except KeyError:
                    aux[waveform_set.Waveforms[idx].Channel] = [idx]

        else:
            idcs = ChannelWSGrid.get_nested_dictionary_template(channel_map)    # idcs contains the endpoints and channels for 
                                                                                # which we can potentially save waveforms.
                                                                                # Contrary to the channel_map == None case,
                                                                                # in this case some of the idcs entries may 
                                                                                # never be filled not even with a single waveform.
                                                                                # We will need to remove those after.
            for idx in range(len(waveform_set.Waveforms)):
                try:
                    aux = idcs[waveform_set.Waveforms[idx].Endpoint]

                except KeyError:
                    continue

                try:
                    aux[waveform_set.Waveforms[idx].Channel].append(idx)

                except KeyError:
                    continue

            empty_channels = {}                             # Now let's remove the channels that are empty.
            for endpoint in idcs.keys():                    # To do so, find those first.
                for channel in idcs[endpoint].keys():
                    if len(idcs[endpoint][channel]) == 0:
                        try:
                            empty_channels[endpoint].append(channel)
                        except KeyError:
                            empty_channels[endpoint] = [channel]

            for endpoint in empty_channels.keys():          # Then remove them. This process is staged to
                for channel in empty_channels[endpoint]:    # prevent a 'RuntimeError: dictionary changed 
                    del idcs[endpoint][channel]             # size during iteration' error

        output = {}

        for endpoint in idcs.keys():
            output[endpoint] = {}

            for channel in idcs[endpoint].keys():
                aux = [waveform_set.Waveforms[idx] for idx in idcs[endpoint][channel]]

                output[endpoint][channel] = ChannelWS(  *aux,
                                                        compute_calib_histo = compute_calib_histo,
                                                        bins_number = bins_number,
                                                        domain = domain,
                                                        variable = variable,
                                                        analysis_label = analysis_label)
        return output
    
    @staticmethod
    def get_nested_dictionary_template(channel_map : ChannelMap) -> Dict[int, Dict[int, List]]:

        """
        This method returns a dictionary which has the same
        structure as the ChWfSets attribute of ChannelWSGrid,
        but whose values are emtpy lists instead of ChannelWS 
        objects. The endpoints and channels that are considered
        for such output are those which are present in the
        input ChannelMap object.

        Parameters
        ----------
        channel_map : ChannelMap
            The ChannelMap object which contains the endpoints
            and channels which will end up in the ouput of
            this method.
    
        Returns
        ----------
        output : dict of dict of list
        """

        output = {}

        for i in range(channel_map.Rows):
            for j in range(channel_map.Columns):
                
                try:
                    aux = output[channel_map.Data[i][j].Endpoint]

                except KeyError:
                    output[channel_map.Data[i][j].Endpoint] = {}
                    aux = output[channel_map.Data[i][j].Endpoint]

                aux[channel_map.Data[i][j].Channel] = []
        
        return output

    def purge(self) -> None:    # Before 2024/06/27, this method was used in
                                # ChannelWSGrid.__init___, because the output
                                # of ChannelWSGrid.clusterize_WaveformSet()
                                # contained channels which were present in its
                                # WaveformSet input, but were not present in the
                                # self.__ch_map attribute. As of such date, 
                                # ChannelWSGrid.clusterize_WaveformSet() is 
                                # fixed and this method is not used anymore, but 
                                # it is kept here in case we need this 
                                # functionality in the future.

        """
        Removes the ChannelWS objects from self.__ch_wf_sets 
        which come from unique channels which are not present
        in self.__ch_map.
        """

        unique_channels_to_remove = {}

        for endpoint in self.__ch_wf_sets.keys():
            for channel in self.__ch_wf_sets[endpoint].keys():

                aux = UniqueChannel(endpoint, channel)
                
                if not self.__ch_map.find_channel(aux)[0]:
                    try:
                        unique_channels_to_remove[aux.Endpoint].append(aux.Channel)     # Keep note of the channel to remove, 
                    except KeyError:                                                    # but not remove it yet, since we are
                        unique_channels_to_remove[aux.Endpoint] = [aux.Channel]         # iterating over the dictionary keys

        for endpoint in unique_channels_to_remove.keys():
            for channel in unique_channels_to_remove[endpoint]:
                del self.__ch_wf_sets[endpoint][channel]

        endpoints_to_remove = []    # Second scan to remove endpoints 
                                    # which have no channels left

        for endpoint in self.__ch_wf_sets.keys():
            if len(self.__ch_wf_sets[endpoint]) == 0:
                endpoints_to_remove.append(endpoint)

        for endpoint in endpoints_to_remove:
            del self.__ch_wf_sets[endpoint]

        return
    
    def plot(self,  *args,
                    figure : Optional[pgo.Figure] = None,
                    share_x_scale : bool = False,
                    share_y_scale : bool = False,
                    mode : str = 'overlay',
                    wfs_per_axes : Optional[int] = 1,
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
                    plot_peaks_fits : bool = False,
                    detailed_label : bool = True,
                    **kwargs) -> pgo.Figure:
        
        """ 
        This method returns a plotly.graph_objects.Figure 
        with a grid of subplots which are arranged according
        to the self.__ch_map attribute. The subplot at position
        i,j may be empty if there is no ChannelWS object in
        self.__ch_wf_sets which matches the UniqueChannel object
        at position i,j in the self.__ch_map attribute. If it
        is not empty, a subplot may contain a waveform
        representation (either overlayed, averaged or heatmapped),
        or a calibration histogram. The type of representation
        is determined by the 'mode' parameter.

        Parameters
        ----------
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
        figure : plotly.graph_objects.Figure
            If it is not None, then it must have been
            generated using plotly.subplots.make_subplots()
            with a 'rows' and 'cols' parameters matching
            the Rows and Columns attribute of self.__ch_map.
            If that's the case, then this method adds the
            plots to this figure and eventually returns 
            it. If it is None, then this method generates
            a new figure and returns it.
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
            every considered waveform, only the averaged 
            waveform of the considered waveforms will be plotted. 
                If it is set to 'heatmap', then a 2D-histogram, 
            whose entries are the union of all of the points 
            of every considered waveform, will be plotted. In 
            the 'heatmap' mode, the baseline of each waveform 
            is subtracted from each waveform before plotting. 
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
            If it is None, then every waveform in each
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
            parameter is defined, then this method will 
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
                If 'mode' is set to 'calibration', then
            this parameter is not used.
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
            is assumed to be 0 (resp. self.PointsPerWf - 1).
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
        plot_peaks_fits : bool
            This parameter only makes a difference if the
            'mode' parameter is set to 'calibration'. In that
            case, then for the calibration histogram of each 
            subplot, this parameter is given to the 'plot_fits' 
            parameter of the CalibrationHistogram.plot() method.
            It means whether to plot the fits of the peaks, if
            available, over the histogram.
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
            This method returns a plotly.graph_objects.Figure 
            with a grid of subplots which are arranged 
            according to the self.__ch_map attribute.
        """

        if figure is not None:
            WaveformSet.check_dimensions_of_suplots_figure( figure,
                                                            self.__ch_map.Rows,
                                                            self.__ch_map.Columns)
            figure_ = figure
        else:
            figure_ = psu.make_subplots(    rows = self.__ch_map.Rows, 
                                            cols = self.__ch_map.Columns)
        fPlotAll = True
        if wfs_per_axes is not None:

            if wfs_per_axes < 1:
                raise Exception(generate_exception_message( 1,
                                                            'ChannelWSGrid.plot()',
                                                            'If defined, the number of waveforms per axes must be positive.'))
            fPlotAll = False

        self.__add_unique_channels_top_annotations(figure_,
                                                   also_add_run_info = True if mode != 'heatmap' else False)    # If mode is 'heatmap', then
                                                                                                                # there is already a right-aligned
                                                                                                                # top annotation which shows the
                                                                                                                # iterator values of the first
                                                                                                                # waveforms. We are not adding a
                                                                                                                # new one so that they don't collide.
        WaveformSet.update_shared_axes_status(  figure_,                    # An alternative way is to specify 
                                                share_x = share_x_scale,    # shared_xaxes=True (or share_yaxes=True)
                                                share_y = share_y_scale)    # in psu.make_subplots(), but, for us, 
                                                                            # that alternative is only doable for 
                                                                            # the case where the given 'figure'
                                                                            # parameter is None.
        if mode == 'overlay':
            for i in range(self.__ch_map.Rows):
                for j in range(self.__ch_map.Columns):

                    try:
                        channel_ws = self.__ch_wf_sets[self.__ch_map.Data[i][j].Endpoint][self.__ch_map.Data[i][j].Channel]

                    except KeyError:
                        WaveformSet._WaveformSet__add_no_data_annotation(   figure_,
                                                                            i + 1,
                                                                            j + 1)
                        continue

                    if fPlotAll:
                        aux_idcs = range(len(channel_ws.Waveforms))
                    else:
                        aux_idcs = range(min(wfs_per_axes, len(channel_ws.Waveforms)))  # If wfs_per_axes is defined, then it has been
                                                                                        # checked to be >=1. If it is not defined, then
                                                                                        # still len(channel_ws.Waveforms) is >=1 (which
                                                                                        # is ensured by WaveformSet.__init__), so the 
                                                                                        # minimum is always >=1.
                    for idx in aux_idcs:

                        aux_name = f"({i+1},{j+1}) - Wf {idx}, Ch {self.__ch_map.Data[i][j]}"

                        channel_ws.Waveforms[idx].plot( figure = figure_,
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
        elif mode == 'average':
            for i in range(self.__ch_map.Rows):
                for j in range(self.__ch_map.Columns):

                    try:
                        channel_ws = self.__ch_wf_sets[self.__ch_map.Data[i][j].Endpoint][self.__ch_map.Data[i][j].Channel]

                    except KeyError:
                        WaveformSet._WaveformSet__add_no_data_annotation(   figure_,
                                                                            i + 1,
                                                                            j + 1)
                        continue

                    if fPlotAll:
                        aux_idcs = range(len(channel_ws.Waveforms))
                    else:
                        aux_idcs = range(min(wfs_per_axes, len(channel_ws.Waveforms)))

                    aux = channel_ws.compute_mean_waveform(wf_idcs = list(aux_idcs))    # WaveformSet.compute_mean_waveform()
                                                                                        # will raise an exception if
                                                                                        # list(aux_idcs) is empty 
                    fAnalyzed = False
                    if analysis_label is not None:
                        
                        _ = aux.analyse(    analysis_label,
                                            *args,
                                            **kwargs)
                        fAnalyzed = True

                    aux_name = f"{len(aux_idcs)} Wf(s)"
                    if detailed_label:
                        aux_name += f": [{WaveformSet.get_string_of_first_n_integers_if_available(list(aux_idcs), queried_no = 2)}]"

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
                raise Exception(generate_exception_message( 2,
                                                            'ChannelWSGrid.plot()',
                                                            "The 'analysis_label' parameter must be defined if the 'mode' parameter is set to 'heatmap'."))
            for i in range(self.__ch_map.Rows):
                for j in range(self.__ch_map.Columns):

                    try:
                        channel_ws = self.__ch_wf_sets[self.__ch_map.Data[i][j].Endpoint][self.__ch_map.Data[i][j].Channel]

                    except KeyError:
                        WaveformSet._WaveformSet__add_no_data_annotation(   figure_,
                                                                            i + 1,
                                                                            j + 1)
                        continue

                    if fPlotAll:
                        aux_idcs = range(len(channel_ws.Waveforms))
                    else:
                        aux_idcs = range(min(wfs_per_axes, len(channel_ws.Waveforms)))

                    aux_name = f"{len(aux_idcs)} Wf(s)"
                    if detailed_label:
                        aux_name += f": [{WaveformSet.get_string_of_first_n_integers_if_available(list(aux_idcs), queried_no = 2)}]"

                    aux_ranges = channel_ws.arrange_time_vs_ADC_ranges( time_range_lower_limit = time_range_lower_limit,
                                                                        time_range_upper_limit = time_range_upper_limit,
                                                                        adc_range_above_baseline = adc_range_above_baseline,
                                                                        adc_range_below_baseline = adc_range_below_baseline)
                
                    figure_ = channel_ws._WaveformSet__subplot_heatmap( figure_,
                                                                        aux_name,
                                                                        i + 1,
                                                                        j + 1,
                                                                        list(aux_idcs),
                                                                        analysis_label,
                                                                        time_bins,
                                                                        adc_bins,
                                                                        aux_ranges,
                                                                        show_color_bar = False) # The color scale is not shown          ## There is a way to make the color scale match for     # https://community.plotly.com/t/trying-to-make-a-uniform-colorscale-for-each-of-the-subplots/32346
                                                                                                # since it may differ from one plot     ## every plot in the grid, though, but comes at the
                                                                                                # to another.                           ## cost of finding the max and min values of the 
                                                                                                                                        ## union of all of the histograms. Such feature may 
                                                                                                                                        ## be enabled in the future, using a boolean input
                                                                                                                                        ## parameter.
                    figure_.add_annotation( xref = "x domain", 
                                            yref = "y domain",      
                                            x = 1.,             # The annotation is right-aligned,
                                            y = 1.25,           # and placed on top of each subplot.
                                            showarrow = False,
                                            text = aux_name,
                                            row = i + 1,
                                            col = j + 1)

        elif mode == 'calibration':
            for i in range(self.__ch_map.Rows):
                for j in range(self.__ch_map.Columns):

                    try:
                        channel_ws = self.__ch_wf_sets[self.__ch_map.Data[i][j].Endpoint][self.__ch_map.Data[i][j].Channel]

                    except KeyError:
                        WaveformSet._WaveformSet__add_no_data_annotation(   figure_,
                                                                            i + 1,
                                                                            j + 1)
                        continue

                    if channel_ws.CalibHisto is None:
                        raise Exception(generate_exception_message( 3,
                                                                    'ChannelWSGrid.plot()',
                                                                    f"In 'calibration' mode, the CalibHisto attribute of each considered ChannelWS object must be defined."))
                    
                    aux_name = f"C.H. of channel {self.__ch_map.Data[i][j]}"

                    channel_ws.CalibHisto.plot( figure_,
                                                name = aux_name,
                                                row = i + 1,
                                                col = j + 1,
                                                plot_fits = plot_peaks_fits,
                                                fit_npoints = 200)
        else:                                                                                                           
            raise Exception(generate_exception_message( 4,
                                                        'ChannelWSGrid.plot()',
                                                        f"The given mode ({mode}) must match either 'overlay', 'average', 'heatmap' or 'calibration'."))
        return figure_

    def __add_unique_channels_top_annotations(self, figure : pgo.Figure,
                                                    also_add_run_info : bool = False) -> pgo.Figure:

        """
        This method is not intended for user usage. It is
        meant to be called uniquely by the ChannelWSGrid.plot() 
        method, where the well-formedness of the input 
        figure has been checked. This method adds annotations 
        on top of each subplot of the given figure. The 
        annotations are the string representation of the
        UniqueChannel object, each of which is placed
        on top of a subplot according to its position in
        the self.__ch_map attribute.

        Parameters
        ----------
        figure : plotly.graph_objects.Figure
            The figure to which the annotations will be added
        also_add_run_info : bool
            If True, then for each subplot for which there
            is a ChannelWS object, say chws, present in the 
            self.__ch_wf_sets attribute, the first run number 
            which appears in the chws.Runs attribute will be
            additionally added to the annotation. For each
            subplot for which there is no ChannelWS object,
            according to the physical position given by the
            self.__ch_map attribute, no additional annotation
            will be added.

        Returns
        ----------
        figure : plotly.graph_objects.Figure
            The given figure with the annotations added
        """

        for i in range(self.__ch_map.Rows):
            for j in range(self.__ch_map.Columns):
                figure.add_annotation(  xref = "x domain", 
                                        yref = "y domain",      
                                        x = 0.,             # The annotation is left-aligned
                                        y = 1.25,           # and on top of each subplot
                                        showarrow = False,
                                        text = str(self.__ch_map.Data[i][j]),   # Implicitly using UniqueChannel.__repr__()
                                        row = i + 1,
                                        col = j + 1)    
        if also_add_run_info:
            for i in range(self.__ch_map.Rows):
                for j in range(self.__ch_map.Columns):
                    try:
                        channel_ws = self.__ch_wf_sets[self.__ch_map.Data[i][j].Endpoint][self.__ch_map.Data[i][j].Channel]
                    
                    except KeyError:
                        continue

                    aux = list(channel_ws.Runs) # Since a WaveformSet must contain at
                                                # least one waveform, it is ensured that
                                                # there is at least one run value here
                    if len(aux)>1:
                        annotation = f"Runs {aux[0]}, ..."
                    else:
                        annotation = f"Run {aux[0]}"

                    figure.add_annotation(  xref = "x domain", 
                                            yref = "y domain",      
                                            x = 1.,             # The run annotation is right-aligned
                                            y = 1.25,
                                            showarrow = False,
                                            text = annotation,
                                            row = i + 1,
                                            col = j + 1)
        return figure
    
    def fit_peaks_of_calibration_histograms(self,   max_peaks : int,
                                                    prominence : float,
                                                    half_points_to_fit : int,
                                                    initial_percentage = 0.1,
                                                    percentage_step = 0.1) -> bool:
        
        """
        This method calls the fit_peaks() method of 
        each CalibrationHistogram object in the
        ChannelWS objects contained in self.__ch_wf_sets
        whose channel is present in the self.__ch_map
        attribute. It returns False if at least one 
        of the fit_peaks() calls returns False, and 
        True if every fit_peaks() call returned True.
        I.e. it returns True if max_peaks peaks were
        successfully found for each histogram, and
        False if only n peaks were found for at
        least one of the histograms, where n < max_peaks.

        Parameters
        ----------
        max_peaks : int
            The maximum number of peaks which will be
            searched for in each calibration histogram.
            It is given to the 'max_peaks' parameter of
            the CalibrationHistogram.fit_peaks() method
            for each calibration histogram.
        prominence : float
            It must be greater than 0.0 and smaller than 
            1.0. It gives the minimal prominence of the 
            peaks to spot. This parameter is passed to the 
            'prominence' parameter of the 
            CalibrationHistogram.fit_peaks() method for 
            each calibration histogram. For more information, 
            check the documentation of such method.
        half_points_to_fit : int
            It must be a positive integer. For each peak in
            each calibration histogram, it gives the number 
            of points to consider on either side of the peak 
            maximum, to fit each gaussian function. It is
            given to the 'half_points_to_fit' parameter of
            the CalibrationHistogram.fit_peaks() method for
            each calibration histogram. For more information, 
            check the documentation of such method.
        initial_percentage : float
            It must be greater than 0.0 and smaller than 1.0.
            This parameter is passed to the 'initial_percentage' 
            parameter of the CalibrationHistogram.fit_peaks()
            method for each calibration histogram. For more 
            information, check the documentation of such method.
        percentage_step : float
            It must be greater than 0.0 and smaller than 1.0.
            This parameter is passed to the 'percentage_step'
            parameter of the CalibrationHistogram.fit_peaks()
            method for each calibration histogram. For more 
            information, check the documentation of such method.

        Returns
        ----------
        output : bool
            True if max_peaks peaks were successfully found for 
            each histogram, and False if only n peaks were found 
            for at least one of the histograms, where n < max_peaks.
        """

        output = True

        for i in range(self.__ch_map.Rows):
            for j in range(self.__ch_map.Columns):

                try:
                    channel_ws = self.__ch_wf_sets[self.__ch_map.Data[i][j].Endpoint][self.__ch_map.Data[i][j].Channel]

                except KeyError:
                    continue

                output *= channel_ws.CalibHisto.fit_peaks(  max_peaks,
                                                            prominence,
                                                            half_points_to_fit,
                                                            initial_percentage = initial_percentage,
                                                            percentage_step = percentage_step)
        return output