import math
import inspect

import uproot
import numba
import numpy as np
from typing import Tuple, List, Callable, Optional
from plotly import graph_objects as pgo
from plotly import subplots as psu

from .WaveformAdcs import WaveformAdcs
from .Waveform import Waveform
from .WfAna import WfAna
from .WfAnaResult import WfAnaResult
from .Exceptions import generate_exception_message

class WaveformSet:

    """
    This class implements a set of waveforms.

    Attributes
    ----------
    Waveforms : list of Waveform objects
        Waveforms[i] gives the i-th waveform in the set.
    PointsPerWf : int
        Number of entries for the Adcs attribute of
        each Waveform object in this WaveformSet object.
    Runs : set of int
        It contains the run number of any run for which
        there is at least one waveform in the set.
    RecordNumbers : dictionary of sets
        It is a dictionary whose keys are runs (int) and
        its values are sets of record numbers (set of int).
        If there is at least one Waveform object within
        this WaveformSet which was acquired during run n,
        then n belongs to RecordNumbers.keys(). RecordNumbers[n]
        is a set of record numbers for run n. If there is at
        least one waveform acquired during run n whose
        RecordNumber is m, then m belongs to RecordNumbers[n].
    AvailableChannels : dictionary of dictionaries of sets
        It is a dictionary whose keys are run numbers (int),
        so that if there is at least one waveform in the set
        which was acquired during run n, then n belongs to
        AvailableChannels.keys(). AvailableChannels[n] is a
        dictionary whose keys are endpoints (int) and its 
        values are sets of channels (set of int). If there 
        is at least one Waveform object within this WaveformSet 
        which was acquired during run n and which comes from 
        endpoint m, then m belongs to AvailableChannels[n].keys(). 
        AvailableChannels[n][m] is a set of channels for 
        endpoint m during run n. If there is at least one 
        waveform for run n, endpoint m and channel p, then p 
        belongs to AvailableChannels[n][m].
    MeanAdcs : WaveformAdcs
        The mean of the adcs arrays for every waveform
        or a subset of waveforms in this WaveformSet. It 
        is a WaveformAdcs object whose TimeStep_ns
        attribute is assumed to match that of the first
        waveform which was used in the average sum.
        Its Adcs attribute contains PointsPerWf entries,
        so that MeanAdcs.Adcs[i] is the mean of 
        self.Waveforms[j].Adcs[i] for every value
        of j or a subset of values of j, within 
        [0, len(self.__waveforms) - 1]. It is not 
        computed by default. I.e. if self.MeanAdcs 
        equals to None, it should be interpreted as 
        unavailable data. Call the 'compute_mean_waveform' 
        method of this WaveformSet to compute it.
    MeanAdcsIdcs : tuple of int
        It is a tuple of integers which contains the indices
        of the waveforms, with respect to this WaveformSet,
        which were used to compute the MeanAdcs.Adcs 
        attribute. By default, it is None. I.e. if 
        self.MeanAdcsIdcs equals to None, it should be 
        interpreted as unavailable data. Call the 
        'compute_mean_waveform' method of this WaveformSet 
        to compute it.

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    def __init__(self,  *waveforms):
        
        """
        WaveformSet class initializer
        
        Parameters
        ----------
        waveforms : unpacked list of Waveform objects
            The waveforms that will be added to the set
        """

        ## Shall we add type checks here?

        if len(waveforms) == 0:
            raise Exception(generate_exception_message( 1,
                                                        'WaveformSet.__init__()',
                                                        'There must be at least one waveform in the set.'))
        self.__waveforms = list(waveforms)

        if not self.check_length_homogeneity():
            raise Exception(generate_exception_message( 2,
                                                        'WaveformSet.__init__()',
                                                        'The length of the given waveforms is not homogeneous.'))
        
        self.__points_per_wf = len(self.__waveforms[0].Adcs)

        self.__runs = set()
        self.update_runs()

        self.__record_numbers = {}
        self.update_record_numbers()

        self.__available_channels = {}
        self.update_available_channels()    # Running on an Apple M2, it took 
                                            # ~ 52 ms to run this line for a
                                            # WaveformSet with 1046223 waveforms
        self.__mean_adcs = None
        self.__mean_adcs_idcs = None

    #Getters
    @property
    def Waveforms(self):
        return self.__waveforms
    
    @property
    def PointsPerWf(self):
        return self.__points_per_wf
    
    @property
    def Runs(self):
        return self.__runs
    
    @property
    def RecordNumbers(self):
        return self.__record_numbers
    
    @property
    def AvailableChannels(self):
        return self.__available_channels
    
    @property
    def MeanAdcs(self):
        return self.__mean_adcs
    
    @property
    def MeanAdcsIdcs(self):
        return self.__mean_adcs_idcs
    
    def get_set_of_endpoints(self) -> set:
            
        """
        This method returns a set which contains every endpoint
        for which there is at least one waveform in this 
        WaveformSet object.

        Returns
        ----------
        output : set of int
        """

        output = set()

        for run in self.__available_channels.keys():
            for endpoint in self.__available_channels[run].keys():
                output.add(endpoint)

        return output
    
    def get_run_collapsed_available_channels(self) -> dict:
            
        """
        This method returns a dictionary of sets of integers,
        say output, whose keys are endpoints. If there is
        at least one waveform within this set that comes from
        endpoint n, then n belongs to output.keys(). output[n]
        is a set of integers, so that if there is at least a
        waveform coming from endpoint n and channel m, then m
        belongs to output[n].

        Returns
        ----------
        output : dictionary of sets
        """

        output = {}

        for run in self.__runs:
            for endpoint in self.__available_channels[run].keys():
                try:
                    aux = output[endpoint]
                except KeyError:
                    output[endpoint] = set()
                    aux = output[endpoint]

                for channel in self.__available_channels[run][endpoint]:
                    aux.add(channel)

        return output
    
    def check_length_homogeneity(self) -> bool:
            
            """
            This method returns True if the Adcs attribute
            of every Waveform object in this WaveformSet
            has the same length. It returns False if else.
            In order to call this method, there must be at
            least one waveform in the set.

            Returns
            ----------
            bool
            """

            if len(self.__waveforms) == 0:
                raise Exception(generate_exception_message( 1,
                                                            'WaveformSet.check_length_homogeneity()',
                                                            'There must be at least one waveform in the set.'))
            length = len(self.__waveforms[0].Adcs)
            for i in range(1, len(self.__waveforms)):
                if len(self.__waveforms[i].Adcs) != length:
                    return False
            return True
    
    def update_runs(self) -> None:
        
        """
        This method clears the self.__runs attribute 
        of this object and then iterates through the 
        whole WaveformSet to fill such attribute 
        according to the waveforms which are currently
        present in this WaveformSet object.

        Returns
        ----------
        None
        """

        self.__runs.clear()

        for wf in self.__waveforms:
            self.__runs.add(wf.RunNumber)
        return
    
    def update_record_numbers(self) -> None:
        
        """
        This method clears the self.__record_numbers 
        attribute of this object and then iterates 
        through the whole WaveformSet to fill such 
        attribute according to the waveforms which 
        are currently present in this WaveformSet 
        object.

        Returns
        ----------
        None
        """

        self.__record_numbers.clear()

        for wf in self.__waveforms:
            try:
                self.__record_numbers[wf.RunNumber].add(wf.RecordNumber)
            except KeyError:
                self.__record_numbers[wf.RunNumber] = set()
                self.__record_numbers[wf.RunNumber].add(wf.RecordNumber)
        return
    
    def update_available_channels(self) -> None:
        
        """
        This method clears the self.__available_channels 
        attribute of this object and then iterates through 
        the whole WaveformSet to fill such attribute 
        according to the waveforms which are currently
        present in this WaveformSet object.

        Returns
        ----------
        None
        """

        self.__available_channels.clear()

        for wf in self.__waveforms:
            try:
                aux = self.__available_channels[wf.RunNumber]

                try:
                    aux[wf.Endpoint].add(wf.Channel)

                except KeyError:
                    aux[wf.Endpoint] = set()
                    aux[wf.Endpoint].add(wf.Channel)

            except KeyError:
                self.__available_channels[wf.RunNumber] = {}
                self.__available_channels[wf.RunNumber][wf.Endpoint] = set()
                self.__available_channels[wf.RunNumber][wf.Endpoint].add(wf.Channel)    
        return
    
    def analyse(self,   label : str,
                        analyser_name : str,
                        baseline_limits : List[int],
                        *args,
                        int_ll : int = 0,
                        int_ul : Optional[int] = None,
                        overwrite : bool = False,
                        **kwargs) -> dict:
        
        """
        For each Waveform in this WaveformSet, this method
        calls its 'analyse' method passing to it the parameters
        given to this method. In turn, Waveform.analyse()
        (actually WaveformAdcs.analyse()) creates a WfAna
        object and adds it to the Analyses attribute of the 
        analysed waveform. It also runs the indicated analyser 
        method (up to the 'analyser_name' parameter) on the 
        waveform, and adds its results to the 'Result' and 
        'Passed' attributes of the newly created WfAna object. 
        Also, it returns a dictionary, say output, whose keys 
        are integers in [0, len(self.__waveforms) - 1]. 
        ouptut[i] matches the output of 
        self.__waveforms[i].analyse(...), which is a dictionary. 
        I.e. the output of this method is a dictionary of 
        dictionaries.

        Parameters
        ----------
        label : str
            For every analysed waveform, this is the key
            for the new WfAna object within its Analyses
            attribute.
        analyser_name : str
            It must match the name of a WfAna method whose first            
            argument must be called 'waveform' and whose type       # The only way to import the WaveformAdcs class in WfAna without having     # This would not be a problem (and we would not    
            annotation must match the WaveformAdcs class or the     # a circular import is to use the typing.TYPE_CHECKING variable, which      # need to grab the analyser method using an 
            'WaveformAdcs' string literal. Such method should       # is only defined for type-checking runs. As a consequence, the type        # string and getattr) if the analyser methods were
            also have a defined return-annotation which must        # annotation should be an string, which the type-checking software          # defined as WaveformAdcs methods or in a separate
            match Tuple[WfAnaResult, bool, dict].                   # successfully associates to the class itself, but which is detected        # module. There might be other downsizes to it such
                                                                    # as so (a string) by inspect.signature().                                  #  as the accesibility to WfAna attributes.
        baseline_limits : list of int
            For every analysed waveform, say wf, it 
            defines the Adcs points which will be used 
            for baseline calculation (it is given to
            the 'baseline_limits' parameter of
            Waveform.analyse() - actually 
            WaveformAdcs.analyse()). It must have an 
            even number of integers which must meet 
            baseline_limits[i] < baseline_limits[i + 1] 
            for all i. The points which are used for 
            baseline calculation are 
            wf.Adcs[baseline_limits[2*i]:baseline_limits[(2*i) + 1]],
            with i = 0,1,...,(len(baseline_limits)/2) - 1. 
            The upper limits are exclusive. For more 
            information check the 'baseline_limits' 
            parameter documentation in the 
            Waveform.analyse() docstring.
        *args
            For each analysed waveform, these are the 
            positional arguments which are given to the
            analyser method by WaveformAdcs.analyse().
        int_ll (resp. int_ul): int
            For every analysed waveform, it defines the
            integration window (it is given to the 'int_ll'
            (resp. 'int_ul') parameter of Waveform.analyse()
            - actually WaveformAdcs.analyse()). int_ll must 
            be smaller than int_ul. These limits are 
            inclusive. If they are not defined, then the
            whole Adcs are considered for each waveform. 
            For more information check the 'int_ll' and 
            'int_ul' parameters documentation in the 
            Waveform.analyse() docstring.
        overwrite : bool
            If True, for every analysed Waveform wf, its
            'analyse' method will overwrite any existing
            WfAna object with the same label (key) within
            its Analyses attribute.
        **kwargs
            For each analysed waveform, these are the
            keyword arguments which are given to the
            analyser method by WaveformAdcs.analyse().

        Returns
        ----------
        output : dict
            output[i] gives the output of 
            self.__waveforms[i].analyse(...), which is a
            dictionary containing any additional information
            of the analysis which was performed over the
            i-th waveform of this WaveformSet. Such 
            dictionary is empty if the analyser method gives 
            no additional information.
        """

        if not self.baseline_limits_are_well_formed(baseline_limits):
            raise Exception(generate_exception_message( 1,
                                                        'WaveformSet.analyse()',
                                                        f"The baseline limits ({baseline_limits}) are not well formed."))
        int_ul_ = int_ul
        if int_ul_ is None:
            int_ul_ = self.PointsPerWf - 1

        if not self.subinterval_is_well_formed(int_ll, int_ul_):
            raise Exception(generate_exception_message( 2,
                                                        'WaveformSet.analyse()',
                                                        f"The integration window ({int_ll}, {int_ul_}) is not well formed."))
        aux = WfAna([0,1],  # Dummy object to access
                    0,      # the analyser instance method
                    1,)
        try:
            analyser = getattr(aux, analyser_name)
        except AttributeError:
            raise Exception(generate_exception_message( 3,
                                                        'WaveformSet.analyse()',
                                                        f"The analyser method '{analyser_name}' does not exist in the WfAna class."))
        try:
            signature = inspect.signature(analyser)
        except TypeError:
            raise Exception(generate_exception_message( 4,
                                                        'WaveformSet.analyse()',
                                                        f"'{analyser_name}' does not match a callable attribute of WfAna."))
        try:
            if list(signature.parameters.keys())[0] != 'waveform':
                raise Exception(generate_exception_message( 5,
                                                            "WaveformSet.analyse()",
                                                            "The name of the first parameter of the given analyser method must be 'waveform'."))
            
            if signature.parameters['waveform'].annotation not in ['WaveformAdcs', WaveformAdcs]:
                raise Exception(generate_exception_message( 6,
                                                            "WaveformSet.analyse()",
                                                            "The 'waveform' parameter of the analyser method must be hinted as a WaveformAdcs object."))
            
            if signature.return_annotation != Tuple[WfAnaResult, bool, dict]:
                raise Exception(generate_exception_message( 7,
                                                            "WaveformSet.analyse()",
                                                            "The return type of the analyser method must be hinted as Tuple[WfAnaResult, bool, dict]."))
        except IndexError:
            raise Exception(generate_exception_message( 8,
                                                        "WaveformSet.analyse()",
                                                        'The given analyser method must take at least one parameter.'))
        output = {}

        for i in range(len(self.__waveforms)):
            output[i] = self.__waveforms[i].analyse(    label,
                                                        analyser_name,
                                                        baseline_limits,
                                                        *args,
                                                        int_ll = int_ll,
                                                        int_ul = int_ul_,
                                                        overwrite = overwrite,
                                                        **kwargs)
        return output
    
    def baseline_limits_are_well_formed(self, baseline_limits : List[int]) -> bool:

        """
        This method returns True if len(baseline_limits) is even and 
        0 <= baseline_limits[0] < baseline_limits[1] < ... < baseline_limits[-1] <= self.PointsPerWf - 1.
        It returns False if else.

        Parameters
        ----------
        baseline_limits : list of int

        Returns
        ----------
        bool
        """

        if len(baseline_limits)%2 != 0:
            return False

        if baseline_limits[0] < 0:
            return False
            
        for i in range(0, len(baseline_limits) - 1):
            if baseline_limits[i] >= baseline_limits[i + 1]:
                return False
                
        if baseline_limits[-1] > self.PointsPerWf - 1:
            return False
        
        return True
    
    def subinterval_is_well_formed(self,    i_low : int, 
                                            i_up : int) -> bool:
        
        """
        This method returns True if 0 <= i_low < i_up <= self.PointsPerWf - 1,
        and False if else.

        Parameters
        ----------
        i_low : int
        i_up : int

        Returns
        ----------
        bool
        """

        if i_low < 0:
            return False
        elif i_up <= i_low:
            return False
        elif i_up > self.PointsPerWf - 1:
            return False
        
        return True
    
    def plot_wfs(self,  *args,
                        nrows : int = 1,
                        ncols : int = 1,
                        figure : Optional[pgo.Figure] = None,
                        wfs_per_axes : Optional[int] = 1,
                        grid_of_wf_idcs : Optional[List[List[List[int]]]] = None,
                        share_x_scale : bool = False,
                        share_y_scale : bool = False,
                        mode : str = 'overlay',
                        analysis_label : Optional[str] = None,
                        plot_analysis_markers : bool = False,
                        show_baseline_limits : bool = False, 
                        show_baseline : bool = True,
                        show_general_integration_limits : bool = False,
                        show_spotted_peaks : bool = True,
                        show_peaks_integration_limits : bool = False,
                        time_bins : int = 512,
                        adc_bins : int = 100,
                        adc_range_above_baseline : int = 100,
                        adc_range_below_baseline : int = 200,
                        detailed_label : bool = True,
                        **kwargs) -> pgo.Figure: 

        """ 
        This method returns a plotly.graph_objects.Figure 
        with a nrows x ncols grid of axes, with plots of
        some of the waveforms in this WaveformSet object.

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
        nrows (resp. ncols) : int
            Number of rows (resp. columns) of the returned 
            grid of axes.
        figure : plotly.graph_objects.Figure
            If it is not None, then it must have been
            generated using plotly.subplots.make_subplots()
            (even if nrows and ncols equal 1). It is the
            caller's responsibility to ensure this.
            If that's the case, then this method adds the
            plots to this figure and eventually returns 
            it. In such case, the number of rows (resp. 
            columns) in such figure must match the 'nrows' 
            (resp. 'ncols') parameter.
        wfs_per_axes : int
            If it is not None, then the argument given to 
            'grid_of_wf_idcs' will be ignored. In this case,
            the number of waveforms considered for each
            axes is wfs_per_axes. P.e. for wfs_per_axes 
            equal to 2, the axes at the first row and first
            column contains information about the first
            two waveforms in the set. The axes in the first 
            row and second column will consider the 
            following two, and so on.
        grid_of_wf_idcs : list of list of list of int
            This list must contain nrows lists, each of which
            must contain ncols lists of integers. 
            grid_of_wf_idcs[i][j] gives the indices of the 
            waveforms, with respect to this WaveformSet, which
            should be considered for plotting in the axes
            which are located at the i-th row and j-th column.
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
            self.__analyses will be the used one. 
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
                In both cases, If True, analysis markers 
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
                                                        'WaveformSet.plot_wfs()',
                                                        'The number of rows and columns must be positive.'))
        fFigureIsGiven = False
        if figure is not None:

            try:
                fig_rows, fig_cols = figure._get_subplot_rows_columns() # Returns two range objects
                fig_rows, fig_cols = list(fig_rows)[-1], list(fig_cols)[-1]

            except Exception:   # Happens if figure was not created using plotly.subplots.make_subplots

                raise Exception(generate_exception_message( 2,
                                                            'WaveformSet.plot_wfs()',
                                                            'The given figure is not a subplot grid.'))
            if fig_rows != nrows or fig_cols != ncols:
                
                raise Exception(generate_exception_message( 3,
                                                            'WaveformSet.plot_wfs()',
                                                            f"The number of rows and columns in the given figure ({fig_rows}, {fig_cols}) must match the nrows ({nrows}) and ncols ({ncols}) parameters."))
            fFigureIsGiven = True

        grid_of_wf_idcs_ = None         # Logically useless

        if wfs_per_axes is not None:    # wfs_per_axes is defined

            if wfs_per_axes < 1:
                raise Exception(generate_exception_message( 4,
                                                            'WaveformSet.plot_wfs()',
                                                            'The number of waveforms per axes must be positive.'))

            grid_of_wf_idcs_ = self.get_grid_of_wf_idcs(nrows,
                                                        ncols,
                                                        wfs_per_axes = wfs_per_axes)

        elif grid_of_wf_idcs is None:   # Nor wf_per_axes, nor 
                                        # grid_of_wf_idcs are defined

            raise Exception(generate_exception_message( 5,
                                                        'WaveformSet.plot_wfs()',
                                                        "The 'grid_of_wf_idcs' parameter must be defined if wfs_per_axes is not."))
        
        elif not WaveformSet.grid_of_lists_is_well_formed(  grid_of_wf_idcs,    # wf_per_axes is not defined, 
                                                            nrows,              # but grid_of_wf_idcs is, but 
                                                            ncols):             # it is not well-formed
            raise Exception(generate_exception_message( 6,
                                                        'WaveformSet.plot_wfs()',
                                                        f"The given grid_of_wf_idcs is not well-formed according to nrows ({nrows}) and ncols ({ncols})."))
        else:   # wf_per_axes is not defined,
                # but grid_of_wf_idcs is,
                # and it is well-formed

            grid_of_wf_idcs_ = grid_of_wf_idcs

        if not fFigureIsGiven:
            
            figure_ = psu.make_subplots(    rows = nrows, 
                                            cols = ncols)
        else:
            figure_ = figure

        WaveformSet.update_shared_axes_status(  figure_,                    # An alternative way is to specify 
                                                share_x = share_x_scale,    # shared_xaxes=True (or share_yaxes=True)
                                                share_y = share_y_scale)    # in psu.make_subplots(), but, for us, 
                                                                            # that alternative is only doable for 
                                                                            # the case where the given 'figure'
                                                                            # parameter is None.
        if mode == 'overlay':
            for i in range(nrows):
                for j in range(ncols):
                    if len(grid_of_wf_idcs_[i][j]) > 0:
                        for k in grid_of_wf_idcs_[i][j]:

                            aux_name = f"({i+1},{j+1}) - Wf {k}, Ch {self.__waveforms[k].Channel}, Ep {self.__waveforms[k].Endpoint}"

                            self.__waveforms[k].plot(   figure = figure_,
                                                        name = aux_name,
                                                        row = i + 1,  # Plotly uses 1-based indexing
                                                        col = j + 1,
                                                        plot_analysis_markers = plot_analysis_markers,
                                                        show_baseline_limits = show_baseline_limits,
                                                        show_baseline = show_baseline,
                                                        show_general_integration_limits = show_general_integration_limits,
                                                        show_spotted_peaks = show_spotted_peaks,
                                                        show_peaks_integration_limits = show_peaks_integration_limits,
                                                        analysis_label = analysis_label)
                    else:
                        
                        WaveformSet.__add_no_data_annotation(   figure_,
                                                                i + 1,
                                                                j + 1)
        elif mode == 'average':
            for i in range(nrows):
                for j in range(ncols):

                    try: 
                        aux = self.compute_mean_waveform(wf_idcs = grid_of_wf_idcs_[i][j])  # WaveformSet.compute_mean_waveform() will raise
                                                                                            # an exception if grid_of_wf_idcs_[i][j] is emtpy

                    except Exception:       ## At some point we should implement a number of exceptions which are self-explanatory,
                                            ## so that we can handle in parallel exceptions due to different reasons if we need it
                        
                        WaveformSet.__add_no_data_annotation(   figure_,
                                                                i + 1,
                                                                j + 1)
                        continue

                    fAnalyzed = False
                    if analysis_label is not None:
                        
                        _ = aux.analyse(    analysis_label,
                                            *args,
                                            **kwargs)
                        fAnalyzed = True

                    aux_name = f"{len(grid_of_wf_idcs_[i][j])} Wf(s)"
                    if detailed_label:
                        aux_name += f": [{WaveformSet.get_string_of_first_n_integers_if_available(grid_of_wf_idcs_[i][j], queried_no = 2)}]"

                    aux.plot(   figure = figure_,
                                name = f"({i+1},{j+1}) - Mean of " + aux_name,
                                row = i + 1,
                                col = j + 1,
                                plot_analysis_markers = plot_analysis_markers if fAnalyzed else False,
                                show_baseline_limits = show_baseline_limits,
                                show_baseline = show_baseline,
                                show_general_integration_limits = show_general_integration_limits,
                                show_spotted_peaks = show_spotted_peaks,
                                show_peaks_integration_limits = show_peaks_integration_limits,
                                analysis_label = analysis_label if (plot_analysis_markers and fAnalyzed) else None)
        elif mode == 'heatmap':

            if analysis_label is None:  # In the 'heatmap' mode, the 'analysis_label' parameter must be defined
                raise Exception(generate_exception_message( 7,
                                                            'WaveformSet.plot_wfs()',
                                                            "The 'analysis_label' parameter must be defined if the 'mode' parameter is set to 'heatmap'."))
            
            aux_ranges =    np.array([  [0,                                 self.PointsPerWf - 1            ],      # Using here that the aim of the time 
                                                                                                                    # offsets of the considered waveforms 
                                                                                                                    # is to slightly align waveforms among 
                                                                                                                    # each other. I.e. the offsets are forced 
                                                                                                                    # by WaveformAdcs.__init__ to belong 
                                                                                                                    # to the [0, N-2] range, where N is 
                                                                                                                    # the number of points of the waveform.
                                                                                                                    # Hence, note that for each considered
                                                                                                                    # waveform wf, a number of points equal
                                                                                                                    # to wf.TimeOffset is lost, in the sense
                                                                                                                    # that they escape the heatmap x-range.
                                        [-1*abs(adc_range_below_baseline),  abs(adc_range_above_baseline)   ]])
            for i in range(nrows):
                for j in range(ncols):
                    if len(grid_of_wf_idcs_[i][j]) > 0:

                        aux_name = f"Heatmap of {len(grid_of_wf_idcs_[i][j])} Wf(s)"
                        if detailed_label:
                            aux_name += f": [{WaveformSet.get_string_of_first_n_integers_if_available(grid_of_wf_idcs_[i][j], queried_no = 2)}]"

                        figure_ = self.__subplot_heatmap(   figure_,
                                                            aux_name,
                                                            i + 1,
                                                            j + 1,
                                                            grid_of_wf_idcs_[i][j],
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

                        WaveformSet.__add_no_data_annotation(   figure_,
                                                                i + 1,
                                                                j + 1)
        else:                                                                                                           
            raise Exception(generate_exception_message( 8,
                                                        'WaveformSet.plot_wfs()',
                                                        f"The given mode ({mode}) must match either 'overlay', 'average', or 'heatmap'."))
        return figure_

    @staticmethod
    def get_string_of_first_n_integers_if_available(input_list : List[int],
                                                    queried_no : int = 3) -> str:

        """
        This method returns an string with the first
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
                                                        'WaveformSet.get_string_of_first_n_integers_if_available()',
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
    
    @staticmethod
    def update_shared_axes_status(  figure : pgo.Figure,
                                    share_x : bool = False,
                                    share_y : bool = True) -> pgo.Figure:
        
        """
        If share_x (resp. share_y) is True, then this
        method makes the x-axis (resp. y-axis) scale 
        of every subplot in the given figure shared.
        If share_x (resp. share_y) is False, then this
        method will reset the shared-status of the 
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
                                                        'WaveformSet.update_shared_axes_status()',
                                                        'The given figure is not a subplot grid.'))
        
        fig_rows, fig_cols = list(fig_rows)[-1], list(fig_cols)[-1]

        aux_x = None if not share_x else 'x'
        aux_y = None if not share_y else 'y'
        
        for i in range(fig_rows):
            for j in range(fig_cols):
                figure.update_xaxes(matches=aux_x, row=i+1, col=j+1)
                figure.update_yaxes(matches=aux_y, row=i+1, col=j+1)

        return figure

    @staticmethod
    def grid_of_lists_is_well_formed(   grid : List[List[List]],
                                        nrows : int,
                                        ncols : int) -> bool:
        
        """
        This method returns True if the given grid contains
        nrows lists, each of which contains ncols lists. It 
        returns False if else.

        Parameters
        ----------
        grid : list of lists of lists
        nrows : int
        ncols : int

        Returns
        ----------
        bool
        """

        if nrows < 1 or ncols < 1:
            raise Exception(generate_exception_message( 1,
                                                        'WaveformSet.grid_of_lists_is_well_formed()',
                                                        'The number of rows and columns must be positive.'))
        if len(grid) != nrows:
            return False
        else:
            for row in grid:
                if len(row) != ncols:
                    return False
        return True

    def get_grid_of_wf_idcs(self,   nrows : int,
                                    ncols : int,
                                    wfs_per_axes : Optional[int] = None,
                                    wf_filter : Optional[Callable[..., bool]] = None,
                                    filter_args : Optional[List[List[List]]] = None,
                                    max_wfs_per_axes : Optional[int] = 5) -> List[List[List[int]]]:
        
        """
        This method returns a list of lists of lists of integers,
        which should be interpreted as iterator values for
        waveforms in this WaveformSet object.

        Parameters
        ----------
        nrows : int
            The length of the returned list.
        ncols : 
            The length of every list within the returned 
            list.
        wfs_per_axes : int
            If it is not None, then it must be a positive
            integer which is smaller or equal to
            math.floor(len(self.Waveforms) / (nrows * ncols)),
            so that the iterator values contained 
            in the output grid are contiguous in
            [0, nrows*ncols*wfs_per_axes - 1]. I.e.
            output[0][0] contains 0, 1, ... , wfs_per_axes - 1,
            output[0][1] contains wfs_per_axes, wfs_per_axes + 1,
            ... , 2*wfs_per_axes - 1, and so on. 
        wf_filter : callable
            This parameter only makes a difference if
            the 'wfs_per_axes' parameter is None. In such
            case, this one must be a callable object whose 
            first parameter must be called 'waveform' and 
            must be hinted as a Waveform object. Also, the
            return type of such callable must be annotated
            as a boolean. If wf_filter is 
                - WaveformSet.match_run or
                - WaveformSet.match_endpoint_and_channel,
            this method can benefit from the information in
            self.Runs and self.AvailableChannels and its
            execution time may be reduced with respect to
            the case where an arbitrary (but compliant) 
            callable is passed to wf_filter.
        filter_args : list of list of list
            This parameter only makes a difference if 
            the 'wfs_per_axes' parameter is None. In such
            case, this parameter must be defined and
            it must contain nrows lists, each of which
            must contain ncols lists. filter_args[i][j],
            for all i and j, is interpreted as a list of
            arguments which will be given to wf_filter
            at some point. The user is responsible for
            giving a set of arguments which comply with
            the signature of the specified wf_filter.
            For more information check the return value 
            documentation.
        max_wfs_per_axes : int
            This parameter only makes a difference if           ## If max_wfs_per_axes applies and 
            the 'wfs_per_axes' parameter is None. In such       ## is a positive integer, it is never
            case, and if 'max_wfs_per_axes' is not None,        ## checked that there are enough waveforms
            then output[i][j] will contain the indices for      ## in the WaveformSet to fill the grid.
            the first max_wfs_per_axes waveforms in this        ## This is an open issue.
            WaveformSet which passed the filter. If it is 
            None, then this function iterates through the 
            whole WaveformSet for every i,j pair. Note that 
            setting this parameter to None may result in a 
            long execution time.

        Returns
        ----------
        output : list of list of list of int
            If the 'wfs_per_axes' parameter is defined, then
            the iterator values contained in the output grid 
            are contiguous in [0, nrows*ncols*wfs_per_axes - 1].
            For more information, check the 'wfs_per_axes'
            parameter documentation. If the 'wfs_per_axes'
            is not defined, then the 'wf_filter' and 'filter_args'
            parameters must be defined and output[i][j] gives 
            the indices of the waveforms in this WaveformSet 
            object, say wf, for which 
            wf_filter(wf, *filter_args[i][j]) returns True.
            In this last case, the number of indices in each
            grid slot may be limited, up to the value given
            to the 'max_wfs_per_axes' parameter.
        """

        if nrows < 1 or ncols < 1:
            raise Exception(generate_exception_message( 1,
                                                        'WaveformSet.get_grid_of_wf_idcs()',
                                                        'The number of rows and columns must be positive.'))
        fFilteringMode = True
        if wfs_per_axes is not None:
            if wfs_per_axes < 1 or wfs_per_axes > math.floor(len(self.__waveforms) / (nrows * ncols)):
                raise Exception(generate_exception_message( 2,
                                                            'WaveformSet.get_grid_of_wf_idcs()',
                                                            f"The given wfs_per_axes ({wfs_per_axes}) must belong to the range [1, math.floor(len(self.__waveforms) / (nrows * ncols))] (={[1, math.floor(len(self.__waveforms) / (nrows * ncols))]})."))
            fFilteringMode = False

        fMaxIsSet = None    # This one should only be defined as
                            # a boolean if fFilteringMode is True
        if fFilteringMode:

            try:
                signature = inspect.signature(wf_filter)
            except TypeError:
                raise Exception(generate_exception_message( 3,
                                                            'WaveformSet.get_grid_of_wf_idcs()',
                                                            "The given wf_filter is not defined or is not callable. It must be suitably defined because the 'wfs_per_axes' parameter is not. At least one of them must be suitably defined."))

            WaveformSet.check_well_formedness_of_generic_waveform_function(signature)

            if filter_args is None:
                raise Exception(generate_exception_message( 4,
                                                            'WaveformSet.get_grid_of_wf_idcs()',
                                                            "The 'filter_args' parameter must be defined if the 'wfs_per_axes' parameter is not."))
            
            elif not WaveformSet.grid_of_lists_is_well_formed(  filter_args,
                                                                nrows,
                                                                ncols):
                    
                    raise Exception(generate_exception_message( 5,
                                                                'WaveformSet.get_grid_of_wf_idcs()',
                                                                f"The shape of the given filter_args list is not nrows ({nrows}) x ncols ({ncols})."))
            fMaxIsSet = False
            if max_wfs_per_axes is not None:
                if max_wfs_per_axes < 1:
                    raise Exception(generate_exception_message( 6,
                                                                'WaveformSet.get_grid_of_wf_idcs()',
                                                                f"The given max_wfs_per_axes ({max_wfs_per_axes}) must be positive."))
                fMaxIsSet = True

        if not fFilteringMode:

            return WaveformSet.get_2D_indices_nested_list(  wfs_per_axes,
                                                            nrows = nrows,
                                                            ncols = ncols)
            
        else:   # fFilteringMode is True and so, wf_filter, 
                # filter_args and fMaxIsSet are defined

            mode_map = {WaveformSet.match_run : 0,
                        WaveformSet.match_endpoint_and_channel : 1}
            try:
                fMode = mode_map[wf_filter]
            except KeyError:
                fMode = 2

            output = WaveformSet.get_2D_empty_nested_list(nrows, ncols)

            if fMode == 0:
                return self.__get_grid_of_wf_idcs_by_run(   output,
                                                            filter_args,
                                                            fMaxIsSet,
                                                            max_wfs_per_axes)
            elif fMode == 1:
                return self.__get_grid_of_wf_idcs_by_endpoint_and_channel(  output,
                                                                            filter_args,
                                                                            fMaxIsSet,
                                                                            max_wfs_per_axes)
            else:
                return self.__get_grid_of_wf_idcs_general(  output,
                                                            wf_filter,
                                                            filter_args,
                                                            fMaxIsSet,
                                                            max_wfs_per_axes)

    @staticmethod
    def match_run(  waveform : Waveform,
                    run : int) -> bool:
        
        """
        This method returns True if the RunNumber attribute
        of the given Waveform object matches run. It returns 
        False if else.

        Parameters
        ----------
        waveform : Waveform
        run : int

        Returns
        ----------
        bool
        """

        return waveform.RunNumber == run
    
    @staticmethod
    def match_endpoint( waveform : Waveform,
                        endpoint : int) -> bool:
        
        """
        This method returns True if the Endpoint attribute
        of the given Waveform object matches endpoint, and 
        False if else.

        Parameters
        ----------
        waveform : Waveform
        endpoint : int

        Returns
        ----------
        bool
        """

        return waveform.Endpoint == endpoint
    
    @staticmethod
    def match_channel(  waveform : Waveform,
                        channel : int) -> bool:
        
        """
        This method returns True if the Channel attribute
        of the given Waveform object matches channel, and 
        False if else.

        Parameters
        ----------
        waveform : Waveform
        channel : int

        Returns
        ----------
        bool
        """

        return waveform.Channel == channel
    
    @staticmethod
    def match_endpoint_and_channel( waveform : Waveform,
                                    endpoint : int,
                                    channel : int) -> bool:
        
        """
        This method returns True if the Endpoint and Channel
        attributes of the given Waveform object match endpoint 
        and channel, respectively.

        Parameters
        ----------
        waveform : Waveform
        endpoint : int
        channel : int

        Returns
        ----------
        bool
        """

        return waveform.Endpoint == endpoint and waveform.Channel == channel
    
    def __get_grid_of_wf_idcs_by_run(self,  blank_grid : List[List[List]],
                                            filter_args : List[List[List]],
                                            fMaxIsSet : bool,
                                            max_wfs_per_axes : Optional[int] = 5) -> List[List[List[int]]]:
        
        """
        This method should only be called by the
        WaveformSet.get_grid_of_wf_idcs() method, where
        the well-formedness checks of the input have
        already been performed. This method generates an
        output as described in such method docstring,
        for the case when wf_filter is WaveformSet.match_run.
        Refer to the WaveformSet.get_grid_of_wf_idcs()
        method documentation for more information.

        Parameters
        ----------
        blank_grid : list of list of list
        filter_args : list of list of list
        fMaxIsSet : bool
        max_wfs_per_axes : int

        Returns
        ----------
        list of list of list of int
        """

        for i in range(len(blank_grid)):
            for j in range(len(blank_grid[i])):

                if filter_args[i][j][0] not in self.__runs:
                    continue

                if fMaxIsSet:   # blank_grid should not be very big (visualization purposes)
                                # so we can afford evaluating the fMaxIsSet conditional here
                                # instead of at the beginning of the method (which would
                                # be more efficient but would entail a more extensive code)

                    counter = 0
                    for k in range(len(self.__waveforms)):
                        if WaveformSet.match_run(   self.__waveforms[k],
                                                    *filter_args[i][j]):
                            blank_grid[i][j].append(k)
                            counter += 1
                            if counter == max_wfs_per_axes:
                                break
                else:
                    for k in range(len(self.__waveforms)):
                        if WaveformSet.match_run(   self.__waveforms[k],
                                                    *filter_args[i][j]):
                            blank_grid[i][j].append(k)        
        return blank_grid
    
    def __get_grid_of_wf_idcs_by_endpoint_and_channel(self, blank_grid : List[List[List]],
                                                            filter_args : List[List[List]],
                                                            fMaxIsSet : bool,
                                                            max_wfs_per_axes : Optional[int] = 5) -> List[List[List[int]]]:
        
        """
        This method should only be called by the 
        WaveformSet.get_grid_of_wf_idcs() method, where 
        the well-formedness checks of the input have 
        already been performed. This method generates an 
        output as described in such method docstring,
        for the case when wf_filter is 
        WaveformSet.match_endpoint_and_channel. Refer to
        the WaveformSet.get_grid_of_wf_idcs() method
        documentation for more information.

        Parameters
        ----------
        blank_grid : list of list of list
        filter_args : list of list of list
        fMaxIsSet : bool
        max_wfs_per_axes : int

        Returns
        ----------
        list of list of list of int
        """

        aux = self.get_run_collapsed_available_channels()

        for i in range(len(blank_grid)):
            for j in range(len(blank_grid[i])):

                if filter_args[i][j][0] not in aux.keys():      # filter_args[i][j][0] is the
                    continue                                    # endpoint we are looking for

                elif filter_args[i][j][1] not in aux[filter_args[i][j][0]]:                         # filter_args[i][j][1] is
                    continue                                                                        # the channel of endpoint 
                                                                                                    # filter_args[i][j][0]
                                                                                                    # which we are looking for
                if fMaxIsSet:   # blank_grid should not be very big (visualization purposes)
                                # so we can afford evaluating the fMaxIsSet conditional here
                                # instead of at the beginning of the method (which would
                                # be more efficient but would entail a more extensive code)

                    counter = 0
                    for k in range(len(self.__waveforms)):
                        if WaveformSet.match_endpoint_and_channel(  self.__waveforms[k],
                                                                    *filter_args[i][j]):
                            blank_grid[i][j].append(k)
                            counter += 1
                            if counter == max_wfs_per_axes:
                                break
                else:
                    for k in range(len(self.__waveforms)):
                        if WaveformSet.match_endpoint_and_channel(  self.__waveforms[k],
                                                                    *filter_args[i][j]):
                            blank_grid[i][j].append(k)
        return blank_grid
    
    def __get_grid_of_wf_idcs_general(self, blank_grid : List[List[List]],
                                            wf_filter : Callable[..., bool],
                                            filter_args : List[List[List]],
                                            fMaxIsSet : bool,
                                            max_wfs_per_axes : Optional[int] = 5) -> List[List[List[int]]]:
        
        """
        This method should only be called by the 
        WaveformSet.get_grid_of_wf_idcs() method, where 
        the well-formedness checks of the input have 
        already been performed. This method generates an 
        output as described in such method docstring,
        for the case when wf_filter is neither
        WaveformSet.match_run nor
        WaveformSet.match_endpoint_and_channel. Refer 
        to the WaveformSet.get_grid_of_wf_idcs() method
        documentation for more information.

        Parameters
        ----------
        blank_grid : list of list of list
        wf_filter : callable
        filter_args : list of list of list
        fMaxIsSet : bool
        max_wfs_per_axes : int

        Returns
        ----------
        list of list of list of int
        """

        for i in range(len(blank_grid)):
            for j in range(len(blank_grid[i])):

                if fMaxIsSet:
                    counter = 0
                    for k in range(len(self.__waveforms)):
                        if wf_filter(   self.__waveforms[k],
                                        *filter_args[i][j]):
                            
                            blank_grid[i][j].append(k)
                            counter += 1
                            if counter == max_wfs_per_axes:
                                break
                else:
                    for k in range(len(self.__waveforms)):
                        if wf_filter(   self.__waveforms[k],
                                        *filter_args[i][j]):
                            blank_grid[i][j].append(k)
        return blank_grid
                            
    @staticmethod
    def get_2D_empty_nested_list(   nrows : int = 1,
                                    ncols : int = 1) -> List[List[List]]:
        
        """
        This method returns a 2D nested list of empty lists
        with nrows rows and ncols columns.
        
        Parameters
        ----------
        nrows (resp. ncols) : int
            Number of rows (resp. columns) of the returned 
            nested list.

        Returns
        ----------
        list of list of list
            A list containing nrows lists, each of them
            containing ncols empty lists.
        """

        if nrows < 1 or ncols < 1:
            raise Exception(generate_exception_message( 1,
                                                        'WaveformSet.get_2D_empty_nested_list()',
                                                        'The number of rows and columns must be positive.'))

        return [[[] for _ in range(ncols)] for _ in range(nrows)]
    
    @staticmethod
    def get_2D_indices_nested_list( indices_per_slot : int,
                                    nrows : int = 1,
                                    ncols : int = 1) -> List[List[List]]:
        
        """
        This method returns a 2D nested list with nrows
        rows and ncols columns. Such nested list, say
        output, contains contiguous positive integers in
        [0, nrows*ncols*indices_per_slot - 1]. I.e.
        output[0][0] contains 0, 1, ... , indices_per_slot - 1,
        output[0][1] contains indices_per_slot, 
        indices_per_slot + 1, ...  , 2*indices_per_slot - 1, 
        and so on. 
        
        Parameters
        ----------
        indices_per_slot : int
            The number of indices contained within each 
            slot in the returned output grid
        nrows (resp. ncols) : int
            Number of rows (resp. columns) of the returned 
            nested list

        Returns
        ----------
        list of list of list
            A list containing nrows lists, each of them
            containing ncols lists, each of them containing
            indices_per_slot integers.
        """

        if nrows < 1 or ncols < 1:
            raise Exception(generate_exception_message( 1,
                                                        'WaveformSet.get_2D_indices_nested_list()',
                                                        f"The given number of rows ({nrows}) and columns ({ncols}) must be positive."))
        if indices_per_slot < 1:
            raise Exception(generate_exception_message( 2,
                                                        'WaveformSet.get_2D_indices_nested_list()',
                                                        f"The given number of indices per slot ({indices_per_slot}) must be positive."))
        
        return [[[k + indices_per_slot*(j + (ncols*i)) for k in range(indices_per_slot)] for j in range(ncols)] for i in range(nrows)]

    @classmethod
    def from_ROOT_file(cls, filepath : str,
                            bulk_data_tree_name : str = 'raw_waveforms',
                            meta_data_tree_name : str = 'metadata',
                            set_offset_wrt_daq_window : bool = False,
                            read_full_streaming_data : bool = False,
                            start_fraction : float = 0.0,
                            stop_fraction : float = 1.0,
                            verbose : bool = True) -> 'WaveformSet':

        """
        Alternative initializer for a WaveformSet object out of the
        waveforms stored in a ROOT file

        Parameters
        ----------
        filepath : str
            Path to the ROOT file to be read. Such ROOT file should 
            have at least two defined TTree objects, so that the 
            name of one of those starts with the string given to the
            'bulk_data_tree_name' parameter - the bulk data tree - 
            and the other one starts with the string given to the 
            'meta_data_tree_name' parameter - the meta data tree. 
            The bulk data TTree should have at least four branches,
            whose names should start with

                - 'adcs'
                - 'channel'
                - 'timestamp'
                - 'record'
                - 'is_fullstream'

            from which the values for the Waveform objects attributes
            Adcs, Channel, Timestamp and RecordNumber will be taken 
            respectively. The 'is_fullstream' branch is used to 
            decide whether a certain waveform should be grabbed 
            or not, depending on the value given to the             ## For the moment, the meta-data tree is not
            'read_full_streaming_data' parameter                    ## read. This needs to change in the near future.
        bulk_data_tree_name (resp. meta_data_tree_name) : str
            Name of the bulk-data (meta-data) tree which will be 
            extracted from the given ROOT file. The first object 
            found within the given ROOT file whose name starts
            with the given string and which is a TTree object, 
            will be identified as the bulk-data (resp. meta-data) 
            tree.
        set_offset_wrt_daq_window : bool
            If True, then the bulk data tree must also have a
            branch whose name starts with 'daq_timestamp'. In
            this case, then the TimeOffset attribute of each
            waveform is set as the difference between its
            value for the 'timestamp' branch and the value
            for the 'daq_timestamp' branch, in such order,
            referenced to the minimum value of such difference
            among all the waveforms. This is useful to align
            waveforms whose time overlap is not null, for 
            plotting and analysis purposes. It is required
            that the time overlap of every waveform is not 
            null, otherwise an exception will be eventually
            raised by the WaveformSet initializer. If False, 
            then the 'daq_timestamp' branch is not queried 
            and the TimeOffset attribute of each waveform 
            is set to 0.
        read_full_streaming_data : bool
            If True (resp. False), then only the waveforms for which 
            the 'is_fullstream' branch in the bulk-data tree has a 
            value equal to True (resp. False) will be considered.
        start_fraction (resp. stop_fraction) : float
            Gives the iterator value for the first (resp. last) 
            waveform which will be a candidate to be loaded into 
            this WaveformSet object. Whether they will be finally 
            read also depends on their value for the 'is_fullstream' 
            branch and the value given to the 'read_full_streaming_data'
            parameter. P.e. setting start_fraction to 0.5, 
            stop_fraction to 0.75 and read_full_streaming_data to 
            True, will result in loading every waveform which belongs
            to the third quarter of the input file and for which 
            the 'is_fullstream' branch equals to True. 
        verbose : bool
            If True, then functioning-related messages will be
            printed.
        """

        if not WaveformSet.fraction_is_well_formed(start_fraction, stop_fraction):
            raise Exception(generate_exception_message( 1,
                                                        'WaveformSet.from_ROOT_file()',
                                                        f"Fraction limits are not well-formed."))
        input_file = uproot.open(filepath)
        
        # meta_data_tree = WaveformSet.find_TTree_in_ROOT_file(   input_file,           ## For the moment, the meta-data tree is not
        #                                                         meta_data_tree_name)  ## read. This needs to change in the near future.
        
        bulk_data_tree = WaveformSet.find_TTree_in_ROOT_file(   input_file,
                                                                bulk_data_tree_name)
        
        is_fullstream_branch = WaveformSet.find_TBranch_in_TTree_file(  bulk_data_tree,
                                                                        'is_fullstream')
        
        wf_start = math.floor(start_fraction*is_fullstream_branch.num_entries)  # Get the start and stop iterator values for
        wf_stop = math.ceil(stop_fraction*is_fullstream_branch.num_entries)     # the chunk which contains the waveforms which
                                                                                # could be potentially read.

        is_fullstream_array = is_fullstream_branch.array(   entry_start = wf_start,
                                                            entry_stop = wf_stop)
    
        aux = np.where(is_fullstream_array)[0] if read_full_streaming_data else np.where(np.logical_not(is_fullstream_array))[0]

        if len(aux) == 0:
            raise Exception(generate_exception_message( 2,
                                                        'WaveformSet.from_ROOT_file()',
                                                        f"No waveforms of the specified type ({'full-stream' if read_full_streaming_data else 'self-trigger'}) were found."))

        idcs_to_retrieve = WaveformSet.__cluster_integers_by_contiguity(aux)

        if verbose:
            print(f"In function WaveformSet.from_ROOT_file(): Found {len(idcs_to_retrieve)} cluster(s) of contiguous {'full-streaming' if read_full_streaming_data else 'self-trigger'} waveforms in the ROOT file.")
            print(f"In function WaveformSet.from_ROOT_file(): Note that, the lesser the clusters the faster the reading process will be.")

        # For reference, reading ~1.6e+3 waveforms in 357 clusters takes ~10s,
        # while reading ~176e+3 waveforms in 1 cluster takes the same ~10s.

        ## If the file to read is highly framented (i.e. there is a lot of clusters, 
        ## then it is highly counterproductive to use this logical structure
        ## (where we read block-by-block) compared to just reading the whole arrays
        ## and then discard what we do not need based on the read 'is_fullstream' array.
        ## That's why we should introduce a criterion based on the number of clusters
        ## i.e. len(idcs_to_retrieve) to decide whether to use this block-reading
        ## structure or not. While lacking a proper criterion, a threshold for the
        ## number of clusters above which just reading the whole arrays, could
        ## be gotten as an input parameter of this method. The block-reading
        ## strategy is worth it, though, when the input file is not very fragmented.
        ## This is an open issue. 
        
        # Note that the indices in idcs_to_retrieve are referred to the block which 
        # we have read. I.e. idcs_to_retrieve[0] being, p.e. [0,3], means that with 
        # respect to the branches in the ROOT file, the first cluster we need to read 
        # goes from index wf_start+0 to index wf_start+3-1 inclusive, or wf_start+3.
        # exclusive. Also note that the 'entry_stop' parameter of uproot.TBranch.array()
        # is exclusive.

        adcs_branch = WaveformSet.find_TBranch_in_TTree_file(   bulk_data_tree,     # It is slightly faster (~106s vs. 114s, for a 809 MB
                                                                'adcs')             # input file running on lxplus9) to read branch by
                                                                                    # branch rather than going for bulk_data_tree.arrays()
        channel_branch = WaveformSet.find_TBranch_in_TTree_file(bulk_data_tree,
                                                                'channel')
        
        timestamp_branch = WaveformSet.find_TBranch_in_TTree_file(  bulk_data_tree,
                                                                    'timestamp')
        
        record_branch = WaveformSet.find_TBranch_in_TTree_file(    bulk_data_tree,
                                                                    'record')

        waveforms = []                      # Using a list comprehension here is slightly slower than a for loop
                                            # (97s vs 102s for 5% of wvfs of a 809 MB file running on lxplus9)

        if not set_offset_wrt_daq_window:   # Code is more extensive this way, but faster than evaluating
                                            # the conditional at each iteration within the loop.

            for interval in idcs_to_retrieve:   # Read the waveforms in contiguous blocks

                branch_start = wf_start + interval[0]
                branch_stop = wf_start + interval[1]

                current_adcs_array = adcs_branch.array( entry_start = branch_start,
                                                        entry_stop = branch_stop)
                
                current_channel_array = channel_branch.array(   entry_start = branch_start,
                                                                entry_stop = branch_stop)
                
                current_timestamp_array = timestamp_branch.array(   entry_start = branch_start,
                                                                    entry_stop = branch_stop)
                
                current_record_array = record_branch.array( entry_start = branch_start,
                                                            entry_stop = branch_stop)
                for i in range(len(current_adcs_array)):

                    endpoint, channel = WaveformSet.get_endpoint_and_channel(current_channel_array[i])

                    waveforms.append(Waveform(  current_timestamp_array[i],
                                                16.,    # TimeStep_ns   ## Hardcoded to 16 ns for now, but
                                                                        ## it must be implemented from the new
                                                                        ## 'metadata' TTree in the ROOT file
                                                np.array(current_adcs_array[i]),
                                            0,      #RunNumber      ## To be implemented from the new
                                                                    ## 'metadata' TTree in the ROOT file
                                                                        ## 'metadata' TTree in the ROOT file
                                                current_record_array[i],
                                                endpoint,
                                                channel,
                                                time_offset = 0))
        else:

            raw_time_offsets = []

            daq_timestamp_branch = WaveformSet.find_TBranch_in_TTree_file(  bulk_data_tree,
                                                                            'daq_timestamp')
            
            for interval in idcs_to_retrieve:   # Read the waveforms in contiguous blocks

                branch_start = wf_start + interval[0]
                branch_stop = wf_start + interval[1]

                current_adcs_array = adcs_branch.array( entry_start = branch_start,
                                                        entry_stop = branch_stop)
                
                current_channel_array = channel_branch.array(   entry_start = branch_start,
                                                                entry_stop = branch_stop)
                
                current_timestamp_array = timestamp_branch.array(   entry_start = branch_start,
                                                                    entry_stop = branch_stop)
                
                current_record_array = record_branch.array( entry_start = branch_start,
                                                            entry_stop = branch_stop)
                
                current_daq_timestamp_array = daq_timestamp_branch.array(   entry_start = branch_start,
                                                                            entry_stop = branch_stop)

                for i in range(len(current_adcs_array)):

                    endpoint, channel = WaveformSet.get_endpoint_and_channel(current_channel_array[i])

                    waveforms.append(Waveform(  current_timestamp_array[i],
                                                16.,    # TimeStep_ns   ## Hardcoded to 16 ns for now, but
                                                                        ## it must be implemented from the new
                                                                        ## 'metadata' TTree in the ROOT file
                                                np.array(current_adcs_array[i]),
                                                0,      #RunNumber      ## To be implemented from the new
                                                                        ## 'metadata' TTree in the ROOT file
                                                current_record_array[i],
                                                endpoint,
                                                channel,
                                                time_offset = 0))
                    
                    raw_time_offsets.append(int(current_timestamp_array[i]) - int(current_daq_timestamp_array[i]))

            time_offsets = WaveformSet.reference_to_minimum(raw_time_offsets)

            for i in range(len(waveforms)):
                waveforms[i]._set_time_offset(time_offsets[i])

        if read_full_streaming_data:

            minimum_length = np.array([ len(wf.Adcs) for wf in waveforms]).min()

            for wf in waveforms:
                wf._WaveformAdcs__truncate_adcs(minimum_length)      
        
        return cls(*waveforms)
    
    @staticmethod
    def __cluster_integers_by_contiguity(increasingly_sorted_integers : np.ndarray) -> List[List[int]]:

        """
        This function gets an unidimensional numpy array of 
        integers, increasingly_sorted_integers, which 
        
            -   must contain at least two elements and
            -   must be strictly increasingly ordered, i.e.
                increasingly_sorted_integers[i] < increasingly_sorted_integers[i+1]
                for all i.

        The first requirement will be checked by this method,
        but it is the caller's responsibility to make sure that
        the second one is met. P.e. the output of 
        np.where(boolean_1d_array)[0], where boolean_1d_array 
        is an unidimensional boolean array, always meets the 
        second requirement.
    
        This function clusters the integers in such array by 
        contiguity. P.e. if increasingly_sorted_integers is
        array([1,2,3,5,6,8,10,11,12,13,16]), then this function 
        will return the following list: 
        '[[1,4],[5,7],[8,9],[10,14],[16,17]]'.
        
        Parameters
        ----------
        increasingly_sorted_integers : np.ndarray
            An increasingly sorted numpy array of integers
            whose length is at least 2.

        Returns
        ----------
        extremals : list of list of int
            output[i] is a list containing two integers,
            so that output[i][0] (resp. output[i][1]) is
            the inclusive (resp. exclusive) lower (resp. 
            upper) bound of the i-th cluster of contiguous
            integers in the input array.
        """

        if increasingly_sorted_integers.ndim != 1:
            raise Exception(generate_exception_message( 1,
                                                        'WaveformSet.__cluster_integers_by_contiguity()',
                                                        'The given numpy array must be unidimensional.'))
        if len(increasingly_sorted_integers) < 2:
            raise Exception(generate_exception_message( 2,
                                                        'WaveformSet.__cluster_integers_by_contiguity()',
                                                        'The given numpy array must contain at least two elements.'))
        extremals = []
        extremals.append([increasingly_sorted_integers[0]])
        
        for i in range(1, len(increasingly_sorted_integers)-1):  # The last integer has an exclusive treatment

            if increasingly_sorted_integers[i] - increasingly_sorted_integers[i-1] != 1:    # We have stepped into a new cluster

                extremals[-1].append(increasingly_sorted_integers[i-1]+1)   # Add one to get the exclusive upper bound
                extremals.append([increasingly_sorted_integers[i]])

        if increasingly_sorted_integers[-1] - increasingly_sorted_integers[-2] != 1:  # Taking care of the last element of the given list

            extremals[-1].append(increasingly_sorted_integers[-2]+1)                                    # Add one to get the 
            extremals.append([increasingly_sorted_integers[-1], increasingly_sorted_integers[-1]+1])    # exclusive upper bound

        else:

            extremals[-1].append(increasingly_sorted_integers[-1]+1)

        return extremals
    
    @staticmethod
    def find_TTree_in_ROOT_file(file : uproot.ReadOnlyDirectory,
                                TTree_pre_name : str) -> uproot.TTree:
        
        """
        This method returns the first object found in the given
        ROOT file whose name starts with the string given to the
        'TTree_pre_name' parameter and which is a TTree object. If 
        no such TTree is found, an exception is raised.

        Parameters
        ----------
        file : uproot.ReadOnlyDirectory
            The ROOT file where to look for the TTree object
        TTree_pre_name : str
            The string which the name of the TTree object must
            start with

        Returns
        ----------
        uproot.TTree
        """

        TTree_name = None
        for key in file.classnames().keys():
            if key.startswith(TTree_pre_name) and file.classnames()[key] == 'TTree':
                TTree_name = key
                break

        if TTree_name is None:
            raise Exception(generate_exception_message( 1,
                                                        'WaveformSet.find_TTree_in_ROOT_file()',
                                                        f"There is no TTree with a name starting with '{TTree_pre_name}'."))
        return file[TTree_name]
    
    @staticmethod
    def find_TBranch_in_TTree_file( TTree : uproot.TTree,
                                    TBranch_pre_name : str) -> uproot.TBranch:
        
        """
        This method returns the first TBranch found in the 
        given TTree whose name starts with the string given 
        to the 'TBranch_pre_name' parameter. If no such 
        TBranch is found, an exception is raised.

        Parameters
        ----------
        TTree : uproot.TTree
            The TTree where to look for the TBranch object
        TBranch_pre_name : str
            The string which the name of the TBranch object 
            must start with

        Returns
        ----------
        uproot.TBranch
        """

        TBranch_name = None
        for key in TTree.keys():
            if key.startswith(TBranch_pre_name):
                TBranch_name = key
                break

        if TBranch_name is None:
            raise Exception(generate_exception_message( 1,
                                                        'WaveformSet.find_TBranch_in_TTree_file()',
                                                        f"There is no TBranch with a name starting with '{TBranch_pre_name}'."))
        return TTree[TBranch_name]
        
    @staticmethod
    def get_endpoint_and_channel(input : int) -> Tuple[int, int]:
    
        """
        Parameters
        ----------
        input : str
            len(input) must be 5. Such input is interpreted as the
            concatenation of the endpoint int(input[0:3]) and the 
            channel int(input[3:5]).

        Returns
        ----------
        int
            The endpoint value
        int
            The channel value
        """

        return int(str(input)[0:3]), int(str(input)[3:5])
    
    @staticmethod
    def fraction_is_well_formed(lower_limit : float = 0.0,
                                upper_limit : float = 1.0) -> bool:
        
        """
        This method returns True if 0.0 <= lower_limit < upper_limit <= 1.0,
        and False if else.

        Parameters
        ----------
        lower_limit : float
        upper_limit : float

        Returns
        ----------
        bool
        """

        if lower_limit < 0.0:
            return False
        elif upper_limit <= lower_limit:
            return False
        elif upper_limit > 1.0:
            return False
        
        return True
    
    def compute_mean_waveform(self, *args,
                                    wf_idcs : Optional[List[int]] = None,
                                    wf_selector : Optional[Callable[..., bool]] = None,
                                    **kwargs) -> WaveformAdcs:

        """
        If wf_idcs is None and wf_selector is None,
        then this method creates a WaveformAdcs
        object whose Adcs attribute is the mean 
        of the adcs arrays for every waveform in 
        this WaveformSet. If wf_idcs is not None, 
        then such mean is computed using the adcs
        arrays of the waveforms whose iterator 
        values, with respect to this WaveformSet, 
        are given in wf_idcs. If wf_idcs is None 
        but wf_selector is not None, then such 
        mean is computed using the adcs arrays
        of the waveforms, wf, within this 
        WaveformSet for which 
        wf_selector(wf, *args, **kwargs) evaluates 
        to True. In any case, the TimeStep_ns
        attribute of the newly created WaveformAdcs
        object assumed to match that of the first
        waveform which was used in the average sum.
        
        In any case, the resulting WaveformAdcs
        object is assigned to the
        self.__mean_adcs attribute. The 
        self.__mean_adcs_idcs attribute is also
        updated with a tuple of the indices of the
        waveforms which were used to compute the
        mean WaveformAdcs. Finally, this method 
        returns the averaged WaveformAdcs object.

        Parameters
        ----------
        *args
            These arguments only make a difference if
            the 'wf_idcs' parameter is None and the
            'wf_selector' parameter is suitable defined.
            For each waveform, wf, these are the 
            positional arguments which are given to
            wf_selector(wf, *args, **kwargs) as *args.
        wf_idcs : list of int
            If it is not None, then it must be a list
            of integers which must be a valid iterator
            value for the __waveforms attribute of this
            WaveformSet. I.e. any integer i within such
            list must satisfy
            0 <= i <= len(self.__waveforms) - 1. Any
            integer which does not satisfy this condition
            is ignored. These integers give the waveforms
            which are averaged.
        wf_selector : callable 
            This parameter only makes a difference if 
            the 'wf_idcs' parameter is None. If that's 
            the case, and 'wf_selector' is not None, then 
            it must be a callable whose first parameter 
            must be called 'waveform' and its type 
            annotation must match the Waveform class. 
            Its return value must be annotated as a 
            boolean. In this case, the mean waveform 
            is averaged over those waveforms, wf, for 
            which wf_selector(wf, *args, **kwargs) 
            evaluates to True.
        *kwargs
            These keyword arguments only make a 
            difference if the 'wf_idcs' parameter is 
            None and the 'wf_selector' parameter is 
            suitable defined. For each waveform, wf, 
            these are the keyword arguments which are 
            given to wf_selector(wf, *args, **kwargs) 
            as **kwargs.

        Returns
        ----------
        output : np.ndarray
            The averaged adcs array
        """

        if len(self.__waveforms) == 0:
            raise Exception(generate_exception_message( 1,
                                                        'WaveformSet.compute_mean_waveform()',
                                                        'There are no waveforms in this WaveformSet object.'))
        if wf_idcs is None and wf_selector is None:

            output = self.__compute_mean_waveform_of_every_waveform()   # Average over every 
                                                                        # waveform in this WaveformSet
        elif wf_idcs is None and wf_selector is not None:

            signature = inspect.signature(wf_selector)

            WaveformSet.check_well_formedness_of_generic_waveform_function(signature)

            output = self.__compute_mean_waveform_with_selector(wf_selector,
                                                                *args,
                                                                **kwargs)
        else:

            fWfIdcsIsWellFormed = False
            for idx in wf_idcs:
                if self.is_valid_iterator_value(idx):

                    fWfIdcsIsWellFormed = True
                    break                       # Just make sure that there 
                                                # is at least one valid 
                                                # iterator value in the given list

            if not fWfIdcsIsWellFormed:
                raise Exception(generate_exception_message( 2,
                                                            'WaveformSet.compute_mean_waveform()',
                                                            'The given list of waveform indices is empty or it does not contain even one valid iterator value in the given list. I.e. there are no waveforms to average.'))

            output = self.__compute_mean_waveform_of_given_waveforms(wf_idcs)   ## In this case we also need to remove indices
                                                                                ## redundancy (if any) before giving wf_idcs to
                                                                                ## WaveformSet.__compute_mean_waveform_of_given_waveforms.
                                                                                ## This is a open issue for now.
        return output
    
    def __compute_mean_waveform_of_every_waveform(self) -> WaveformAdcs:
        
        """
        This method should only be called by the
        WaveformSet.compute_mean_waveform() method,
        where any necessary well-formedness checks 
        have already been performed. It is called by 
        such method in the case where both the 'wf_idcs' 
        and the 'wf_selector' input parameters are 
        None. This method sets the self.__mean_adcs
        and self.__mean_adcs_idcs attributes according
        to the WaveformSet.compute_mean_waveform()
        method documentation. It also returns the 
        averaged WaveformAdcs object. Refer to the 
        WaveformSet.compute_mean_waveform() method 
        documentation for more information.

        Returns
        ----------
        output : np.ndarray
            The averaged adcs array
        """

        aux = self.Waveforms[0].Adcs                # WaveformSet.compute_mean_waveform() 
                                                    # has already checked that there is at 
                                                    # least one waveform in this WaveformSet
        for i in range(1, len(self.__waveforms)):
            aux += self.Waveforms[i].Adcs

        output = WaveformAdcs(  self.__waveforms[0].TimeStep_ns,
                                aux/len(self.__waveforms),
                                time_offset = 0)
        
        self.__mean_adcs = output
        self.__mean_adcs_idcs = tuple(range(len(self.__waveforms)))

        return output
    
    def __compute_mean_waveform_with_selector(self, wf_selector : Callable[..., bool],
                                                    *args,
                                                    **kwargs) -> WaveformAdcs:
        
        """
        This method should only be called by the
        WaveformSet.compute_mean_waveform() method,
        where any necessary well-formedness checks 
        have already been performed. It is called by 
        such method in the case where the 'wf_idcs'
        parameter is None and the 'wf_selector' 
        parameter is suitably defined. This method 
        sets the self.__mean_adcs and 
        self.__mean_adcs_idcs attributes according
        to the WaveformSet.compute_mean_waveform()
        method documentation. It also returns the 
        averaged WaveformAdcs object. Refer to the 
        WaveformSet.compute_mean_waveform() method 
        documentation for more information.

        Parameters
        ----------
        wf_selector : callable
        *args
        **kwargs

        Returns
        ----------
        output : np.ndarray
            The averaged adcs array
        """

        added_wvfs = []

        aux = np.zeros((self.__points_per_wf,))

        for i in range(len(self.__waveforms)):
            if wf_selector(self.__waveforms[i], *args, **kwargs):
                aux += self.__waveforms[i].Adcs
                added_wvfs.append(i)
                
        if len(added_wvfs) == 0:
            raise Exception(generate_exception_message( 1,
                                                        'WaveformSet.__compute_mean_waveform_with_selector()',
                                                        'No waveform in this WaveformSet object passed the given selector.'))
    
        output = WaveformAdcs(  self.__waveforms[added_wvfs[0]].TimeStep_ns,
                                aux/len(added_wvfs),
                                time_offset = 0)
        
        self.__mean_adcs = output
        self.__mean_adcs_idcs = tuple(added_wvfs)

        return output
    
    def __compute_mean_waveform_of_given_waveforms(self, wf_idcs : List[int]) -> WaveformAdcs:
        
        """
        This method should only be called by the
        WaveformSet.compute_mean_waveform() method,
        where any necessary well-formedness checks 
        have already been performed. It is called by 
        such method in the case where the 'wf_idcs'
        parameter is not None, regardless the input
        given to the 'wf_selector' parameter. This 
        method sets the self.__mean_adcs and 
        self.__mean_adcs_idcs attributes according
        to the WaveformSet.compute_mean_waveform()
        method documentation. It also returns the 
        averaged WaveformAdcs object. Refer to the 
        WaveformSet.compute_mean_waveform() method 
        documentation for more information.

        Parameters
        ----------
        wf_idcs : list of int

        Returns
        ----------
        output : np.ndarray
            The averaged adcs array
        """

        added_wvfs = []

        aux = np.zeros((self.__points_per_wf,))

        for idx in wf_idcs:
            try:                # WaveformSet.compute_mean_waveform() only checked that there 
                                # is at least one valid iterator value, but we need to handle
                                # the case where there are invalid iterator values

                aux += self.__waveforms[idx].Adcs
            except IndexError:
                continue        # Ignore the invalid iterator values as specified in the 
                                # WaveformSet.compute_mean_waveform() method documentation
            else:
                added_wvfs.append(idx)

        output = WaveformAdcs(  self.__waveforms[added_wvfs[0]].TimeStep_ns,
                                aux/len(added_wvfs),                            # len(added_wvfs) must be at least 1. 
                                                                                # This was already checked by 
                                                                                # WaveformSet.compute_mean_waveform()
                                time_offset = 0)
        self.__mean_adcs = output
        self.__mean_adcs_idcs = tuple(added_wvfs)

        return output

    def is_valid_iterator_value(self, iterator_value : int) -> bool:

        """
        This method returns True if
        0 <= iterator_value <= len(self.__waveforms) - 1,
        and False if else.
        """

        if iterator_value < 0:
            return False
        elif iterator_value <= len(self.__waveforms) - 1:
            return True
        else:
            return False
        
    def filter(self,    wf_filter : Callable[..., bool],
                        *args,
                        actually_filter : bool = False,
                        return_the_staying_ones : bool = True,
                        **kwargs) -> List[int]:
        
        """
        This method filters the waveforms in this WaveformSet
        using the given wf_filter callable. I.e. for each
        Waveform object, wf, in this WaveformSet, it runs
        wf_filter(wf, *args, **kwargs). This method returns
        a list of indices for the waveforms which got the
        same result from the filter.

        Parameters
        ----------
        wf_filter : callable 
            It must be a callable whose first parameter 
            must be called 'waveform' and its type
            annotation must match the Waveform class. 
            Its return value must be annotated as a 
            boolean. The waveforms that are filtered
            out are those for which 
            wf_filter(waveform, *args, **kwargs)
            evaluates to False.
        *args
            For each waveform, wf, these are the 
            positional arguments which are given to
            wf_filter(wf, *args, **kwargs) as *args.
        actually_filter : bool
            If False, then no changes are done to 
            this WaveformSet object. If True, then 
            the waveforms which are filtered out 
            are deleted from the self.__waveforms 
            attribute of this WaveformSet object. 
            If so, the self.__runs, 
            self.__record_numbers and the 
            self.__available_channels attributes
            are updated accordingly, and the
            the self.__mean_adcs and the 
            self.__mean_adcs_idcs are reset to None. 
        return_the_staying_ones : bool
            If True (resp. False), then this method 
            returns the indices of the waveforms which 
            passed (resp. didn't pass) the filter, i.e.
            those for which the filter evaluated to 
            True (resp. False).
        *kwargs
            For each waveform, wf, these are the 
            keyword arguments which are given to
            wf_filter(wf, *args, **kwargs) as *kwargs

        Returns
        ----------
        output : list of int
            If return_the_staying_ones is True (resp.
            False), then this list contains the indices,
            with respect to the self.__waveforms list, 
            for the waveforms, wf, for which 
            wf_filter(wf, *args, **kwargs) evaluated to
            True (resp. False).
        """

        signature = inspect.signature(wf_filter)

        WaveformSet.check_well_formedness_of_generic_waveform_function(signature)
        
        staying_ones, dumped_ones = [], []      # Better fill the two lists during the WaveformSet scan and then return
                                                # the desired one, rather than filling just the dumped_ones one and
                                                # then computing its negative in case return_the_staying_ones is True
        for i in range(len(self.__waveforms)):      
            if wf_filter(self.__waveforms[i], *args, **kwargs):
                staying_ones.append(i)
            else:
                dumped_ones.append(i)

        if actually_filter:

            for idx in reversed(dumped_ones):       # dumped_ones is increasingly ordered, so 
                del self.Waveforms[idx]             # iterate in reverse order for waveform deletion

            self.update_runs()                      # If actually_filter, then we need to update 
            self.update_record_numbers()            # the self.__runs, self.__record_numbers and 
            self.update_available_channels()        # self.__available_channels

            self.__mean_adcs = None                 # We also need to reset the attributes regarding the mean
            self.__mean_adcs_idcs = None            # waveform, for which some of the waveforms might have been removed

        if return_the_staying_ones:
            return staying_ones
        else:
            return dumped_ones
        
    @classmethod
    def from_filtered_WaveformSet(cls,  original_WaveformSet : 'WaveformSet',
                                        wf_filter : Callable[..., bool],
                                        *args,
                                        **kwargs) -> 'WaveformSet':
        
        """
        This method returns a new WaveformSet object
        which contains only the waveforms from the
        given original_WaveformSet object which passed
        the given wf_filter callable, i.e. those Waveform
        objects, wf, for which
        wf_filter(wf, *args, **kwargs) evaluated to True.
        To do so, this method calls the WaveformSet.filter()
        instance method of the Waveformset given to the
        'original_WaveformSet' parameter by setting the
        its 'actually_filter' parameter to True.

        Parameters
        ----------
        original_WaveformSet : WaveformSet
            The WaveformSet object which will be filtered
            so as to create the new WaveformSet object
        wf_filter : callable
            It must be a callable whose first parameter
            must be called 'waveform' and its type
            annotation must match the Waveform class.
            Also, its return value must be annotated
            as a boolean. The well-formedness of
            the given callable is not checked by
            this method, but checked by the 
            WaveformSet.filter() instance method of
            the original_WaveformSet object, whose
            'wf_filter' parameter receives the input
            given to the 'wf_filter' parameter of this
            method. The waveforms which end up staying 
            in the returned WaveformSet object are those
            within the original_WaveformSet object,
            wf, for which wf_filter(wf, *args, **kwargs)
            evaluated to True.
        *args
            For each waveform, wf, these are the 
            positional arguments which are given to
            wf_filter(wf, *args, **kwargs) as *args.
        **kwargs
            For each waveform, wf, these are the 
            keyword arguments which are given to
            wf_filter(wf, *args, **kwargs) as **kwargs
        
        Returns
        ----------
        WaveformSet
            A new WaveformSet object which contains
            only the waveforms from the given 
            original_WaveformSet object which passed
            the given wf_filter callable.
        """

        staying_wfs_idcs = original_WaveformSet.filter( wf_filter,
                                                        *args,
                                                        actually_filter = False,
                                                        return_the_staying_ones = True,
                                                        **kwargs)
        
        waveforms = [ original_WaveformSet.Waveforms[idx] for idx in staying_wfs_idcs ] 
        
        ## About the waveforms that we will handle to the new WaveformSet object:
        ## Shall they be a deep copy? If they are not, maybe some of the Waveform
        ## objects that belong to both - the original and the filtered - WaveformSet
        ## objects are not independent, but references to the same Waveform objects 
        ## in memory. This could be an issue if we want, p.e. to run different 
        ## analyses on the different WaveformSet objects. I.e. running an analysis
        ## on the filtered waveformset could modify the analysis on the same waveform
        ## in the original waveformset. This would not be an issue, though, if we 
        ## want to partition the original waveformset into disjoint waveformsets, and
        ## never look back on the original waveformset, p.e. if we want to partition 
        ## the original waveformset according to the endpoints. This needs to be 
        ## checked, because it might be an open issue.

        return cls(*waveforms)  
        
    @staticmethod
    def check_well_formedness_of_generic_waveform_function(wf_function_signature : inspect.Signature) -> None:

        """
        This method gets an argument, wf_function_signature, 
        and returns None if the following conditions are met:

            -   such signature takes at least one argument
            -   the first argument of such signature
                is called 'waveform'
            -   the type annotation of such argument 
                must be either the WaveformAdcs class,
                the Waveform class, the 'WaveformAdcs'
                string literal or the 'Waveform' string
                literal
            -   the return type of such signature 
                is annotated as a boolean value

        If any of these conditions are not met, this
        method raises an exception.
                
        Parameters
        ----------
        wf_function_signature : inspect.Signature

        Returns
        ----------
        bool
        """

        try:
            if list(wf_function_signature.parameters.keys())[0] != 'waveform':
                raise Exception(generate_exception_message( 1,
                                                            "WaveformSet.check_well_formedness_of_generic_waveform_function()",
                                                            "The name of the first parameter of the given signature must be 'waveform'."))
        except IndexError:
            raise Exception(generate_exception_message( 2,
                                                        "WaveformSet.check_well_formedness_of_generic_waveform_function()",
                                                        'The given signature must take at least one parameter.'))
        
        if wf_function_signature.parameters['waveform'].annotation not in [WaveformAdcs, 'WaveformAdcs', Waveform, 'Waveform']:
            raise Exception(generate_exception_message( 3,
                                                        "WaveformSet.check_well_formedness_of_generic_waveform_function()",
                                                        "The 'waveform' parameter of the given signature must be hinted as a WaveformAdcs (or an inherited class) object."))
        if wf_function_signature.return_annotation != bool:
            raise Exception(generate_exception_message( 4,
                                                        "WaveformSet.check_well_formedness_of_generic_waveform_function()",
                                                        "The return type of the given signature must be hinted as a boolean."))
        return
    
    @staticmethod
    @numba.njit(nogil=True, parallel=False)
    def histogram1d(samples : np.ndarray,
                    bins : int,
                    domain : np.ndarray) -> np.ndarray:     # Not calling it 'range' because 
                                                            # it is a reserved keyword in Python
        """
        This method returns an unidimensional integer numpy 
        array which is the 1D histogram of the given samples.

        Parameters
        ----------
        samples : np.ndarray
            An unidimensional numpy array where samples[i] 
            gives the i-th sample.
        bins : int
            The number of bins
        domain : np.ndarray
            A 2x1 numpy array where (domain[0], domain[1])
            gives the range to consider for the histogram.
            Any sample which falls outside this range is 
            ignored.

        Returns
        ----------
        result : np.ndarray
            An unidimensional integer numpy array which 
            is the 1D histogram of the given samples
        """

        result = np.zeros(bins, dtype=np.uint64)

        inverse_step = 1. / ((domain[1] - domain[0]) / bins)

        for t in range(samples.shape[0]):

            i = (samples[t] - domain[0]) * inverse_step

            if 0 <= i < bins:
                result[int(i)] += 1

        return result

    @staticmethod
    @numba.njit(nogil=True, parallel=False)                 
    def histogram2d(samples : np.ndarray, 
                    bins : np.ndarray,                      # ~ 20 times faster than numpy.histogram2d
                    ranges : np.ndarray) -> np.ndarray:     # for a dataset with ~1.8e+8 points

        """
        This method returns a bidimensional integer numpy 
        array which is the 2D histogram of the given samples.

        Parameters
        ----------
        samples : np.ndarray
            A 2xN numpy array where samples[0, i] (resp.
            samples[1, i]) gives, for the i-th point in the
            samples set, the value for the coordinate which 
            varies along the first (resp. second) axis of 
            the returned bidimensional matrix.
        bins : np.ndarray
            A 2x1 numpy array where bins[0] (resp. bins[1])
            gives the number of bins to be considered along
            the coordinate which varies along the first 
            (resp. second) axis of the returned bidimensional 
            matrix.
        ranges : np.ndarray
            A 2x2 numpy array where (ranges[0,0], ranges[0,1])
            (resp. (ranges[1,0], ranges[1,1])) gives the 
            range for the coordinate which varies along the 
            first (resp. second) axis of the returned 
            bidimensional. Any sample which falls outside 
            these ranges is ignored.

        Returns
        ----------
        result : np.ndarray
            A bidimensional integer numpy array which is the
            2D histogram of the given samples.
        """

        result = np.zeros((bins[0], bins[1]), dtype=np.uint64)

        inverse_step = 1. / ((ranges[:, 1] - ranges[:, 0]) / bins)

        for t in range(samples.shape[1]):

            i = (samples[0, t] - ranges[0, 0]) * inverse_step[0]
            j = (samples[1, t] - ranges[1, 0]) * inverse_step[1]

            if 0 <= i < bins[0] and 0 <= j < bins[1]:       # Using this condition is slightly faster than               
                result[int(i), int(j)] += 1                 # using four nested if-conditions (one for each        
                                                            # one of the four conditions). For a dataset with             
        return result                                       # 178993152 points, the average time (for 30        
                                                            # calls to this function) gave ~1.06 s vs ~1.22 s

    @staticmethod
    def __add_no_data_annotation(   figure : pgo.Figure,
                                    row : int,
                                    col : int) -> pgo.Figure:
        
        """
        This method should only be called by the
        WaveformSet.plot_wfs() method, where the 
        the well-formedness checks of the input 
        have already been performed. No checks 
        are performed in this method. This method
        adds an empty trace and a centered annotation
        displaying 'No data' to the given figure at
        the given row and column. Finally, this
        method returns the figure.

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

    def __subplot_heatmap(self, figure : pgo.Figure,
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
        This method should only be called by the
        WaveformSet.plot_wfs() method, where the 
        data-availability and the well-formedness 
        checks of the input have already been 
        performed. No checks are performed in
        this method. For each subplot in the grid 
        plot generated by the WaveformSet.plot_wfs()
        methods when its 'mode' parameter is
        set to 'heatmap', such method delegates
        plotting the heatmap to the current method.
        This method takes the given figure, and 
        plots on it the heatmap of the union of 
        the waveforms whose indices are contained 
        within the given 'wf_idcs' list. The 
        position of the subplot where this heatmap 
        is plotted is given by the 'row' and 'col' 
        parameters. Finally, this method returns 
        the figure.

        Parameters
        ----------
        figure : pgo.Figure
            The figure where the heatmap will be
            plotted
        name : str
            The name of the heatmap. It is given
            to the 'heatmap' parameter of 
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
            to the self.__waveforms list, which
            will be added to the heatmap.
        analysis_label : str
            For each considered waveform, it is the
            key for its Analyses attribute which gives
            the WfAna object whose computed baseline
            is subtracted from the waveform prior to
            addition to the heatmap. This method does
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
                                        self.PointsPerWf,
                                        dtype = np.float32) + self.Waveforms[idx].TimeOffset for idx in wf_idcs])

        aux_y = np.hstack([self.Waveforms[idx].Adcs - self.Waveforms[idx].Analyses[analysis_label].Result.Baseline for idx in wf_idcs])

        aux = WaveformSet.histogram2d(  np.vstack((aux_x, aux_y)), 
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
    
    def plot_calibration_histogram(self,    nrows : int = 1,                                                ## This is a quick solution a la WaveformSet.plot_wfs()
                                            ncols : int = 1,                                                ## which is useful to produce calibration plots in
                                            figure : Optional[pgo.Figure] = None,                           ## self-trigger cases where a general integration window
                                            wfs_per_axes : Optional[int] = 100,                             ## can be defined. Eventually, a method like this should
                                            grid_of_wf_idcs : Optional[List[List[List[int]]]] = None,       ## inspect the Analyses attribute of each waveform
                                            analysis_label : Optional[str] = None,                          ## in search for the spotted WfPeaks and their integrals.
                                            bins : int = 250,
                                            domain : Tuple[float, float] = (-20000., 60000.),               # It's the regular range, but here it is called 'domain'
                                            share_x_scale : bool = False,                                   # to not collide with the 'range' reserved keyword
                                            share_y_scale : bool = False,                                   
                                            detailed_label : bool = True) -> pgo.Figure:                    ## Also, most of the code of this function is copied from            
                                                                                                            ## that of WaveformSet.plot_wfs(). A way to avoid this is
                                                                                                            ## to incorporate the histogram functionality into the
                                                                                                            ## WaveformSet.plot_wfs() method, but I don't think that's
                                                                                                            ## a good idea, though. Maybe we should find a way to 
                                                                                                            ## encapsulate the shared code into an static method.    
        """                                                                                                 
        This method returns a plotly.graph_objects.Figure                                                   
        with a nrows x ncols grid of axes, with plots of                                                    
        the calibration histograms which include a subset
        of the waveforms in this WaveformSet object.

        Parameters
        ----------
        nrows (resp. ncols) : int
            Number of rows (resp. columns) of the returned 
            grid of axes.
        figure : plotly.graph_objects.Figure
            If it is not None, then it must have been
            generated using plotly.subplots.make_subplots()
            (even if nrows and ncols equal 1). It is the
            caller's responsibility to ensure this.
            If that's the case, then this method adds the
            plots to this figure and eventually returns 
            it. In such case, the number of rows (resp. 
            columns) in such figure must match the 'nrows' 
            (resp. 'ncols') parameter.
        wfs_per_axes : int
            If it is not None, then the argument given to 
            'grid_of_wf_idcs' will be ignored. In this case,
            the number of waveforms considered for each
            axes is wfs_per_axes. P.e. for wfs_per_axes 
            equal to 100, the axes at the first row and 
            first column contains a calibration histogram 
            with 100 entries, each of which comes from the
            integral of the first 100 waveforms in this
            WaveformSet object. The axes in the first 
            row and second column will consider the 
            following 100 waveforms, and so on.
        grid_of_wf_idcs : list of list of list of int
            This list must contain nrows lists, each of 
            which must contain ncols lists of integers. 
            grid_of_wf_idcs[i][j] gives the indices of the 
            waveforms, with respect to this WaveformSet, whose
            integrals will be part of the calibration
            histogram which is located at the i-th row 
            and j-th column.
        analysis_label : str
            This parameter gives the key for the WfAna 
            object within the Analyses attribute of each 
            considered waveform from where to take the 
            integral value to add to the calibration
            histogram. Namely, if such WfAna object is
            x, then x.Result.Integral is the considered
            integral. If 'analysis_label' is None, 
            then the last analysis added to 
            self.__analyses will be the used one.
        bins : int
            A positive integer giving the number of bins 
            in each histogram
        domain : tuple of float
            It must contain two floats, so that domain[0]
            is smaller than domain[1]. It is the range
            of each histogram.
        share_x_scale (resp. share_y_scale) : bool
            If True, the x-axis (resp. y-axis) scale will be 
            shared among all the subplots.
        detailed_label : bool
            Whether to show the iterator values of the two 
            first available waveforms (which contribute to
            the calibration histogram) in the label of
            each histogram.
             
        Returns
        ----------
        figure : plotly.graph_objects.Figure
            The figure with the grid plot of the waveforms
        """

        if nrows < 1 or ncols < 1:
            raise Exception(generate_exception_message( 1,
                                                        'WaveformSet.plot_calibration_histogram()',
                                                        'The number of rows and columns must be positive.'))
        fFigureIsGiven = False
        if figure is not None:

            try:
                fig_rows, fig_cols = figure._get_subplot_rows_columns() # Returns two range objects
                fig_rows, fig_cols = list(fig_rows)[-1], list(fig_cols)[-1]

            except Exception:   # Happens if figure was not created using plotly.subplots.make_subplots

                raise Exception(generate_exception_message( 2,
                                                            'WaveformSet.plot_calibration_histogram()',
                                                            'The given figure is not a subplot grid.'))
            if fig_rows != nrows or fig_cols != ncols:
                
                raise Exception(generate_exception_message( 3,
                                                            'WaveformSet.plot_calibration_histogram()',
                                                            f"The number of rows and columns in the given figure ({fig_rows}, {fig_cols}) must match the nrows ({nrows}) and ncols ({ncols}) parameters."))
            fFigureIsGiven = True

        grid_of_wf_idcs_ = None         # Logically useless

        if wfs_per_axes is not None:    # wfs_per_axes is defined

            if wfs_per_axes < 1:
                raise Exception(generate_exception_message( 4,
                                                            'WaveformSet.plot_calibration_histogram()',
                                                            'The number of waveforms per axes must be positive.'))

            grid_of_wf_idcs_ = self.get_grid_of_wf_idcs(nrows,
                                                        ncols,
                                                        wfs_per_axes = wfs_per_axes)

        elif grid_of_wf_idcs is None:   # Nor wf_per_axes, nor 
                                        # grid_of_wf_idcs are defined

            raise Exception(generate_exception_message( 5,
                                                        'WaveformSet.plot_calibration_histogram()',
                                                        "The 'grid_of_wf_idcs' parameter must be defined if wfs_per_axes is not."))
        
        elif not WaveformSet.grid_of_lists_is_well_formed(  grid_of_wf_idcs,    # wf_per_axes is not defined, 
                                                            nrows,              # but grid_of_wf_idcs is, but 
                                                            ncols):             # it is not well-formed
            raise Exception(generate_exception_message( 6,
                                                        'WaveformSet.plot_calibration_histogram()',
                                                        f"The given grid_of_wf_idcs is not well-formed according to nrows ({nrows}) and ncols ({ncols})."))
        else:   # wf_per_axes is not defined,
                # but grid_of_wf_idcs is,
                # and it is well-formed

            grid_of_wf_idcs_ = grid_of_wf_idcs

        if bins < 1:
            raise Exception(generate_exception_message( 7,
                                                        'WaveformSet.plot_calibration_histogram()',
                                                        f"The given number of bins ({bins}) is not positive."))
        
        if domain[0] >= domain[1]:
            raise Exception(generate_exception_message( 8,
                                                        'WaveformSet.plot_calibration_histogram()',
                                                        f"The given domain ({domain}) is not well-formed."))
        if not fFigureIsGiven:
            
            figure_ = psu.make_subplots(    rows = nrows, 
                                            cols = ncols)
        else:
            figure_ = figure

        WaveformSet.update_shared_axes_status(  figure_,                    # An alternative way is to specify 
                                                share_x = share_x_scale,    # shared_xaxes=True (or share_yaxes=True)
                                                share_y = share_y_scale)    # in psu.make_subplots(), but, for us, 
                                                                            # that alternative is only doable for 
                                                                            # the case where the given 'figure'
                                                                            # parameter is None.

        step = (domain[1] - domain[0]) / bins
                                                                        
        for i in range(nrows):
            for j in range(ncols):
                if len(grid_of_wf_idcs_[i][j]) > 0:

                    aux_name = f"{len(grid_of_wf_idcs_[i][j])} Wf(s)"
                    if detailed_label:
                         aux_name += f": [{WaveformSet.get_string_of_first_n_integers_if_available(grid_of_wf_idcs_[i][j], queried_no = 2)}]"
                         
                    data = WaveformSet.histogram1d( np.array([self.Waveforms[idc].get_analysis(analysis_label).Result.Integral for idc in grid_of_wf_idcs_[i][j]]), ## This one might be slow !!!!
                                                    bins,
                                                    domain)
                    
                    figure.add_trace(   pgo.Scatter(    x = np.linspace(domain[0] + (step / 2.0), 
                                                                        domain[1] - (step / 2.0), 
                                                                        num = bins,
                                                                        endpoint = True),
                                                        y = data,
                                                        mode = 'lines',
                                                        line = dict(color='black', 
                                                                    width=0.5),
                                                        name = f"({i+1},{j+1}) - C. H. of " + aux_name,),
                                        row = i + 1, 
                                        col = j + 1)
                else:

                    WaveformSet.__add_no_data_annotation(   figure_,
                                                            i + 1,
                                                            j + 1)
        return figure_
    
    @staticmethod
    def reference_to_minimum(input : List[int]) -> List[int]:

        """
        This method returns a list of integers, say output,
        so that output[i] is equal to input[i] minus the
        minimum value within input.

        Parameters
        ----------
        input : list of int

        Returns
        ----------
        list of int
        """

        aux = np.array(input).min()

        return [ input[i] - aux for i in range(len(input)) ]