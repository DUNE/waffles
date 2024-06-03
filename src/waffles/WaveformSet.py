import math
import inspect
from typing import Tuple, List, Callable, Optional

import uproot
import numpy as np
from plotly import graph_objects as pgo
from plotly import subplots as psu

from src.waffles.NWaveform import Waveform
from src.waffles.WfAna import WfAna
from src.waffles.WfAnaResult import WfAnaResult
from src.waffles.Exceptions import generate_exception_message

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
    AvailableChannels : dictionary
        It is a dictionary whose keys are endpoints (int) 
        and its values are sets of channels (set of int).
        If there is at least one Waveform object within
        this WaveformSet which comes from endpoint n, then
        n belongs to AvailableChannels.keys(). 
        AvailableChannels[n] is a set of channels for 
        endpoint n. If there is at least one waveform for
        endpoint n and channel m, then m belongs to 
        AvailableChannels[n].
    MeanAdcs : np.ndarray
        The mean of the adcs arrays for a every waveform
        or a subset of waveforms in this WaveformSet. It 
        is an unidimensional float numpy array with 
        PointsPerWf entries, so that MeanAdcs[i] is the 
        mean of self.Waveforms[j].Adcs[i] for every value
        of j or a subset of values of j, within 
        [0, len(self.__waveforms) - 1]. It is not 
        computed by default. I.e. if self.MeanAdcs 
        equals to None, it should be interpreted as 
        unavailable data. Call the 'compute_mean_waveform' 
        method of this WaveformSet to compute it.
    MeanAdcsIdcs : tuple of int
        It is a tuple of integers which contains the indices
        of the waveforms, with respect to this WaveformSet,
        which were used to compute the MeanAdcs attribute.
        By default, it is None. I.e. if self.MeanAdcsIdcs
        equals to None, it should be interpreted as
        unavailable data. Call the 'compute_mean_waveform'
        method of this WaveformSet to compute it.

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
        
        self.__waveforms = list(waveforms)

        if not self.check_length_homogeneity():
            raise Exception(generate_exception_message( 1,
                                                        'WaveformSet.__init__',
                                                        'The length of the given waveforms is not homogeneous.'))
        
        self.__points_per_wf = len(self.__waveforms[0].Adcs)

        self.__runs = set()
        self.update_runs()

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
    def AvailableChannels(self):
        return self.__available_channels
    
    @property
    def MeanAdcs(self):
        return self.__mean_adcs
    
    @property
    def MeanAdcsIdcs(self):
        return self.__mean_adcs_idcs
    
    def check_length_homogeneity(self) -> bool:
            
            """
            This method returns True if the Adcs attribute
            of every Waveform object in this WaveformSet
            has the same length. It returns False if else.

            Returns
            ----------
            bool
            """

            length = len(self.__waveforms[0].Adcs)
            for i in range(1, len(self.__waveforms)):
                if len(self.__waveforms[i].Adcs) != length:
                    return False
            return True
    
    def update_runs(self) -> None:
        
        """
        This method iterates through the whole WaveformSet
        and updates the self.__runs attribute of this object. 

        Returns
        ----------
        None
        """

        for wf in self.__waveforms:
            self.__runs.add(wf.RunNumber)
        return
    
    def update_available_channels(self) -> None:
        
        """
        This method iterates through the whole WaveformSet
        and updates the self.__available_channels attribute of 
        this object. 

        Returns
        ----------
        None
        """

        for wf in self.__waveforms:
            try:
                self.__available_channels[wf.Endpoint].add(wf.Channel)
            except KeyError:
                self.__available_channels[wf.Endpoint] = set()
                self.__available_channels[wf.Endpoint].add(wf.Channel)
        return
    
    def analyse(self,   label : str,
                        analyser_name : str,
                        baseline_limits : List[int],
                        int_ll : int = 0,
                        int_ul : Optional[int] = None,
                        *args,
                        overwrite : bool = False,
                        **kwargs) -> dict:
        
        """
        For each Waveform in this WaveformSet, this method
        calls its 'analyse' method passing to it the parameters
        given to this method. In turn, Waveform.analyse()
        creates a WfAna object and adds it to the Analyses 
        attribute of the analysed waveform. It also runs the
        indicated analyser method (up to the 'analyser_name'
        parameter) on the waveform, and adds its results to 
        the 'Result' and 'Passed' attributes of the newly 
        created WfAna object. Also, it returns a dictionary,
        say output, whose keys are integers in 
        [0, len(self.__waveforms) - 1]. ouptut[i] matches
        the output of self.__waveforms[i].analyse(...), 
        which is a dictionary. I.e. the output of this method 
        is a dictionary of dictionaries.

        Parameters
        ----------
        label : str
            For every analysed waveform, this is the key
            for the new WfAna object within its Analyses
            attribute.
        analyser_name : str
            It must match the name of a WfAna method whose first            
            argument must be called 'waveform' and whose type           # The only way to import the Waveform class in WfAna without having         # This would not be a problem (and we would not    
            annotation must match the Waveform class or the             # a circular import is to use the typing.TYPE_CHECKING variable, which      # need to grab the analyser method using an 
            'Waveform' string literal. Such method should also          # is only defined for type-checking runs. As a consequence, the type        # string and getattr) if the analyser methods were
            have a defined return-annotation which must match           # annotation should be an string, which the type-checking software          # defined as Waveform methods or in a separate module.
            Tuple[WfAnaResult, bool, dict].                             # successfully associates to the class itself, but which is detected        # There might be other downsizes to it such as the
                                                                        # as so (a string) by inspect.signature().                                  # accesibility to WfAna attributes.
        baseline_limits : list of int
            For every analysed waveform, it defines
            the Adcs points which will be used for
            baseline calculation (it is given to
            the 'baseline_limits' parameter of
            Waveform.analyse()). It must have an 
            even number of integers which must meet 
            baseline_limits[i] < baseline_limits[i + 1] 
            for all i. The points which are used for 
            baseline calculation are 
            self.__adcs[baseline_limits[2*i]:baseline_limits[(2*i) + 1]],
            with i = 0,1,...,(len(baseline_limits)/2) - 1. 
            The upper limits are exclusive. For more 
            information check the 'baseline_limits' 
            parameter documentation in the 
            Waveform.analyse() docstring.
        int_ll (resp. int_ul): int
            For every analysed waveform, it defines the
            integration window (it is given to the 'int_ll'
            (resp. 'int_ul') parameter of Waveform.analyse()).
            int_ll must be smaller than int_ul. These limits 
            are inclusive. If they are not defined, then the
            whole Adcs are considered for each waveform. 
            For more information check the 'int_ll' and 
            'int_ul' parameters documentation in the 
            Waveform.analyse() docstring.
        *args
            For each analysed waveform, these are the 
            positional arguments which are given to the
            analyser method by Waveform.analyse().
        overwrite : bool
            If True, for every analysed Waveform wf, its
            'analyze' method will overwrite any existing
            WfAna object with the same label (key) within
            its Analyses attribute.
        **kwargs
            For each analysed waveform, these are the
            keyword arguments which are given to the
            analyser method by Waveform.analyse().

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
                                                            "WaveformSet.analyse",
                                                            "The name of the first parameter of the given analyser method must be 'waveform'."))
            
            if signature.parameters['waveform'].annotation not in ['Waveform', Waveform]:
                raise Exception(generate_exception_message( 6,
                                                            "WaveformSet.analyse",
                                                            "The 'waveform' parameter of the analyser method must be hinted as a Waveform object."))
            
            if signature.return_annotation != Tuple[WfAnaResult, bool, dict]:
                raise Exception(generate_exception_message( 7,
                                                            "WaveformSet.analyse",
                                                            "The return type of the analyser method must be hinted as Tuple[WfAnaResult, bool]."))
        except IndexError:
            raise Exception(generate_exception_message( 8,
                                                        "WaveformSet.analyse",
                                                        'The given analyser method must take at least one parameter.'))
        output = {}

        for i in range(len(self.__waveforms)):
            output[i] = self.__waveforms[i].analyse(    label,
                                                        analyser_name,
                                                        baseline_limits,
                                                        int_ll = int_ll,
                                                        int_ul = int_ul_,
                                                        *args,
                                                        overwrite = overwrite,
                                                        **kwargs)
        return output
    
    def baseline_limits_are_well_formed(self, baseline_limits : List[int]) -> bool:

        """
        This method returns True if len(baseline_limits) is even and 
        0 <= baseline_limites[0] < baseline_limits[1] < ... < baseline_limits[-1] <= self.PointsPerWf - 1.
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
    
    def plot(self,  nrows : int = 1,
                    ncols : int = 1,
                    wfs_per_axes : Optional[int] = 1,
                    grid_of_wf_idcs : Optional[List[List[List[int]]]] = None,
                    plot_analysis_markers : bool = False,
                    show_baseline_limits : bool = False, 
                    show_baseline : bool = True,
                    show_general_integration_limits : bool = False,
                    show_spotted_peaks : bool = True,
                    show_peaks_integration_limits : bool = False,
                    analysis_label : Optional[str] = None) -> pgo.Figure: 


        """ 
        This method returns a plotly.graph_objects.Figure 
        with a nrows x ncols grid of axes, with plots of
        some of the waveforms in this WaveformSet object.

        Parameters
        ----------
        nrows (resp. ncols) : int
            Number of rows (resp. columns) of the returned 
            grid of axes.
        wfs_per_axes : int
            If it is not None, then the argument given to 
            'grid_of_wf_idcs' will be ignored. In this case,
            each axes contains wfs_per_axes waveforms.
            P.e. for wfs_per_axes equal to 2, the axes
            for the first row and first column will contain
            the first two waveforms in the set, the axes
            in the first row and second column will contain
            the following two, and so on.
        grid_of_wf_idcs : list of list of list of int
            This list must contain nrows lists, each of which
            must contain ncols lists of integers. 
            grid_of_wf_idcs[i][j] gives the indices of the 
            waveforms, with respect to this WaveformSet, which 
            should be plotted in the axes located at the i-th 
            row and j-th column.
        plot_analysis_markers : bool
            This parameter is given to the 'plot_analysis_markers' 
            argument of the Waveform.plot() method for each 
            waveform in this WaveformSet. If True, analysis markers
            for the waveforms will potentially be plotted together 
            with each waveform. For more information, check the 
            'plot_analysis_markers' parameter documentation in the 
            Waveform.plot() method. If False, no analysis markers 
            will be plot.
        show_baseline_limits : bool
            This parameter only makes a difference if
            'plot_analysis_markers' is set to True. In that case,
            this parameter means whether to plot vertical lines
            framing the intervals which were used to compute
            the baseline.
        show_baseline : bool
            This parameter only makes a difference if
            'plot_analysis_markers' is set to True. In that case,
            this parameter means whether to plot an horizontal
            line matching the computed baseline
        show_general_integration_limits : bool
            This parameter only makes a difference if
            'plot_analysis_markers' is set to True. In that case,
            this parameter means whether to plot vertical lines
            framing the general integration interval.
        show_spotted_peaks : bool
            This parameter only makes a difference if
            'plot_analysis_markers' is set to True. In that case,
            this parameter means whether to plot a triangle
            marker over each spotted peak.
        show_peaks_integration_limits : bool
            This parameter only makes a difference if
            'plot_analysis_markers' is set to True. In that case,
            this parameter means whether to plot two vertical
            lines framing the integration interval for each
            spotted peak.
        analysis_label : str
            This parameter is given to the 'analysis_label' 
            parameter of the Waveform.plot() method for each
            waveform in this WaveformSet. It only makes a difference 
            if 'plot_analysis_markers' is set to True. In that case, 
            'analysis_label' is the key for the WfAna object within 
            the Analysis attribute of each plotted waveform from 
            where to take the information for the analysis markers 
            plot. If 'analysis_label' is None, then the last analysis 
            added to self.__analyses will be the used one.            

        Returns
        ----------
        figure : plotly.graph_objects.Figure
            The figure with the grid plot
        """

        if nrows < 1 or ncols < 1:
            raise Exception(generate_exception_message( 1,
                                                        'WaveformSet.plot()',
                                                        'The number of rows and columns must be positive.'))
        fArbitraryWfs = False
        if wfs_per_axes is not None:
            if wfs_per_axes < 1:
                raise Exception(generate_exception_message( 2,
                                                            'WaveformSet.plot()',
                                                            'The number of waveforms per axes must be positive.'))
            fArbitraryWfs = True

        elif grid_of_wf_idcs is None:
            raise Exception(generate_exception_message( 3,
                                                        'WaveformSet.plot()',
                                                        "The 'grid_of_wf_idcs' parameter must be defined if wfs_per_axes is not."))
        
        elif not WaveformSet.grid_of_lists_is_well_formed(  grid_of_wf_idcs,
                                                            nrows,
                                                            ncols):
                
                raise Exception(generate_exception_message( 4,
                                                            'WaveformSet.plot()',
                                                            f"The given grid_of_wf_idcs is not well-formed according to nrows ({nrows}) and ncols ({ncols})."))
        figure = psu.make_subplots( rows = nrows, 
                                    cols = ncols)
        if fArbitraryWfs:
            counter = 0
            for i in range(nrows):
                for j in range(ncols):
                    for k in range(wfs_per_axes):

                        self.__waveforms[counter].plot( figure = figure,
                                                        name = f"Wf {counter}, Ch {self.__waveforms[counter].Channel}, Ep {self.__waveforms[counter].Endpoint}",
                                                        row = i + 1,  # Plotly uses 1-based indexing
                                                        col = j + 1,
                                                        plot_analysis_markers = plot_analysis_markers,
                                                        show_baseline_limits = show_baseline_limits,
                                                        show_baseline = show_baseline,
                                                        show_general_integration_limits = show_general_integration_limits,
                                                        show_spotted_peaks = show_spotted_peaks,
                                                        show_peaks_integration_limits = show_peaks_integration_limits,
                                                        analysis_label = analysis_label)
                        counter += 1
        else:
            for i in range(nrows):
                for j in range(ncols):
                    for k in grid_of_wf_idcs[i][j]:

                        self.__waveforms[k].plot(   figure = figure,
                                                    name = f"Wf {k}, Ch {self.__waveforms[k].Channel}, Ep {self.__waveforms[k].Endpoint}",
                                                    row = i + 1,  # Plotly uses 1-based indexing
                                                    col = j + 1,
                                                    plot_analysis_markers = plot_analysis_markers,
                                                    show_baseline_limits = show_baseline_limits,
                                                    show_baseline = show_baseline,
                                                    show_general_integration_limits = show_general_integration_limits,
                                                    show_spotted_peaks = show_spotted_peaks,
                                                    show_peaks_integration_limits = show_peaks_integration_limits,
                                                    analysis_label = analysis_label)
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
                                    wf_filter : Callable[..., bool],
                                    filter_args : List[List[List]],
                                    max_wfs_per_axes : Optional[int] = 5) -> List[List[List[int]]]:
        
        """
        This method returns a list of lists of lists of integers.

        Parameters
        ----------
        nrows : int
            The length of the returned list. It must match
            the length of the filter_args list.
        ncols : 
            The length of every list within the returned 
            list. It must match the length of every list
            within the filter_args list.
        wf_filter : callable
            A callable object whose first parameter must be
            called 'waveform' and must be hinted as a Waveform
            object. Such callable must return a boolean value.
            If wf_filter is 
                - WaveformSet.match_run or
                - WaveformSet.match_endpoint_and_channel,
            this method can benefit from the information in
            self.Runs and self.AvailableChannels and its
            execution time may be reduced with respect to
            the case where an arbitrary (but compliant) 
            callable is passed to wf_filter.
        filter_args : list of list of list
            filter_args[i][j], for all i and j, is a list of 
            arguments which will be given to wf_filter at
            some point. The user is responsible for giving
            a set of arguments which comply with the signature
            of the specified wf_filter. For more information 
            check the return value documentation.
        max_wfs_per_axes : int
            If it is not None, then output[i][j] will contain
            the indices for the first max_wfs_per_axes waveforms
            in this WaveformSet which passed the filter.
            If it is None, then this function iterates through
            the whole WaveformSet for every i,j pair.
            Note that setting this parameter to None may
            result in a long execution time.

        Returns
        ----------
        output : list of list of list of int
            output[i][j] gives the indices of the waveforms
            in this WaveformSet object, say wf, for which
            wf_filter(wf, *filter_args[i][j]) returns True.
        """

        if nrows < 1 or ncols < 1:
            raise Exception(generate_exception_message( 1,
                                                        'WaveformSet.get_grid_of_wf_idcs()',
                                                        'The number of rows and columns must be positive.'))

        if not WaveformSet.grid_of_lists_is_well_formed(filter_args,
                                                        nrows,
                                                        ncols):
            
            raise Exception(generate_exception_message( 2,
                                                        'WaveformSet.get_grid_of_wf_idcs()',
                                                        f"The shape of the given filter_args list is not nrows ({nrows}) x ncols ({ncols})."))
        
        signature = inspect.signature(wf_filter)

        if list(signature.parameters.keys())[0] != 'waveform':
            raise Exception(generate_exception_message( 3,
                                                        'WaveformSet.get_grid_of_wf_idcs()',
                                                        "The name of the first parameter of the given filter must be 'waveform'."))
        
        if signature.parameters['waveform'].annotation != Waveform:
            raise Exception(generate_exception_message( 4,
                                                        'WaveformSet.get_grid_of_wf_idcs()',
                                                        "The 'waveform' parameter of the filter must be hinted as a Waveform object."))
        fMaxIsSet = False
        if max_wfs_per_axes is not None:
            if max_wfs_per_axes < 1:
                raise Exception(generate_exception_message( 5,
                                                            'WaveformSet.get_grid_of_wf_idcs()',
                                                            'The number of waveforms per axes must be positive.'))
            fMaxIsSet = True

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
        of the given waveform matches run. It returns False
        if else.

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
    def match_channel(  waveform : Waveform,
                            channel : int) -> bool:
        
        """
        This method returns True if the Channel attribute
        of the given waveform matches channel, and False
        if else.

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
        attributes of the given waveform match endpoint and 
        channel, respectively.

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

        for i in range(len(blank_grid)):
            for j in range(len(blank_grid[i])):

                if filter_args[i][j][0] not in self.__available_channels.keys():    # filter_args[i][j][0] is the
                    continue                                                        # endpoint we are looking for

                elif filter_args[i][j][1] not in self.__available_channels[filter_args[i][j][0]]:   # filter_args[i][j][1] is
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

    @classmethod
    def from_ROOT_file(cls, filepath : str,
                            tree_to_look_for : str ='raw_waveforms',
                            start_fraction : float = 0.0,
                            stop_fraction : float = 1.0) -> 'WaveformSet':

        """
        Alternative initializer for a WaveformSet object out of the
        waveforms stored in a ROOT file

        Parameters
        ----------
        filepath : str
            Path to the ROOT file to be read. Such ROOT file should 
            have a defined TTree object whose name matches tree_to_look_for.
            Such TTree should have at least three branches, with names
            'channel', 'timestamp', 'adcs', from which the values for           ## For the moment, the timestamp branch may
            the Waveform objects attributes Channel, Timestamp and Adcs         ## be called 'timestamps'
            will be taken respectively.
        tree_to_look_for : str
            Name of the tree which will be extracted from the given
            ROOT file
        start_fraction (resp. stop_fraction) : float
            Gives the iterator value for the first (resp. last) waveform
            which will be loaded into this WaveformSet object. P.e. 
            setting start_fraction to 0.5 and stop_fraction to 0.75 
            will result in loading the waveforms that belong to the 
            third quarter of the input file.
        """

        if not WaveformSet.fraction_is_well_formed(start_fraction, stop_fraction):
            raise Exception(generate_exception_message( 1,
                                                        'WaveformSet.from_ROOT_file()',
                                                        f"Fraction limits are not well-formed"))
        input_file = uproot.open(filepath)

        try:
            aux = input_file[tree_to_look_for+';1']     # Assuming that ROOT appends
        except KeyError:                                # ';1' to its trees names
            raise Exception(generate_exception_message( 2,
                                                        'WaveformSet.from_ROOT_file()',
                                                        f"TTree {tree_to_look_for} not found in {filepath}"))
        if 'channel' not in aux.keys():
            raise Exception(generate_exception_message( 3,
                                                        'WaveformSet.from_ROOT_file()',
                                                        f"Branch 'channel' not found in the given TTree"))
        if 'timestamp' not in aux.keys() and 'timestamps' not in aux.keys():    ## Temporal
            raise Exception(generate_exception_message( 4,
                                                        'WaveformSet.from_ROOT_file()',
                                                        f"Branch 'timestamp' not found in the given TTree"))
        if 'adcs' not in aux.keys():
            raise Exception(generate_exception_message( 5,
                                                        'WaveformSet.from_ROOT_file()',
                                                        f"Branch 'adcs' not found in the given TTree"))
        
        adcs = aux['adcs']  # adcs is an uproot.TBranch object

        wf_start = math.floor(start_fraction*adcs.num_entries)
        wf_stop = math.ceil(stop_fraction*adcs.num_entries)

        channels = aux['channel'].array(entry_start = wf_start, 
                                        entry_stop = wf_stop)         # It is slightly faster (~106s vs. 114s, for a
                                                                    # 809 MB input file running on lxplus9) to read
        adcs = aux['adcs'].array(   entry_start = wf_start,           # branch by branch rather than going for aux.arrays()
                                    entry_stop = wf_stop)          
        try:
            timestamps = aux['timestamp'].array(entry_start = wf_start,
                                                entry_stop = wf_stop)   
        except uproot.exceptions.KeyInFileError:    
            timestamps = aux['timestamps'].array(   entry_start = wf_start,
                                                    entry_stop = wf_stop) ## Temporal

        waveforms = []                  # Using a list comprehension here is slightly slower than a for loop
        for i in range(len(adcs)):      # (97s vs 102s for 5% of wvfs of a 809 MB file running on lxplus9)

            endpoint, channel = WaveformSet.get_endpoint_and_channel(channels[i])

            waveforms.append(Waveform(  timestamps[i],
                                        0,      # TimeStep_ns   ## To be implemented from the new
                                                                ## 'metadata' TTree in the ROOT file
                                        np.array(adcs[i]),
                                        0,      #RunNumber      ## To be implemented from the new
                                                                ## 'metadata' TTree in the ROOT file
                                        endpoint,
                                        channel))      
        return cls(*waveforms)

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
    
    def compute_mean_waveform(self, wf_selector : Optional[Callable[..., bool]] = None,
                                    *args,
                                    **kwargs) -> None:

        """
        If wf_selector is None, then this method 
        computes mean of the adcs arrays for every 
        waveform in this WaveformSet. If wf_selector
        is not None, then this method computes the
        mean of the adcs arrays of the waveforms, wf,
        within this WaveformSet for which 
        wf_selector(wf, *args, **kwargs) evaluates 
        to True. The result is assigned to the
        self.__mean_adcs attribute. The 
        self.__mean_adcs_idcs attribute is also
        updated with a tuple of the indices of the
        waveforms which were used to compute the
        mean waveform.

        Parameters
        ----------
        wf_selector : callable                                      
            If it is not None, then it must be a callable           # As for WaveformSet.analyse(), requiring the type
            whose first parameter must be called 'waveform'         # annotation to be either Waveform or 'Waveform'
            and its type annotation must match the Waveform         # lets us cover the case where the wf_selector
            class or the 'Waveform' string literal. Its             # is defined in a module where the Waveform class
            return value must be annotated as a boolean.            # cannot be imported due to circular-import issues.
        *args
            For each waveform, wf, these are the 
            positional arguments which are given to
            wf_selector(wf, *args, **kwargs) as *args.
        *kwargs
            For each waveform, wf, these are the 
            keyword arguments which are given to
            wf_selector(wf, *args, **kwargs) as **kwargs.

        Returns
        ----------
        None
        """

        if len(self.__waveforms) == 0:
            raise Exception(generate_exception_message( 1,
                                                        'WaveformSet.compute_mean_waveform()',
                                                        'There are no waveforms in this WaveformSet object.'))
        if wf_selector is None:

            aux = self.Waveforms[0].Adcs

            for i in range(1, len(self.__waveforms)):
                aux += self.Waveforms[i].Adcs

            self.__mean_adcs = aux/len(self.__waveforms)
            self.__mean_adcs_idcs = tuple(range(len(self.__waveforms)))

        else:

            signature = inspect.signature(wf_selector)

            if list(signature.parameters.keys())[0] != 'waveform':
                raise Exception(generate_exception_message( 2,
                                                            "WaveformSet.compute_mean_waveform",
                                                            "The name of the first parameter of the given waveform-selector method must be 'waveform'."))
            
            if signature.parameters['waveform'].annotation not in ['Waveform', Waveform]:
                raise Exception(generate_exception_message( 3,
                                                            "WaveformSet.compute_mean_waveform",
                                                            "The 'waveform' parameter of the waveform-selector method must be hinted as a Waveform object."))
            
            if signature.return_annotation != bool:
                raise Exception(generate_exception_message( 4,
                                                            "WaveformSet.compute_mean_waveform",
                                                            "The return type of the waveform-selector method must be hinted as a boolean."))
            added_wvfs = []

            aux = np.zeros((self.__points_per_wf,))

            for i in range(len(self.__waveforms)):
                if wf_selector(self.__waveforms[i], *args, **kwargs):
                    aux += self.__waveforms[i].Adcs
                    added_wvfs.append(i)
                    
            if len(added_wvfs) == 0:
                raise Exception(generate_exception_message( 5,
                                                            'WaveformSet.compute_mean_waveform()',
                                                            'No waveform in this WaveformSet object passed the given selector.'))
            
            self.__mean_adcs = aux/len(added_wvfs)
            self.__mean_adcs_idcs = tuple(added_wvfs)

        return