import math
import inspect

import numba
import numpy as np
from typing import Tuple, List, Dict, Callable, Optional
from plotly import graph_objects as pgo
from plotly import subplots as psu

from waffles.data_classes.WaveformAdcs import WaveformAdcs
from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.WfAna import WfAna
from waffles.data_classes.WfAnaResult import WfAnaResult
from waffles.data_classes.Map import Map
from waffles.data_classes.ChannelMap import ChannelMap

import waffles.utils.filtering_utils as wuf

from waffles.Exceptions import generate_exception_message

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
        self.__update_runs(other_runs = None)

        self.__record_numbers = {}
        self.__update_record_numbers(other_record_numbers = None)

        self.__available_channels = {}
        self.__update_available_channels(other_available_channels = None)   # Running on an Apple M2, it took 
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
    
    def __update_runs(self, other_runs : Optional[set] = None) -> None:
        
        """
        This method is not intended to be called by the user.
        This method updates the self.__runs attribute of this
        object. Its behaviour is different depending on whether
        the 'other_runs' parameter is None or not. Check its
        documentation for more information.

        Parameters
        ----------
        other_runs : set of int
            If it is None, then this method clears the self.__runs 
            attribute of this object and then iterates through the 
            whole WaveformSet to fill such attribute according to 
            the waveforms which are currently present in this 
            WaveformSet object. If the 'other_runs' parameter is 
            defined, then it must be a set of integers, as expected 
            for the Runs attribute of a WaveformSet object. In this 
            case, the entries within other_runs which are not
            already present in self.__runs, are added to self.__runs.
            The well-formedness of this parameter is not checked by
            this method. It is the caller's responsibility to ensure
            it.

        Returns
        ----------
        None
        """

        if other_runs is None:
            self.__reset_runs()
        else:
            self.__runs = self.__runs.union(other_runs)

        return
    
    def __reset_runs(self) -> None:

        """
        This method is not intended for user usage.
        This method must only be called by the
        WaveformSet.__update_runs() method. It clears
        the self.__runs attribute of this object and 
        then iterates through the whole WaveformSet to 
        fill such attribute according to the waveforms 
        which are currently present in this WaveformSet 
        object.
        """

        self.__runs.clear()

        for wf in self.__waveforms:
            self.__runs.add(wf.RunNumber)

        return
    
    def __update_record_numbers(self, other_record_numbers : Optional[Dict[int, set]] = None) -> None:
        
        """
        This method is not intended to be called by the user.
        This method updates the self.__record_numbers attribute
        of this object. Its behaviour is different depending on
        whether the 'other_record_numbers' parameter is None or
        not. Check its documentation for more information.

        Parameters
        ----------
        other_record_numbers : dictionary of sets of int
            If it is None, then this method clears the
            self.__record_numbers attribute of this object
            and then iterates through the whole WaveformSet
            to fill such attribute according to the waveforms
            which are currently present in this WaveformSet.
            If the 'other_record_numbers' parameter is defined,
            then it must be a dictionary of sets of integers,
            as expected for the RecordNumbers attribute of a
            WaveformSet object. In this case, the information 
            in other_record_numbers is merged into the
            self.__record_numbers attribute of this object,
            according to the meaning of the self.__record_numbers
            attribute. The well-formedness of this parameter 
            is not checked by this method. It is the caller's 
            responsibility to ensure it.

        Returns
        ----------
        None
        """

        if other_record_numbers is None:
            self.__reset_record_numbers()

        else:
            for run in other_record_numbers.keys():
                if run in self.__record_numbers.keys():     # If this run is present in both, this WaveformSet and
                                                            # the incoming one, then carefully merge the information
                    
                    self.__record_numbers[run] = self.__record_numbers[run].union(other_record_numbers[run])

                else:                                       # If this run is present in the incoming WaveformSet but not in self, then
                                                            # simply get the information from the incoming WaveformSet as a block

                    self.__record_numbers[run] = other_record_numbers[run]
        return
    
    def __reset_record_numbers(self) -> None:

        """
        This method is not intended for user usage.
        This method must only be called by the
        WaveformSet.__update_record_numbers() method. It clears
        the self.__record_numbers attribute of this object and 
        then iterates through the whole WaveformSet to fill such 
        attribute according to the waveforms which are currently 
        present in this WaveformSet object.
        """

        self.__record_numbers.clear()

        for wf in self.__waveforms:
            try:
                self.__record_numbers[wf.RunNumber].add(wf.RecordNumber)
            except KeyError:
                self.__record_numbers[wf.RunNumber] = set()
                self.__record_numbers[wf.RunNumber].add(wf.RecordNumber)
        return

    def __update_available_channels(self, other_available_channels : Optional[Dict[int, Dict[int, set]]] = None) -> None:
        
        """
        This method is not intended to be called by the user.
        This method updates the self.__available_channels 
        attribute of this object. Its behaviour is different 
        depending on whether the 'other_available_channels' 
        parameter is None or not. Check its documentation for 
        more information.

        Parameters
        ----------
        other_available_channels : dictionary of dictionaries of sets
            If it is None, then this method clears the
            self.__available_channels attribute of this object
            and then iterates through the whole WaveformSet
            to fill such attribute according to the waveforms
            which are currently present in this WaveformSet.
            If the 'other_available_channels' parameter is 
            defined, then it must be a dictionary of dictionaries
            of sets of integers, as expected for the 
            AvailableChannels attribute of a WaveformSet object. 
            In this case, the information in other_available_channels 
            is merged into the self.__available_channels attribute 
            of this object, according to the meaning of the 
            self.__available_channels attribute. The well-
            formedness of this parameter is not checked by this 
            method. It is the caller's responsibility to ensure it.

        Returns
        ----------
        None
        """

        if other_available_channels is None:
            self.__reset_available_channels()

        else:
            for run in other_available_channels.keys():
                if run in self.__available_channels.keys():     # If this run is present in both, this WaveformSet and
                                                                # the incoming one, then carefully merge the information

                    for endpoint in other_available_channels[run].keys():
                        if endpoint in self.__available_channels[run].keys():   # If this endpoint for this run is present
                                                                                # in both waveform sets, then carefully
                                                                                # merge the information.
                            
                            self.__available_channels[run][endpoint] = self.__available_channels[run][endpoint].union(other_available_channels[run][endpoint])

                        else:   # If this endpoint for this run is present in the incoming WaveformSet but not in
                                # self, then simply get the information from the incoming WaveformSet as a block
                                
                            self.__available_channels[run][endpoint] = other_available_channels[run][endpoint]

                else:       # If this run is present in the incoming WaveformSet but not in self, then
                            # simply get the information from the incoming WaveformSet as a block

                    self.__available_channels[run] = other_available_channels[run]
        return
    
    def __reset_available_channels(self) -> None:

        """
        This method is not intended for user usage.
        This method must only be called by the
        WaveformSet.__update_available_channels() method. It clears
        the self.__available_channels attribute of this object and 
        then iterates through the whole WaveformSet to fill such 
        attribute according to the waveforms which are currently 
        present in this WaveformSet object.
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
                        amp_ll : int = 0,
                        amp_ul : Optional[int] = None,
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
        amp_ll (resp. amp_ul): int
            For every analysed waveform, it defines the
            interval considered for the amplitude calculation 
            (it is given to the 'amp_ll' (resp. 'amp_ul') 
            parameter of Waveform.analyse() - actually 
            WaveformAdcs.analyse()). amp_ll must 
            be smaller than amp_ul. These limits are 
            inclusive. If they are not defined, then the
            whole Adcs are considered for each waveform. 
            For more information check the 'amp_ll' and 
            'amp_ul' parameters documentation in the 
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
        amp_ul_ = amp_ul
        if amp_ul_ is None:
            amp_ul_ = self.PointsPerWf - 1

        if not self.subinterval_is_well_formed(amp_ll, amp_ul_):
            raise Exception(generate_exception_message( 3,
                                                        'WaveformSet.analyse()',
                                                        f"The amplitude window ({amp_ll}, {amp_ul_}) is not well formed."))
        aux = WfAna([0,1],  # Dummy object to access
                    0,      # the analyser instance method
                    1,
                    0,
                    1)
        try:
            analyser = getattr(aux, analyser_name)
        except AttributeError:
            raise Exception(generate_exception_message( 4,
                                                        'WaveformSet.analyse()',
                                                        f"The analyser method '{analyser_name}' does not exist in the WfAna class."))
        try:
            signature = inspect.signature(analyser)
        except TypeError:
            raise Exception(generate_exception_message( 5,
                                                        'WaveformSet.analyse()',
                                                        f"'{analyser_name}' does not match a callable attribute of WfAna."))
        try:
            if list(signature.parameters.keys())[0] != 'waveform':
                raise Exception(generate_exception_message( 6,
                                                            "WaveformSet.analyse()",
                                                            "The name of the first parameter of the given analyser method must be 'waveform'."))
            
            if signature.parameters['waveform'].annotation not in ['WaveformAdcs', WaveformAdcs]:
                raise Exception(generate_exception_message( 7,
                                                            "WaveformSet.analyse()",
                                                            "The 'waveform' parameter of the analyser method must be hinted as a WaveformAdcs object."))
            
            if signature.return_annotation != Tuple[WfAnaResult, bool, dict]:
                raise Exception(generate_exception_message( 8,
                                                            "WaveformSet.analyse()",
                                                            "The return type of the analyser method must be hinted as Tuple[WfAnaResult, bool, dict]."))
        except IndexError:
            raise Exception(generate_exception_message( 9,
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
                                                        amp_ll = amp_ll,
                                                        amp_ul = amp_ul_,
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

    def get_map_of_wf_idcs(self,    nrows : int,
                                    ncols : int,
                                    wfs_per_axes : Optional[int] = None,
                                    wf_filter : Optional[Callable[..., bool]] = None,
                                    filter_args : Optional[Map] = None,
                                    max_wfs_per_axes : Optional[int] = 5) -> Map:
        
        """
        This method returns a Map of lists of integers,
        i.e. a Map object whose Type attribute equals 
        list. The contained integers should be interpreted 
        as iterator values for waveforms in this WaveformSet 
        object.

        Parameters
        ----------
        nrows : int
            The number of rows of the returned Map object
        ncols : int
            The number of columns of the returned Map object
        wfs_per_axes : int
            If it is not None, then it must be a positive
            integer which is smaller or equal to
            math.floor(len(self.Waveforms) / (nrows * ncols)),
            so that the iterator values contained 
            in the output Map are contiguous in
            [0, nrows*ncols*wfs_per_axes - 1]. I.e.
            output.Data[0][0] contains 0, 1, ... , wfs_per_axes - 1,
            output.Data[0][1] contains wfs_per_axes, wfs_per_axes + 1,
            ... , 2*wfs_per_axes - 1, and so on. 
        wf_filter : callable
            This parameter only makes a difference if
            the 'wfs_per_axes' parameter is None. In such
            case, this one must be a callable object whose 
            first parameter must be called 'waveform' and 
            must be hinted as a Waveform object. Also, the
            return type of such callable must be annotated
            as a boolean. If wf_filter is 
                - wuf.match_run or
                - wuf.match_endpoint_and_channel,
            this method can benefit from the information in
            self.Runs and self.AvailableChannels and its
            execution time may be reduced with respect to
            the case where an arbitrary (but compliant) 
            callable is passed to wf_filter.
        filter_args : Map
            This parameter only makes a difference if 
            the 'wfs_per_axes' parameter is None. In such
            case, this parameter must be defined and
            it must be a Map object whose Rows (resp.
            Columns) attribute match nrows (resp. ncols).
            Its Type attribute must be list. 
            filter_args.Data[i][j], for all i and j, is 
            interpreted as a list of arguments which will 
            be given to wf_filter at some point. The user 
            is responsible for giving a set of arguments 
            which comply with the signature of the 
            specified wf_filter. For more information 
            check the return value documentation.
        max_wfs_per_axes : int
            This parameter only makes a difference if           ## If max_wfs_per_axes applies and 
            the 'wfs_per_axes' parameter is None. In such       ## is a positive integer, it is never
            case, and if 'max_wfs_per_axes' is not None,        ## checked that there are enough waveforms
            then output.Data[i][j] will contain the indices     ## in the WaveformSet to fill the map.
            for the first max_wfs_per_axes waveforms in this    ## This is an open issue.
            WaveformSet which passed the filter. If it is 
            None, then this function iterates through the 
            whole WaveformSet for every i,j pair. Note that 
            setting this parameter to None may result in a 
            long execution time for big waveform sets.

        Returns
        ----------
        output : Map
            It is a Map object whose Type attribute is list.
            Namely, output.Data[i][j] is a list of integers.
            If the 'wfs_per_axes' parameter is defined, then
            the iterator values contained in the output Map 
            are contiguous in [0, nrows*ncols*wfs_per_axes - 1].
            For more information, check the 'wfs_per_axes'
            parameter documentation. If the 'wfs_per_axes'
            is not defined, then the 'wf_filter' and 'filter_args'
            parameters must be defined and output.Data[i][j] 
            gives the indices of the waveforms in this WaveformSet 
            object, say wf, for which 
            wf_filter(wf, *filter_args.Data[i][j]) returns True.
            In this last case, the number of indices in each
            entry may be limited, up to the value given to the 
            'max_wfs_per_axes' parameter.
        """

        if nrows < 1 or ncols < 1:
            raise Exception(generate_exception_message( 1,
                                                        'WaveformSet.get_map_of_wf_idcs()',
                                                        'The number of rows and columns must be positive.'))
        fFilteringMode = True
        if wfs_per_axes is not None:
            if wfs_per_axes < 1 or wfs_per_axes > math.floor(len(self.__waveforms) / (nrows * ncols)):
                raise Exception(generate_exception_message( 2,
                                                            'WaveformSet.get_map_of_wf_idcs()',
                                                            f"The given wfs_per_axes ({wfs_per_axes}) must belong to the range [1, math.floor(len(self.__waveforms) / (nrows * ncols))] (={[1, math.floor(len(self.__waveforms) / (nrows * ncols))]})."))
            fFilteringMode = False

        fMaxIsSet = None    # This one should only be defined as
                            # a boolean if fFilteringMode is True
        if fFilteringMode:

            try:
                signature = inspect.signature(wf_filter)
            except TypeError:
                raise Exception(generate_exception_message( 3,
                                                            'WaveformSet.get_map_of_wf_idcs()',
                                                            "The given wf_filter is not defined or is not callable. It must be suitably defined because the 'wfs_per_axes' parameter is not. At least one of them must be suitably defined."))

            wuf.check_well_formedness_of_generic_waveform_function(signature)

            if filter_args is None:
                raise Exception(generate_exception_message( 4,
                                                            'WaveformSet.get_map_of_wf_idcs()',
                                                            "The 'filter_args' parameter must be defined if the 'wfs_per_axes' parameter is not."))
            
            elif not Map.list_of_lists_is_well_formed(  filter_args.Data,
                                                        nrows,
                                                        ncols):
                    
                    raise Exception(generate_exception_message( 5,
                                                                'WaveformSet.get_map_of_wf_idcs()',
                                                                f"The shape of the given filter_args list is not nrows ({nrows}) x ncols ({ncols})."))
            fMaxIsSet = False
            if max_wfs_per_axes is not None:
                if max_wfs_per_axes < 1:
                    raise Exception(generate_exception_message( 6,
                                                                'WaveformSet.get_map_of_wf_idcs()',
                                                                f"The given max_wfs_per_axes ({max_wfs_per_axes}) must be positive."))
                fMaxIsSet = True

        if not fFilteringMode:

            return WaveformSet.get_contiguous_indices_map(  wfs_per_axes,
                                                            nrows = nrows,
                                                            ncols = ncols)
            
        else:   # fFilteringMode is True and so, wf_filter, 
                # filter_args and fMaxIsSet are defined

            mode_map = {wuf.match_run : 0,
                        wuf.match_endpoint_and_channel : 1}
            try:
                fMode = mode_map[wf_filter]
            except KeyError:
                fMode = 2

            output = Map.from_unique_value( nrows,
                                            ncols,
                                            list,
                                            [],
                                            independent_copies = True)
            if fMode == 0:
                return self.__get_map_of_wf_idcs_by_run(output,
                                                        filter_args,
                                                        fMaxIsSet,
                                                        max_wfs_per_axes)
            elif fMode == 1:
                return self.__get_map_of_wf_idcs_by_endpoint_and_channel(   output,
                                                                            filter_args,
                                                                            fMaxIsSet,
                                                                            max_wfs_per_axes)
            else:
                return self.__get_map_of_wf_idcs_general(   output,
                                                            wf_filter,
                                                            filter_args,
                                                            fMaxIsSet,
                                                            max_wfs_per_axes)
    
    def __get_map_of_wf_idcs_by_endpoint_and_channel(self,  blank_map : Map,
                                                            filter_args : ChannelMap,
                                                            fMaxIsSet : bool,
                                                            max_wfs_per_axes : Optional[int] = 5) -> Map:
        
        """
        This method should only be called by the 
        WaveformSet.get_map_of_wf_idcs() method, where 
        the well-formedness checks of the input have 
        already been performed. This method generates an 
        output as described in such method docstring,
        for the case when wf_filter is 
        wuf.match_endpoint_and_channel. Refer to
        the WaveformSet.get_map_of_wf_idcs() method
        documentation for more information.

        Parameters
        ----------
        blank_map : Map
        filter_args : ChannelMap
        fMaxIsSet : bool
        max_wfs_per_axes : int

        Returns
        ----------
        Map
        """

        aux = self.get_run_collapsed_available_channels()

        for i in range(blank_map.Rows):
            for j in range(blank_map.Columns):

                if filter_args.Data[i][j].Endpoint not in aux.keys():
                    continue

                elif filter_args.Data[i][j].Channel not in aux[filter_args.Data[i][j].Endpoint]:
                    continue    
          
                if fMaxIsSet:   # blank_map should not be very big (visualization purposes)
                                # so we can afford evaluating the fMaxIsSet conditional here
                                # instead of at the beginning of the method (which would
                                # be more efficient but would entail a more extensive code)

                    counter = 0
                    for k in range(len(self.__waveforms)):
                        if wuf.match_endpoint_and_channel(  self.__waveforms[k],
                                                            filter_args.Data[i][j].Endpoint,
                                                            filter_args.Data[i][j].Channel):
                            blank_map.Data[i][j].append(k)
                            counter += 1
                            if counter == max_wfs_per_axes:
                                break
                else:
                    for k in range(len(self.__waveforms)):
                        if wuf.match_endpoint_and_channel(  self.__waveforms[k],
                                                            filter_args.Data[i][j].Endpoint,
                                                            filter_args.Data[i][j].Channel):
                            blank_map.Data[i][j].append(k)
        return blank_map
    
    def __get_map_of_wf_idcs_general(self,  blank_map : Map,
                                            wf_filter : Callable[..., bool],
                                            filter_args : Map,
                                            fMaxIsSet : bool,
                                            max_wfs_per_axes : Optional[int] = 5) -> List[List[List[int]]]:
        
        """
        This method should only be called by the 
        WaveformSet.get_map_of_wf_idcs() method, where 
        the well-formedness checks of the input have 
        already been performed. This method generates an 
        output as described in such method docstring,
        for the case when wf_filter is neither
        wuf.match_run nor wuf.match_endpoint_and_channel. 
        Refer to the WaveformSet.get_map_of_wf_idcs() 
        method documentation for more information.

        Parameters
        ----------
        blank_map : Map
        wf_filter : callable
        filter_args : Map
        fMaxIsSet : bool
        max_wfs_per_axes : int

        Returns
        ----------
        list of list of list of int
        """

        for i in range(blank_map.Rows):
            for j in range(blank_map.Columns):

                if fMaxIsSet:
                    counter = 0
                    for k in range(len(self.__waveforms)):
                        if wf_filter(   self.__waveforms[k],
                                        *filter_args.Data[i][j]):
                            
                            blank_map.Data[i][j].append(k)
                            counter += 1
                            if counter == max_wfs_per_axes:
                                break
                else:
                    for k in range(len(self.__waveforms)):
                        if wf_filter(   self.__waveforms[k],
                                        *filter_args.Data[i][j]):
                            blank_map.Data[i][j].append(k)
        return blank_map
                            
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

            wuf.check_well_formedness_of_generic_waveform_function(signature)

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

        wuf.check_well_formedness_of_generic_waveform_function(signature)
        
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

            self.__update_runs(other_runs = None)                               # If actually_filter, then we need to update 
            self.__update_record_numbers(other_record_numbers = None)           # the self.__runs, self.__record_numbers and 
            self.__update_available_channels(other_available_channels = None)   # self.__available_channels

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
    def histogram1d(samples : np.ndarray,
                    bins : int,
                    domain : np.ndarray,
                    keep_track_of_idcs : bool = False) -> Tuple[np.ndarray, List[List[int]]]:   # Not calling it 'range' because 
                                                                                                # it is a reserved keyword in Python

        """
        This method returns a tuple with two elements. The
        first one is an unidimensional integer numpy 
        array, say counts, which is the 1D histogram of the 
        given samples. I.e. counts[i] gives the number of
        samples that fall into the i-th bin of the histogram,
        with i = 0, 1, ..., bins - 1. The second element
        of the returned tuple is a list containing bins 
        empty lists. If keep_track_of_idcs is True, then 
        the returned list of lists contains integers, so 
        that the i-th list contains the indices of the 
        samples which fall into the i-th bin of the 
        histogram. It is the caller's responsibility to 
        make sure that the given input parameters are 
        well-formed. No checks are performed here.

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
        keep_track_of_idcs : bool
            If True, then the second element of the returned
            tuple is not empty

        Returns
        ----------
        counts : np.ndarray
            An unidimensional integer numpy array which 
            is the 1D histogram of the given samples
        idcs : list of list of int
            A list containing bins empty lists. If 
            keep_track_of_idcs is True, then the i-th 
            list contains the indices of the samples,
            with respect to the input samples array,
            which fall into the i-th bin of the histogram.
        """

        counts, formatted_idcs = WaveformSet.__histogram1d( samples,
                                                            bins,
                                                            domain,
                                                            keep_track_of_idcs = keep_track_of_idcs)
        deformatted_idcs = [ [] for _ in range(bins) ]

        if keep_track_of_idcs:
            for i in range(0, len(formatted_idcs), 2):
                deformatted_idcs[formatted_idcs[i + 1]].append(formatted_idcs[i])

        return counts, deformatted_idcs
    
    @staticmethod
    @numba.njit(nogil=True, parallel=False)
    def __histogram1d(  samples : np.ndarray,
                        bins : int,
                        domain : np.ndarray,
                        keep_track_of_idcs : bool = False) -> Tuple[np.ndarray, List[List[int]]]:   # Not calling it 'range' because 
                                                                                                    # it is a reserved keyword in Python
        """
        This method is not intended for user usage. It 
        must only be called by the WaveformSet.histogram1d() 
        method. This is the low-level optimized numerical 
        implementation of the histogramming process.

        Parameters
        ----------
        samples : np.ndarray
        bins : int
        domain : np.ndarray
        keep_track_of_idcs : bool

        Returns
        ----------
        counts : np.ndarray
            An unidimensional integer numpy array which 
            is the 1D histogram of the given samples
        formatted_idcs : list of int
            A list of integers. If keep_track_of_idcs is
            False, then this list is empty. If it is True,
            then this list is 2*len(samples) long at most.
            This list is such that formatted_idcs[2*i]
            gives the index of the i-th sample from
            samples which actually fell into the 
            specified domain, and formatted_idcs[2*i + 1] 
            gives the index of the bin where such sample 
            falls into.
        """

        # Of course, building a list of lists with the indices of the 
        # samples which fell into each bin would be more conceptually 
        # stragihtforward, but appending to a list which is contained 
        # within another list is apparently not allowed within a numba.njit 
        # function. So, we are using this format, which only implies 
        # appending to a flat list, and it will be latter de-formatted 
        # into a list of lists in the upper level WaveformSet.histogram1d() 
        # method, which is not numba decorated and should not perform
        # very demanding operations.

        counts = np.zeros(bins, dtype = np.uint64)
        formatted_idcs = []

        inverse_step = 1. / ((domain[1] - domain[0]) / bins)

        if not keep_track_of_idcs:
            for t in range(samples.shape[0]):
                i = (samples[t] - domain[0]) * inverse_step

                if 0 <= i < bins:
                    counts[int(i)] += 1
        else:
            for t in range(samples.shape[0]):
                i = (samples[t] - domain[0]) * inverse_step

                if 0 <= i < bins:
                    aux = int(i)
                    counts[aux] += 1

                    formatted_idcs.append(t)
                    formatted_idcs.append(aux)

        return counts, formatted_idcs
    
    ## WaveformSet.plot_calibration_histogram() is not supported anymore,
    ## and so, it may not work. ChannelWSGrid.plot() with its 'mode' 
    ## input parameter set to calibration already covers this feature. 
    ## It is not worth the effort to fix this method for WaveformSet, 
    ## it will be deleted soon.

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
            the Analyses attribute will be the used one.
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

            grid_of_wf_idcs_ = self.get_map_of_wf_idcs( nrows,
                                                        ncols,
                                                        wfs_per_axes = wfs_per_axes)

        elif grid_of_wf_idcs is None:   # Nor wf_per_axes, nor 
                                        # grid_of_wf_idcs are defined

            raise Exception(generate_exception_message( 5,
                                                        'WaveformSet.plot_calibration_histogram()',
                                                        "The 'grid_of_wf_idcs' parameter must be defined if wfs_per_axes is not."))
        
        elif not Map.list_of_lists_is_well_formed(  grid_of_wf_idcs,    # wf_per_axes is not defined, 
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
                         
                    data, _ = WaveformSet.histogram1d(  np.array([self.Waveforms[idc].get_analysis(analysis_label).Result.Integral for idc in grid_of_wf_idcs_[i][j]]), ## Trying to grab the WfAna object
                                                        bins,                                                                                                           ## waveform by waveform using 
                                                        domain,                                                                                                         ## WaveformAdcs.get_analysis()
                                                        keep_track_of_idcs = False)                                                                                     ## might be slow. Find a different
                                                                                                                                                                        ## solution if this becomes a 
                    figure.add_trace(   pgo.Scatter(    x = np.linspace(domain[0] + (step / 2.0),                                                                       ## a problem at some point.
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

    def merge(self, other : 'WaveformSet') -> None:

        """
        This method merges the given other WaveformSet
        object into this WaveformSet object. For every
        waveform in the given other WaveformSet object,
        it is appended to the list of waveforms of this
        WaveformSet object. The self.__runs, 
        self.__record_numbers and self.__available_channels
        are updated accordingly. The self.__mean_adcs and
        self.__mean_adcs_idcs are reset to None.

        Parameters
        ----------
        other : WaveformSet
            The WaveformSet object to be merged into this
            WaveformSet object. The PointsPerWf attribute
            of the given WaveformSet object must be equal
            to the PointsPerWf attribute of this WaveformSet
            object. Otherwise, an exception is raised.

        Returns
        ----------
        None
        """

        if other.PointsPerWf != self.PointsPerWf:
            raise Exception(generate_exception_message( 1,
                                                        'WaveformSet.merge()',
                                                        f"The given WaveformSet object has waveforms with lengths ({other.PointsPerWf}) different to the ones in this WaveformSet object ({self.PointsPerWf})."))
        for wf in other.Waveforms:
            self.__waveforms.append(wf)

        self.__update_runs(other_runs = other.Runs)
        self.__update_record_numbers(other_record_numbers = other.RecordNumbers)
        self.__update_available_channels(other_available_channels = other.AvailableChannels)

        self.__mean_adcs = None
        self.__mean_adcs_idcs = None

        return
