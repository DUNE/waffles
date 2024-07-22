import numpy as np
from typing import List, Optional
from collections import OrderedDict
from plotly import graph_objects as pgo

from waffles.data_classes.WfAna import WfAna
from waffles.Exceptions import generate_exception_message

class WaveformAdcs:

            # It is useful to have such a class so that tools which only need the Adcs information
            # can be run even in situations where a waveform does not have a defined timestamp,
            # endpoint or any other attribute which could be used to identify a waveform at a higher 
            # level. For example, the waveform which is the result of a averaging over every waveform
            # for a certain channel could be analyzed so as to compute its baseline, but its timestamp
            # is not defined, i.e. it makes no sense.

    """
    This class implements the Adcs array of a waveform.

    Attributes
    ----------
    TimeStep_ns : float
        The time step (in nanoseconds) for this waveform
    Adcs : unidimensional numpy array of integers
        The readout for this waveform, in # of ADCs
    TimeOffset : int
        A time offset, in units of TimeStep_ns (i.e.
        time ticks) which will be used as a relative
        alignment among different WaveformAdcs
        objects for plotting and analysis purposes. 
        It must be semipositive and smaller than 
        len(self.__adcs)-1. It is set to 0 by default.
    Analyses : OrderedDict of WfAna objects

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """
                                # The restrictions over the TimeOffset attribute
                                # ensure that there are always at least two points
                                # left in the [0, 1, ..., len(self.__adcs) - 1] range,
                                # so that baselines, integrals and amplitudes can be 
                                # computed using points in that range.

    def __init__(self,  time_step_ns : float,
                        adcs : np.ndarray,
                        time_offset : int = 0):
        
        """
        WaveformAdcs class initializer
        
        Parameters
        ----------
        time_step_ns : float
        adcs : unidimensional numpy array of integers
        time_offset : int
            It must be semipositive and smaller than 
            len(self.__adcs)-1. It is set to 0 by 
            default.
        """

        ## Shall we add add type checks here?

        self.__time_step_ns = time_step_ns
        self.__adcs = adcs
        self.__set_time_offset(time_offset)      # WaveformSet._set_time_offset() 
                                                # takes care of the proper checks
       
        self.__analyses = OrderedDict() # Initialize the analyses 
                                        # attribute as an empty 
                                        # OrderedDict.

        ## Do we need to add trigger primitives as attributes?

    #Getters
    @property
    def TimeStep_ns(self):
        return self.__time_step_ns
    
    @property
    def Adcs(self):
        return self.__adcs
    
    @property
    def TimeOffset(self):
        return self.__time_offset
    
    @property
    def Analyses(self):
        return self.__analyses
    
#   #Setters                            # For the moment there are no setters for 
#   @TimeStep_ns.setter                 # the attributes of WaveformAdcs. I.e. you
#   def TimeStep_ns(self, input):       # can only set the value of its attributes
#       self.__time_step_ns = input     # through WaveformAdcs.__init__. Here's an
#       return                          # example of what a setter would look like, though.

    def __set_time_offset(self, input : float) -> None:

        """
        This method is not intended for user usage. It is 
        a setter for the TimeOffset attribute. 
        
        Parameters
        ----------
        input : float
            The value which will be assigned to the TimeOffset
            attribute. It must be semipositive and smaller than 
            len(self.__adcs)-1.
        
        Returns
        ----------
        None
        """

        if input < 0 or input >= len(self.__adcs)-1:
            
            raise Exception(generate_exception_message( 1,
                                                        'WaveformAdcs.__set_time_offset()',
                                                        f"The given time offset ({input}) must belong to the [0, {len(self.__adcs)-2}] interval."))
        else:
            self.__time_offset = input

        return

    def __truncate_adcs(self, number_of_points_to_keep : int) -> None:

        """
        This method is not intended for user usage. It truncates 
        the self.__adcs attribute array to the first 
        'number_of_points_to_keep' points.

        Parameters
        ----------
        number_of_points_to_keep : int

        Returns
        ----------
        None
        """

        self.__adcs = self.__adcs[:number_of_points_to_keep]    # Numpy handles the case where number_of_points_to_keep
                                                                # is greater than the length of self.__adcs.

    def confine_iterator_value(self, input : int) -> int:

        """
        Confines the input integer to the range [0, len(self.__adcs) - 1].
        I.e returns 0 if input is negative, returns input if input belongs
        to the range [0, len(self.__adcs) - 1], and returns len(self.__adcs) - 1
        in any other case.

        Parameters
        ----------
        input : int

        Returns
        ----------
        int
        """
    
        if input < 0:
            return 0
        elif input < len(self.__adcs):
            return input
        else:
            return len(self.__adcs) - 1

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
        This method creates a WfAna object and adds it to the
        self.__analyses dictionary using label as its key.
        To do so, it grabs the WfAna instance method whose name
        matches analyser_name and runs it on this WaveformAdcs
        object. Then, this method does two things:
        
            -   first, it adds the two first outputs of such
                analyser method to the 'Result' and 'Passed' 
                attributes of the newly created WfAna object,
                respectively.
            -   second, it returns the third output of the
                analyser method, which should be a dictionary
                containing any additional information that the
                analyser method wants to return. Such dictionary
                is empty if no additional information is
                provided by the analyser method.

        Parameters
        ----------
        label : str
            Key for the new WfAna object within the self.__analyses
            OrderedDict
        analyser_name : str
            It must match the name of a WfAna method whose first
            argument must be called 'waveform' and whose type   
            annotation must match the WaveformAdcs class or the     
            'WaveformAdcs' string literal. Such method should also  
            have a defined return-annotation which must match   
            Tuple[WfAnaResult, bool, dict]. It is the caller's
            responsibility to check such conditions for this
            parameter. No checks are performed here for this
            input.
        baseline_limits : list of int                                   
            Given to the 'baseline_limits' parameter of                 
            WfAna.__init__. It must have an even number
            of integers which must meet 
            baseline_limits[i] < baseline_limits[i + 1] for
            all i. The points which are used for 
            baseline calculation are 
            self.__adcs[baseline_limits[2*i]:baseline_limits[(2*i) + 1]],
            with i = 0,1,...,(len(baseline_limits)/2) - 1. 
            The upper limits are exclusive. It is the
            caller's responsibility to ensure the
            well-formedness of this input. No checks are
            performed here for 'baseline_limits'.
        *args
            Positional arguments which are given to the 
            analyser method.
        int_ll (resp. int_ul): int
            Given to the 'int_ll' (resp. 'int_ul') parameter of
            WfAna.__init__. Iterator value for the first (resp. 
            last) point of self.__adcs that falls into the 
            integration window. int_ll must be smaller than 
            int_ul. These limits are inclusive. If they are 
            not defined, then the whole self.__adcs is considered.
            It is the caller's responsibility to ensure the 
            well-formedness of this input. No checks are
            performed here for this parameter.
        amp_ll (resp. amp_ul): int
            Given to the 'amp_ll' (resp. 'amp_ul') parameter of
            WfAna.__init__. Iterator value for the first (resp. 
            last) point of self.__adcs that is considered for
            the amplitude calculation. amp_ll must be smaller 
            than amp_ul. These limits are inclusive. If they are 
            not defined, then the whole self.__adcs is considered.
            It is the caller's responsibility to ensure the 
            well-formedness of this input. No checks are
            performed here for this parameter.
        overwrite : bool
            If True, the method will overwrite any existing
            WfAna object with the same label (key) within
            self.__analyses.
        **kwargs
            Keyword arguments which are given to the analyser
            method.

        Returns
        ----------
        output_3 : dict
            The third output of the analyser method, which
            should be a dictionary containing any additional
            information that the analyser method wants to
            return. Note that the analyser method must return
            a dictionary as its third output, even it its
            an empty one.
        """

        if label in self.__analyses.keys() and not overwrite:
            raise Exception(generate_exception_message( 1,
                                                        'WaveformAdcs.analyse()',
                                                        f"There is already an analysis with label '{label}'. If you want to overwrite it, set the 'overwrite' parameter to True."))
        else:

            aux = WfAna(baseline_limits,
                        int_ll,
                        int_ul,
                        amp_ll,
                        amp_ul)
            
            analyser = getattr(aux, analyser_name)

            output_1, output_2, output_3 = analyser(self,   *args, 
                                                            **kwargs)
            aux.Result = output_1
            aux.Passed = output_2

            self.__analyses[label] = aux

            return output_3
    
    def get_analysis(self, label : Optional[str] = None) -> WfAna:

        """
        If the 'label' parameter is defined, then this 
        method returns the WfAna object which has such 
        label within the self.__analyses OrderedDict. 
        If there is no analysis which such label, then
        this method raises a KeyError. If the 'label'
        parameter is not defined, then this method returns
        the last WfAna object added to self.__analyses. If
        there are no analyses, then this method raises an
        exception.

        Parameters
        ----------
        label : str
            The key for the WfAna object within the
            self.__analyses OrderedDict.

        Returns
        ----------
        output : WfAna
            The WfAna object which has the given label
        """

        if label is None:
            try:
                output = next(reversed(self.__analyses.values()))  # Grabbing the last analysis
            except StopIteration:
                raise Exception(generate_exception_message( 1,
                                                            'WaveformAdcs.get_analysis()',
                                                            'The waveform has not been analysed yet.'))
        else:
            try:
                output = self.__analyses[label]
            except KeyError:
                raise Exception(generate_exception_message( 2,
                                                            'WaveformAdcs.get_analysis()',
                                                            f"There is no analysis with label '{label}'."))
        return output