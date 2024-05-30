import inspect
from typing import Tuple, List, Optional
from collections import OrderedDict

import numpy as np

from src.waffles.WfAna import WfAna
from src.waffles.WfAnaResult import WfAnaResult
from src.waffles.Exceptions import generate_exception_message

class Waveform:

    """
    This class implements a waveform.

    Attributes
    ----------
    Timestamp : int
        The timestamp value for this waveform
    TimeStep_ns : float
        The time step (in nanoseconds) for this waveform
    Adcs : unidimensional numpy array of integers
        The readout for this waveform, in # of ADCs
    RunNumber : int
        Number of the run from which this waveform was
        acquired
    Endpoint : int
        Endpoint number from which this waveform was
        acquired
    Channel : int
        Channel number for this waveform
    Analyses : OrderedDict of WfAna objects. 

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    def __init__(self,  timestamp : int,
                        time_step_ns : float,
                        adcs : np.ndarray,
                        run_number : int,
                        endpoint : int,
                        channel : int):
        
        """
        Waveform class initializer
        
        Parameters
        ----------
        timestamp : int
        time_step_ns : float
        adcs : unidimensional numpy array of integers
        run_number : int
        endpoint : int
        channel : int
        """

        ## Shall we add add type checks here?
    
        self.__timestamp = timestamp
        self.__time_step_ns = time_step_ns
        self.__adcs = adcs
        self.__run_number = run_number
        self.__endpoint = endpoint
        self.__channel = channel
        self.__analyses = OrderedDict() # Initialize the analyses 
                                        # attribute as an empty 
                                        # OrderedDict.

        ## Do we need to add trigger primitives as attributes?
    
    #Getters
    @property
    def Timestamp(self):
        return self.__timestamp
    
    @property
    def TimeStep_ns(self):
        return self.__time_step_ns
    
    @property
    def Adcs(self):
        return self.__adcs
    
    @property
    def RunNumber(self):
        return self.__run_number
    
    @property
    def Endpoint(self):
        return self.__endpoint
    
    @property
    def Channel(self):
        return self.__channel
    
    @property
    def Analyses(self):
        return self.__analyses
    
#   #Setters                       # For the moment there are no setters for 
#   @Timestamp.setter              # the attributes of Waveform. I.e. you can
#   def Timestamp(self, input):    # only set the value of its attributes
#       self.__timestamp = input     # through Waveform.__init__. Here's an example
#       return                     # of what a setter would look like, though.

    def confine_iterator_value(self, input : int) -> int:

        """
        Confines the input integer to the range [0, len(self.__adcs)-1].
        I.e returns 0 if input is negative, returns input if input belongs
        to the range [0, len(self.__adcs)-1], and returns len(self.__adcs)-1
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
            return len(self.__adcs)-1

    def analyse(self,   label : str,
                        analyser_name : str,
                        baseline_limits : List[int],
                        int_ll : int = 0,
                        int_ul : Optional[int] = None,
                        *args,
                        overwrite : bool = False,
                        **kwargs) -> None:

        """
        This method creates a WfAna object and adds it to the
        self.__analyses dictionary using label as its key.
        This method grabs the analyser method from the WfAna
        class, up to the given analyser_name, runs it on this 
        Waveform object, and adds its results to the 'Result' 
        and 'Passed' attributes of the newly created WfAna object.

        Parameters
        ----------
        label : str
            Key for the new WfAna object within the self.__analyses
            OrderedDict
        analyser_name : str
            It must match the name of a WfAna method whose first 
            argument must be called 'waveform' and must be hinted 
            as a Waveform object. Such method should also have a
            defined return-annotation which must match
            Tuple[WfAnaResult, bool].
        baseline_limits : list of int
            Given to the 'baseline_limits' parameter of 
            WfAna.__init__. It must have an even number 
            of integers which must meet 
            baseline_limits[i] < baseline_limits[i+1] for
            all i. The points which are used for 
            baseline calculation are 
            self.__adcs[baseline_limits[2*i]:baseline_limits[2*i+1]],
            with i = 0,1,...,(len(baseline_limits)/2)-1. 
            The upper limits are exclusive.
        int_ll (resp. int_ul): int
            Given to the 'int_ll' (resp. 'int_ul') parameter of
            WfAna.__init__. Iterator value for the first (resp. 
            last) point of self.Adcs that falls into the 
            integration window. int_ll must be smaller than 
            int_ul. These limits are inclusive. If they are 
            not defined, then the whole self.Adcs is considered.
        *args
            Positional arguments which are given to the 
            analyser method.
        overwrite : bool
            If True, the method will overwrite any existing
            WfAna object with the same label (key) within
            self.__analyses.
        **kwargs
            Keyword arguments which are given to the analyser
            method.

        Returns
        ----------
        None
        """

        if label in self.__analyses.keys() and not overwrite:
            raise Exception(generate_exception_message( 1,
                                                        'Waveform.analyse()',
                                                        f"There is already an analysis with label '{label}'. If you want to overwrite it, set the 'overwrite' parameter to True."))
        else:

            ## *DISCLAIMER: The following two 'if' statements might make the run time go 
            ## prohibitively high when running analyses sequentially over a large WaveformSet. 
            ## If that's the case, these checks might be implemented at the WaveformSet level, 
            ## or simply removed.

            if not self.baseline_limits_are_well_formed(baseline_limits):
                raise Exception(generate_exception_message( 2,
                                                            'Waveform.analyse()',
                                                            f"The baseline limits ({baseline_limits}) are not well formed."))
            int_ul_ = int_ul
            if int_ul_ is None:
                int_ul_ = len(self.__adcs)-1

            if not self.subinterval_is_well_formed(int_ll, int_ul_):
                raise Exception(generate_exception_message( 3,
                                                            'Waveform.analyse()',
                                                            f"The integration window ({int_ll}, {int_ul_}) is not well formed."))
            aux = WfAna(baseline_limits,
                        int_ll,
                        int_ul_)
            try:
                analyser = getattr(aux, analyser_name)
            except AttributeError:
                raise Exception(generate_exception_message( 4,
                                                            'Waveform.analyse()',
                                                            f"The analyser method '{analyser_name}' does not exist in the WfAna class."))
            try:
                signature = inspect.signature(analyser)
            except TypeError:
                raise Exception(generate_exception_message( 5,
                                                            'Waveform.analyse()',
                                                            f"'{analyser_name}' does not match a callable attribute of WfAna."))
            try:

                ## DISCLAIMER: Same problem here for the following
                ## three 'if' statements as for the disclaimer above.

                if list(signature.parameters.keys())[0] != 'waveform':
                    raise Exception(generate_exception_message( "Waveform.analyse",
                                                                6,
                                                                "The name of the first parameter of the given analyser method must be 'waveform'."))
                if signature.parameters['waveform'].annotation != Waveform:
                    raise Exception(generate_exception_message( "Waveform.analyse",
                                                                7,
                                                                "The 'waveform' parameter of the analyser method must be hinted as a Waveform object."))
                
                if signature.return_annotation != Tuple[WfAnaResult, bool]:
                    raise Exception(generate_exception_message( "Waveform.analyse",
                                                                8,
                                                                "The return type of the analyser method must be hinted as Tuple[WfAnaResult, bool]."))
            except IndexError:
                raise Exception(generate_exception_message( "Waveform.analyse",
                                                            9,
                                                            "The given filter must take at least one parameter."))
            output_1, output_2 = analyser(*args, **kwargs)

            aux.Result = output_1
            aux.Passed = output_2

            self.__analyses[label] = aux

            return
        
    def subinterval_is_well_formed(self,    i_low : int, 
                                            i_up : int) -> bool:
        
        """
        This method returns True if 0 <= i_low < i_up <= len(self.__adcs)-1,
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
        elif i_up > len(self.__adcs)-1:
            return False
        
        return True
    
    def baseline_limits_are_well_formed(self, baseline_limits : List[int]) -> bool:

        """
        This method returns True if len(baseline_limits) is even and 
        0 <= baseline_limites[0] < baseline_limits[1] < ... < baseline_limits[-1] <= len(self.__adcs)-1.
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
            
        for i in range(0, len(baseline_limits)-1):
            if baseline_limits[i] >= baseline_limits[i+1]:
                return False
                
        if baseline_limits[-1] > len(self.__adcs)-1:
            return False
        
        return True
        
    def get_global_channel(self):

        """
        Returns
        ----------
        int
            An integer value for the readout channel with respect to a numbering 
            scheme which identifies the endpoint and the APA channel at the same
            time
        """

        pass