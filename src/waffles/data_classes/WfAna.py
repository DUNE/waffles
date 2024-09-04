from abc import ABC, abstractmethod

from waffles.data_classes.WaveformAdcs import WaveformAdcs
from waffles.data_classes.IPDict import IPDict
from waffles.data_classes.WfAnaResult import WfAnaResult

import waffles.Exceptions as we


class WfAna(ABC):
    """Stands for Waveform Analysis. This abstract
    class is intended to be the base class for
    any class which implements a certain type of
    analysis which is performed over an arbitrary
    WaveformAdcs object.

    Attributes
    ----------
    InputParameters: IPDict
        An IPDict object (a dictionary) containing the
        input parameters of this analysis. The keys (resp.
        values) are the names (resp. values) of the input
        parameters.
    Result: WfAnaResult
        A WfAnaResult object (a dictionary) containing
        the result of the analysis

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    def __init__(self, input_parameters: IPDict):
        """WfAna class initializer. It is assumed that it is
        the caller responsibility to check the well-formedness
        of the input parameter, according to the attributes
        documentation in the class documentation. No checks
        are perfomed here.

        Parameters
        ----------
        input_parameters: IPDict
        """

        self.__input_parameters = input_parameters

        # To be determined a posteriori
        # by the analyse() instance method
        self.__result = None


    # Getters
    @property
    def InputParameters(self):
        return self.__input_parameters

    @property
    def Result(self):
        return self.__result

    # Not adding setters for the attributes
    # of this class:
    # - InputParameters should be fixed
    #   since the initialization
    # - Result should be set by the
    #   analyse() instance method

    @abstractmethod
    def analyse(
        self,
        waveform: WaveformAdcs,
        *args,
        **kwargs
    ):
        """This abstract method serves as a template 
        for the analyser method that MUST be implemented
        for whichever derived class of WfAna. This
        method must be responsible for creating an
        object of class WfAnaResult and assigning it
        to self.__result.

        Parameters
        ----------
        waveform: WaveformAdcs
            The WaveformAdcs object which will be
            analysed
        *args
            Additional positional arguments
        **kwargs
            Additional keyword arguments

        Returns
        ----------
        None
        """

        # Maybe call here a number of helper
        # methods to perform an analysis and
        # create the WfAnaResult object

        self.__result = WfAnaResult()
        return

    @staticmethod
    @abstractmethod
    @we.handle_missing_data
    def check_input_parameters(input_parameters: IPDict) -> None:
        """This abstract method, which MUST be 
        implemented for whichever derived class of 
        WfAna, is responsible for checking whether 
        the input parameters are well-formed. It should 
        raise an exception if the input parameters are 
        not well-formed. It should end execution normally
        and return None if they are well-formed.

        For efficiency purposes, the aim is to call
        it at the WaveformSet level (before instantiating
        the first WfAna (or derived) object) for
        cases where the same IPDict is given for every
        WaveformAdcs object to be analysed. In these
        cases, just one check is performed, instead
        of N checks, where N is the number of WaveformAdcs
        objects and N-1 checks are redundant. That's the
        reason why this method is static.

        Parameters
        ----------
        input_parameters: IPDict
            The input parameters to be checked

        Returns
        ----------
        bool
            True if the input parameters are well-formed,
            False otherwise
        """

        pass
