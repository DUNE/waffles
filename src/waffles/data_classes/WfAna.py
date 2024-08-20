from abc import ABC, abstractmethod

from waffles.data_classes.WaveformAdcs import waveform_adcs
from waffles.data_classes.IPDict import ip_dict
from waffles.data_classes.WfAnaResult import wf_ana_result

import waffles.Exceptions as we


class wf_ana(ABC):

    """
    Stands for Waveform Analysis. This abstract
    class is intended to be the base class for
    any class which implements a certain type of
    analysis which is performed over an arbitrary
    waveform_adcs object.

    Attributes
    ----------
    InputParameters : ip_dict
        An ip_dict object (a dictionary) containing the
        input parameters of this analysis. The keys (resp.
        values) are the names (resp. values) of the input
        parameters.
    Result : WfAnaResult
        A WfAnaResult object (a dictionary) containing
        the result of the analysis

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    def __init__(self, input_parameters: ip_dict):
        """
        WfAna class initializer. It is assumed that it is
        the caller responsibility to check the well-formedness
        of the input parameter, according to the attributes
        documentation in the class documentation. No checks
        are perfomed here.

        Parameters
        ----------
        input_parameters : ip_dict
        """

        self.__input_parameters = input_parameters
        self.__result = None
        # To be determined a posteriori
        # by the analyse() instance method

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
            waveform: waveform_adcs,
            *args,
            **kwargs):
        """
        This abstract method serves as a template for
        the analyser method that MUST be implemented
        for whichever derived class of WfAna. This
        method must be responsible for creating an
        object of class WfAnaResult and assigning it
        to self.__result.

        Parameters
        ----------
        waveform : waveform_adcs
            The waveform_adcs object which will be
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

        self.__result = wf_ana_result()
        return

    @staticmethod
    @abstractmethod
    @we.handle_missing_data
    def check_input_parameters(input_parameters: ip_dict) -> None:
        """
        This abstract method, which MUST be implemented
        for whichever derived class of WfAna, is
        responsible for checking whether the input
        parameters are well-formed. It should raise
        an exception if the input parameters are not
        well-formed. It should end execution normally
        and return None if they are well-formed.

        For efficiency purposes, the aim is to call
        it at the WaveformSet level (before instantiating
        the first WfAna (or derived) object) for
        cases where the same ip_dict is given for every
        waveform_adcs object to be analysed. In these
        cases, just one check is performed, instead
        of N checks, where N is the number of waveform_adcs
        objects and N-1 checks are redundant. That's the
        reason why this method is static.

        Parameters
        ----------
        input_parameters : ip_dict
            The input parameters to be checked

        Returns
        ----------
        bool
            True if the input parameters are well-formed,
            False otherwise
        """

        pass
