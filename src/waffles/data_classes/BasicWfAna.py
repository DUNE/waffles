
import numpy as np

from waffles.data_classes.WaveformAdcs import WaveformAdcs                                                    
from waffles.data_classes.IPDict import IPDict
from waffles.data_classes.WfAna import WfAna
from waffles.data_classes.WfAnaResult import WfAnaResult

import waffles.utils.check_utils as wuc
import waffles.Exceptions as we



class BasicWfAna(WfAna):
    

    """
    Stands for Basic Waveform Analysis. This class
    inherits from WfAna. It implements a basic 
    analysis which is performed over a certain 
    WaveformAdcs object.
    
    Attributes
    ----------
    InputParameters : IPDict (inherited from WfAna)
    BaselineLimits : list of int
        It must have an even number of integers which
        must meet BaselineLimits[i] < BaselineLimits[i + 1].
        Given a WaveformAdcs object, wf, the points which
        are used for baseline calculation are
        wf.Adcs[BaselineLimits[2*i] - wf.TimeOffset : BaselineLimits[(2*i) + 1] - wf.TimeOffset],
        with i = 0,1,...,(len(BaselineLimits)/2) - 1. The 
        upper limits are exclusive.
    IntLl (resp. IntUl) : int
        Stands for integration lower (resp. upper) limit.
        Iterator value for the first (resp. last) point 
        of the waveform that falls into the integration
        window. IntLl must be smaller than IntUl. These
        limits are inclusive. I.e. the points which are
        used for the integral calculation are
        wf.Adcs[IntLl - wf.TimeOffset : IntUl + 1 - wf.TimeOffset]. 
    AmpLl (resp. AmpUl) : int
        Stands for amplitude lower (resp. upper) limit.
        Iterator value for the first (resp. last) point 
        of the waveform that is considered to compute
        the amplitude of the waveform. AmpLl must be smaller 
        than AmpUl. These limits are inclusive. I.e., the 
        points which are used for the amplitude calculation 
        are wf.Adcs[AmpLl - wf.TimeOffset : AmpUl + 1 - wf.TimeOffset].
    Result : WfAnaResult (inherited from WfAna)

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    @we.handle_missing_data
    def __init__(self,  input_parameters : IPDict):
        
        """
        BasicWfAna class initializer. It is assumed that it is
        the caller responsibility to check the well-formedness 
        of the input parameters, according to the attributes
        documentation in the class documentation. No checks 
        are perfomed here.
        
        Parameters
        ----------
        input_parameters : IPDict
            This IPDict must contain the following keys:
                - 'baseline_limits' (list of int)
                - 'int_ll' (int)
                - 'int_ul' (int)
                - 'amp_ll' (int)
                - 'amp_ul' (int)
        """
        
        self.__baseline_limits = input_parameters['baseline_limits']
        self.__int_ll = input_parameters['int_ll']
        self.__int_ul = input_parameters['int_ul']
        self.__amp_ll = input_parameters['amp_ll']
        self.__amp_ul = input_parameters['amp_ul']

        super().__init__(input_parameters)

    #Getters
    @property
    def BaselineLimits(self):
        return self.__baseline_limits
    
    @property
    def IntLl(self):
        return self.__int_ll
    
    @property
    def IntUl(self):
        return self.__int_ul
    
    @property
    def AmpLl(self):
        return self.__amp_ll
    
    @property
    def AmpUl(self):
        return self.__amp_ul
    
    def analyse(self, waveform : WaveformAdcs) -> None:

        """
        With respect to the given WaveformAdcs object, this analyser 
        method does the following:

            - It computes the baseline as the median of the points
            that are considered, according to the documentation of
            the self.__baseline_limits attribute.
            - It calculates the integral of 
            waveform.Adcs[IntLl - waveform.TimeOffset : IntUl + 1 - waveform.TimeOffset]. 
            To do so, it assumes that the temporal resolution of 
            the waveform is constant and approximates its integral 
            to waveform.TimeStep_ns*np.sum( -b + waveform.Adcs[IntLl - waveform.TimeOffset : IntUl + 1 - waveform.TimeOffset]),
            where b is the computed baseline.
            - It calculates the amplitude of 
            waveform.Adcs[AmpLl - waveform.TimeOffset : AmpUl + 1 - waveform.TimeOffset].

        Note that for these computations to be well-defined, it is
        assumed that

            - BaselineLimits[0] - wf.TimeOffset >= 0
            - BaselineLimits[-1] - wf.TimeOffset <= len(wf.Adcs)
            - IntLl - wf.TimeOffset >= 0
            - IntUl - wf.TimeOffset < len(wf.Adcs)
            - AmpLl - wf.TimeOffset >= 0
            - AmpUl - wf.TimeOffset < len(wf.Adcs)

        For the sake of efficiency, these checks are not done.
        It is the caller's responsibility to ensure that these 
        requirements are met.

        Parameters
        ----------
        waveform : WaveformAdcs
            The WaveformAdcs object which will be analysed

        Returns
        ----------
        None
        """
                    
        split_baseline_samples = [  waveform.Adcs[ self.__baseline_limits[2*i] - waveform.TimeOffset : self.__baseline_limits[(2*i) + 1] - waveform.TimeOffset]
                                    for i in range(len(self.__baseline_limits)//2) ]
        
        baseline_samples = np.concatenate(split_baseline_samples)
        baseline = np.median(baseline_samples)

        self._WfAna__result = WfAnaResult(  baseline = baseline,
                                            baseline_min = None,            # Might be set to np.min(baseline_samples) (resp. 
                                            baseline_max = None,            # np.max(baseline_samples), ~np.std(baseline_samples)) 
                                            baseline_rms = None,            # for a deeper analysis for which we need (and
                                                                            # can afford the computation time) for this data

                                            integral = waveform.TimeStep_ns*(((self.__int_ul - self.__int_ll + 1)*baseline) - np.sum(waveform.Adcs[ self.__int_ll - waveform.TimeOffset : self.__int_ul + 1 - waveform.TimeOffset])),   ## Assuming that the waveform is
                                                                                                                                                                                                                                        ## inverted and using linearity
                                                                                                                                                                                                                                        ## to avoid some multiplications
                                            amplitude = + np.max(waveform.Adcs[self.__amp_ll - waveform.TimeOffset : self.__amp_ul + 1 - waveform.TimeOffset]) 
                                                        - np.min(waveform.Adcs[self.__amp_ll - waveform.TimeOffset : self.__amp_ul + 1 - waveform.TimeOffset]))
        return
    
    @staticmethod
    @we.handle_missing_data
    def check_input_parameters( input_parameters : IPDict,
                                points_no : int) -> None:

        """
        This method performs three checks:

            - It checks whether the baseline limits, say bl, are 
            well-formed, i.e. whether they meet

                0 <= bl[0] < bl[1] < ... < bl[-1] <= points_no - 1

            - It checks whether the integration window, say 
            (int_ll, int_ul) is well-formed, i.e. whether it meets

                0 <= int_ll < int_ul <= points_no - 1

            - It checks whether the amplitude window, say
            (amp_ll, amp_ul) is well-formed, i.e. whether it meets

                0 <= amp_ll < amp_ul <= points_no - 1

        If any of these checks fail, an exception is raised.

        Parameters
        ----------
        input_parameters : IPDict
            The input parameters to be checked. It is the IPDict
            that can be potentially given to BasciWfAna.__init__
            to instantiate a BasicWfAna object.
        points_no : int
            The number of points in any waveform that could be
            analysed. It is assumed to be the same for all the
            waveforms.

        Returns
        ----------
        None
        """

        if not wuc.baseline_limits_are_well_formed( input_parameters['baseline_limits'],
                                                    points_no):
            
            raise Exception(we.generate_exception_message(  1,
                                                            'BasicWfAna.check_input_parameters()',
                                                            f"The baseline limits ({input_parameters['baseline_limits']}) are not well formed."))
        int_ul_ = input_parameters['int_ul']
        if int_ul_ is None:
            int_ul_ = points_no - 1

        if not wuc.subinterval_is_well_formed(  input_parameters['int_ll'],
                                                int_ul_,
                                                points_no):
            
            raise Exception(we.generate_exception_message(  2,
                                                            'BasicWfAna.check_input_parameters()',
                                                            f"The integration window ({input_parameters['int_ll']}, {int_ul_}) is not well formed. It must be a subset of [0, {points_no})."))
        amp_ul_ = input_parameters['amp_ul']
        if amp_ul_ is None:
            amp_ul_ = points_no - 1

        if not wuc.subinterval_is_well_formed(  input_parameters['amp_ll'],
                                                amp_ul_,
                                                points_no):
            
            raise Exception(we.generate_exception_message(  3,
                                                            'BasicWfAna.check_input_parameters()',
                                                            f"The amplitude window ({input_parameters['amp_ll']}, {amp_ul_}) is not well formed. It must be a subset of [0, {points_no})."))