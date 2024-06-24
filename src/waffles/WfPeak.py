from typing import Optional

from .Exceptions import generate_exception_message

class WfPeak:

    """
    Stands for Waveform Peak. This class implements a
    peak which has been spotted in the Adcs attribute
    of a certain Waveform object.
    
    Attributes
    ----------
    Position : int
        The iterator value for the point within 
        the waveform Adcs attribute where the 
        peak was spotted
    Baseline : float
        The baseline value which was used for the
        Amplitude and Integral attributes evaluation
    IntLl (resp. IntUl): int
        Stands for integration lower (resp. upper) 
        limit. Iterator value for the first (resp. 
        last) point of the waveform Adcs range which 
        was used to compute this WfPeak Integral 
        attribute. IntLl must be smaller than IntUl. 
        Both limits are inclusive.
    Amplitude : float
        Amplitude of this peak
    Integral : float
        Integral of this peak

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    def __init__(self,  position : int,
                        baseline : Optional[float] = None,
                        int_ll : Optional[int] = None,
                        int_ul : Optional[int] = None,
                        amplitude : Optional[float] = None,
                        integral : Optional[float] = None):
        
        """
        WfPeak class initializer. The only requirement to initialize
        a WfPeak object is to provide its position. The rest of the
        attributes can be setted later using the 
        WfPeak.set_amplitude_and_integral() instance method.
        Attributes which are not provided are set to None until
        otherwise specified. Thus, retrieving an attribute which
        is None should be interpreted as unavailable data.
        
        Parameters
        ----------
        position : int
            It must be semipositive.
        baseline : float
        int_ll (resp. int_ul) : int
            If defined, it must be semipositive.
        amplitude : float
            If defined, it must be positive.
        integral : float

        """
        ## Shall we add type checks here?

        if position < 0:                                                            # If this check makes the execution time 
            raise Exception(generate_exception_message( 1,                          # be prohibitively high, it may be removed
                                                        'WfPeak.__init__()',
                                                        f"The provided position is negative ({position})."))
        self.__position = position

        self.__baseline = None      # These might never be set by 
        self.__int_ll = None        # self.set_amplitude_and_integral()
        self.__int_ul = None        # We want them to be None at least.
        self.__amplitude = None
        self.__integral = None

        self.set_amplitude_and_integral(baseline = baseline,
                                        int_ll = int_ll,
                                        int_ul = int_ul,
                                        amplitude = amplitude,
                                        integral = integral)

    #Getters
    @property
    def Position(self):
        return self.__position
    
    @property
    def Baseline(self):
        return self.__baseline
    
    @property
    def IntLl(self):
        return self.__int_ll
    
    @property
    def IntUl(self):
        return self.__int_ul

    @property
    def Amplitude(self):
        return self.__amplitude
    
    @property
    def Integral(self):
        return self.__integral

    def set_amplitude_and_integral(self,    baseline : Optional[float] = None,                # Attributes should be set mutually
                                            int_ll : Optional[int] = None,                    # to make sure that the baseline
                                            int_ul : Optional[int] = None,                    # in self.__baseline matches the
                                            amplitude : Optional[float] = None,               # one that was used to compute
                                            integral : Optional[float] = None) -> None:       # self.__amplitude and self.__integral
        
        """
        Method to jointly set the

            - self.__baseline, 
            - self.__int_ll,
            - self.__int_ul,
            - self.__amplitude and
            - self.__integral 

        attributes. Note that this method will only set the 
        self.__amplitude if the self.__baseline is set.
        On the other hand, to set self.__integral it is 
        also mandatory to define self.__int_ll and 
        self.__int_ul.

        Parameters
        ----------
        baseline : float
            It is loaded into the self.__baseline attribute.
        int_ll (resp. int_ul) : int
            If defined, it must be a semipositive integer. It 
            is loaded into the self.__int_ll (resp. self.__int_ul) 
            attribute. int_ll must be smaller than int_ul.
        amplitude : float
            If defined, it must be a positive float. It is loaded 
            into the self.__amplitude attribute if the 'baseline' 
            input parameter is defined.
        integral : float
            It is loaded into the self.__integral attribute
            if the 'baseline', 'int_ll' and 'int_ul' input 
            parameters are defined.

        Returns
        ----------
        None
        """

        ## Shall we add type checks here?

        if baseline is not None:    # If the baseline is not defined, then there is no information to set
                                    # the amplitude nor the integral. In this case we are also ignoring
                                    # the 'int_ll' and 'int_ul' input parameters.

            self.__baseline = baseline      # At this point the amplitude can be defined

            if amplitude is not None:
                if amplitude <= 0.0:                                    # If this check makes the execution time 
                                                                        # be prohibitively high, it may be removed
                    raise Exception(generate_exception_message( 1,
                                                                'WfPeak.set_amplitude_and_integral()',
                                                                f"The provided amplitude is negative ({amplitude})."))                
            
            self.__amplitude = amplitude    # Set the amplitude even if it is None

            self.reset_integral()   # Reset the integral to None, since the baseline has changed. 
                                    # This is to avoid inconsistencies between the baseline used 
                                    # to compute the integral and the one that is stored in
                                    # self.__baseline as of now.

            if int_ll is not None:  # If the lower integration limit is not defined, then
                                    # there is no information to set the integral attribute. 

                if int_ll < 0:                                      # If the execution time goes prohibitively high due to this
                                                                    # and the int_ul < int_ll checks, we shall remove them.                                    
                    raise Exception(generate_exception_message( 2,
                                                                'WfPeak.set_amplitude_and_integral()',
                                                                f"The provided int_ll is negative ({int_ll})."))

                if int_ul is not None:      # Check for int_ul only if int_ll is OK
                    if int_ul <= int_ll:
                        raise Exception(generate_exception_message( 3,
                                                                    'WfPeak.set_amplitude_and_integral()',
                                                                    f"The provided int_ul ({int_ul}) is smaller than or equal to int_ll ({int_ll})."))
                    
                    self.__int_ll = int_ll  # Set both limits only if the 
                    self.__int_ul = int_ul  # input for both is well-formed

                    # At this point the integral can be defined
                    self.__integral = integral
        return
    
    def reset_integral(self) -> None:

        """
        Method to reset the
            
                - self.__int_ll,
                - self.__int_ul and
                - self.__integral

        attributes to None. 

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """

        self.__int_ll = None
        self.__int_ul = None
        self.__integral = None
        return