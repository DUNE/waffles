import numpy as np

from waffles.data_classes.WaveformAdcs import WaveformAdcs

class Waveform(WaveformAdcs):

    """
    This class implements a waveform which includes
    information which is relative to detector readout.
    It inherits from the WaveformAdcs class.

    Attributes
    ----------
    Timestamp : int
        The timestamp value for this waveform
    TimeStep_ns : float (inherited from WaveformAdcs)
    Adcs : unidimensional numpy array of integers (inherited from WaveformAdcs)
    RunNumber : int
        Number of the run from which this waveform was
        acquired
    RecordNumber : int
        Number of the record within which this waveform
        was acquired
    Endpoint : int
        Endpoint number from which this waveform was
        acquired
    Channel : int
        Channel number for this waveform
    TimeOffset : int (inherited from WaveformAdcs)
    Analyses : OrderedDict of WfAna objects (inherited from WaveformAdcs)

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    def __init__(self,  timestamp : int,
                        time_step_ns : float,
                        adcs : np.ndarray,
                        run_number : int,
                        record_number : int,
                        endpoint : int,
                        channel : int,
                        time_offset : int = 0):
        
        """
        Waveform class initializer
        
        Parameters
        ----------
        timestamp : int
        time_step_ns : float
            It is given to the 'time_step_ns' parameter of
            the base class initializer.
        adcs : unidimensional numpy array of integers
            It is given to the 'adcs' parameter of the base
            class initializer.
        run_number : int
        record_number : int
        endpoint : int
        channel : int
        time_offset : int
            It is given to the 'time_offset' parameter of the
            base class initializer. It must be semipositive 
            and smaller than len(self.__adcs)-1. Its default 
            value is 0.
        """

        ## Shall we add add type checks here?
    
        self.__timestamp = timestamp
        self.__run_number = run_number
        self.__record_number = record_number
        self.__endpoint = endpoint
        self.__channel = channel

        ## Do we need to add trigger primitives as attributes?

        super().__init__(   time_step_ns, 
                            adcs,
                            time_offset = time_offset)

    #Getters
    @property
    def Timestamp(self):
        return self.__timestamp
    
    @property
    def RunNumber(self):
        return self.__run_number
    
    @property
    def RecordNumber(self):
        return self.__record_number
    
    @property
    def Endpoint(self):
        return self.__endpoint
    
    @property
    def Channel(self):
        return self.__channel
    
#   #Setters                        # For the moment there are no setters for 
#   @Timestamp.setter               # the attributes of Waveform. I.e. you can
#   def Timestamp(self, input):     # only set the value of its attributes
#       self.__timestamp = input    # through Waveform.__init__. Here's an example
#       return                      # of what a setter would look like, though.
        
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