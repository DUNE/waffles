class Waveform:

    """
    This class implements a waveform.

    Attributes
    ----------
    Timestamp : int
        The timestamp value for this waveform
    Adcs : unidimensional numpy array of integers
        The readout for this waveform, in # of ADCs
    Run_number : int
        Number of the run from which this waveform was
        acquired.
    Endpoint : int
        Endpoint number from which this waveform was
        acquired
    Channel : int
        Channel number for this waveform
    Analyses : list of WfAnaResult objects
    """

    def __init__(self,  timestamp,
                        adcs,
                        run_number,
                        endpoint,
                        channel):
        
        """Waveform class initializer
        
        Parameters
        ----------
        timestamp : int
        adcs : unidimensional numpy array of integers
        run_number : int
        endpoint : int
        channel : int
        """

        # Do we want to add type checks here?
    
        self.__timestamp = timestamp
        self.__adcs = adcs
        self.__run_number = run_number
        self.__endpoint = endpoint
        self.__channel = channel
        self.__analyses = []    # Initialize the analyses 
                                # list as an empty list
    
    #Getters
    @property
    def Timestamp(self):
        return self.__timestamp
    
    @property
    def Adcs(self):
        return self.__adcs
    
    @property
    def Run_number(self):
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
#       self.__timestamp=input     # through Waveform.__init__. Here's an example
#       return                     # of what a setter would look like, though.

    def analyze(self, analysis_name):

        """
        Performs an analysis over this waveform object (self) 
        and adds it to the self.__analyses attribute.

        Parameters
        ----------
        param1 : str
            An ID describing which analysis to perform  # To be implemented

        Returns
        ----------
        None
        """

        pass

    def get_global_channel(self):

        """
        Returns
        ----------
        int
            An integer value for the readout channel with respect to a numbering 
            scheme which identifies the endpoint and the APA channel at the same
            time.
        """

        pass