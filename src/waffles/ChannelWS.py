import numpy as np
from typing import Dict, Optional

from .WaveformSet import WaveformSet
from .CalibrationHistogram import CalibrationHistogram
from src.waffles.Exceptions import generate_exception_message

class ChannelWS(WaveformSet):

    """
    Stands for Channel Waveform Set. This class inherits 
    from the WaveformSet class. It implements a set of 
    Waveform objects for which its Endpoint attribute is 
    the same accross the whole set, and their Channel 
    attribute is also homogeneous. 

    Attributes
    ----------
    Waveforms : list of Waveform objects (inherited from WaveformSet)
    PointsPerWf : int (inherited from WaveformSet)
    Runs : set of int (inherited from WaveformSet)
    RecordNumbers : dictionary of sets (inherited from WaveformSet)
    AvailableChannels : dictionary of dictionaries of sets (inherited from WaveformSet)
    MeanAdcs : WaveformAdcs (inherited from WaveformSet)
    MeanAdcsIdcs : tuple of int (inherited from WaveformSet)
    Endpoint : int
        Endpoint number for this set of waveforms
    Channel : int
        Channel number for this set of waveforms
    CalibHisto : CalibrationHistogram
        A calibration histogram for this set of waveforms.
        It is not computed by default. I.e. if 
        self.CalibHisto equals to None, it should be 
        interpreted as unavailable data.

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    def __init__(self,  *waveforms,
                        compute_calib_histo : bool = False,
                        bins_number : Optional[int] = None,
                        domain : Optional[np.ndarray] = None,
                        variable : str = 'integral',
                        analysis_label : Optional[str] = None):
        
        """
        ChannelWS class initializer
        
        Parameters
        ----------
        waveforms : unpacked list of Waveform objects
            The waveforms that will be added to the set.
            Their Endpoint and Channel attributes must be
            homogeneous. Otherwise, an exception will be
            raised.
        compute_calib_histo : bool
            If True, then the calibration histogram for 
            this ChannelWS object will be computed, up to 
            the input given to the 'variable' parameter.
            If False, then the calibration histogram for
            this ChannelWS object will be set to None.
        bins_number : int
            This parameter only makes a difference if
            'compute_calib_histo' is set to True.
            If so, this parameter must be defined.
            In that case, it gives the number of bins
            that the calibration histogram will have.
            It must be greater than 1.
        domain : np.ndarray
            This parameter only makes a difference if
            'compute_calib_histo' is set to True.
            If so, this parameter must be defined.
            A 2x1 numpy array where (domain[0], domain[1])
            gives the range to consider for the
            calibration histogram. Any sample which falls 
            outside this range is ignored.
        variable : str
            This parameter only makes a difference if
            'compute_calib_histo' is set to True.
            In that case, if variable is set to 'integral', 
            then the calibration histogram will be 
            computed using the integral of the waveforms, 
            up to the input given to the 'analysis_label' 
            parameter. If variable is set to 'amplitude', 
            then the calibration histogram will be 
            computed using the amplitude of the waveforms.           ## Not implemented yet
            The default behaviour, which is used if
            the input is different from 'integral' or
            'amplitude', is that of 'integral'.
        analysis_label : str
            This parameter only makes a difference if
            'compute_calib_histo' is set to True
            and 'variable' is set to 'integral'.
            In such case, this parameter gives the key
            for the WfAna object within the Analyses 
            attribute of each considered waveform 
            from where to take the integral value to 
            add to the calibration histogram. Namely, 
            if such WfAna object is x, then 
            x.Result.Integral is the considered
            integral. If 'analysis_label' is None, 
            then the last analysis added to 
            the Analyses attribute will be the used 
            one. If there is not even one analysis, 
            then an exception will be raised.
        """

        ## Shall we add type checks here?

        super().__init__(*waveforms)

        self.__endpoint = None
        self.__channel = None
        self.update_endpoint_and_channel()

        self.__calib_histo = None

        if compute_calib_histo:

            if bins_number is None:
                raise Exception(generate_exception_message( 1,
                                                            'ChannelWS.__init__()',
                                                            'The bins number must be provided if the calibration histogram is to be computed.'))
            if domain is None:
                raise Exception(generate_exception_message( 2,
                                                            'ChannelWS.__init__()',
                                                            'The domain must be provided if the calibration histogram is to be computed.'))

            self.__calib_histo = CalibrationHistogram.from_WaveformSet( self,
                                                                        bins_number,
                                                                        domain,
                                                                        variable = variable,
                                                                        analysis_label = analysis_label)

    #Getters
    @property
    def Endpoint(self):
        return self.__endpoint
    
    @property
    def Channel(self):
        return self.__channel
    
    @property
    def CalibHisto(self):
        return self.__calib_histo
    
    def update_endpoint_and_channel(self) -> None:

        """
        This method checks the information returned by
        self.get_run_collapsed_available_channels(), to ensure that
        the Endpoint and the Channel attributes of every Waveform 
        object within this set is homogeneous. If it is not, then 
        it raises an exception. If they are, then the Endpoint and 
        Channel attributes of this ChannelWS object are updated 
        accordingly.

        Returns
        ----------
        None
        """

        aux = self.get_run_collapsed_available_channels()

        if(len(aux) != 1):  
            raise Exception(generate_exception_message( 1,
                                                        'ChannelWS.update_endpoint_and_channel()',
                                                        'Every Waveform object within this set must have the same Endpoint attribute.'))
        else: 
            endpoint = next(iter(aux.keys()))
            if len(aux[endpoint]) != 1:
                raise Exception(generate_exception_message( 2,
                                                            'ChannelWS.update_endpoint_and_channel()',
                                                            'Every Waveform object within this set must have the same Channel attribute.'))
            else:
                channel = next(iter(aux[endpoint]))

        self.__endpoint = endpoint
        self.__channel = channel

        return