from typing import Dict

from src.waffles.WaveformSet import WaveformSet
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

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    def __init__(self,  *waveforms):
        
        """
        ChannelWS class initializer
        
        Parameters
        ----------
        waveforms : unpacked list of Waveform objects
            The waveforms that will be added to the set.
            Their Endpoint and Channel attributes must be
            homogeneous. Otherwise, an exception will be
            raised.
        """

        ## Shall we add type checks here?

        super().__init__(*waveforms)

        self.__endpoint = None
        self.__channel = None
        self.update_endpoint_and_channel()

    #Getters
    @property
    def Endpoint(self):
        return self.__endpoint
    
    @property
    def Channel(self):
        return self.__channel
    
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

    @staticmethod
    def clusterize_WaveformSet(waveform_set : WaveformSet) -> Dict[int, Dict[int, 'ChannelWS']]:

        """
        This function returns a dictionary, say output, 
        whose keys are endpoint values for which there 
        is at least one Waveform object in the given 
        WaveformSet object. The values of such dictionary 
        are dictionaries, whose keys are channel values 
        for which there is at least one Waveform object 
        in this WaveformSet object. The values for the 
        deeper-level dictionaries are ChannelWS objects,
        which are initialized by this static method,
        in a way that output[i][j] is the ChannelWS object 
        which contains all of the Waveform objects within 
        the given WaveformSet object which come from 
        endpoint i and channel j.

        This method is useful to partition the given 
        WaveformSet object into WaveformSet objects 
        (actually ChannelWS objects, which inherit from 
        the WaveformSet class but require the Endpoint 
        and the Channel attribute of its constituent 
        Waveform objects to be homogeneous) which are 
        subsets of the given WaveformSet object, and 
        whose Waveform objects have homogeneous endpoint 
        and channel values.

        Returns
        ----------
        output : dict of dict of ChannelWS
        """

        idcs = {}

        for idx in range(len(waveform_set.Waveforms)):
            try:
                aux = idcs[waveform_set.Waveforms[idx].Endpoint]

            except KeyError:
                idcs[waveform_set.Waveforms[idx].Endpoint] = {}
                aux = idcs[waveform_set.Waveforms[idx].Endpoint]

            try:
                aux[waveform_set.Waveforms[idx].Channel].append(idx)
            except KeyError:
                aux[waveform_set.Waveforms[idx].Channel] = [idx]

        output = {}

        for endpoint in idcs.keys():
            output[endpoint] = {}

            for channel in idcs[endpoint].keys():
                aux = [waveform_set.Waveforms[idx] for idx in idcs[endpoint][channel]]
                output[endpoint][channel] = ChannelWS(*aux)

        return output