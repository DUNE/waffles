from waffles.data_classes.Event import Event
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
from waffles.data_classes.BeamInfo import BeamInfo
from waffles.data_classes.WaveformSet import WaveformSet
from typing import List

class BeamEvent(Event):
    """This class implements a BeamEvent. It inherits from Event and
    extends the base class with beam information (BeamInfo)

    Attributes
    ----------
    beam_info: BeamInfo
        Information about the particle generating this event

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    def __init__(
        self,
        beam_info: BeamInfo,
        channel_grids: List[ChannelWsGrid]= None,
            wfset: WaveformSet=None,
        ref_timestamp: int = 0,
        first_timestamp: int = 0,
        last_timestamp: int = 0,
        run_number: int = 0,
        record_number: int = 0,
        event_number: int = 0):
        

        # Shall we add add type checks here?

        self.__beam_info = beam_info

        # initialize the base class
        super().__init__(
            channel_grids,
            wfset,
            ref_timestamp,
            first_timestamp,
            last_timestamp,
            run_number,
            record_number,
            event_number)
        
    # Getters
    @property
    def beam_info(self):
        return self.__beam_info
