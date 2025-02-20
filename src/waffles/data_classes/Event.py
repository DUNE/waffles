import numpy as np
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
from waffles.data_classes.WaveformSet import WaveformSet
from typing import List

class Event():
    """This class implements a Event.

    Attributes
    ----------
    channel_grids: List[ChannelWsGrid] NOT YET FULLY FUNCTIONAL
        Is meant con contain an organised set of WaveformSets, for
        each channel, all of them belonging to the event
    
    wset: WaveformSet
        The WaveformSet containing all waveforms associated to this event

    ref_timestamp: int
        is the reference time for the event
        (for beam events it coincides with the daq_window_timestamp)

    first_timestamp: int
        the timestamp of the first waveform iun the WaveformSet

    last_timestamp: int
        the timestamp of the last waveform iun the WaveformSet

    run_number: int

    record_number: int

    event_number: int
    
    
    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    def __init__(
        self, 
        channel_grids: List[ChannelWsGrid]= None,
        wfset: WaveformSet = None,
        ref_timestamp: int = 0,
        first_timestamp: int = 0,
        last_timestamp: int = 0,
        run_number: int = 0,
        record_number: int = 0,
        event_number: int = 0):


        # Shall we add add type checks here?

        self.__channel_grids = channel_grids
        self.__wfset = wfset        
        self.__ref_timestamp = ref_timestamp
        self.__first_timestamp = first_timestamp
        self.__last_timestamp = last_timestamp
        self.__run_number = run_number
        self.__record_number = record_number
        self.__event_number = event_number
        

    # Getters
    @property
    def channel_wfs(self):
        return self.__channel_grids

    @property
    def wfset(self):
        return self.__wfset
    
    @property
    def ref_timestamp(self):
        return self.__ref_timestamp
    
    @property
    def first_timestamp(self):
        return self.__first_timestamp
    
    @property
    def last_timestamp(self):
        return self.__last_timestamp
    
    @property
    def run_number(self):
        return self.__run_number

    @property
    def record_number(self):
        return self.__record_number
    
    @property
    def event_number(self):
        return self.__event_number
    

    def get_nchannels(self):
        available_channels = []
        for i in self.__channel_grids:
            if not i:
                continue
            for endpoint in i.ch_wf_sets.keys():
                for channel in i.ch_wf_sets[endpoint].keys():
                    available_channels.append(channel)

        return len(available_channels)
    
    def get_channels(self):
        available_channels = []
        for ch_grid in self.__channel_grids:
            if not ch_grid: continue
            available_channels = {}
            for endpoint in ch_grid.ch_wf_sets.keys():
                available_channels[endpoint] = []
                for channel in ch_grid.ch_wf_sets[endpoint].keys():
                    available_channels[endpoint].append(channel)

        return available_channels
    

    def get_wfset_in_channel(self, endpoint: int, channel: int):
        wfset = None
        for ch_grid in self.__channel_grids:
            if not ch_grid:continue
            for ep in ch_grid.ch_wf_sets.keys():
                if ep != endpoint: continue
                for ch in ch_grid.ch_wf_sets[endpoint].keys():
                    if channel == ch:
                        return ch_grid.ch_wf_sets[endpoint][ch]  
        
        return wfset
    
    def get_wf_in_channel(self, endpoint: int, channel: int):
        return self.get_wfset_in_channel(endpoint,channel).waveforms[0]


    
    

    
    
