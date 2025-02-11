from typing import List

from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
from waffles.data_classes.ChannelMap import ChannelMap
from waffles.data_classes.Event import Event
from waffles.np04_data.ProtoDUNE_HD_APA_maps import APA_map
from waffles.plotting.plot import plot_ChannelWsGrid
from waffles.data_classes.WaveformSet import WaveformSet

def events_from_wfset(wfset0: WaveformSet, delta_t: int = 1024, 
        channel_map: List[ChannelMap] = APA_map,
        ngrids: int = 4,
        nevents: int = 100000000000,
        ) -> List[Event]:


    # sort waveforms in WaveforSet by timestamp
    waveforms = sorted(wfset0.waveforms, key=lambda Waveform: Waveform.timestamp)
    wfset = WaveformSet(*waveforms)

    # ----------- organize waveforms in events
    previous_wf = None
    dw_wfs = [[]]*ngrids
    events = []
    detector_grids = [None]*ngrids
    event_number = 1
    
    for i in range(ngrids):
        dw_wfs[i] = []

    for wf in wfset.waveforms: 

        # get the grid index for this waveform
        gi = get_grid_index(wf)

        if previous_wf==None:
            ini_ts = wf.daq_window_timestamp
            first_ts = wf.timestamp
        #elif wf.daq_window_timestamp != previous_wf.daq_window_timestamp: 
        elif wf.timestamp - previous_wf.timestamp > delta_t:            
            # create a new list of grids
            detector_grids = [None]*ngrids            
            for i in range(ngrids):
                if len(dw_wfs[i]) > 0:
                    dw_wfset = WaveformSet(*dw_wfs[i])
                    detector_grids[i] = ChannelWsGrid(channel_map[i+1], dw_wfset)
                else:
                    detector_grids[i] = None
                    
                dw_wfs[i] = []        

            # create Event with 4 channel grids, one for each APA
            event = Event(detector_grids, 
                          previous_wf.daq_window_timestamp - ini_ts, 
                          first_ts - ini_ts, 
                          previous_wf.timestamp - ini_ts, 
                          previous_wf.run_number, 
                          previous_wf.record_number,
                          event_number)                                
            events.append(event)
            event_number +=1

            first_ts = wf.timestamp
 
            #figure = plot_ChannelWsGrid(event.channel_wfs[apa-1])
            #figure.write_image("plots.pdf")

        dw_wfs[gi-1].append(wf)
        previous_wf = wf
        
        if len(events) >= nevents:
            return events

    return events


def get_grid_index(wf: Waveform):

    # get the apa for that waveform      
    if    wf.endpoint <  109: gi = 1
    elif  wf.endpoint == 109: gi = 2 
    elif  wf.endpoint == 111: gi = 3
    elif  wf.endpoint >  111: gi = 4  

    return gi 
