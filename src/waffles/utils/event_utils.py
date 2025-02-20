from typing import List

from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
from waffles.data_classes.ChannelMap import ChannelMap
from waffles.data_classes.Event import Event
from waffles.data_classes.BeamEvent import BeamEvent
from waffles.np04_data.ProtoDUNE_HD_APA_maps import APA_map
from waffles.plotting.plot import plot_ChannelWsGrid
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.BeamInfo import BeamInfo

def events_from_wfset_and_beam_info(
        wfset: WaveformSet,
        beam_infos: List[BeamInfo],
        delta_t_max: int = 1024, 
        channel_map: List[ChannelMap] = APA_map,
        ngrids: int = 4,
        ) -> List[BeamEvent]:


    events = []

    i=0
    # loop over beam events
    for b in beam_infos:
#        print (i,b.t, b.tof)

        dw_wfs = [[]]*ngrids
        for j in range(ngrids):
            dw_wfs[j] = []

        record = 0
        run = b.run
        event_number=b.event
        wfs = []

        t_first = 1e20
        t_last = 0 

        i+=1        
        # loop over waveforms
        for w in wfset.waveforms:

            # select waveforms in the same daq window            
            if w.daq_window_timestamp != b.t:
                continue

            # select the ones close in time to the beam time stamp
            delta_t =  abs(int(w.timestamp) - int(b.t))
            if delta_t>delta_t_max:
                continue
                
#            print ('  -', w.channel, w.endpoint, min(w.adcs), delta_t)        
            gi = get_grid_index(w)
            dw_wfs[gi-1].append(w)
            wfs.append(w)
            record = w.record_number,
            
            if w.timestamp < t_first:
                t_first = w.timestamp
            if w.timestamp > t_last:
                t_last = w.timestamp                
            

        # create a new list of grids
        detector_grids = [None]*ngrids            
        for j in range(ngrids):
            if len(dw_wfs[j]) > 0:
                dw_wfset = WaveformSet(*dw_wfs[j])
                detector_grids[j] = ChannelWsGrid(channel_map[j+1], dw_wfset)
            else:
                detector_grids[j] = None


        wfset_ev = None
        if len(wfs)>0:
            wfset_ev = WaveformSet(*wfs)

        # create the beam event
        event = BeamEvent(b,     # beam info
                          detector_grids,
                          wfset_ev,
                          b.t,
                          t_first,
                          t_last,
                          run, 
                          record,
                          event_number)                                
        
        # add the event to the list
        events.append(event)    

    return events

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
