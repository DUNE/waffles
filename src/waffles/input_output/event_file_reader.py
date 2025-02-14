import os
import _pickle as pickle    # Making sure that cPickle is used
from pathlib import Path
from typing import List, Optional

from waffles.data_classes.WaveformSet import WaveformSet
from waffles.input_output.pickle_file_reader import WaveformSet_from_pickle_file
from waffles.input_output.raw_root_reader    import BeamInfo_from_root_file
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.BeamEvent import BeamEvent
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
from waffles.data_classes.ChannelMap import ChannelMap
from waffles.data_classes.Event import Event
from waffles.np04_data.ProtoDUNE_HD_APA_maps import APA_map

import waffles.utils.event_utils as eu

import waffles.Exceptions as we

def events_from_pickle_and_beam_files(
        path_to_pickle_file : str,
        path_to_root_file : str,
        delta_t_max: int
    ) -> List[BeamEvent]:
                                
    """
    This function gets a path to two files, the first file should be
    a pickle of a list of Event objects, and the second a root file with beam info.
    It creates Event objects combining the information from both files and returns
    a list of Event objects.

    Parameters
    ----------
    path_to_pickle_file: str
        Path to the WaveformSet file which will be loaded. Its extension
        must match '.pkl'.

    path_to_root_file: str
        Path to the beam file which will be loaded. Its extension
        must match '.root'.

    Returns
    ----------        
    output: list of Event objects
    """

    if not os.path.isfile(path_to_pickle_file):
        raise Exception(
            we.GenerateExceptionMessage(
                1, 
                'events_from_pickle_and_beam_files()',
                f"The given path ({path_to_pickle_file}) "
                "does not point to an existing file."))

    if not os.path.isfile(path_to_root_file):
        raise Exception(
            we.GenerateExceptionMessage(
                1, 
                'events_from_pickle_and_beam_files()',
                f"The given path ({path_to_root_file}) "
                "does not point to an existing file."))
    
    # read all waveforms from the pickle file
    wfset = WaveformSet_from_pickle_file(path_to_pickle_file)

    # read all beam events from the root file
    beam_infos  = BeamInfo_from_root_file(path_to_root_file) 


    channel_map = APA_map
    ngrids = 4
    
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
            gi = eu.get_grid_index(w)
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



def events_from_pickle_file(
        path_to_pickle_file : str
    ) -> List[BeamEvent]:


    """
    This function gets a path to a file which should be
    a pickle of a list of Event objects, and loads it using 
    the pickle library. It returns the resulting list of Events.

    Parameters
    ----------
    path_to_pickle_file: str
        Path to the file which will be loaded. Its extension
        must match '.pkl'.

    Returns
    ----------        
    output: list of Event objects
    """

    if os.path.isfile(path_to_pickle_file):
        with open(path_to_pickle_file, 'rb') as file:
            output = pickle.load(file)
    else:
        raise Exception(
            we.GenerateExceptionMessage(
                1, 
                'events_from_pickle_file()',
                f"The given path ({path_to_pickle_file}) "
                "does not point to an existing file."))
        
    if not isinstance(output[0], Event):
        raise Exception(
            we.GenerateExceptionMessage(2,
            'events_from_pickle_file()',
            "The object loaded from the given "
            "file is not a Event object."))
    
    return output
    
