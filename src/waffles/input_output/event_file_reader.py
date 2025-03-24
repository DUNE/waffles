import os
import _pickle as pickle    # Making sure that cPickle is used
from pathlib import Path
from typing import List, Optional

from waffles.data_classes.WaveformSet import WaveformSet
from waffles.input_output.pickle_file_reader import WaveformSet_from_pickle_file
from waffles.input_output.raw_root_reader    import BeamInfo_from_root_file
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.BeamEvent import BeamEvent
from waffles.data_classes.Event import Event
from waffles.utils.event_utils import events_from_wfset_and_beam_info
from waffles.np04_data.ProtoDUNE_HD_APA_maps import APA_map

import waffles.Exceptions as we

def events_from_pickle_and_beam_files(
        path_to_pickle_file : str,
        path_to_root_file : str,
        delta_t_max: int,
        library: str='uproot'
    ) -> List[BeamEvent]:
                                
    """
    This function gets a path to two files, the first file should be
    a pickle of a WaveformSet object, and the second a root file with beam info.
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

    delta_t_max:
        [-delta_tmax, +delta_tmax] will be the time interval around the beam time where
        waveforms will be considered for an event
    library: str
        pyroot or uproot

    Returns
    ----------        
    output: List[BeamEvent]
        list of Event objects
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
    beam_infos  = BeamInfo_from_root_file(path_to_root_file, library=library) 
    
    # do association between waveforms and beam and create events
    events = events_from_wfset_and_beam_info(wfset,
                                             beam_infos,
                                             delta_t_max,
                                             channel_map=APA_map,
                                             ngrids=4)
                
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
    output: List[BeamEvent]
        list of Event objects        
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
    
