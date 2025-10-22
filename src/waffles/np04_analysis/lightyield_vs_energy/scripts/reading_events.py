from waffles.input_output.event_file_reader import events_from_pickle_file
from waffles.data_classes.IPDict import IPDict
from waffles.data_classes.BasicWfAna import BasicWfAna
import numpy as np
import json

# NON FUNZIONA  
def particle_selection(events, selected_particle):
    """Select events based on the particle type."""
    selected_events = []
    for event in events:
        if event.beam_info.particle == selected_particle:
            selected_events.append(event)
    return selected_events

# selected_particle = 'e'
# beam_energy = 1
# wfset = particle_selection(events, selected_particle)

##################################################################################

def get_events_info(events):
    t0 = events[0].ref_timestamp
    for e in events:
        if e.wfset:
            print('yes')
            nwfs = len(e.wfset.waveforms)
            print (e.record_number,
                       e.event_number,
                       e.first_timestamp-t0,
                       (e.last_timestamp-e.first_timestamp)*0.016,
                       ', p =', e.beam_info.p,
                       ', nwfs =', nwfs,
                       ', c0 =', e.beam_info.c0,
                       ', c1 =', e.beam_info.c1,
                       ', tof =', e.beam_info.tof)


def events_filter(event):
    a=1
    if a>0:    
        return True
    else:
        return False
    

def unify_wfset(events):
    i = 0
    for e in events:
        if e.wfset and events_filter(e):    
            if i == 0:
                unified_wfset = e.wfset
            else:
                unified_wfset.merge(e.wfset)
            i += 1
    return unified_wfset    
    
pickle_file_path = '/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/beam_example/output/events.pkl'
events = events_from_pickle_file(pickle_file_path)
wfset = unify_wfset(events)