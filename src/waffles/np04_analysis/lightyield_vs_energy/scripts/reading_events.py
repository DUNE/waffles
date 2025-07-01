from waffles.input_output.event_file_reader import events_from_pickle_file
from waffles.data_classes.IPDict import IPDict
from waffles.data_classes.BasicWfAna import BasicWfAna
import numpy as np
import json


def read_events_from_pickle_file(events):
    
    t0 = events[0].ref_timestamp

    for e in events:
        if e.wfset:
            print('yes')
            nwfs = len(e.wfset.waveforms)
            
            
            b_ll = 0
            b_ul = 50
            int_ll = 55
            int_ul = 115
            
            # baseline limits
            bl = [b_ll, b_ul, 900, 1000]
            
            peak_finding_kwargs = dict( prominence = 20,rel_height=0.5,width=[0,75])
            ip = IPDict(baseline_limits=bl,
                        int_ll=int_ll,int_ul=int_ul,amp_ll=int_ll,amp_ul=int_ul,
                        points_no=10,
                        peak_finding_kwargs=peak_finding_kwargs,
                        baseline_method='EasyMedian'
                        )
            analysis_kwargs = dict(  return_peaks_properties = False)
            checks_kwargs   = dict( points_no = e.wfset.points_per_wf)
            
            a=e.wfset.analyse('standard',BasicWfAna,ip,checks_kwargs = checks_kwargs,overwrite=True)
            
            print (e.record_number,
                       e.event_number,
                       e.first_timestamp-t0,
                       (e.last_timestamp-e.first_timestamp)*0.016,
                       ', p =', e.beam_info.p,
                       ', nwfs =', nwfs,
                       ', c0 =', e.beam_info.c0,
                       ', c1 =', e.beam_info.c1,
                       ', tof =', e.beam_info.tof)
            
        
def particle_selection(events, selected_particle):
    """Select events based on the particle type."""
    selected_events = []
    for event in events:
        if event.beam_info.particle == selected_particle:
            selected_events.append(event)
    return selected_events
    
    
    
pickle_file_path = '/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/beam_example/output/events.pkl'
events = events_from_pickle_file(pickle_file_path)

#read_events_from_pickle_file(pickle_file_path)


selected_particle = 'e'
beam_energy = 1
wfset = particle_selection(events, selected_particle)

