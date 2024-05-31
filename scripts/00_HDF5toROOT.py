from hdf5libs import HDF5RawDataFile

# import daqdataformats
from daqdataformats import FragmentType
from rawdatautils.unpack.daphne import *
from rawdatautils.unpack.utils  import *

import detdataformats
import fddetdataformats
import trgdataformats

import os, click, subprocess, stat
import numpy as np

from rich.progress import track
from uproot import recreate as rc
from uproot import update as up
from uproot.models.TString import Model_TString as TString

def extract_fragment_info(frag):

    # waveforms
    frag_id   = str(frag).split(' ')[3][:-1]
    frh       = frag.get_header()
    fragType  = frh.fragment_type
    scr_id    = frag.get_header().element_id.id
    threshold = None
    baseline  = None
    
    if fragType == FragmentType.kDAPHNE.value:  # For self trigger
        trigger        = 'self_trigger'
        frame_obj      = fddetdataformats.DAPHNEFrame
        daphne_headers = [frame_obj(frag.get_data(iframe*frame_obj.sizeof())).get_header() for iframe in range(get_n_frames(frag))]
        threshold      = np.array([header.threshold for header in daphne_headers])
        baseline       = np.array([header.baseline for header in daphne_headers])
        timestamps     = np_array_timestamp(frag)
        adcs           = np_array_adc(frag)
        channels       = np_array_channels(frag)
    elif fragType == 13:  # For full_stream
        trigger = 'full_stream'
        timestamps = np_array_timestamp_stream(frag)
        adcs = np_array_adc_stream(frag)
        channels = np_array_channels_stream(frag)[0]

    # trigger primitive
    trg_obj    = trgdataformats.TriggerPrimitive
    trg_header = [trg_obj(frag.get_data(i_tp*trg_obj.sizeof())) for i_tp in range(int(frag.get_data_size()/trg_obj.sizeof()))]
        
    tp_data = [(tp.time_start, tp.time_peak, tp.time_over_threshold, tp.adc_integral, tp.adc_peak, tp.channel, tp.type, tp.flag) for tp in trg_header if tp.type.value == 2]
    
    if tp_data:
        time_start, time_peak, time_over_threshold, adc_integral, adc_peak = zip(*filtered_data)
    else:
        time_start = time_peak = time_over_threshold = adc_integral = adc_peak = []
    
    return trigger, frag_id, scr_id, channels, adcs, timestamps, threshold, baseline, time_start, time_peak, time_over_threshold, adc_integral, adc_peak

def to_awkward_array(data):
    if not isinstance(data, ak.Array):
        return ak.Array(data)
    return data

@click.command()
@click.option("--path", '-p', default = '/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles', help="Insert the run number, ex: 026102")
@click.option("--run" , '-r', default = None, help="Insert the run number, ex: 026102")
@click.option("--debug",'-b', default = False, help="Insert the run number, ex: 026102", type=bool)

def main(path, run, debug):

    ## Check if the run number is provided ##
    if run is None: 
        print("\033[35mPlease provide a run(s) number(s) to be analysed, separated by commas:)\033[0m")
        run = [int(input("Run(s) number(s): "))]
        
    if len(run)!=1: runs_list = list(map(int, list(run.split(","))))
    else: runs_list = run

    map_id = {'104': [1, 2, 3, 4], '105': [5, 6, 7, 8], '107': [9, 10], '109': [11], '111': [12], '112': [13], '113': [14]}
    for run in runs_list:
        run = str(run).zfill(6) # check if run have 6 digits and fill with zeros if not
        det = 'HD_PDS'

        run_path = f'{path}/1_rucio_paths/{run}.txt'
        
        try: 
            with open(f'{run_path}', "r") as run_list: pass
            print(f"\033[92mFound the file {run_path}\n\033[0m")
        except FileNotFoundError: 
            subprocess.call(f"python get_rucio.py --runs {run}", shell=True)     
            run_path = f'{path}/1_rucio_paths/{run}.txt'
        
        path_root = f'{path}/2_daq_root/run_{run}'
        
        try: 
            os.mkdir(path_root)
            os.chmod(f'{path_root}', stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        except FileExistsError: print("DATA STRUCTURE ALREADY EXISTS") 

        with open(f'{run_path}', "r") as run_list: run_paths = run_list.readlines()
        files = [run_path.rstrip('\n') for run_path in run_paths]
        
        for raw_file in files:
            h5_file    = HDF5RawDataFile(raw_file)
            run_date   = h5_file.get_attribute('creation_timestamp')
            run_id     = raw_file.split('_')[3]
            root_file  = rc(f'{path_root}/{run}_{run_id}.root')
            records    = h5_file.get_all_record_ids()
            
            # Iterate through records 
            for r in track(records, description=f'Dumping {raw_file.split("/")[-1]} ...'):     
                pds_geo_ids = list(h5_file.get_geo_ids_for_subdetector(r, detdataformats.DetID.string_to_subdetector(det)))
                
                ##TODO: store repeated data in log file?
                ##TODO: check if the repeated nrecord has only the first frag repeated or all of them

                if debug: 
                    print("\nTrigger Timestamp from header:", trigger_ts)
                    print("NRECORD: ", r)
                
                # Iterate through geo_ids
                for gid in pds_geo_ids:

                    frag = h5_file.get_frag(r, gid)
                    trigger, frag_id, scr_id, channels, adcs, timestamps, threshold, baseline, time_start, time_peak, time_over_threshold, adc_integral, adc_peak = extract_fragment_info(frag)
                            
                    endpoint = int(find_endpoint(map_id, scr_id))
                    channels = 100*int(endpoint) + channels 

                    if debug: 
                        print("GEO ID: ", gid)
                        print("FRAG ID: ", frag_id)
                        print("EP: ", endpoint)
                        print("CH: ", channels)
                        print("ADCS: ",adcs)

                    data = {
                        'channel'   : channels,
                        'adcs'      : adcs if trigger == 'self_trigger' else adcs.transpose(),
                        'timestamps': timestamps if trigger == 'self_trigger' else [[timestamps[0]]]*len(channels)
                    }

                    data_tp ={
                        'time_start'          : time_start,
                        'time_peak'           : time_peak,
                        'time_over_threshold' : time_over_threshold,
                        'adc_integral'        : adc_integral,
                        'adc_peak'            : adc_peak
                    }
                    
                    if trigger == 'self_trigger':
                        
                        data['threshold']  = threshold
                        data['baseline']   = baseline
                        data['timestamps'] = timestamps

                        adc_size = len(adcs)
                        started = started_st 

                    else: 
                        adc_size = len(adcs.transpose())
                        started = started_fs

                    try:
                        root_file[f'{trigger}/raw_waveforms_{adc_size}']      = data
                        root_file[f'{trigger}/trigger_primitives_{adc_size}'] = data_tp'
 
                    except:
                        root_file[f'{trigger}/raw_waveforms_{adc_size}'].extend(data)
                        root_file[f'{trigger}/trigger_primitives_{adc_size}'].extend(data_tp)
                       
                    
            root_file['metadata'] = ({'run': (int(run),), 'nrecords': (len(records),), 'detector': (TString(det),), 'date' : (TString(run_date),), 'ticks_to_nanoseconds': (float(1/16),), 'adc_to_volts': (float((1.5*3.2)/(2^(14)-1)),) })
if __name__ == "__main__":
    main()
