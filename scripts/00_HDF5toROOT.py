from hdf5libs import HDF5RawDataFile

# import daqdataformats
from daqdataformats import FragmentType
from rawdatautils.unpack.daphne import *
from rawdatautils.unpack.utils  import *

import detdataformats
import fddetdataformats

import os, click, subprocess, stat
import numpy as np

from tqdm import tqdm
from uproot import recreate as rc
from uproot.models.TString import Model_TString as TString
from multiprocessing import Pool

map_id     = {'104': [1, 2, 3, 4], '105': [5, 6, 7, 8], '107': [9, 10], '109': [11], '111': [12], '112': [13], '113': [14]}
def find_endpoint(map_id, target_value):
    for key, value_list in map_id.items():
        if target_value in value_list:
            return key

def extract_fragment_info(frag):

    # waveforms
    frag_id   = str(frag).split(' ')[3][:-1]
    frh       = frag.get_header()
    fragType  = frh.fragment_type
    scr_id    = frag.get_header().element_id.id
    threshold = None
    baseline  = None
    trigger_sample_value = None
    
    if fragType == FragmentType.kDAPHNE.value:  # For self trigger
        trigger              = 'self_trigger'
        frame_obj            = fddetdataformats.DAPHNEFrame
        daphne_headers       = [frame_obj(frag.get_data(iframe*frame_obj.sizeof())).get_header() for iframe in range(get_n_frames(frag))]
        threshold            = np.array([header.threshold for header in daphne_headers])
        baseline             = np.array([header.baseline for header in daphne_headers])
        trigger_sample_value = np.array([header.trigger_sample_value for header in daphne_headers])

        timestamps     = np_array_timestamp(frag)
        adcs           = np_array_adc(frag)
        channels       = np_array_channels(frag)
    elif fragType == 13:  # For full_stream
        trigger = 'full_stream'
        timestamps = np_array_timestamp_stream(frag)
        adcs = np_array_adc_stream(frag)
        channels = np_array_channels_stream(frag)[0]
       
    return trigger, frag_id, scr_id, channels, adcs, timestamps, threshold, baseline, trigger_sample_value

def root_creator(inputs):
    raw_file, path_root, run, debug = inputs
                
    det        = 'HD_PDS'
    h5_file    = HDF5RawDataFile(raw_file)
    run_date   = h5_file.get_attribute('creation_timestamp')
    run_id     = raw_file.split('_')[3]
    root_file  = rc(f'{path_root}/{run}_{run_id}.root')
    records    = h5_file.get_all_record_ids()

    # Iterate through records
    #for r in records:
    for r in tqdm(records):
        pds_geo_ids = list(h5_file.get_geo_ids_for_subdetector(r, detdataformats.DetID.string_to_subdetector(det)))

        if debug: 
            print("\nTrigger Timestamp from header:", trigger_ts)
            print("NRECORD: ", r)
        
        # Iterate through geo_ids
        for gid in pds_geo_ids:

            frag = h5_file.get_frag(r, gid)
            trigger, frag_id, scr_id, channels, adcs, timestamps, threshold, baseline, trigger_sample_value = extract_fragment_info(frag)
                    
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
                'timestamps': timestamps if trigger == 'self_trigger' else [[timestamps[0]]]*len(channels)
            }

            if trigger == 'self_trigger':
                data['adcs']                 = adcs
                data['threshold']            = threshold
                data['baseline']             = baseline
                data['trigger_sample_value'] = trigger_sample_value
                
            else:
                adcs         = adcs.transpose()
                adcs         = [selected_adcs[:262100] for selected_adcs in adcs]
                data['adcs'] = adcs

            try:
                root_file[f'raw_waveforms_{trigger}'].extend(data)
            except:
                root_file[f'raw_waveforms_{trigger}'] = data
               
    root_file['metadata'] = ({'run': (int(run),), 'nrecords': (len(records),), 'detector': (TString(det),), 'date' : (TString(run_date),), 'ticks_to_nanoseconds': (float(1/16),), 'adc_to_volts': (float((1.5*3.2)/(2^(14)-1)),) })

@click.command()
@click.option("--path", '-p', default = '/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles', help="Insert the desired path.")
@click.option("--run" , '-r', default = None, help="Insert the run number, ex: 026102")
@click.option("--debug",'-b', default = False, help="Insert the run number, ex: 026102", type=bool)

def main(path, run, debug):

    ## Check if the run number is provided ##
    if run is None: 
        print("\033[35mPlease provide a run(s) number(s) to be analysed, separated by commas:)\033[0m")
        run = [int(input("Run(s) number(s): "))]
        
    if len(run)!=1: runs_list = list(map(int, list(run.split(","))))
    else: runs_list = run
    
    for run in runs_list:
        run = str(run).zfill(6) # check if run have 6 digits and fill with zeros if not
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
        files  = [run_path.rstrip('\n') for run_path in run_paths]
        inputs = [[raw_file, path_root, run, debug] for raw_file in files]
        
        with Pool(processes=len(files)) as pool:
            process = pool.map(root_creator, inputs)
        
if __name__ == "__main__":
    main()

