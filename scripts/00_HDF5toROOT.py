from hdf5libs import HDF5RawDataFile

# import daqdataformats
from daqdataformats import FragmentType
from rawdatautils.unpack.daphne import *
from rawdatautils.unpack.utils  import *

import detdataformats
import fddetdataformats

import os, click, subprocess, stat
import numpy as np
from array import array

from tqdm import tqdm
from ROOT import TFile, TTree, std
from multiprocessing import Pool, current_process, cpu_count

map_id     = {'104': [1, 2, 3, 4], '105': [5, 6, 7, 9], '107': [10, 8], '109': [11], '111': [12], '112': [13], '113': [14]}
def find_endpoint(map_id, target_value):
    for key, value_list in map_id.items():
        if target_value in value_list:
            return key

def split_list(original_list, n_splits):
    avg = len(original_list) / float(n_splits)
    out = []
    last = 0.0

    while last < len(original_list):
        out.append(original_list[int(last):int(last + avg)])
        last += avg

    return out

def check_PDS(raw_file):
    h5_file  = HDF5RawDataFile(raw_file)
    record   = list(h5_file.get_all_record_ids())[0]
    gid      = list(h5_file.get_geo_ids_for_subdetector(record, detdataformats.DetID.string_to_subdetector('HD_PDS')))
    if len(gid) == 0:
        output = False
    else:
        output = True
    return output

def extract_fragment_info(frag, trig):
    frag_id = str(frag).split(' ')[3][:-1]
    frh     = frag.get_header()
    trh     = trig.get_header()
    
    scr_id           = frh.element_id.id
    fragType         = frh.fragment_type
    window_begin_dts = frh.window_begin
    
    trigger_timestamp = trh.trigger_timestamp
    daq_pretrigger    = window_begin_dts - trigger_timestamp
    
    threshold = -1
    baseline  = -1
    trigger_sample_value = -1
    
    if fragType == FragmentType.kDAPHNE.value:  # For self trigger
        trigger              = 'self_trigger'
        frame_obj            = fddetdataformats.DAPHNEFrame
        daphne_headers       = [frame_obj(frag.get_data(iframe*frame_obj.sizeof())).get_header() for iframe in range(get_n_frames(frag))]
        threshold            = daphne_headers[0].threshold  #[header.threshold for header in daphne_headers]
        baseline             = [header.baseline for header in daphne_headers]
        trigger_sample_value = [header.trigger_sample_value for header in daphne_headers]

        timestamps = np_array_timestamp(frag)
        adcs       = np_array_adc(frag)
        channels   = np_array_channels(frag)
        
    elif fragType == 13:  # For full_stream
        trigger    = 'full_stream'
        timestamps = np_array_timestamp_stream(frag)
        adcs       = np_array_adc_stream(frag)
        channels   = np_array_channels_stream(frag)[0]
       
    return trigger, frag_id, scr_id, channels, adcs, timestamps, threshold, baseline, trigger_sample_value, daq_pretrigger

def root_creator(inputs):
    raw_file, path_root, run, debug, overwrite = inputs

    det      = 'HD_PDS'
    h5_file  = HDF5RawDataFile(raw_file)
    run_date = h5_file.get_attribute('creation_timestamp')
    
    run_id     = raw_file.split('_')[3]
    run_flow   = raw_file.split('_')[4]
    datawriter = dataflow = raw_file.split('_')[6]
    run_numb = (raw_file.split('_')[7]).split('.')[0]

    root_name = f'run{run}_{run_id}_{run_flow}_datawriter_{datawriter}_{run_numb}.root'   
    if overwrite:
        root_file = TFile( f'{path_root}/{root_name}' , 'RECREATE')
    else:
        created_files = [f for f in os.listdir(path_root) if os.path.isfile(os.path.join(path_root, f))]
        if root_name not in created_files:
            root_file = TFile( f'{path_root}/{root_name}' , 'RECREATE')
        else:
            print(f'File {root_name} already exists and will not be overwritten.')
            return 0
            
        
    fWaveformTree = TTree("raw_waveforms", "raw_waveforms")

    adcs                 = std.vector('float')()
    timestamps           = array('l', [0])
    channel              = array('i', [0])
    baseline             = array('i', [0])
    trigger_sample_value = array('i', [0])
    is_fullstream        = array('b', [0])
    daq_timestamp        = array('l', [0])
    record               = array('i', [0])
    
    fWaveformTree.Branch("record", record, "record/I")
    fWaveformTree.Branch("daq_timestamp", daq_timestamp, "daq_timestamp/L")
    fWaveformTree.Branch("adcs", adcs)#, "adcs[{}]/F".format(len(adcs)))
    fWaveformTree.Branch("timestamps", timestamps, "timestamps/L")
    fWaveformTree.Branch("channel", channel, "channel/I")
    fWaveformTree.Branch("baseline", baseline, "baseline/I")
    fWaveformTree.Branch("trigger_sample_value", trigger_sample_value, "trigger_sample_value/I")
    fWaveformTree.Branch("is_fullstream", is_fullstream, "is_fullstream/O")

    fMetaDataTree = TTree("metadata", "metadata")

    edp           = std.vector('int')()
    threshold     = std.vector('int')()
    run_          = array('i', [0])
    nrecords      = array('i', [0])
    date          = array('l', [0])
    ticks_to_nsec = array('f', [0])
    adc_to_volts  = array('f', [0])

    fMetaDataTree.Branch("endpoint", edp)
    fMetaDataTree.Branch("threshold", threshold)
    fMetaDataTree.Branch("run", run_, "run/I")
    fMetaDataTree.Branch("nrecords", nrecords, "nrecords/I")
    fMetaDataTree.Branch("date", date, "date/L")
    fMetaDataTree.Branch("ticks_to_nsec", ticks_to_nsec, "ticks_to_nsec/F")
    fMetaDataTree.Branch("adc_to_volts", adc_to_volts, "adc_to_volts/F")
    
    active_endpoints = set()
    threshold_list   = []

    records = h5_file.get_all_record_ids()
    current = current_process()
    for r in tqdm(records, position = current._identity[0]-1, desc = f'{root_name}'):
        pds_geo_ids = list(h5_file.get_geo_ids_for_subdetector(r, detdataformats.DetID.string_to_subdetector(det)))

        for gid in pds_geo_ids:
            try:
                frag = h5_file.get_frag(r, gid)
                trig = h5_file.get_trh(r)
                
                trigger, frag_id, scr_id, channels_frag, adcs_frag, timestamps_frag, threshold_frag, baseline_frag, trigger_sample_value_frag, daq_pretrigger_frag = extract_fragment_info(frag, trig)
                        
                endpoint = int(find_endpoint(map_id, scr_id))
                channels_frag = 100 * int(endpoint) + channels_frag 
                
                if debug: 
                    print("GEO ID:", gid)
                    print("FRAG ID:", frag_id)
                    print("EP:", endpoint)
                    print("CH:", channels)
                    print("ADCS:", adcs)
    
                if trigger == 'full_stream':
                    adcs_frag                 = adcs_frag.transpose()
                    timestamps_frag           = [timestamps_frag[0]] * len(channels_frag)
                    baseline_frag             = [-1] * len(channels_frag)
                    trigger_sample_value_frag = [-1] * len(channels_frag)
                    is_fullstream_frag        = [True] * len(channels_frag)
                elif trigger == 'self_trigger':
                    is_fullstream_frag = [False] * len(channels_frag)
                
                if endpoint not in active_endpoints: 
                    active_endpoints.add(endpoint)
                    threshold_list.append(threshold_frag)
                    
                for index, ch in enumerate(channels_frag):
    
                    adcs.clear()
                    for value in adcs_frag[index]:
                        adcs.push_back(value)
                    timestamps[0]           = timestamps_frag[index]
                    channel[0]              = ch
                    baseline[0]             = baseline_frag[index]
                    trigger_sample_value[0] = trigger_sample_value_frag[index]
                    is_fullstream[0]        = is_fullstream_frag[index]
                    daq_timestamp[0]        = daq_pretrigger_frag
                    record[0]               = r[0]
                    fWaveformTree.Fill()
    
            except:
                print(f'Corrupted PDS data on record {r} and GeoID {gid}')

    fWaveformTree.Write("", TFile.kOverwrite)    

    for edp_value in active_endpoints:
        edp.push_back(edp_value)
    for threshold_value in threshold_list:
        threshold.push_back(int(threshold_value))

    run_[0]      = int(run)
    nrecords[0]  = len(records)
    date[0]      = np.uint64(run_date)
    ticks_to_nsec[0] = 1/16
    adc_to_volts[0]  = (1.5 * 3.2)/(2 ** 14 - 1)
    fMetaDataTree.Fill()
    
    fMetaDataTree.Write("", TFile.kOverwrite)
    root_file.Close()
    
@click.command()
@click.option("--path", '-p', default = '/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles', help="Insert the desired path.")
@click.option("--run" , '-r', default = None, help="Insert the run number, ex: 026102")
@click.option("--debug",'-b', default = False, help="To debug, make -b True", type=bool)
@click.option("--overwrite", '-o', default = False, help="If you want to overwrite make -o True", type= bool)

def main(path, run, debug, overwrite):

    ## Check if the run number is provided ##
    if run is None: 
        print("\033[35mPlease provide a run(s) number(s) to be analysed, separated by commas:)\033[0m")
        run = [int(input("Run(s) number(s): "))]
        
    if len(run)!=1: runs_list = list(map(int, list(run.split(","))))
    else: runs_list = run

    num_cores_mp = cpu_count()
    
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

        with open(f'{run_path}', "r") as run_list: 
            run_paths = run_list.readlines()

        files = [run_path.rstrip('\n') for run_path in run_paths]
        if check_PDS(files[0]):
     
            print(f'Cores avaliable:{num_cores_mp}')
            print(f'Files to be processed: {len(run_paths)}')
        
            if num_cores_mp > len(run_paths): 
                cores    = len(run_paths)
                steps    = 1
            else: 
                cores    = num_cores_mp
                steps    = len(run_paths)//num_cores_mp
            
            split_files = split_list(files, steps)
            
            for st in range(steps):
                print(f'\n - Step {st} -> {cores} cores')
                with Pool(cores) as pool:
                    inputs  = [[raw_file, path_root, run, debug, overwrite] for raw_file in split_files[st]]
                    process = pool.map(root_creator, inputs, chunksize=1)
            
        else: print(f'No PDS data on run {run}!')
            
        
if __name__ == "__main__":
    main()

