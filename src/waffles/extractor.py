from hdf5libs import HDF5RawDataFile

import daqdataformats
import detdataformats
from daqdataformats import FragmentType
from rawdatautils.unpack.daphne import *

import numpy as np
import click
from uproot import recreate as rc

def extract_fragment_info(frag):
    frag_id = str(frag).split(' ')[3][:-1]
    fragType = frag.get_header().fragment_type
    
    if fragType == FragmentType.kDAPHNE.value:  # For self trigger
        trigger = 'self_trigger'
        timestamps = np_array_timestamp(frag)
        adcs = np_array_adc(frag)
        channels = np_array_channels(frag)
    elif fragType == 13:  # For full_stream
        trigger = 'full_stream'
        timestamps = np_array_timestamp_stream(frag)
        adcs = np_array_adc_stream(frag)
        channels = np_array_channels_stream(frag)[0]
    
    return trigger, frag_id, channels, adcs, timestamps

@click.command()
@click.option("--run", '-r', default = None, help="Insert the run number, ex: 026102")

def main(run):
    det         = 'HD_PDS'
    run_path    = f'/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/rucio_paths/{run}.txt'
    root_file   = rc(f'/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/root_files/{run}.root')

    with open(f'{run_path}', "r") as run_list:
        run_paths = run_list.readlines()
    files = [run_path.rstrip('\n') for run_path in run_paths]
    
    for file in files:
        print(f'Reading {file}...')
        h5_file   = HDF5RawDataFile(file)
        records   = h5_file.get_all_record_ids()
        tr_ref    = None
        
        # Iterate through records        
        for r in records:
            pds_geo_ids = list(h5_file.get_geo_ids_for_subdetector(r, detdataformats.DetID.string_to_subdetector(det)))
            
            # Iterate through geo_ids
            for gid in pds_geo_ids:
                frag = h5_file.get_frag(r, gid)
                tr_header = frag.get_header().trigger_timestamp

                if hex(gid)[3] == '0': endpoint = hex(gid)[2]
                else                 : endpoint = hex(gid)[2:4]

                # Filtering data with different timestamp on the header
                if tr_header != tr_ref:
        
                    trigger, frag_id, channels, adcs, timestamps = extract_fragment_info(frag)
                    channels = 100*int(endpoint) + channels   

                    if trigger == 'full_stream': adcs = adcs.transpose()
                   
                    if tr_ref is None:
                        root_file[f'waveform_primitive']       = ({'channel': channels,'adcs': np.array(adcs)})
                        root_file[f'timestamps']               = ({'timestamps': timestamps})
                    else:
                        root_file[f'waveform_primitive'].extend({'channel': channels,'adcs': np.array(adcs)})
                        root_file[f'timestamps'].extend ({'timestamps': timestamps})            
                           
                tr_ref = tr_header
                    
if __name__ == "__main__":
    main()
