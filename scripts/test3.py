import os
import subprocess
import shlex
import re
import numpy as np
from hdf5libs import HDF5RawDataFile
from daqdataformats import FragmentType
import detdataformats
import fddetdataformats
from rawdatautils.unpack.daphne import *
from rawdatautils.unpack.utils import *
from typing import Optional

from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.Waveform import Waveform

def get_local_hdf5_file(filepath, local_dir="/tmp"):
    if filepath.startswith("root://"):
        local_file = os.path.join(local_dir, os.path.basename(filepath))
        if not os.path.exists(local_file):
            print(f"Copying {filepath} to {local_file}...")
            try:
                subprocess.run(shlex.split(f"xrdcp {filepath} {local_file}"), check=True)
                print("File copied successfully!")
            except subprocess.CalledProcessError as e:
                print(f"Error copying file: {e}")
                exit(1)
        return local_file
    return filepath

def extract_run_number(filepath: str) -> int:
    match = re.search(r'run(\d{5,})', filepath)
    return int(match.group(1)) if match else -1

def extract_waveform_set(frag, trig, run_number, record_number):
    try:
        frag_id = str(frag).split(' ')[3][:-1]
        frh = frag.get_header()
        trh = trig.get_header()

        scr_id = frh.element_id.id
        fragType = frh.fragment_type
        daq_pretrigger = frh.window_begin - trh.trigger_timestamp
        trigger_ts = frag.get_trigger_timestamp()

        trigger, timestamps, adcs, channels = None, None, None, None

        if fragType == FragmentType.kDAPHNE.value:
            trigger = 'self_trigger'
            timestamps = np_array_timestamp(frag)
            adcs = np_array_adc(frag)
            channels = np_array_channels(frag)
        elif fragType == FragmentType.kDAPHNEStream:
            trigger = 'full_stream'
            timestamps = np_array_timestamp_stream(frag)
            adcs = np_array_adc_stream(frag)
            channels = np_array_channels_stream(frag)[0]
        else:
            return None

        if timestamps is None or adcs is None or channels is None:
            raise ValueError("Failed to extract waveform data: timestamps, adcs, or channels are None.")

        timestamps = np.asarray(timestamps, dtype=int)
        adcs = np.asarray(adcs, dtype=int)
        channels = np.asarray(channels, dtype=int)

        if len(adcs) != len(channels) or len(adcs) != len(timestamps):
            raise ValueError("Mismatched array lengths for timestamps, adcs, and channels.")

        waveforms = [
            Waveform(
                timestamp=int(timestamps[i]),
                time_step_ns=1.0,  # Placeholder value
                daq_window_timestamp=int(trigger_ts),
                adcs=np.array(adcs[i], dtype=int),
                run_number=int(run_number),
                record_number=int(record_number),
                endpoint=int(scr_id),
                channel=int(channels[i])
            ) for i in range(len(channels))
        ]

        return WaveformSet(*waveforms)
    except Exception as e:
        print(f"Error in extract_waveform_set: {e}")
        return None

def WaveformSet_from_hdf5_file(filepath: str, read_full_streaming_data: bool = False,
                               truncate_wfs_to_minimum: bool = False, nrecord_start_fraction: float = 0.0,
                               nrecord_stop_fraction: float = 1.0, subsample: int = 1,
                               wvfm_count: int = int(1e9), ch: Optional[dict] = {},
                               det: str = 'HD_PDS', temporal_copy_directory: str = '/tmp',
                               erase_temporal_copy: bool = False) -> WaveformSet:
    local_file = get_local_hdf5_file(filepath, temporal_copy_directory)
    h5_file = HDF5RawDataFile(local_file)
    records = h5_file.get_all_record_ids()
    run_number = extract_run_number(filepath)
    merged_waveform_set = None

    for r in records:
        geo_ids = list(h5_file.get_geo_ids_for_subdetector(r, detdataformats.DetID.string_to_subdetector(det)))
        for gid in geo_ids:
            frag = h5_file.get_frag(r, gid)
            if frag.get_data_size() == 0:
                continue
            
            fragType = frag.get_header().fragment_type
            if read_full_streaming_data and fragType == FragmentType.kDAPHNE:
                continue
            if not read_full_streaming_data and fragType == FragmentType.kDAPHNEStream:
                continue
            
            trig = h5_file.get_trh(r)
            waveform_set = extract_waveform_set(frag, trig, run_number, record_number=r)
            if waveform_set:
                if merged_waveform_set is None:
                    merged_waveform_set = waveform_set
                else:
                    merged_waveform_set.merge(waveform_set)
    
    if erase_temporal_copy and filepath.startswith("root://"):
        os.remove(local_file)
    
    return merged_waveform_set

if __name__ == "__main__":
    file_path = "root://fndca1.fnal.gov:1094/pnfs/fnal.gov/usr/dune/tape_backed/dunepro//hd-protodune/raw/2024/detector/physics/None/00/02/73/43/np04hd_raw_run027343_0000_dataflow0_datawriter_0_20240621T134132.hdf5"
    merged_waveform_set = WaveformSet_from_hdf5_file(file_path)
    print("Merged WaveformSet:", merged_waveform_set)
