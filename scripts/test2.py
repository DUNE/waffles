import os
import subprocess
import shlex
import h5py
import numpy as np
from multiprocessing import Pool, cpu_count
from daqdataformats import FragmentType
from hdf5libs import HDF5RawDataFile
from rawdatautils.unpack.daphne import *
from rawdatautils.unpack.utils import *
import detdataformats
import fddetdataformats

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

def extract_fragment_data(frag, trig):
    try:
        frag_id = str(frag).split(' ')[3][:-1]
        frh = frag.get_header()
        trh = trig.get_header()
        run_number = frh.run_number
        scr_id = frh.element_id.id
        fragType = frh.fragment_type
        daq_pretrigger = frh.window_begin - trh.trigger_timestamp
        trigger_ts = frag.get_trigger_timestamp()

        timestamps, adcs, channels, baseline, trigger_sample_value = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

        if fragType == FragmentType.kDAPHNE.value:
            frame_obj = fddetdataformats.DAPHNEFrame
            frame_size = frame_obj.sizeof()
            num_frames = frag.get_data_size() // frame_size

            daphne_headers = [
                frame_obj(frag.get_data(iframe * frame_size)).get_header()
                for iframe in range(num_frames)
            ]

            baseline = np.array([header.baseline for header in daphne_headers])
            trigger_sample_value = np.array([header.trigger_sample_value for header in daphne_headers])

            timestamps = np_array_timestamp(frag)
            adcs = np_array_adc(frag)
            channels = np_array_channels(frag)

        elif fragType == FragmentType.kDAPHNEStream:
            timestamps = np_array_timestamp_stream(frag)
            adcs = np_array_adc_stream(frag)
            channels = np_array_channels_stream(frag)[0]

        return run_number,frag_id, scr_id, channels, adcs, timestamps, baseline, trigger_sample_value, trigger_ts, daq_pretrigger

    except Exception as e:
        print(f"Error in extract_fragment_data for fragment {frag}: {e}")
        return None

def process_file(file_path):
    local_file = get_local_hdf5_file(file_path)
    h5_file = HDF5RawDataFile(local_file)
    records = h5_file.get_all_record_ids()
    raw_data = []

    for r in records:
        geo_ids = list(h5_file.get_geo_ids_for_subdetector(r, detdataformats.DetID.string_to_subdetector("HD_PDS")))
        for gid in geo_ids:
            frag = h5_file.get_frag(r, gid)
            trig = h5_file.get_trh(r)
            extracted_data = extract_fragment_data(frag, trig)
            if extracted_data:
                raw_data.append(extracted_data)

    return raw_data

class WaveformSet:
    def __init__(self, waveforms):
        self.waveforms = waveforms

    def __repr__(self):
        return f"WaveformSet with {len(self.waveforms)} waveforms"

def create_waveformset(raw_data):
    waveforms = []
    for data in raw_data:
        run_number,frag_id, scr_id, channels, adcs, timestamps, baseline, trigger_sample_value, trigger_ts, daq_pretrigger = data
        waveforms.append({
            "run_number":run_number,
            "frag_id": frag_id,
            "scr_id": scr_id,
            "channels": channels,
            "adcs": adcs,
            "timestamps": timestamps,
            "baseline": baseline,
            "trigger_sample_value": trigger_sample_value,
            "trigger_ts": trigger_ts,
            "daq_pretrigger": daq_pretrigger,
        })

    return WaveformSet(waveforms)

if __name__ == "__main__":
    file_path = "root://fndca1.fnal.gov:1094/pnfs/fnal.gov/usr/dune/tape_backed/dunepro//hd-protodune/raw/2024/detector/physics/None/00/02/73/43/np04hd_raw_run027343_0000_dataflow0_datawriter_0_20240621T134132.hdf5"
    
    raw_waveform_data = process_file(file_path)
    waveform_set = create_waveformset(raw_waveform_data)
    
    print(waveform_set)