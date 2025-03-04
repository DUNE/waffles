import click
import json
import os
from pathlib import Path
from waffles.utils.utils import print_colored
import waffles.input_output.raw_hdf5_reader as reader
from waffles.input_output.persistence_utils import WaveformSet_to_file
from multiprocessing import Pool, cpu_count
from numba import jit
import numpy as np
import subprocess
import shlex
import h5py
from daqdataformats import FragmentType
from hdf5libs import HDF5RawDataFile
from rawdatautils.unpack.daphne import *
from rawdatautils.unpack.utils import *
import detdataformats
import fddetdataformats
from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.WaveformSet import WaveformSet

# Function to copy file locally if it's remote
def get_local_hdf5_file(filepath, local_dir="/tmp"):
    if filepath.startswith("root://"):  # Check if it's an XRootD path
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
    return filepath  # Return original if it's already local

@jit(nopython=True)
def extract_fragment_info(frag, trig):
    frag_id = str(frag).split(' ')[3][:-1]
    frh = frag.get_header()
    trh = trig.get_header()

    scr_id = frh.element_id.id
    fragType = frh.fragment_type
    window_begin_dts = frh.window_begin
    trigger_timestamp = trh.trigger_timestamp
    daq_pretrigger = window_begin_dts - trigger_timestamp

    timestamps = np.array([])  # NumPy array for efficiency
    adcs = np.array([])
    channels = np.array([])
    trigger = 'unknown'
    trigger_ts = frag.get_trigger_timestamp()

    if fragType == FragmentType.kDAPHNE.value:
        trigger = 'self_trigger'
        frame_obj = fddetdataformats.DAPHNEFrame
        daphne_headers = [frame_obj(frag.get_data(iframe * frame_obj.sizeof())).get_header() for iframe in range(get_n_frames(frag))]
        baseline = np.array([header.baseline for header in daphne_headers])
        trigger_sample_value = np.array([header.trigger_sample_value for header in daphne_headers])
        timestamps = np_array_timestamp(frag)  # NumPy optimizations
        adcs = np_array_adc(frag)
        channels = np_array_channels(frag)

    elif fragType == FragmentType.kDAPHNEStream:
        trigger = 'full_stream'
        timestamps = np_array_timestamp_stream(frag)
        adcs = np_array_adc_stream(frag)
        channels = np_array_channels_stream(frag)[0]

    return trigger, frag_id, scr_id, channels, adcs, timestamps, baseline, trigger_sample_value, trigger_ts, daq_pretrigger

# Function to create a WaveformSet per fragment
def waveform_from_fragment(frag, trig):
    extracted_info = extract_fragment_info(frag, trig)
    waveforms = [Waveform(*extracted_info)]
    return WaveformSet(*waveforms)

@click.command(help="\033[34mTest waveform processing using JSON configuration.\033[0m")
@click.option("--config", required=True, help="Path to JSON configuration file.", type=str)
def test_waveform_processing(config):
    try:
        with open(config, 'r') as f:
            config_data = json.load(f)

        required_keys = ["run", "rucio_dir", "output_dir", "ch"]
        missing_keys = [key for key in required_keys if key not in config_data]
        if missing_keys:
            raise ValueError(f"Missing required keys in config file: {missing_keys}")

        print_colored(f"Loaded configuration: {config_data}", color="INFO")
        print_colored("Testing waveform processing...", color="DEBUG")

        filepaths = reader.get_filepaths_from_rucio(f"{config_data['rucio_dir']}/{str(config_data['run']).zfill(6)}.txt")
        if config_data.get("max_files") != "all":
            filepaths = filepaths[:int(config_data["max_files"])]

        for file in filepaths:
            print_colored(f"Processing file: {file}", color="INFO")
            local_file = get_local_hdf5_file(file)
            h5_file = HDF5RawDataFile(local_file)
            records = h5_file.get_all_record_ids()
            waveform_sets = []

            for r in records:
                pds_geo_ids = h5_file.get_geo_ids_for_subdetector(r, detdataformats.DetID.string_to_subdetector("HD_PDS"))
                for gid in pds_geo_ids:
                    frag = h5_file.get_frag(r, gid)
                    trig = h5_file.get_trh(r)
                    waveform_set = waveform_from_fragment(frag, trig)
                    waveform_sets.append(waveform_set)

            merged_waveform_set = WaveformSet(*[wf for ws in waveform_sets for wf in ws.waveforms])
            print(f"Extracted {len(merged_waveform_set.waveforms)} waveforms from {file}")

        print_colored("Test completed successfully.", color="SUCCESS")
    except Exception as e:
        print_colored(f"An error occurred: {e}", color="ERROR")

if __name__ == "__main__":
    test_waveform_processing()
