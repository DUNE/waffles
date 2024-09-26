import os, io, click, subprocess, stat, math, shlex
from array import array
from tqdm import tqdm
import numpy as np
from XRootD import client
from typing import List, Optional
from hdf5libs import HDF5RawDataFile

# import daqdataformats
from daqdataformats import FragmentType
from rawdatautils.unpack.daphne import *
from rawdatautils.unpack.utils  import *
import detdataformats
import fddetdataformats

from multiprocessing import Pool, current_process, cpu_count

import waffles.utils.check_utils as wuc
from waffles.Exceptions import GenerateExceptionMessage
from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.WaveformSet import WaveformSet

# these functions should probably have a shared location, since they are used elsewhere...
map_id = {'104': [1, 2, 3, 4], '105': [5, 6, 7, 9], '107': [
    10, 8], '109': [11], '111': [12], '112': [13], '113': [14]}


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
    h5_file = HDF5RawDataFile(raw_file)
    record = list(h5_file.get_all_record_ids())[0]
    gid = list(h5_file.get_geo_ids_for_subdetector(
        record, detdataformats.DetID.string_to_subdetector('HD_PDS')))
    if len(gid) == 0:
        output = False
    else:
        output = True
    return output


def extract_fragment_info(frag, trig):
    frag_id = str(frag).split(' ')[3][:-1]
    frh = frag.get_header()
    trh = trig.get_header()

    scr_id = frh.element_id.id
    fragType = frh.fragment_type
    window_begin_dts = frh.window_begin

    trigger_timestamp = trh.trigger_timestamp
    daq_pretrigger = window_begin_dts - trigger_timestamp

    threshold = -1
    baseline = -1
    trigger_sample_value = -1

    if fragType == FragmentType.kDAPHNE.value:  # For self trigger
        trigger              = 'self_trigger'
        frame_obj            = fddetdataformats.DAPHNEFrame
        daphne_headers       = [frame_obj(frag.get_data(iframe*frame_obj.sizeof())).get_header() for iframe in range(get_n_frames(frag))]
        # threshold            = daphne_headers[0].threshold  #[header.threshold for header in daphne_headers]
        baseline             = [header.baseline for header in daphne_headers]
        trigger_sample_value = [header.trigger_sample_value for header in daphne_headers]

        timestamps = np_array_timestamp(frag)
        adcs = np_array_adc(frag)
        channels = np_array_channels(frag)

    elif fragType == 13:  # For full_stream
        trigger = 'full_stream'
        timestamps = np_array_timestamp_stream(frag)
        adcs = np_array_adc_stream(frag)
        channels = np_array_channels_stream(frag)[0]

    return trigger, frag_id, scr_id, channels, adcs, timestamps, threshold, baseline, trigger_sample_value, daq_pretrigger


def filepath_is_hdf5_file_candidate(filepath: str) -> bool:
    """
    This function returns True if the given file path points
    to a file which exists and whose extension is '.hdf5' or '.h5'. It
    returns False if else.

    Parameters
    ----------
    filepath : str
        The file path to be checked.

    Returns
    ----------
    bool
    """

    if os.path.isfile(filepath):
        if filepath.endswith('.hdf5') or filepath.endswith('.h5'):
            return True

    return False


def get_filepaths_from_rucio(rucio_filepath) -> list:
    """
    Function to convert the info from the rucio txt file to a
    filepath_list to be used as an input for WaveformSet_from_*_files.
    It will also perform a quality check to ensure that the *.hdf5
    are stil exist there. If not a exception to re-generate them
    will appear.

    Parameters
    ----------
    rucio_filepath : str
        Path to the txt file storing the rucio paths of your run.
    """

    if not os.path.isfile(rucio_filepath):
            raise Exception(generate_exception_message( 1,
                                                        'get_filepaths_from_rucio()',
                                                        f"The given rucio_filepath ({rucio_filepath}) is not a valid file."))

    with open(rucio_filepath, 'r') as file:
        lines = file.readlines()

    filepaths = [line.strip() for line in lines]
    quality_check = filepaths[0]
    if "eos" in quality_check:
        print("Your files are stored in /eos/")
        if not os.path.isfile(filepaths[0]):
                raise Exception(generate_exception_message( 2,
                                                            'get_filepaths_from_rucio()',
                                                            f"The given filepaths[0] ({quality_check}) is not a valid file."))
    else:
        print("\nYour files are stored around the world. \n[WARNING] Check you have a correct configuration to use XRootD" )

    return filepaths

def WaveformSet_from_hdf5_files(filepath_list : List[str] = [],
                                read_full_streaming_data : bool = False,
                                folderpath : Optional[str] = None,
                                nrecord_start_fraction : float = 0.0, \
                                nrecord_stop_fraction : float = 1.0, \
                                subsample : int = 1, \
                                wvfm_count : int = 1e9) -> WaveformSet:
    """
    Alternative initializer for a WaveformSet object that reads waveforms directly from hdf5 files.
    The WaveformSet object made from each hdf5 file is combined into a single WaveformSet object.

    Parameters
    ----------
    filepath_list : str
        Path to the hdf5 file to be read.
    read_full_streaming_data : bool
        If True (resp. False), then only the waveforms for which
        the 'is_fullstream' parameter in the fragment has a
        value equal to True (resp. False) will be considered.
    folderpath : str
        If given, then the value given to the 'filepath_list'
        parameter is ignored, and the list of filepaths to be
        read is generated by listing all the files in the given
        folder.
    nrecord_start_fraction : float
        Used to select at which record to start reading.
        In particular floor(nrecord_start_fraction*(total records)) is the first record.
    nrecord_stop_fraction : float
        Used to select at which record to stop reading.
        In particular ceiling(nrecord_stop_fraction*(total records)) is the last record.
    subsample : int
        Select a subsampling of waveforms from the file.
        So 1 (default) selects every waveform, 2 selects every other, 3 selects every third, etc.
        Can combine with nrecord selection parameters.
    wvfm_count : int
        Select total number of waveforms to save.
    """
    if folderpath is not None:

        if not os.path.isdir(folderpath):
            raise Exception(GenerateExceptionMessage(1,
                                                     'WaveformSet_from_hdf5_files()',
                                                     f"The given folderpath ({folderpath}) is not a valid directory."))

        valid_filepaths = [os.path.join(folderpath, filename)
                           for filename in os.listdir(folderpath)
                           if filepath_is_hdf5_file_candidate(os.path.join(folderpath, filename))]
    else:
        valid_filepaths = [filepath
                           # Remove possible duplicates
                           for filepath in set(filepath_list)
                           if filepath_is_hdf5_file_candidate(filepath)]
    output = WaveformSet_from_hdf5_file(
        filepath_list[0], read_full_streaming_data, nrecord_start_fraction, nrecord_stop_fraction, subsample, wvfm_count)

    for filepath in filepath_list[1:]:
        aux = WaveformSet_from_hdf5_file(filepath, read_full_streaming_data, nrecord_start_fraction, nrecord_stop_fraction, subsample, wvfm_count)
        output.merge(aux)

    return output

def WaveformSet_from_hdf5_file( filepath : str,
                                read_full_streaming_data : bool = False, \
                                nrecord_start_fraction : float = 0.0, \
                                nrecord_stop_fraction : float = 1.0, \
                                subsample : int = 1, \
                                wvfm_count : int = 1e9) -> WaveformSet:
    """
    Alternative initializer for a WaveformSet object that reads waveforms directly from hdf5 files.

    Parameters
    ----------
    filepath : str
        Path to the hdf5 file to be read.
    read_full_streaming_data : bool
        If True (resp. False), then only the waveforms for which
        the 'is_fullstream' parameter in the fragment has a
        value equal to True (resp. False) will be considered.
    nrecord_start_fraction : float
        Used to select at which record to start reading.
        In particular floor(nrecord_start_fraction*(total records)) is the first record.
    nrecord_stop_fraction : float
        Used to select at which record to stop reading.
        In particular ceiling(nrecord_stop_fraction*(total records)) is the last record.
    subsample : int
        Select a subsampling of waveforms from the file.
        So 1 (default) selects every waveform, 2 selects every other, 3 selects every third, etc.
        Can combine with nrecord selection parameters.
    wvfm_count : int
        Select total number of waveforms to save.
    """

    if "/eos" not in filepath:
        print("Using XROOTD")

        subprocess.call(shlex.split(f"xrdcp {filepath} /tmp/."), shell=False)
        filepath = f"/tmp/{filepath.split('/')[-1]}"

    h5_file = HDF5RawDataFile(filepath)
    det        = 'HD_PDS'
    run_date   = h5_file.get_attribute('creation_timestamp')
    run_id     = filepath.split('/')[-1].split('_')[3]
    run_flow   = filepath.split('/')[-1].split('_')[4]
    datawriter = dataflow = filepath.split('/')[-1].split('_')[6]
    run_numb   = int((filepath.split('/')[-1].split('_')[2]).strip('run'))
    #run_numb  = (filepath.split('_')[7]).split('.')[0]
    print('run_numb=',run_numb)

    waveforms = []
    active_endpoints = set()
    threshold_list = []

    records = h5_file.get_all_record_ids()
    if nrecord_stop_fraction > 1.0:
        nrecord_stop_fraction = 1.0
    if nrecord_start_fraction > 1.0 or nrecord_start_fraction < 0.0:
        raise ValueError('Invalid value for nrecord_start_fraction. Must be >=0 or <=1.')
    nrecord_start_index = int(np.floor(nrecord_start_fraction*(len(records)-1)))
    nrecord_stop_index = int(np.ceil(nrecord_stop_fraction*(len(records)-1)))
    records = records[nrecord_start_index:nrecord_stop_index+1]
    # print(f'total number of records = {len(records)}')

    wvfm_index = 0
    for i, r in tqdm(enumerate(records)):
        pds_geo_ids = list(h5_file.get_geo_ids_for_subdetector(
            r, detdataformats.DetID.string_to_subdetector(det)))

        for gid in pds_geo_ids:
            frag = h5_file.get_frag(r, gid)
            trig = h5_file.get_trh(r)

            trigger, frag_id, scr_id, channels_frag, adcs_frag, timestamps_frag, threshold_frag, baseline_frag, trigger_sample_value_frag, daq_pretrigger_frag = extract_fragment_info(
                frag, trig)

            endpoint = int(find_endpoint(map_id, scr_id))

            # if debug:
            #    print("GEO ID:", gid)
            #    print("FRAG ID:", frag_id)
            #    print("EP:", endpoint)
            #    print("CH:", channels)
            #    print("ADCS:", adcs)

            if trigger == 'full_stream':
                adcs_frag = adcs_frag.transpose()
                timestamps_frag = [timestamps_frag[0]] * len(channels_frag)
                baseline_frag = [-1] * len(channels_frag)
                trigger_sample_value_frag = [-1] * len(channels_frag)
                is_fullstream_frag = [True] * len(channels_frag)
            elif trigger == 'self_trigger':
                is_fullstream_frag = [False] * len(channels_frag)

            if endpoint not in active_endpoints:
                active_endpoints.add(endpoint)
                threshold_list.append(threshold_frag)

            for index, ch in enumerate(channels_frag):

                adcs = []
                adcs = adcs_frag[index]
                # for value in adcs_frag[index]:
                #    #adcs.push_back(int(value))
                #    adcs.append(int(value))
                if read_full_streaming_data == is_fullstream_frag[index]:
                    if not wvfm_index % subsample:
                        waveforms.append(Waveform(timestamps_frag[index],
                                                  16.,    # time_step_ns
                                                  np.array(adcs),
                                                  run_numb,
                                                  r[0],
                                                  endpoint,
                                                  ch,
                                                  time_offset=0))
                    wvfm_index += 1
                    if wvfm_index >= wvfm_count:
                        return WaveformSet(*waveforms)

    return WaveformSet(*waveforms)
