import os
import subprocess
import shlex
import logging
import hashlib

from typing import List, Optional, Dict
from functools import partial
from multiprocessing import Pool, cpu_count
from numba import jit
from XRootD import client
from tqdm import tqdm
from collections import Counter
import numpy as np


from daqdataformats import FragmentType
from hdf5libs import HDF5RawDataFile
from rawdatautils.unpack.daphne import (
    np_array_timestamp,
    np_array_adc,
    np_array_channels,
    np_array_timestamp_stream,
    np_array_adc_stream,
    np_array_channels_stream
)
from rawdatautils.unpack.utils import *
from waffles.Exceptions import GenerateExceptionMessage
from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.WaveformSet import WaveformSet
import detdataformats
import fddetdataformats
import trgdataformats
import waffles.input_output.input_utils as wiu


# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    fmt='[%(levelname)s] %(asctime)s - %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

def local_copy_path_for(filepath: str, tmp_dir: str = "/tmp") -> str:
    """
    Generates a stable local copy path in `tmp_dir` based on a short hash of `filepath`.
    That way if you call it multiple times for the same remote file, it returns the same path.
    """
    hash_str = hashlib.md5(filepath.encode('utf-8')).hexdigest()[:8]
    base_name = os.path.basename(filepath)
    return os.path.join(tmp_dir, f"{base_name}_{hash_str}")


REMOTE_PREFIXES = ("root://", "davs://", "https://", "http://")

def is_remote_path(fp: str) -> bool:
    """Return True for any URL that points outside the local filesystem."""
    return fp.startswith(REMOTE_PREFIXES)


def xrdcp_if_not_exists(filepath: str,
                       tmp_dir: str = "/tmp",
                       logger: logging.Logger = None,
                       use_lock: bool = False) -> str:
    """
    Checks if a stable local copy of `filepath` exists in `tmp_dir`.
    If not, downloads via xrdcp. Returns the local file path.

    If `use_lock=True` and 'filelock' is installed, we lock on the final local path
    to prevent multiple processes from partially overwriting each other in parallel.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    local_path = local_copy_path_for(filepath, tmp_dir)

    # If the file already exists, assume it's fine (or add your own validation).
    if os.path.exists(local_path):
        logger.info(f"Local copy already present, reusing: {local_path}")
        return local_path

    # Optional concurrency lock
    lockfile_path = local_path + ".lock"
    if use_lock and LOCK_AVAILABLE:
        file_lock = filelock.FileLock(lockfile_path)
    else:
        file_lock = None

    if file_lock:
        with file_lock:  # acquire the lock before checking again
            # Re-check inside lock in case another process grabbed it in the meantime
            if os.path.exists(local_path):
                logger.info(f"Local copy found after lock, reusing: {local_path}")
                return local_path

            cmd = f"xrdcp --streams 8 {filepath} {local_path}"
            logger.info(f"Copying file with XRootD under lock: {cmd}")
            retcode = subprocess.call(shlex.split(cmd), shell=False)
            if retcode != 0:
                raise RuntimeError(f"xrdcp failed with return code {retcode} for file {filepath}")
            logger.info(f"File copied to {local_path}")

    else:
        # No lock scenario
        cmd = f"xrdcp --streams 8 {filepath} {local_path}"
        logger.info(f"Copying file with XRootD: {cmd}")
        retcode = subprocess.call(shlex.split(cmd), shell=False)
        if retcode != 0:
            raise RuntimeError(f"xrdcp failed with return code {retcode} for file {filepath}")
        logger.info(f"File copied to {local_path}")

    return local_path


def get_inv_map_id(det):
    if det == 'HD_PDS':
        map_id = {
            '104': [1, 2, 3, 4],
            '105': [5, 6, 7, 9],
            '107': [10, 8],
            '109': [11],
            '111': [12],
            '112': [13],
            '113': [14]
        }
    elif det == 'VD_Membrane_PDS':
        map_id = {'107': [700, 701, 51]}
    elif det == 'VD_Cathode_PDS':
        map_id = {'106': [723, 722, 721, 720, 21, 22, 23]}
    else:
        raise ValueError(f"det '{det}' is not recognized.")
    inv_map_id = {v: k for k, vals in map_id.items() for v in vals}
    return inv_map_id


def find_endpoint(map_id, target_value):
    return map_id.get(target_value, None)


def extract_fragment_info(frag, trig):
    frh = frag.get_header()
    run_number = frh.run_number
    scr_id = frh.element_id.id
    fragType = frh.fragment_type
    timestamps = []
    adcs = []
    channels = []
    trigger = 'unknown'
    trigger_ts = frag.get_trigger_timestamp()

    if fragType == FragmentType.kDAPHNE.value:
        trigger = 'self_trigger'
        timestamps = np_array_timestamp(frag)
        adcs = np_array_adc(frag)
        channels = np_array_channels(frag)
    elif fragType == FragmentType.kDAPHNEStream.value:
        trigger = 'full_stream'
        timestamps = np_array_timestamp_stream(frag)
        adcs = np_array_adc_stream(frag)
        channels = np_array_channels_stream(frag)[0]

    return run_number, trigger, scr_id, channels, adcs, timestamps, trigger_ts


def filepath_is_hdf5_file_candidate(filepath: str) -> bool:
    # 1) Remote paths are always accepted as “candidates”
    if is_remote_path(filepath):
        return filepath.endswith((".hdf5", ".h5"))

    # 2) Otherwise we insist the file exists *and* ends with .h5/.hdf5
    return os.path.isfile(filepath) and filepath.endswith((".hdf5", ".h5", ".hdf5.copied", ".h5.copied"))


def get_filepaths_from_rucio(rucio_filepath) -> list:
    # The Rucio list file itself *must* be local
    if not os.path.isfile(rucio_filepath):
        raise Exception(GenerateExceptionMessage(
            1, 'get_filepaths_from_rucio()',
            f"The given rucio_filepath ({rucio_filepath}) is not a valid file."
        ))

    with open(rucio_filepath, 'r') as fh:
        lines = fh.readlines()

    filepaths = [
        line.strip().replace('root://eospublic.cern.ch:1094/', '')
        for line in lines if 'tpwriter' not in line
    ]

    if not filepaths:
        logger.warning("No file paths found in the Rucio file.")
        return []

    quality_check = filepaths[0]

    #  ──► only check local files with os.path.isfile
    if is_remote_path(quality_check):
        logger.info("First entry is a remote EOS/WebDAV path; skipping local-file check.")
    else:
        if "eos" in quality_check:
            logger.info("Your files are stored in /eos/")
        if not os.path.isfile(quality_check):
            raise Exception(GenerateExceptionMessage(
                2, 'get_filepaths_from_rucio()',
                f"The given filepaths[0] ({quality_check}) is not a valid local file."
            ))

    return filepaths

def WaveformSet_from_hdf5_files(filepath_list: List[str] = [],
                                read_full_streaming_data: bool = False,
                                truncate_wfs_method: str = "",
                                folderpath: Optional[str] = None,
                                nrecord_start_fraction: float = 0.0,
                                nrecord_stop_fraction: float = 1.0,
                                subsample: int = 1,
                                wvfm_count: int = 1e9,
                                ch: Optional[dict] = {},
                                det: str = 'HD_PDS',
                                temporal_copy_directory: str = '/tmp',
                                erase_temporal_copy: bool = True
                                ) -> WaveformSet:
    """
    Creates a WaveformSet from multiple HDF5 files by sequentially reading
    them in the current process.
    """
    if folderpath is not None:
        if not os.path.isdir(folderpath):
            raise Exception(GenerateExceptionMessage(
                1,
                'WaveformSet_from_hdf5_files()',
                f"The given folderpath ({folderpath}) is not a valid directory."
            ))
        valid_filepaths = []
        for filename in os.listdir(folderpath):
            full_path = os.path.join(folderpath, filename)
            if filepath_is_hdf5_file_candidate(full_path):
                valid_filepaths.append(full_path)
    else:
        valid_filepaths = [
            filepath for filepath in set(filepath_list)
            if filepath_is_hdf5_file_candidate(filepath)
        ]

    if not valid_filepaths:
        logger.warning("No valid HDF5 file paths were found.")
        return WaveformSet()

    output = None
    for filepath in tqdm(valid_filepaths, desc="Reading HDF5 files (sequential)"):
        try:
            aux = WaveformSet_from_hdf5_file(
                filepath,
                read_full_streaming_data,
                truncate_wfs_method,
                nrecord_start_fraction,
                nrecord_stop_fraction,
                subsample,
                wvfm_count,
                ch,
                det,
                temporal_copy_directory=temporal_copy_directory,
                erase_temporal_copy=erase_temporal_copy
            )
        except Exception as error:
            logger.error(f"Error reading file {filepath}: {error}")
            logger.error("Skipping this file...")
            continue

        if output is None:
            output = aux
        else:
            output.merge(aux)

        logger.info(f"WaveformSet length so far: {len(output.waveforms)}")

    return output if output is not None else WaveformSet()


def WaveformSet_from_hdf5_file(filepath: str,
                               read_full_streaming_data: bool = False,
                               truncate_wfs_method: str = "",
                               nrecord_start_fraction: float = 0.0,
                               nrecord_stop_fraction: float = 1.0,
                               subsample: int = 1,
                               wvfm_count: int = 1e9,
                               ch: Optional[dict] = {},
                               det: str = 'HD_PDS',
                               temporal_copy_directory: str = '/tmp',
                               erase_temporal_copy: bool = True,
                               record_chunk_size: int = 200,
                               choose_minimum: bool = False,
                               ) -> WaveformSet:
    """
    Reads a single HDF5 file and constructs a WaveformSet. Records are processed
    in chunks (default size = 200) to reduce peak memory usage.

    Args:
        filepath (str): The path to the HDF5 file.
        read_full_streaming_data (bool): If True, read DAPHNEStream fragments (full stream).
        truncate_wfs_method (str): Method to truncate waveforms. Options are:
            - "minimum": Truncate all waveforms to the minimum length.
            - "MPV": Use the most probable waveform length.
            - "choose": User chooses a length interactively.
        nrecord_start_fraction (float): Fraction of records to skip from start.
        nrecord_stop_fraction (float): Fraction of records to skip from end.
        subsample (int): Keep 1 out of every 'subsample' waveforms.
        wvfm_count (int): Maximum number of waveforms to read.
        ch (dict): Channels to read, e.g. {'104': [1,2], '105': [3,4]}.
        det (str): Detector type (e.g., 'HD_PDS').
        temporal_copy_directory (str): Local dir for XRootD copy.
        erase_temporal_copy (bool): Remove local copy after reading if True.
        record_chunk_size (int): Number of records to process at once (to reduce memory usage).

    Returns:
        WaveformSet: The WaveformSet of waveforms from the file.
    """
    fUsedXRootD = False
    # Attempt local copy if outside known local paths
    if is_remote_path(filepath) or not os.path.isfile(filepath):
        if wiu.write_permission(temporal_copy_directory):
            temp_path = os.path.join(temporal_copy_directory, os.path.basename(filepath))
            if not os.path.exists(temp_path):
                cmd = f"xrdcp --streams 8 {filepath} {temporal_copy_directory}"
                logger.info(f"Copying file with XRootD: {cmd}")
                subprocess.call(shlex.split(cmd), shell=False)
            filepath = temp_path
            fUsedXRootD = True
        else:
            raise Exception(GenerateExceptionMessage(
                1,
                'WaveformSet_from_hdf5_file()',
                f"Attempting to temporarily copy {filepath} into {temporal_copy_directory}, "
                "but no write permission."
            ))

    # Open the HDF5 file
    h5_file = HDF5RawDataFile(filepath)
    waveforms = []
    active_endpoints = set()

    records = h5_file.get_all_record_ids()

    valid_pairs = {
        (int(endpoint), channel)
        for endpoint, channels in ch.items()
        for channel in channels
    }

    # Validate fraction ranges
    if nrecord_stop_fraction > 1.0:
        nrecord_stop_fraction = 1.0
    if nrecord_start_fraction < 0.0 or nrecord_start_fraction > 1.0:
        raise ValueError(
            "Invalid value for nrecord_start_fraction (must be between 0 and 1)."
        )

    total_records = len(records)
    start_index = int(np.floor(nrecord_start_fraction * (total_records - 1)))
    stop_index = int(np.ceil(nrecord_stop_fraction * (total_records - 1)))
    records = records[start_index:stop_index + 1]

    wvfm_index = 0
    inv_map_id = get_inv_map_id(det)

    # Process records in chunks
    for chunk_start in range(0, len(records), record_chunk_size):
        chunk_end = chunk_start + record_chunk_size
        record_chunk = records[chunk_start:chunk_end]

        for r in record_chunk:
            pds_geo_ids = list(h5_file.get_geo_ids_for_subdetector(
                r, detdataformats.DetID.string_to_subdetector(det)
            ))

            try:
                trig = h5_file.get_trh(r)
                trigger_type_bits = trig.get_trigger_type()
            except Exception as e:
                logger.warning(f"Corrupted fragment:\n {r}\n{gid}\nError: {e}")
                continue

            for gid in pds_geo_ids:
                try:
                    frag = h5_file.get_frag(r, gid)
                except Exception as e:
                    logger.warning(f"Corrupted fragment:\n {r}\n{gid}\nError: {e}")
                    continue

                if frag.get_data_size() == 0:
                    # logger.warning(f"Empty fragment:\n {frag}\n{r}\n{gid}")
                    continue

                if read_full_streaming_data and frag.get_fragment_type() == FragmentType.kDAPHNE:
                    continue
                if (not read_full_streaming_data) and frag.get_fragment_type() == FragmentType.kDAPHNEStream:
                    continue

                (run_number, trigger, scr_id,
                 channels_frag, adcs_frag, timestamps_frag,
                 trigger_ts) = extract_fragment_info(frag, trig)

                endpoint = find_endpoint(inv_map_id, scr_id)
                if endpoint is None:
                    continue
                endpoint = int(endpoint)

                if trigger == 'full_stream':
                    adcs_frag = adcs_frag.transpose()
                    timestamps_frag = [timestamps_frag[0]] * len(channels_frag)
                    is_fullstream_frag = [True] * len(channels_frag)
                elif trigger == 'self_trigger':
                    is_fullstream_frag = [False] * len(channels_frag)
                else:
                    is_fullstream_frag = [False] * len(channels_frag)

                if endpoint not in active_endpoints:
                    active_endpoints.add(endpoint)

                for index, ch_id in enumerate(channels_frag):
                    if valid_pairs and (endpoint, ch_id) not in valid_pairs:
                        continue

                    if read_full_streaming_data == is_fullstream_frag[index]:
                        if not wvfm_index % subsample:
                            wv = Waveform(
                                timestamps_frag[index],
                                16.,
                                trigger_ts,
                                np.array(adcs_frag[index]),
                                run_number,
                                r[0],
                                endpoint,
                                ch_id,
                                time_offset=0,
                                starting_tick=0,
                                trigger_type=trigger_type_bits
                            )
                            waveforms.append(wv)

                        wvfm_index += 1
                        if wvfm_index >= wvfm_count:
                            if truncate_wfs_method == "minimum" and waveforms:
                                min_len = min(len(wf.adcs) for wf in waveforms)
                                for wf in waveforms:
                                    wf._WaveformAdcs__slice_adcs(0, min_len)

                            if fUsedXRootD and erase_temporal_copy and os.path.exists(filepath):
                                os.remove(filepath)
                            return WaveformSet(*waveforms)

    # Finished reading all chunks
    if truncate_wfs_method == "minimum" and waveforms:
        min_len = min(len(wf.adcs) for wf in waveforms)
        for wf in waveforms:
            wf._WaveformAdcs__slice_adcs(0, min_len)
    elif truncate_wfs_method:
        allwaveformslengths = Counter([len(wf.adcs) for wf in waveforms])
        allwaveformslengths = sorted(allwaveformslengths.items(), key=lambda x: x[1], reverse=True)
        if truncate_wfs_method == "MPV":
            print(f"Most common waveform length: {allwaveformslengths[0][0]} with {allwaveformslengths[0][1]} occurrences")
            print("Showing the next 4 most common lengths:")
            for length, count in allwaveformslengths[0:5]:
                print(f"Length {length} with {count} occurrences")
            
            slice_len = allwaveformslengths[0][0] # Most common length
        elif truncate_wfs_method == "choose":
            print("Choose a length from the following options:")
            for i, (length, count) in enumerate(allwaveformslengths):
                print(f"{i}: {length} with {count} occurrences")
            choice = int(input("Enter the index of the length you want to choose: "))
            if choice < 0 or choice >= len(allwaveformslengths):
                raise ValueError("Invalid choice index.")
            slice_len = allwaveformslengths[choice][0]
        else:
            raise ValueError(f"Unknown truncate_wfs_method: {truncate_wfs_method}. "
                             "Use 'minimum', 'MPV', or 'choose'.")
        waveforms_full = waveforms.copy()  # Keep original for reference
        waveforms = []  
        for wf in waveforms_full:
            if len(wf.adcs) > slice_len:
                wf._WaveformAdcs__slice_adcs(0, slice_len)
            if len(wf.adcs) == slice_len:
                waveforms.append(wf)


    if fUsedXRootD and erase_temporal_copy and os.path.exists(filepath):
        os.remove(filepath)

    return WaveformSet(*waveforms)


def process_record(r):
    # This was just a placeholder in your code, referencing global objects.
    # It won't work as-is unless you define h5_file, gid in a higher scope.
    # You might remove or adapt it, but we keep it for completeness.
    return extract_fragment_info(h5_file.get_frag(r, gid), h5_file.get_trh(r))


# -----------------------------------------------------------------------------
# PARALLEL VERSION of WaveformSet_from_hdf5_files
# -----------------------------------------------------------------------------
def WaveformSet_from_hdf5_files_parallel(filepath_list: List[str] = [],
                                         read_full_streaming_data: bool = False,
                                         truncate_wfs_method: str = "",
                                         folderpath: Optional[str] = None,
                                         nrecord_start_fraction: float = 0.0,
                                         nrecord_stop_fraction: float = 1.0,
                                         subsample: int = 1,
                                         wvfm_count: int = 1e9,
                                         ch: Optional[dict] = {},
                                         det: str = 'HD_PDS',
                                         temporal_copy_directory: str = '/tmp',
                                         erase_temporal_copy: bool = True,
                                         record_chunk_size: int = 200,
                                         n_processes: int = 4
                                         ) -> WaveformSet:
    """
    Parallel version of WaveformSet_from_hdf5_files. Uses multiprocessing to read each file
    in a separate process, then merges the results.

    Before dispatching to worker processes, we pre-copy any remote files (using xrdcp_if_not_exists)
    so that every worker works with a local file and no concurrent XRootD copies occur.
    """
    # 1) Determine valid filepaths (same as original)
    if folderpath is not None:
        if not os.path.isdir(folderpath):
            raise Exception(GenerateExceptionMessage(
                1,
                'WaveformSet_from_hdf5_files_parallel()',
                f"The given folderpath ({folderpath}) is not a valid directory."
            ))
        valid_filepaths = []
        for filename in os.listdir(folderpath):
            full_path = os.path.join(folderpath, filename)
            if filepath_is_hdf5_file_candidate(full_path):
                valid_filepaths.append(full_path)
    else:
        valid_filepaths = [
            filepath for filepath in set(filepath_list)
            if filepath_is_hdf5_file_candidate(filepath)
        ]

    if not valid_filepaths:
        logger.warning("No valid HDF5 file paths were found for parallel reading.")
        return WaveformSet()

    logger.info(f"Starting parallel read of {len(valid_filepaths)} file(s) with {n_processes} workers.")

    # 2) Pre-copy remote files if needed.
    #    For each file in valid_filepaths, if it is remote (e.g. starts with "root://") or not local,
    #    we download it once using xrdcp_if_not_exists. This avoids race conditions in the workers.
    pre_copied_filepaths = []
    for fp in valid_filepaths:
        # Check if fp is remote by testing if it starts with "root://"
        if fp.startswith("root://") or (not os.path.isfile(fp)):
            # xrdcp_if_not_exists uses a stable local filename (e.g. with a hash) and a file lock.
            local_fp = xrdcp_if_not_exists(fp, tmp_dir=temporal_copy_directory, logger=logger, use_lock=True)
            pre_copied_filepaths.append(local_fp)
        else:
            pre_copied_filepaths.append(fp)

    # 3) Prepare partial function with fixed parameters.
    #    Note: We pass the local file paths now so that each worker simply opens the file.
    func = partial(
        WaveformSet_from_hdf5_file,
        read_full_streaming_data=read_full_streaming_data,
        truncate_wfs_method=truncate_wfs_method,
        nrecord_start_fraction=nrecord_start_fraction,
        nrecord_stop_fraction=nrecord_stop_fraction,
        subsample=subsample,
        wvfm_count=wvfm_count,
        ch=ch,
        det=det,
        temporal_copy_directory=temporal_copy_directory,
        erase_temporal_copy=erase_temporal_copy,
        record_chunk_size=record_chunk_size
    )

    # 4) Parallel execution: Each worker processes a file (which is now local)
    with Pool(processes=n_processes) as pool:
        results = list(tqdm(pool.imap(func, pre_copied_filepaths),
                            total=len(pre_copied_filepaths),
                            desc="Reading HDF5 files (parallel)"))

    # 5) Merge all WaveformSets
    combined = WaveformSet()
    for wset in results:
        combined.merge(wset)
    logger.info(f"Finished parallel read. Combined WaveformSet length: {len(combined.waveforms)}")

    return combined
