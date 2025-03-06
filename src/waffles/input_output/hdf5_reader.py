import os
import sys
import uuid
import hashlib
import logging
import subprocess
import shlex
from functools import partial
from typing import List, Optional, Dict
from multiprocessing import Pool

import numpy as np

# Optional concurrency lock
try:
    import filelock
    LOCK_AVAILABLE = True
except ImportError:
    filelock = None
    LOCK_AVAILABLE = False

from tqdm import tqdm
from XRootD import client
from numba import jit

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
import waffles.input_output.input_utils as wiu


# ------------------------------------------------------------------------------
# Helper functions for stable local file copies
# ------------------------------------------------------------------------------

def local_copy_path_for(filepath: str, tmp_dir: str = "/tmp") -> str:
    """
    Generates a stable local copy path in `tmp_dir` based on a short hash of `filepath`.
    That way if you call it multiple times for the same remote file, it returns the same path.
    """
    hash_str = hashlib.md5(filepath.encode('utf-8')).hexdigest()[:8]
    base_name = os.path.basename(filepath)
    return os.path.join(tmp_dir, f"{base_name}_{hash_str}")


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


# ------------------------------------------------------------------------------
# HDF5Reader class
# ------------------------------------------------------------------------------

class HDF5Reader:
    """
    Class-based wrapper for reading HDF5 files (DAPHNE data) either sequentially or in parallel.
    Includes logic for chunked reading, channel/fragment filters, and local XRootD copy with reuse.
    """

    def __init__(self):
        # Basic logger setup
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt='[%(levelname)s] %(asctime)s - %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    # --------------------------------------------------------------------------
    # Detector ID <--> Endpoint mapping
    # --------------------------------------------------------------------------
    def get_inv_map_id(self, det: str) -> Dict[int, str]:
        """
        Returns the 'inverse' mapping from integer scr_id (geo_id) to string (e.g. '104'),
        for a given detector type. 
        """
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
        elif det in ('VD_Membrane_PDS', 'VD_Cathode_PDS'):
            map_id = {'107': [51]}
        else:
            raise ValueError(f"Detector '{det}' not recognized.")

        # invert: e.g. 1->'104', 2->'104', etc.
        inv_map_id = {v: k for k, vals in map_id.items() for v in vals}
        return inv_map_id

    def find_endpoint(self, map_dict: Dict[int, str], target_value: int) -> str:
        """Look up the string endpoint given the integer scr_id."""
        return map_dict[target_value]

    # --------------------------------------------------------------------------
    # Fragment extraction
    # --------------------------------------------------------------------------
    def extract_fragment_info(self, frag, trig):
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
        elif fragType == FragmentType.kDAPHNEStream:
            trigger = 'full_stream'
            timestamps = np_array_timestamp_stream(frag)
            adcs = np_array_adc_stream(frag)
            channels = np_array_channels_stream(frag)[0]

        return run_number, trigger, scr_id, channels, adcs, timestamps, trigger_ts

    # --------------------------------------------------------------------------
    # Rucio file utility
    # --------------------------------------------------------------------------
    def get_filepaths_from_rucio(self, rucio_filepath: str) -> List[str]:
        """
        Reads a plain-text 'Rucio' file that lists HDF5 paths, skipping lines w/ 'tpwriter'.
        """
        if not os.path.isfile(rucio_filepath):
            raise Exception(GenerateExceptionMessage(
                1,
                'get_filepaths_from_rucio()',
                f"The given rucio_filepath ({rucio_filepath}) is not a valid file."
            ))

        with open(rucio_filepath, 'r') as file:
            lines = file.readlines()

        # Remove 'tpwriter' lines, fix public.cern paths
        filepaths = [ln.strip().replace('root://eospublic.cern.ch:1094/', '') for ln in lines]
        filepaths = [ln for ln in filepaths if 'tpwriter' not in ln]

        if not filepaths:
            self.logger.warning("No file paths found in the Rucio file.")
            return []

        if "eos" in filepaths[0]:
            self.logger.info("Your files are stored in /eos/")
            if not os.path.isfile(filepaths[0]):
                raise Exception(GenerateExceptionMessage(
                    2,
                    'get_filepaths_from_rucio()',
                    f"The given filepaths[0] ({filepaths[0]}) is not a valid file."
                ))
        else:
            self.logger.warning(
                "Your files are stored around the world.\n"
                "[WARNING] Check you have a correct configuration to use XRootD."
            )

        return filepaths

    def filepath_is_hdf5_file_candidate(self, filepath: str) -> bool:
        """
        1) If path starts with 'root://', treat as valid for further XRootD reading.
        2) Otherwise, must be a local file that ends with .h5 or .hdf5.
        """
        if filepath.startswith("root://"):
            return True
        if os.path.isfile(filepath) and (filepath.endswith('.hdf5') or filepath.endswith('.h5')):
            return True
        return False

    # --------------------------------------------------------------------------
    # Main single-file read
    # --------------------------------------------------------------------------
    def WaveformSet_from_hdf5_file(
        self,
        filepath: str,
        read_full_streaming_data: bool = False,
        truncate_wfs_to_minimum: bool = False,
        nrecord_start_fraction: float = 0.0,
        nrecord_stop_fraction: float = 1.0,
        subsample: int = 1,
        wvfm_count: int = 1e9,
        ch: Optional[dict] = {},
        det: str = 'HD_PDS',
        temporal_copy_directory: str = '/tmp',
        erase_temporal_copy: bool = True,
        record_chunk_size: int = 200,
        use_file_lock: bool = True
    ) -> WaveformSet:
        """
        Reads a single HDF5 file into a WaveformSet. Optionally uses
        local caching in /tmp, reusing existing copies if present.
        """
        fUsedXRootD = False

        # If it's a remote path (not in /eos, /nfs, /afs, etc.), copy locally or reuse existing
        if ("/eos" not in filepath and
            "/nfs" not in filepath and
            "/afs" not in filepath and
            not os.path.isfile(filepath)):
            if wiu.write_permission(temporal_copy_directory):
                # Reuse local copy if exists, else xrdcp
                local_path = xrdcp_if_not_exists(
                    filepath,
                    tmp_dir=temporal_copy_directory,
                    logger=self.logger,
                    use_lock=use_file_lock
                )
                filepath = local_path
                fUsedXRootD = True
            else:
                raise Exception(GenerateExceptionMessage(
                    1,
                    'WaveformSet_from_hdf5_file()',
                    f"Attempting to temporarily copy {filepath} into {temporal_copy_directory}, "
                    "but no write permission."
                ))

        # Open HDF5 file
        h5_file = HDF5RawDataFile(filepath)
        waveforms = []
        active_endpoints = set()

        records = h5_file.get_all_record_ids()

        # Build valid channel-endpoint pairs
        valid_pairs = {
            (int(endpoint), channel)
            for endpoint, channels in ch.items()
            for channel in channels
        }

        # Validate fraction ranges
        if nrecord_stop_fraction > 1.0:
            nrecord_stop_fraction = 1.0
        if nrecord_start_fraction < 0.0 or nrecord_start_fraction > 1.0:
            raise ValueError("Invalid value for nrecord_start_fraction (must be between 0 and 1).")

        total_records = len(records)
        start_index = int(np.floor(nrecord_start_fraction * (total_records - 1)))
        stop_index = int(np.ceil(nrecord_stop_fraction * (total_records - 1)))
        records = records[start_index:stop_index + 1]

        inv_map_id = self.get_inv_map_id(det)
        wvfm_index = 0

        # Read data in chunks
        for chunk_start in range(0, len(records), record_chunk_size):
            chunk_end = chunk_start + record_chunk_size
            record_chunk = records[chunk_start:chunk_end]

            for r in record_chunk:
                # Get geo ids for subdetector
                pds_geo_ids = list(h5_file.get_geo_ids_for_subdetector(
                    r, detdataformats.DetID.string_to_subdetector(det)
                ))
                trig = h5_file.get_trh(r)

                for gid in pds_geo_ids:
                    try:
                        frag = h5_file.get_frag(r, gid)
                    except Exception as e:
                        self.logger.warning(f"Corrupted fragment:\n {r}\n{gid}\nError: {e}")
                        continue

                    if frag.get_data_size() == 0:
                        self.logger.warning(f"Empty fragment:\n {frag}\n{r}\n{gid}")
                        continue

                    # Skip undesired types
                    if read_full_streaming_data and frag.get_fragment_type() == FragmentType.kDAPHNE:
                        continue
                    if (not read_full_streaming_data) and frag.get_fragment_type() == FragmentType.kDAPHNEStream:
                        continue

                    (run_number, trigger, scr_id,
                     channels_frag, adcs_frag, timestamps_frag,
                     trigger_ts) = self.extract_fragment_info(frag, trig)

                    endpoint_str = self.find_endpoint(inv_map_id, scr_id)
                    endpoint = int(endpoint_str)

                    if trigger == 'full_stream':
                        # shape: adcs_frag=(ch, samples). We transpose if needed
                        adcs_frag = adcs_frag.transpose()  # (samples, ch)
                        timestamps_frag = [timestamps_frag[0]] * len(channels_frag)
                        is_fullstream_frag = [True] * len(channels_frag)
                    elif trigger == 'self_trigger':
                        is_fullstream_frag = [False] * len(channels_frag)
                    else:
                        is_fullstream_frag = [False] * len(channels_frag)

                    if endpoint not in active_endpoints:
                        active_endpoints.add(endpoint)

                    # Now loop over each channel in the fragment
                    for index, ch_id in enumerate(channels_frag):
                        if (endpoint, ch_id) not in valid_pairs:
                            continue

                        # If user wants streaming data, we only pick full_stream waveforms, etc.
                        if read_full_streaming_data == is_fullstream_frag[index]:
                            # Subsampling
                            if not (wvfm_index % subsample):
                                # The Waveform constructor expects positional arguments: 
                                # (timestamp, time_step_ns, daq_window_timestamp, adcs, run_number, record_number, endpoint, channel)
                                # Example below uses the first element of timestamps_frag as an int:
                                wv = Waveform(
                                    int(timestamps_frag[index][0]) if hasattr(timestamps_frag[index], '__getitem__') else int(timestamps_frag[index]),
                                    16.0,                # time_step_ns
                                    trigger_ts,          # daq_window_timestamp
                                    np.array(adcs_frag[index]),
                                    run_number,
                                    r[0],                # record_number
                                    endpoint,
                                    ch_id,
                                    time_offset=0,
                                    starting_tick=0
                                )
                                waveforms.append(wv)

                            wvfm_index += 1
                            if wvfm_index >= wvfm_count:
                                # If we've read enough waveforms, optionally truncate
                                if truncate_wfs_to_minimum and waveforms:
                                    min_len = min(len(wf.adcs) for wf in waveforms)
                                    for wf in waveforms:
                                        wf._WaveformAdcs__slice_adcs(0, min_len)

                                # Cleanup if we copied
                                if fUsedXRootD and erase_temporal_copy and os.path.exists(filepath):
                                    os.remove(filepath)
                                return WaveformSet(*waveforms)

        # After reading all records
        if truncate_wfs_to_minimum and waveforms:
            min_len = min(len(wf.adcs) for wf in waveforms)
            for wf in waveforms:
                wf._WaveformAdcs__slice_adcs(0, min_len)

        if fUsedXRootD and erase_temporal_copy and os.path.exists(filepath):
            os.remove(filepath)

        return WaveformSet(*waveforms)

    # --------------------------------------------------------------------------
    # Many-files read (sequential)
    # --------------------------------------------------------------------------
    def WaveformSet_from_hdf5_files(
        self,
        filepath_list: List[str] = [],
        read_full_streaming_data: bool = False,
        truncate_wfs_to_minimum: bool = False,
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
        use_file_lock: bool = True
    ) -> WaveformSet:
        """
        Creates a WaveformSet from multiple HDF5 files by sequentially reading them.
        """
        # Collect valid filepaths
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
                if self.filepath_is_hdf5_file_candidate(full_path):
                    valid_filepaths.append(full_path)
        else:
            valid_filepaths = [
                fp for fp in set(filepath_list)
                if self.filepath_is_hdf5_file_candidate(fp)
            ]

        if not valid_filepaths:
            self.logger.warning("No valid HDF5 file paths were found.")
            return WaveformSet()

        output = None
        for filepath in tqdm(valid_filepaths, desc="Reading HDF5 files (sequential)"):
            try:
                aux = self.WaveformSet_from_hdf5_file(
                    filepath=filepath,
                    read_full_streaming_data=read_full_streaming_data,
                    truncate_wfs_to_minimum=truncate_wfs_to_minimum,
                    nrecord_start_fraction=nrecord_start_fraction,
                    nrecord_stop_fraction=nrecord_stop_fraction,
                    subsample=subsample,
                    wvfm_count=wvfm_count,
                    ch=ch,
                    det=det,
                    temporal_copy_directory=temporal_copy_directory,
                    erase_temporal_copy=erase_temporal_copy,
                    record_chunk_size=record_chunk_size,
                    use_file_lock=use_file_lock
                )
            except Exception as error:
                self.logger.error(f"Error reading file {filepath}: {error}")
                self.logger.error("Skipping this file...")
                continue

            if output is None:
                output = aux
            else:
                output.merge(aux)

            self.logger.info(f"WaveformSet length so far: {len(output.waveforms)}")

        return output if output is not None else WaveformSet()

    # --------------------------------------------------------------------------
    # Many-files read (parallel)
    # --------------------------------------------------------------------------
    def WaveformSet_from_hdf5_files_parallel(
        self,
        filepath_list: List[str] = [],
        read_full_streaming_data: bool = False,
        truncate_wfs_to_minimum: bool = False,
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
        n_processes: int = 4,
        use_file_lock: bool = True
    ) -> WaveformSet:
        """
        Parallel version of reading multiple HDF5 files. Each file is processed by a separate
        worker process, then results are merged.
        """
        # 1) Collect valid filepaths
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
                if self.filepath_is_hdf5_file_candidate(full_path):
                    valid_filepaths.append(full_path)
        else:
            valid_filepaths = [
                fp for fp in set(filepath_list)
                if self.filepath_is_hdf5_file_candidate(fp)
            ]

        if not valid_filepaths:
            self.logger.warning("No valid HDF5 file paths were found for parallel reading.")
            return WaveformSet()

        self.logger.info(f"Starting parallel read of {len(valid_filepaths)} file(s) with {n_processes} workers.")

        # 2) Prepare partial function
        func = partial(
            self.WaveformSet_from_hdf5_file,
            read_full_streaming_data=read_full_streaming_data,
            truncate_wfs_to_minimum=truncate_wfs_to_minimum,
            nrecord_start_fraction=nrecord_start_fraction,
            nrecord_stop_fraction=nrecord_stop_fraction,
            subsample=subsample,
            wvfm_count=wvfm_count,
            ch=ch,
            det=det,
            temporal_copy_directory=temporal_copy_directory,
            erase_temporal_copy=erase_temporal_copy,
            record_chunk_size=record_chunk_size,
            use_file_lock=use_file_lock
        )

        # 3) Parallel execution
        with Pool(processes=n_processes) as pool:
            results = list(
                tqdm(
                    pool.imap(func, valid_filepaths),
                    total=len(valid_filepaths),
                    desc="Reading HDF5 files (parallel)"
                )
            )

        # 4) Merge results
        combined = WaveformSet()
        for wset in results:
            combined.merge(wset)

        self.logger.info(f"Finished parallel read. Combined WaveformSet length: {len(combined.waveforms)}")
        return combined

# End of file
