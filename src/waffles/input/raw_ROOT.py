import os
import math
import numpy as np
import uproot
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import ROOT
except ImportError:
    print("[raw_ROOT_reader.py]: Could not import ROOT module. Do not use 'pyroot' library options.")
    pass

from typing import List, Optional

from waffles.data_classes.WaveformSet import WaveformSet
import waffles.utils.check_utils as wuc
import waffles.input.input_utils as wii
from waffles.Exceptions import generate_exception_message


def read_single_root_file(filepath, library, bulk_data_tree_name, meta_data_tree_name,
                          set_offset_wrt_daq_window, read_full_streaming_data,
                          truncate_wfs_to_minimum, start_fraction, stop_fraction, subsample, verbose):
    if verbose:
        print(f"Reading file {filepath} ...")
    return WaveformSet_from_ROOT_file(filepath,
                                      library,
                                      bulk_data_tree_name=bulk_data_tree_name,
                                      meta_data_tree_name=meta_data_tree_name,
                                      set_offset_wrt_daq_window=set_offset_wrt_daq_window,
                                      read_full_streaming_data=read_full_streaming_data,
                                      truncate_wfs_to_minimum=truncate_wfs_to_minimum,
                                      start_fraction=start_fraction,
                                      stop_fraction=stop_fraction,
                                      subsample=subsample,
                                      verbose=verbose)


def WaveformSet_from_ROOT_files(library: str,
                                folderpath: Optional[str] = None,
                                filepath_list: List[str] = [],
                                bulk_data_tree_name: str = 'raw_waveforms',
                                meta_data_tree_name: str = 'metadata',
                                set_offset_wrt_daq_window: bool = False,
                                read_full_streaming_data: bool = False,
                                truncate_wfs_to_minimum: bool = False,
                                start_fraction: float = 0.0,
                                stop_fraction: float = 1.0,
                                subsample: int = 1,
                                verbose: bool = True) -> WaveformSet:
    if folderpath is not None:
        if not os.path.isdir(folderpath):
            raise Exception(generate_exception_message(1,
                                                       'WaveformSet_from_ROOT_files()',
                                                       f"The given folderpath ({folderpath}) is not a valid directory."))
        valid_filepaths = [os.path.join(folderpath, filename)
                           for filename in os.listdir(folderpath)
                           if wii.filepath_is_ROOT_file_candidate(os.path.join(folderpath, filename))]
    else:
        valid_filepaths = [filepath
                           # Remove possible duplicates
                           for filepath in set(filepath_list)
                           if wii.filepath_is_ROOT_file_candidate(filepath)]
    if len(valid_filepaths) == 0:
        raise Exception(generate_exception_message(2,
                                                   'WaveformSet_from_ROOT_files()',
                                                   f"No valid ROOT files were found in the given folder or filepath list."))
    if verbose:
        print(f"In function WaveformSet_from_ROOT_files(): Found {
              len(valid_filepaths)} different valid ROOT files: \n\n", end='')
        for filepath in valid_filepaths:
            print(f"\t - {filepath}\n", end='')
        print("\n", end='')

    output = None

    with ThreadPoolExecutor() as executor:
        future_to_filepath = {executor.submit(read_single_root_file, filepath, library, bulk_data_tree_name,
                                              meta_data_tree_name, set_offset_wrt_daq_window, read_full_streaming_data,
                                              truncate_wfs_to_minimum, start_fraction, stop_fraction, subsample, verbose): filepath
                              for filepath in valid_filepaths}

        for future in as_completed(future_to_filepath):
            try:
                result = future.result()
                if output is None:
                    output = result
                else:
                    output.merge(result)
            except Exception as exc:
                print(f"Generated an exception: {exc}")

    if verbose:
        print(f"In function WaveformSet_from_ROOT_files(): Reading finished")

    return output


def WaveformSet_from_ROOT_file(filepath: str,
                               library: str,
                               bulk_data_tree_name: str = 'raw_waveforms',
                               meta_data_tree_name: str = 'metadata',
                               set_offset_wrt_daq_window: bool = False,
                               read_full_streaming_data: bool = False,
                               truncate_wfs_to_minimum: bool = False,
                               start_fraction: float = 0.0,
                               stop_fraction: float = 1.0,
                               subsample: int = 1,
                               verbose: bool = True) -> WaveformSet:
    if not wuc.fraction_is_well_formed(start_fraction, stop_fraction):
        raise Exception(generate_exception_message(1,
                                                   'WaveformSet_from_ROOT_file()',
                                                   f"Fraction limits are not well-formed."))
    if library not in ['uproot', 'pyroot']:
        raise Exception(generate_exception_message(2,
                                                   'WaveformSet_from_ROOT_file()',
                                                   f"The given library ({library}) is not supported."))
    elif library == 'uproot':
        input_file = uproot.open(filepath)
    else:
        input_file = ROOT.TFile(filepath)

    meta_data_tree, _ = wii.find_TTree_in_ROOT_TFile(
        input_file, meta_data_tree_name, library)
    bulk_data_tree, _ = wii.find_TTree_in_ROOT_TFile(
        input_file, bulk_data_tree_name, library)
    is_fullstream_branch, is_fullstream_branch_name = wii.find_TBranch_in_ROOT_TTree(
        bulk_data_tree, 'is_fullstream', library)

    aux = is_fullstream_branch.num_entries if library == 'uproot' else is_fullstream_branch.GetEntries()

    # Get the start and stop iterator values for
    wf_start = math.floor(start_fraction * aux)
    # the chunk which contains the waveforms which
    wf_stop = math.ceil(stop_fraction * aux)
    # could be potentially read.
    if library == 'uproot':
        is_fullstream_array = is_fullstream_branch.array(
            entry_start=wf_start, entry_stop=wf_stop)
    else:
        is_fullstream_array = wii.get_1d_array_from_pyroot_TBranch(bulk_data_tree, is_fullstream_branch_name,
                                                                   i_low=wf_start,
                                                                   i_up=wf_stop,
                                                                   ROOT_type_code='O')

    aux = np.where(is_fullstream_array)[0] if read_full_streaming_data else np.where(
        np.logical_not(is_fullstream_array))[0]

    if len(aux) == 0:
        raise Exception(generate_exception_message(3,
                                                   'WaveformSet_from_ROOT_file()',
                                                   f"No waveforms of the specified type ({'full-stream' if read_full_streaming_data else 'self-trigger'}) were found."))
    if library == 'uproot':
        waveforms = wii.__build_waveforms_list_from_ROOT_file_using_uproot(aux, bulk_data_tree, meta_data_tree,
                                                                           set_offset_wrt_daq_window=set_offset_wrt_daq_window,
                                                                           first_wf_index=wf_start,
                                                                           verbose=verbose)
    else:
        waveforms = wii.__build_waveforms_list_from_ROOT_file_using_pyroot(aux, bulk_data_tree, meta_data_tree,
                                                                           set_offset_wrt_daq_window=set_offset_wrt_daq_window,
                                                                           first_wf_index=wf_start,
                                                                           subsample=subsample,
                                                                           verbose=verbose)
    if truncate_wfs_to_minimum:
        minimum_length = np.array([len(wf.Adcs) for wf in waveforms]).min()
        for wf in waveforms:
            wf._WaveformAdcs__truncate_adcs(minimum_length)

    return WaveformSet(*waveforms)
