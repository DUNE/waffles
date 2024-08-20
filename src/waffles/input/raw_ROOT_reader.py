import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import uproot

try:
    import ROOT
    ROOT_IMPORTED = True
except ImportError:
    print(
        "[raw_ROOT_reader.py]: Could not import ROOT module. "
        "'pyroot' library options will not be available."
    )
    ROOT_IMPORTED = False

import waffles.utils.check_utils as wuc
import waffles.input.input_utils as wii
from waffles.data_classes.WaveformSet import waveform_set
from waffles.Exceptions import generate_exception_message


def waveform_set_from_root_files(
    library: str,
    folderpath: Optional[str] = None,
    filepath_list: Optional[List[str]] = None,
    bulk_data_tree_name: str = 'raw_waveforms',
    meta_data_tree_name: str = 'metadata',
    set_offset_wrt_daq_window: bool = False,
    read_full_streaming_data: bool = False,
    truncate_wfs_to_minimum: bool = False,
    start_fraction: float = 0.0,
    stop_fraction: float = 1.0,
    subsample: int = 1,
    verbose: bool = True
) -> waveform_set:
    """
    Initializes a waveform_set object from the waveforms stored in a list
    of ROOT files. Validates filepaths, reads them into a waveform_set,
    and merges all into a single waveform_set object.

    Parameters:
    - library: The ROOT processing library ('uproot' or 'pyroot').
    - folderpath: The path to the folder containing ROOT files.
    - filepath_list: List of ROOT file paths.
    - bulk_data_tree_name: Name of the tree containing bulk waveform data.
    - meta_data_tree_name: Name of the tree containing metadata.
    - set_offset_wrt_daq_window: Whether to set offset with respect to DAQ window.
    - read_full_streaming_data: Whether to read full streaming data.
    - truncate_wfs_to_minimum: Whether to truncate waveforms to minimum length.
    - start_fraction: Start fraction for waveform selection.
    - stop_fraction: Stop fraction for waveform selection.
    - subsample: Subsampling factor for waveform data.
    - verbose: Whether to print verbose output.

    Returns:
    - A waveform_set object containing the waveforms from the ROOT files.
    """

    if folderpath:
        folder = Path(folderpath)
        if not folder.is_dir():
            raise Exception(
                generate_exception_message(
                    1,
                    'waveform_set_from_root_files',
                    f"The given folderpath ({
                        folderpath}) is not a valid directory."
                )
            )
        valid_filepaths = [
            f for f in folder.glob('*.root')
            if wii.filepath_is_ROOT_file_candidate(f)
        ]
    else:
        valid_filepaths = [
            Path(f) for f in set(filepath_list or [])
            if wii.filepath_is_ROOT_file_candidate(f)
        ]

    if not valid_filepaths:
        raise Exception(
            generate_exception_message(
                2,
                'waveform_set_from_root_files',
                f"No valid ROOT files were found in the given folder "
                f"'{folderpath}' or filepath list."
            )
        )

    if verbose:
        file_count = len(valid_filepaths)
        print(f"Found {file_count} valid ROOT files:")
        print('\n'.join(f"\t - {f}" for f in valid_filepaths))
        print(f"Reading file 1/{file_count} ...")

    output = waveform_set_from_root_file(
        valid_filepaths[0],
        library,
        bulk_data_tree_name=bulk_data_tree_name,
        meta_data_tree_name=meta_data_tree_name,
        set_offset_wrt_daq_window=set_offset_wrt_daq_window,
        read_full_streaming_data=read_full_streaming_data,
        truncate_wfs_to_minimum=truncate_wfs_to_minimum,
        start_fraction=start_fraction,
        stop_fraction=stop_fraction,
        subsample=subsample,
        verbose=verbose
    )

    for count, filepath in enumerate(valid_filepaths[1:], start=2):
        if verbose:
            print(f"Reading file {count}/{file_count} ...")

        aux = waveform_set_from_root_file(
            filepath,
            library,
            bulk_data_tree_name=bulk_data_tree_name,
            meta_data_tree_name=meta_data_tree_name,
            set_offset_wrt_daq_window=set_offset_wrt_daq_window,
            read_full_streaming_data=read_full_streaming_data,
            truncate_wfs_to_minimum=truncate_wfs_to_minimum,
            start_fraction=start_fraction,
            stop_fraction=stop_fraction,
            subsample=subsample,
            verbose=verbose
        )
        output.merge(aux)

    if verbose:
        print("Reading finished")

    return output


def waveform_set_from_root_file(
    filepath: str,
    library: str,
    bulk_data_tree_name: str = 'raw_waveforms',
    meta_data_tree_name: str = 'metadata',
    set_offset_wrt_daq_window: bool = False,
    read_full_streaming_data: bool = False,
    truncate_wfs_to_minimum: bool = False,
    start_fraction: float = 0.0,
    stop_fraction: float = 1.0,
    subsample: int = 1,
    verbose: bool = True
) -> waveform_set:
    """
    Initializes a waveform_set object from the waveforms stored in a ROOT file.

    Parameters:
    - filepath: The path to the ROOT file.
    - library: The ROOT processing library ('uproot' or 'pyroot').
    - bulk_data_tree_name: Name of the tree containing bulk waveform data.
    - meta_data_tree_name: Name of the tree containing metadata.
    - set_offset_wrt_daq_window: Whether to set offset with respect to DAQ window.
    - read_full_streaming_data: Whether to read full streaming data.
    - truncate_wfs_to_minimum: Whether to truncate waveforms to minimum length.
    - start_fraction: Start fraction for waveform selection.
    - stop_fraction: Stop fraction for waveform selection.
    - subsample: Subsampling factor for waveform data.
    - verbose: Whether to print verbose output.

    Returns:
    - A waveform_set object containing the waveforms from the ROOT file.
    """

    if not wuc.fraction_is_well_formed(start_fraction, stop_fraction):
        raise Exception(
            generate_exception_message(
                1,
                'waveform_set_from_root_file',
                "Fraction limits are not well-formed."
            )
        )

    if library not in ['uproot', 'pyroot']:
        raise Exception(
            generate_exception_message(
                2,
                'waveform_set_from_root_file',
                f"The given library ({library}) is not supported."
            )
        )

    input_file = uproot.open(
        filepath) if library == 'uproot' else ROOT.TFile(filepath)

    meta_data_tree, _ = wii.find_ttree_in_root_tfile(
        input_file, meta_data_tree_name, library)
    bulk_data_tree, _ = wii.find_ttree_in_root_tfile(
        input_file, bulk_data_tree_name, library)

    is_fullstream_branch, is_fullstream_branch_name = wii.find_tbranch_in_root_ttree(
        bulk_data_tree, 'is_fullstream', library
    )

    num_entries = (
        is_fullstream_branch.num_entries if library == 'uproot'
        else is_fullstream_branch.GetEntries()
    )
    wf_start = math.floor(start_fraction * num_entries)
    wf_stop = math.ceil(stop_fraction * num_entries)

    if library == 'uproot':
        is_fullstream_array = is_fullstream_branch.array(
            entry_start=wf_start, entry_stop=wf_stop
        )
    else:
        is_fullstream_array = wii.get_1d_array_from_pyroot_TBranch(
            bulk_data_tree, is_fullstream_branch_name,
            i_low=wf_start, i_up=wf_stop, ROOT_type_code='O'
        )

    selected_indices = np.where(is_fullstream_array)[0] if read_full_streaming_data else np.where(
        np.logical_not(is_fullstream_array)
    )[0]

    if len(selected_indices) == 0:
        raise Exception(
            generate_exception_message(
                3,
                'waveform_set_from_root_file',
                f"No waveforms of the specified type ({
                    'full-stream' if read_full_streaming_data else 'self-trigger'}) were found."
            )
        )

    if library == 'uproot':
        waveforms = wii.__build_waveforms_list_from_root_file_using_uproot(
            selected_indices, bulk_data_tree, meta_data_tree,
            set_offset_wrt_daq_window=set_offset_wrt_daq_window,
            first_wf_index=wf_start, verbose=verbose
        )
    else:
        waveforms = wii.__build_waveforms_list_from_root_file_using_pyroot(
            selected_indices, bulk_data_tree, meta_data_tree,
            set_offset_wrt_daq_window=set_offset_wrt_daq_window,
            first_wf_index=wf_start, subsample=subsample, verbose=verbose
        )

    if truncate_wfs_to_minimum:
        min_length = min(len(wf.adcs) for wf in waveforms)
        for wf in waveforms:
            wf._waveform_adcs__truncate_adcs(min_length)

    return waveform_set(*waveforms)
