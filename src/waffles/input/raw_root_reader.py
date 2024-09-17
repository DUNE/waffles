import os
import math
from pathlib import Path

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
    pass

from typing import List, Optional

import waffles.utils.check_utils as wuc
import waffles.input.input_utils as wii
from waffles.data_classes.WaveformSet import WaveformSet
import waffles.Exceptions as we

def WaveformSet_from_root_files(
    library: str,
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
    verbose: bool = True
) -> WaveformSet:
    """Alternative initializer for a WaveformSet object out of the
    waveforms stored in a list of ROOT files. This function
    checks that the given filepaths, up to the 'folderpath' and
    'filepath_list' input parameters, are valid ROOT files, 
    according to the filepath_is_root_file_candidate() function. 
    The valid ROOT files are read into a WaveformSet, and every 
    WaveformSet is merged together into a single one. The resulting 
    WaveformSet is returned.

    Parameters
    ----------
    library: str
        For every valid filepath, this parameter is passed to
        the WaveformSet_from_root_file() 'library' parameter.
        For more information, check such function docstring.
    folderpath: str
        If given, then the value given to the 'filepath_list'
        parameter is ignored, and the list of filepaths to be 
        read is generated by listing all the files in the given 
        folder. The ROOT files should have the same structure 
        as the one described in the WaveformSet_from_root_file() 
        docstring for the 'filepath' parameter.
    filepath_list: list of strings
        This parameter only makes a difference if the 'folderpath'
        parameter is not defined. It is the list of paths to the 
        ROOT files to be read. Such ROOT files should have the 
        same structure as the one described in the 
        WaveformSet_from_root_file() docstring for the 'filepath' 
        parameter.
    bulk_data_tree_name (resp. meta_data_tree_name) : str
        For every given filepath, this parameter is passed to
        the WaveformSet_from_root_file() 'bulk_data_tree_name'
        (resp. 'meta_data_tree_name') parameter. For more 
        information, check such function docstring.
    set_offset_wrt_daq_window: bool
        For every given filepath, this parameter is passed to
        the WaveformSet_from_root_file() 'set_offset_wrt_daq_window'
        parameter. For more information, check such function
        docstring.
    read_full_streaming_data: bool
        For every given filepath, this parameter is passed to
        the WaveformSet_from_root_file() 'read_full_streaming_data'
        parameter. For more information, check such function
        docstring.
    truncate_wfs_to_minimum: bool
        For every given filepath, this parameter is passed to
        the WaveformSet_from_root_file() 'truncate_wfs_to_minimum'
        parameter. For more information, check such function
        docstring.
    start_fraction (resp. stop_fraction): float
        For every given filepath, this parameter is passed to
        the WaveformSet_from_root_file() 'start_fraction' (resp.
        'stop_fraction') parameter. For more information, check
        such function docstring.
    subsample: int
        For every given filepath, this parameter is passed to
        the WaveformSet_from_root_file() 'subsample' parameter.
        For more information, check such function docstring.
    verbose: bool
        For every given filepath, this parameter is passed to
        the WaveformSet_from_root_file() 'verbose' parameter.
        For more information, check such function docstring.
    
    Returns
    ----------
    output: WaveformSet
        The WaveformSet object which contains all the waveforms
        read from the given ROOT files.
    """

    if folderpath is not None:

        folder = Path(folderpath)
        if not folder.is_dir():

            raise Exception(we.GenerateExceptionMessage(
                1,
                'WaveformSet_from_root_files()',
                f"The given folderpath ({folderpath})"
                " is not a valid directory."))
        
        valid_filepaths = [ os.path.join(folderpath, filename) 
                            for filename in os.listdir(folderpath) 
                            if wii.filepath_is_root_file_candidate(
                                os.path.join(
                                    folderpath, 
                                    filename))]
    else:

        # Remove possible duplicates

        valid_filepaths = [ Path(filepath)
                            for filepath in set(filepath_list)
                            if wii.filepath_is_root_file_candidate(
                                filepath)]
        
    if len(valid_filepaths) == 0:
        raise Exception(we.GenerateExceptionMessage(
            2,
            'WaveformSet_from_root_files()',
            f"No valid ROOT files were found in the "
            "given folder or filepath list."))
    
    if verbose:

        print(f"In function WaveformSet_from_root_files():"
              f" Found {len(valid_filepaths)} different"
              " valid ROOT file(s): \n\n", end = '')
        
        for filepath in valid_filepaths:
            print(f"\t - {filepath}\n", end = '')

        print("\n", end = '')
        print(f"In function WaveformSet_from_root_files():"
              f" Reading file 1/{len(valid_filepaths)} ...")
    
    # There's at least one entry in valid_filepaths
    # The first WaveformSet is handled separatedly, so
    # that the rest of them can be merged into this one

    output = WaveformSet_from_root_file(
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
        verbose=verbose)
    
    count = 2

    # If len(valid_filepaths) == 1, then this loop is not executed
    for filepath in valid_filepaths[1:]:

        if verbose:
            print(f"In function WaveformSet_from_root_files():"
                  f" Reading file {count}/{len(valid_filepaths)} ...")
            count += 1

        aux = WaveformSet_from_root_file(
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
            verbose=verbose)
        
        output.merge(aux)

    if verbose:
        print(f"In function WaveformSet_from_root_files():"
              " Reading finished")

    return output

def WaveformSet_from_root_file( 
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
) -> WaveformSet:
    """Alternative initializer for a WaveformSet object out of the
    waveforms stored in a ROOT file

    Parameters
    ----------
    filepath: str
        Path to the ROOT file to be read. Such ROOT file should 
        have at least two defined TTree objects, so that the 
        name of one of those starts with the string given to the
        'bulk_data_tree_name' parameter - the bulk data tree - 
        and the other one starts with the string given to the 
        'meta_data_tree_name' parameter - the meta data tree.
            The meta data TTree must have at least two branches, 
        whose names start with 
        
            - 'run' 
            - 'ticks_to_nsec' 
            
        from which the values for the Waveform objects attributes 
        RunNumber and TimeStep_ns will be taken respectively.
            The bulk data TTree must have at least five branches,
        whose names should start with

            - 'adcs'
            - 'channel'
            - 'timestamp'
            - 'record'
            - 'is_fullstream'

        from which the values for the Waveform objects attributes
        Adcs, Channel, Timestamp and RecordNumber will be taken 
        respectively. The 'is_fullstream' branch is used to 
        decide whether a certain waveform should be grabbed 
        or not, depending on the value given to the
        'read_full_streaming_data' parameter.
    library: str
        The library to be used to read the input ROOT file. 
        The supported values are 'uproot' and 'pyroot'. If 
        pyroot is selected, then it is assumed that the 
        types of the branches in the meta-data tree are the
        following ones:

            - 'run'             : 'i', i.e. a 32 bit unsigned integer
            - 'ticks_to_nsec'   : 'F', i.e. a 32 bit floating point
    
        while the types of the branches in the bulk-data tree 
        should be:

            - 'adcs'            : vector<short>
            - 'channel'         : 'S', i.e. a 16 bit signed integer
            - 'timestamp'       : 'l', i.e. a 64 bit unsigned integer
            - 'record'          : 'i', i.e. a 32 bit unsigned integer
            - 'is_fullstream'   : 'O', i.e. a boolean

        Additionally, if set_offset_wrt_daq_window is True,
        then the 'daq_timestamp' branch must be of type 'l',
        i.e. a 64 bit unsigned integer. Type checks are not
        implemented here. If these requirements are not met,
        the read data may be corrupted or a a segmentation 
        fault may occur in the reading process.
    bulk_data_tree_name (resp. meta_data_tree_name): str
        Name of the bulk-data (meta-data) tree which will be 
        extracted from the given ROOT file. The first object 
        found within the given ROOT file whose name starts
        with the given string and which is a TTree object, 
        will be identified as the bulk-data (resp. meta-data) 
        tree.
    set_offset_wrt_daq_window: bool
        If True, then the bulk data tree must also have a
        branch whose name starts with 'daq_timestamp'. In
        this case, then the TimeOffset attribute of each
        waveform is set as the difference between its
        value for the 'timestamp' branch and the value
        for the 'daq_timestamp' branch, in such order,
        referenced to the minimum value of such difference
        among all the waveforms. This is useful to align
        waveforms whose time overlap is not null, for 
        plotting and analysis purposes. It is required
        that the time overlap of every waveform is not 
        null, otherwise an exception will be eventually
        raised by the WaveformSet initializer. If False, 
        then the 'daq_timestamp' branch is not queried 
        and the TimeOffset attribute of each waveform 
        is set to 0.
    read_full_streaming_data: bool
        If True (resp. False), then only the waveforms for which 
        the 'is_fullstream' branch in the bulk-data tree has a 
        value equal to True (resp. False) will be considered.
    truncate_wfs_to_minimum: bool
        If True, then the waveforms will be truncated to
        the minimum length among all the waveforms in the input 
        file before being handled to the WaveformSet class 
        initializer. If False, then the waveforms will be 
        read and handled to the WaveformSet initializer as 
        they are. Note that WaveformSet.__init__() will raise 
        an exception if the given waveforms are not homogeneous 
        in length, so this parameter should be set to False 
        only if the user is sure that all the waveforms in 
        the input file have the same length.
    start_fraction (resp. stop_fraction): float
        Gives the iterator value for the first (resp. last) 
        waveform which will be a candidate to be loaded into 
        the created WaveformSet object. Whether they will be 
        finally read also depends on their value for the 
        'is_fullstream' branch and the value given to the 
        'read_full_streaming_data' parameter. P.e. setting 
        start_fraction to 0.5, stop_fraction to 0.75 and 
        read_full_streaming_data to True, will result in 
        loading every waveform which belongs to the third 
        quarter of the input file and for which the 
        'is_fullstream' branch equals to True.
    subsample: int
        This feature is only enabled for the case when
        library == 'pyroot'. Otherwise, this parameter
        is ignored. It matches one plus the number of 
        waveforms to be skipped between two consecutive 
        read waveforms. I.e. if it is set to one, then 
        every waveform will be read. If it is set to two, 
        then every other waveform will be read, and so 
        on. This feature can be combined with the 
        start_fraction and stop_fraction parameters. P.e. 
        if start_fraction (resp. stop_fraction, subsample) 
        is set to 0.25 (resp. 0.5, 2), then every other 
        waveform in the second quarter of the input file 
        will be read.
    verbose: bool
        If True, then functioning-related messages will be
        printed.
    """

    if not wuc.fraction_is_well_formed(start_fraction, stop_fraction):
        raise Exception(we.GenerateExceptionMessage(
            1,
            'WaveformSet_from_root_file()',
            f"Fraction limits are not well-formed."))
    
    if library not in ['uproot', 'pyroot']:
        raise Exception(we.GenerateExceptionMessage(
            2,
            'WaveformSet_from_root_file()',
            f"The given library ({library}) is not supported."))
    
    elif library == 'uproot':
        input_file = uproot.open(filepath)
    else:
        input_file = ROOT.TFile(filepath)
    
    meta_data_tree, _ = wii.find_ttree_in_root_tfile(
        input_file,
        meta_data_tree_name,
        library)

    bulk_data_tree, _ = wii.find_ttree_in_root_tfile(
        input_file,
        bulk_data_tree_name,
        library)
    
    is_fullstream_branch, is_fullstream_branch_name = \
    wii.find_tbranch_in_root_ttree(
        bulk_data_tree,
        'is_fullstream',
        library)

    aux = is_fullstream_branch.num_entries \
    if library == 'uproot' \
    else is_fullstream_branch.GetEntries()

    # Get the start and stop iterator values for
    # the chunk which contains the waveforms which
    # could be potentially read.

    wf_start = math.floor(start_fraction*aux)
    wf_stop = math.ceil(stop_fraction*aux)
      
    if library == 'uproot':
        is_fullstream_array = is_fullstream_branch.array(
            entry_start=wf_start,
            entry_stop=wf_stop)
    else:

        is_fullstream_array = wii.get_1d_array_from_pyroot_tbranch(
            bulk_data_tree,
            is_fullstream_branch_name,
            i_low=wf_start, 
            i_up=wf_stop,
            ROOT_type_code='O')

    aux = np.where(is_fullstream_array)[0] if read_full_streaming_data \
    else np.where(np.logical_not(is_fullstream_array))[0]
    
    # One could consider summing wf_start to every entry of aux at 
    # this point, so that the __build... helper functions do not 
    # need to take both parameters idcs_to_retrieve and 
    # first_wf_index. However, for the library == 'uproot' case, 
    # it is more efficient to clusterize first (which is done within 
    # the helper function for the uproot case), then sum wf_start. 
    # That's why we carry both parameters until then.

    if len(aux) == 0:
        raise Exception(we.GenerateExceptionMessage(
            3,
            'WaveformSet_from_root_file()',
            f"No waveforms of the specified type "
            f"({'full-stream' if read_full_streaming_data else 'self-trigger'})"
            " were found."))
    
    if library == 'uproot':

        waveforms = wii.__build_waveforms_list_from_root_file_using_uproot(
            aux,
            bulk_data_tree,
            meta_data_tree,
            set_offset_wrt_daq_window=set_offset_wrt_daq_window,
            first_wf_index=wf_start,
            verbose=verbose)
    else:
    
        waveforms = wii.__build_waveforms_list_from_root_file_using_pyroot(
            aux,
            bulk_data_tree,
            meta_data_tree,
            set_offset_wrt_daq_window=set_offset_wrt_daq_window,
            first_wf_index=wf_start,
            subsample=subsample,
            verbose=verbose)
        
    if truncate_wfs_to_minimum:
                
        minimum_length = np.array([len(wf.Adcs) for wf in waveforms]).min()

        for wf in waveforms:
            wf._WaveformAdcs__truncate_adcs(minimum_length)

    return WaveformSet(*waveforms)