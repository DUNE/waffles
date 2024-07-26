import math

import numpy as np
import uproot

try: 
    import ROOT
except ImportError: 
    print("[raw_ROOT_reader.py]: Could not import ROOT module. Do not use 'pyroot' library options.")
    pass

from waffles.data_classes.WaveformSet import WaveformSet

import waffles.utils.check_utils as wuc
import waffles.input.input_utils as wii

from waffles.Exceptions import generate_exception_message

def WaveformSet_from_ROOT_file( filepath : str,
                                library : str,
                                bulk_data_tree_name : str = 'raw_waveforms',
                                meta_data_tree_name : str = 'metadata',
                                set_offset_wrt_daq_window : bool = False,
                                read_full_streaming_data : bool = False,
                                truncate_wfs_to_minimum : bool = False,
                                start_fraction : float = 0.0,
                                stop_fraction : float = 1.0,
                                subsample : int = 1,
                                verbose : bool = True) -> WaveformSet:

    """
    Alternative initializer for a WaveformSet object out of the
    waveforms stored in a ROOT file

    Parameters
    ----------
    filepath : str
        Path to the ROOT file to be read. Such ROOT file should 
        have at least two defined TTree objects, so that the 
        name of one of those starts with the string given to the
        'bulk_data_tree_name' parameter - the bulk data tree - 
        and the other one starts with the string given to the 
        'meta_data_tree_name' parameter - the meta data tree. 
        The bulk data TTree should have at least four branches,
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
        or not, depending on the value given to the                 ## For the moment, the meta-data tree is not
        'read_full_streaming_data' parameter.                       ## read. This needs to change in the near future.
    library : str
        The library to be used to read the input ROOT file. 
        The supported values are 'uproot' and 'pyroot'. If 
        pyroot is selected, then it is assumed that the 
        types of the branches in the bulk-data tree are the 
        following ones:

            - 'adcs'            : vector<short>
            - 'channel'         : 'S', i.e. a 16 bit signed integer
            - 'timestamp'       : 'l', i.e. a 64 bit unsigned integer
            - 'record'          : 'i', i.e. a 32 bit unsigned integer
            - 'is_fullstream'   : 'O', i.e. a boolean

        Additionally, if set_offset_wrt_daq_window is True,
        then the 'daq_timestamp' branch must be of type 'l',
        i.e. a 64 bit unsigned integer. Type checks are not
        implemented here. If these requirements are not met,
        a segmentation fault may occur in the reading process.
    bulk_data_tree_name (resp. meta_data_tree_name) : str
        Name of the bulk-data (meta-data) tree which will be 
        extracted from the given ROOT file. The first object 
        found within the given ROOT file whose name starts
        with the given string and which is a TTree object, 
        will be identified as the bulk-data (resp. meta-data) 
        tree.
    set_offset_wrt_daq_window : bool
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
    read_full_streaming_data : bool
        If True (resp. False), then only the waveforms for which 
        the 'is_fullstream' branch in the bulk-data tree has a 
        value equal to True (resp. False) will be considered.
    truncate_wfs_to_minimum : bool
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
    start_fraction (resp. stop_fraction) : float
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
    subsample : int
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
    verbose : bool
        If True, then functioning-related messages will be
        printed.
    """

    if not wuc.fraction_is_well_formed(start_fraction, stop_fraction):
        raise Exception(generate_exception_message( 1,
                                                    'WaveformSet_from_ROOT_file()',
                                                    f"Fraction limits are not well-formed."))
    if library not in ['uproot', 'pyroot']:
        raise Exception(generate_exception_message( 2,
                                                    'WaveformSet_from_ROOT_file()',
                                                    f"The given library ({library}) is not supported."))
    elif library == 'uproot':
        input_file = uproot.open(filepath)
    else:
        input_file = ROOT.TFile(filepath)
    
    try:
        meta_data_tree, _ = wii.find_TTree_in_ROOT_TFile(   input_file,
                                                            meta_data_tree_name,
                                                            library)
    except NameError:
        meta_data_tree = None           ## To enable compatibility with old runs when the meta data tree was not defined, we are handling 
                                        ## this exception here. This can be done for now, because we are not reading yet any information 
                                        ## from such tree, but setting meta_data_tree to None (i.e. passing None to 
                                        ## __build_waveforms_list_from_ROOT_file_using_uproot or 
                                        ## __build_waveforms_list_from_ROOT_file_using_pyroot) will be unacceptable in the near future.

    bulk_data_tree, _ = wii.find_TTree_in_ROOT_TFile(   input_file,
                                                        bulk_data_tree_name,
                                                        library)
    
    is_fullstream_branch, is_fullstream_branch_name = wii.find_TBranch_in_ROOT_TTree(   bulk_data_tree,
                                                                                        'is_fullstream',
                                                                                        library)

    aux = is_fullstream_branch.num_entries if library == 'uproot' else is_fullstream_branch.GetEntries()

    wf_start = math.floor(start_fraction*aux)   # Get the start and stop iterator values for
    wf_stop = math.ceil(stop_fraction*aux)      # the chunk which contains the waveforms which
                                                # could be potentially read.
    if library == 'uproot':
        is_fullstream_array = is_fullstream_branch.array(   entry_start = wf_start,
                                                            entry_stop = wf_stop)
    else:

        is_fullstream_array = wii.get_1d_array_from_pyroot_TBranch( bulk_data_tree,
                                                                    is_fullstream_branch_name,
                                                                    i_low = wf_start, 
                                                                    i_up = wf_stop,
                                                                    ROOT_type_code = 'O')

    aux = np.where(is_fullstream_array)[0] if read_full_streaming_data else np.where(np.logical_not(is_fullstream_array))[0]
    
    # One could consider summing wf_start to every entry of aux at this point, so that the __build... helper
    # functions do not need to take both parameters idcs_to_retrieve and first_wf_index. However, for
    # the library == 'uproot' case, it is more efficient to clusterize first (which is done within the
    # helper function for the uproot case), then sum wf_start. That's why we carry both parameters until then.

    if len(aux) == 0:
        raise Exception(generate_exception_message( 3,
                                                    'WaveformSet_from_ROOT_file()',
                                                    f"No waveforms of the specified type ({'full-stream' if read_full_streaming_data else 'self-trigger'}) were found."))
    if library == 'uproot':

        waveforms = wii.__build_waveforms_list_from_ROOT_file_using_uproot( aux,
                                                                            bulk_data_tree,
                                                                            meta_data_tree,
                                                                            set_offset_wrt_daq_window = set_offset_wrt_daq_window,
                                                                            first_wf_index = wf_start,
                                                                            verbose = verbose)
    else:
    
        waveforms = wii.__build_waveforms_list_from_ROOT_file_using_pyroot( aux,
                                                                            bulk_data_tree,
                                                                            meta_data_tree,
                                                                            set_offset_wrt_daq_window = set_offset_wrt_daq_window,
                                                                            first_wf_index = wf_start,
                                                                            subsample = subsample,
                                                                            verbose = verbose)
    if truncate_wfs_to_minimum:
                
        minimum_length = np.array([len(wf.Adcs) for wf in waveforms]).min()

        for wf in waveforms:
            wf._WaveformAdcs__truncate_adcs(minimum_length)

    return WaveformSet(*waveforms)