from pathlib import Path
import array
import numpy as np
import uproot
import tempfile

try:
    import ROOT
    ROOT_IMPORTED = True
except ImportError:
    print(
        "[input_utils.py]: Could not import ROOT module. "
        "'pyroot' library options will not be available."
    )
    ROOT_IMPORTED = False
    pass

from typing import Union, List, Tuple, Optional

from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.BeamInfo import BeamInfo
import waffles.utils.numerical_utils as wun
import waffles.Exceptions as we


def find_ttree_in_root_tfile(
    file: Union[uproot.ReadOnlyDirectory, 'ROOT.TFile'],
    TTree_pre_name: str,
    library: str
) -> Union[uproot.TTree, 'ROOT.TTree']:
    """This function returns the first object found in the given
    ROOT file whose name starts with the string given to the
    'TTree_pre_name' parameter and which is a TTree object,
    and the full exact name of the returned TTree object. If
    no such TTree is found, a NameError exception is raised.

    Parameters
    ----------
    file: uproot.ReadOnlyDirectory or ROOT.TFile
        The ROOT file where to look for the TTree object
    TTree_pre_name: str
        The string which the name of the TTree object must
        start with
    library: str
        The library used to open the ROOT file. It can be
        either 'uproot' or 'pyroot'. If 'uproot' (resp.
        'pyroot'), then the 'file' parameter must be of
        type uproot.ReadOnlyDirectory (resp. ROOT.TFile).

    Returns
    ----------
    output: tuple of ( uproot.TTree or ROOT.TTree, str, )
        The first element of the returned tuple is the
        TTree object found in the given TFile whose
        name starts with the string given to the
        'TTree_pre_name' parameter. The second element
        is the full name, within the given TFile, of the
        returned TTree.
    """

    if library == 'uproot':
        if not isinstance(
                file, uproot.ReadOnlyDirectory):
            raise Exception(we.GenerateExceptionMessage(
                1,
                'find_ttree_in_root_tfile()',
                'Since the uproot library was specified, the input file'
                ' must be of type uproot.ReadOnlyDirectory.'))
    elif library == 'pyroot':
        if not isinstance(file, ROOT.TFile):
            raise Exception(we.GenerateExceptionMessage(
                2,
                'find_ttree_in_root_tfile()',
                'Since the pyroot library was specified, the input file '
                'must be of type ROOT.TFile.'))
    else:
        raise Exception(we.GenerateExceptionMessage(
            3,
            'find_ttree_in_root_tfile()',
            f"The library '{library}' is not supported. Either 'uproot' "
            "or 'pyroot' must be given."))
    
    TTree_name = None

    if library == 'uproot':
        for key in file.classnames().keys():
            if key.startswith(TTree_pre_name) and file.classnames()[key] == 'TTree':
                TTree_name = key
                break
    else:
        for key in file.GetListOfKeys():
            if key.GetName().startswith(TTree_pre_name) and key.GetClassName() == 'TTree':
                TTree_name = key.GetName()
                break

    if TTree_name is None:
        raise NameError(we.GenerateExceptionMessage(
            4,
            'find_ttree_in_root_tfile()',
            f"There is no TTree with a name starting with '{TTree_pre_name}'."))
    
    return file[TTree_name], TTree_name


def find_tbranch_in_root_ttree(
    tree: Union[uproot.TTree, 'ROOT.TTree'],
    TBranch_pre_name: str,
    library: str,
    require_exact_match: bool = True
) -> Tuple[Union[uproot.TBranch, 'ROOT.TBranch'], str]:
    """This function returns the first TBranch found in 
    the given ROOT TTree whose name matches the string
    given to the 'TBranch_pre_name' parameter, and the
    full exact name of the returned TBranch. The required
    matching between the given TBranch_pre_name and the
    returned TBranch might be exact or not, depending
    on the input given to the 'require_exact_match'
    parameter. If no such TBranch is found, a NameError
    exception is raised.

    Parameters
    ----------
    tree: uproot.TTree or ROOT.TTree
        The TTree where to look for the TBranch object
    TBranch_pre_name: str
        The string which the name of the TBranch object
        must match to some extent
    library: str
        The library used to read the TBranch from the
        given tree. It can be either 'uproot' or 'pyroot'.
        If 'uproot' (resp. 'pyroot'), then the 'tree'
        parameter must be of type uproot.TTree (resp.
        ROOT.TTree).
    require_exact_match: bool
        If True, then the name of the returned TBranch
        must exactly match the string given to the
        'TBranch_pre_name' parameter. If False, then
        the first TBranch found whose name starts with
        the given identifier will be returned. In
        either case, the exact name of the returned
        TBranch will be also returned.

    Returns
    ----------
    output: tuple of ( uproot.TBranch or ROOT.TBranch, str, )
        The first element of the returned tuple is the
        TBranch object found in the given TTree whose
        name matches to some extent the string given to the
        'TBranch_pre_name' parameter. The second element
        is the full name, within the given TTree, of the
        returned TBranch.
    """

    if library == 'uproot':
        if not isinstance(tree, uproot.TTree):
            raise Exception(we.GenerateExceptionMessage(
                1,
                'find_tbranch_in_root_ttree()',
                'Since the uproot library was specified, '
                'the input tree must be of type uproot.TTree.'))
    elif library == 'pyroot':
        if not isinstance(tree, ROOT.TTree):
            raise Exception(we.GenerateExceptionMessage(
                2,
                'find_tbranch_in_root_ttree()',
                'Since the pyroot library was specified, '
                'the input tree must be of type ROOT.TTree.'))
    else:
        raise Exception(we.GenerateExceptionMessage(
            3,
            'find_tbranch_in_root_ttree()',
            f"The library '{library}' is not supported. "
            "Either 'uproot' or 'pyroot' must be given."))
    TBranch_name = None

    get_list_of_branches = lambda tree, library: \
        tree.keys() if library == 'uproot' \
            else tree.GetListOfBranches()
    
    get_name_of_branch = lambda branch, library: \
        branch if library == 'uproot' \
            else branch.GetName()

    if require_exact_match:
        for branch in get_list_of_branches(tree, library):
            branch_name = get_name_of_branch(branch, library)
            if branch_name == TBranch_pre_name:
                TBranch_name = branch_name
                break
    else:
        for branch in get_list_of_branches(tree, library):
            branch_name = get_name_of_branch(branch, library)
            if branch_name.startswith(TBranch_pre_name):
                TBranch_name = branch_name
                break

    if TBranch_name is None:
        raise NameError(we.GenerateExceptionMessage(
            4,
            'find_tbranch_in_root_ttree()',
            "There is no TBranch with a name "+
            ("matching" if require_exact_match
            else "starting with")+
            f" '{TBranch_pre_name}'."))

    output = (
        tree[TBranch_name] if library == 'uproot' else tree.GetBranch(TBranch_name), TBranch_name)
    return output


def root_to_array_type_code(input: str) -> str:
    """This function gets a length-one string which matches
    a code type used in ROOT TTree and TBranch objects.
    It returns its equivalent in python array module.
    If the code is not recognized, a ValueError exception
    is raised.

    Parameters
    ----------
    input: str
        A length-one string which matches a code type
        used in ROOT TTree and TBranch objects. The
        available codes are:

            - B: an 8 bit signed integer
            - b: an 8 bit unsigned integer
            - S: a 16 bit signed integer
            - s: a 16 bit unsigned integer
            - I: a 32 bit signed integer
            - i: a 32 bit unsigned integer
            - F: a 32 bit floating point
            - D: a 64 bit floating point
            - L: a 64 bit signed integer
            - l: a 64 bit unsigned integer
            - G: a long signed integer, stored as 64 bit
            - g: a long unsigned integer, stored as 64 bit
            - O: [the letter o, not a zero] a boolean (bool)

        For more information, check
        https://root.cern/doc/master/classTTree.html

    Returns
    ----------
    output: str
        The equivalent code in python array module. The possible
        outputs are

            - 'b': signed char (int, 1 byte)
            - 'B': unsigned char (int, 1 byte)
            - 'h': signed short (int, 2 bytes)
            - 'H': unsigned short (int, 2 bytes)
            - 'i': signed int (int, 2 bytes)
            - 'I': unsigned int (int, 2 bytes)
            - 'l': signed long (int, 4 bytes)
            - 'L': unsigned long (int, 4 bytes)
            - 'q': signed long long (int, 8 bytes)
            - 'Q': unsigned long long (int, 8 bytes)
            - 'f': float (float, 4 bytes)
            - 'd': double (float, 8 bytes)

        For more information, check
        https://docs.python.org/3/library/array.html
    """

    map = {
        'B': 'b',
        'b': 'B',
        'O': 'B',
        'S': 'h',
        's': 'H',
        'I': 'l',
        'i': 'L',
        'G': 'q',
        'L': 'q',
        'g': 'Q',
        'l': 'Q',
        'F': 'f',
        'D': 'd'}
    try:
        output = map[input]
    except KeyError:
        raise ValueError(we.GenerateExceptionMessage(
            1,
            'root_to_array_type_code()',
            f"The given data type ({input}) is not recognized."))
    else:
        return output


def get_1d_array_from_pyroot_tbranch(
    tree: 'ROOT.TTree',
    branch_name: str,
    i_low: int = 0,
    i_up: Optional[int] = None,
    ROOT_type_code: str = 'S',
    require_exact_match: bool = True
) -> np.ndarray:
    """This function returns a 1D numpy array containing the
    values of the branch whose name matches the string
    given to the 'branch_name' parameter in the given ROOT
    TTree object. The values are taken from the entries
    of the TTree object whose iterator values are in the
    range [i_low, i_up). I.e. the lower (resp. upper) bound
    is inclusive (resp. exclusive). The required
    matching between the given branch_name and the
    branch whose data is returned, might be exact or not,
    depending on the input given to the 'require_exact_match'
    parameter.

    Parameters
    ----------
    tree: ROOT.TTree
        The ROOT TTree object where to look for the TBranch
    branch_name: str
        The string which the name of the TBranch object
        must match to some extent
    i_low: int
        The inclusive lower bound of the range of entries
        to be read from the branch. It must be non-negative.
        It is set to 0 by default. It must be smaller than
        i_up.
    i_up: int
        The exclusive upper bound of the range of entries
        to be read from the branch. If it is not defined,
        then it is set to the length of the considered branch.
        It it is defined, then it must be smaller or equal
        to the length of the considered branch, and greater
        than i_low.
    ROOT_type_code: str
        The data type of the branch to be read. The valid
        values can be checked in the docstring of the
        root_to_array_type_code() function.
    require_exact_match: bool
        If True, then the name of the TBranch whose
        data is returned, must exactly match the string
        given to the 'branch_name' parameter. If False,
        then the first TBranch found whose name starts
        with the given identifier is the one whose data
        will be returned.

    Returns
    ----------
    output: np.ndarray
    """

    try:
        branch, exact_branch_name = find_tbranch_in_root_ttree(
            tree,
            branch_name,
            'pyroot',
            require_exact_match=require_exact_match
        )
    except NameError:
        raise NameError(
            we.GenerateExceptionMessage(
                1,
                'get_1d_array_from_pyroot_tbranch()',
                "There is no TBranch with a name "+
                ("matching" if require_exact_match
                else "starting with")+
                f" '{branch_name}' in the given tree."
            )
        )
    
    if i_up is None:
        i_up_ = branch.GetEntries()
    else:
        i_up_ = i_up

    if i_low < 0 or i_low >= i_up_ or i_up_ > branch.GetEntries():
        raise Exception(we.GenerateExceptionMessage(
            2,
            'get_1d_array_from_pyroot_tbranch()',
            f"The given range [{i_low}, {i_up_}) "
            "is not well-defined for this branch."))

    # root_to_array_type_code()
    # will raise a ValueError if
    # ROOT_type_code is not recognized.

    retrieval_address = array.array(
        root_to_array_type_code(ROOT_type_code),
        [0])
    
    # Specifying a dtype here might speed up the process
    output = np.empty((i_up_ - i_low,))

    tree.SetBranchAddress(exact_branch_name, retrieval_address)

    for i in range(i_low, i_up_):
        tree.GetEntry(i)
        output[i - i_low] = retrieval_address[0]

    # This is necessary to avoid a segmentation fault. Indeed,
    # from https://root.cern/doc/master/classTTree.html :
    # 'The pointer whose address is passed to TTree::Branch
    # must not be destroyed (i.e. go out of scope) until the
    # TTree is deleted or TTree::ResetBranchAddress is called.'

    tree.ResetBranchAddresses()

    return output


def split_endpoint_and_channel(input: int) -> Tuple[int, int]:
    """
    Parameters
    ----------
    input: str
        len(input) must be 5. It is the caller's responsibility
        to ensure this. Such input is interpreted as the
        concatenation of the endpoint int(input[0:3]) and the
        channel int(input[3:5]).

    Returns
    ----------
    int
        The endpoint value
    int
        The channel value
    """

    return int(str(input)[0:3]), int(str(input)[3:5])


def __build_waveforms_list_from_root_file_using_uproot(
    idcs_to_retrieve: np.ndarray,
    bulk_data_tree: uproot.TTree,
    meta_data_tree: uproot.TTree,
    set_offset_wrt_daq_window: bool = False,
    first_wf_index: int = 0,
    verbose: bool = True
) -> List[Waveform]:
    """This is a helper function which must only be called by the
    WaveformSet_from_ROOT_file() function. This function reads
    a subset of waveforms from the given uproot.TTree and appends
    them one by one to a list of Waveform objects, which is
    finally returned by this function. When the uproot library
    is specified, WaveformSet_from_ROOT_file() delegates such
    task to this helper function.

    Parameters
    ----------
    idcs_to_retrieve: np.ndarray
        A numpy array of (strictly) increasingly-ordered
        integers which contains the indices of the waveforms
        to be read from the TTree given to the bulk_data_tree
        parameter. These indices are referred to the
        first_wf_index iterator value of the bulk data tree.
    bulk_data_tree (resp. meta_data_tree): uproot.TTree
        The tree from which the bulk data (resp. meta data)
        of the waveforms will be read. Branches whose name
        start with 'adcs', 'channel', 'timestamp', 
        'daq_timestamp' and 'record' (resp. 'run' and 
        'ticks_to_nsec') will be required.
    set_offset_wrt_daq_window: bool
        If True, then the time_offset attribute of each
        Waveform is set as the difference between its
        value for the 'timestamp' branch and the value
        for the 'daq_timestamp' branch, in such order,
        referenced to the minimum value of such difference
        among all the waveforms. This is useful to align
        waveforms whose time overlap is not null, for
        plotting and analysis purposes.
    first_wf_index: int
        The index of the first Waveform of the chunk in
        the bulk data, which can be potentially read.
        WaveformSet_from_ROOT_file() calculates this
        value based on its 'start_fraction' input parameter.
    verbose: bool
        If True, then functioning-related messages will be
        printed.

    Returns
    ----------
    waveforms: list of Waveform
    """

    clustered_idcs_to_retrieve = wun.cluster_integers_by_contiguity(
        idcs_to_retrieve)

    if verbose:

        print("In function __build_waveforms_list_from_ROOT_file_using_uproot"
              f"(): Found {len(clustered_idcs_to_retrieve)} cluster(s) of "
              "contiguous waveforms of the selected type "
              "(self-trigger or full-stream) in the ROOT file.")
        print("In function "
              "__build_waveforms_list_from_ROOT_file_using_uproot():"
              " Note that, the lesser the clusters the faster the reading"
              " process will be.")

    # For reference, reading ~1.6e+3 waveforms in 357 clusters takes ~10s,
    # while reading ~176e+3 waveforms in 1 cluster takes the same ~10s.

    # If the file to read is highly framented (i.e. there is a lot of clusters,
    # then it is highly counterproductive to use this logical structure
    # (where we read block-by-block) compared to just reading the whole arrays
    # and then discard what we do not need based on the read 'is_fullstream'
    # array. That's why we should introduce a criterion based on the number
    # of clusters i.e. len(clustered_idcs_to_retrieve) to decide whether
    # to use this block-reading structure or not. While lacking a proper
    # criterion, a threshold for the number of clusters above which just
    # reading the whole arrays, could be gotten as an input parameter of
    # this function. The block-reading strategy is worth it, though, when
    # the input file is not very fragmented.
    # This is an open issue.

    # Note that the indices in clustered_idcs_to_retrieve are referred to
    # the block which we have read. I.e. clustered_idcs_to_retrieve[0] being,
    # p.e. [0,3], means that with respect to the branches in the ROOT file,
    # the first cluster we need to read goes from index wf_start+0 to index
    # wf_start+3-1 inclusive, or wf_start+3. exclusive.
    # Also note that the 'entry_stop' parameter of uproot.TBranch.array()
    # is exclusive.

    meta_data = __read_metadata_from_root_file_using_uproot(meta_data_tree)

    adcs_branch, _ = find_tbranch_in_root_ttree(
        bulk_data_tree,
        'adcs',
        'uproot',
        require_exact_match=False
    )

    channel_branch, _ = find_tbranch_in_root_ttree(
        bulk_data_tree,
        'channel',
        'uproot',
        require_exact_match=False
    )

    timestamp_branch, _ = find_tbranch_in_root_ttree(
        bulk_data_tree,
        'timestamp',
        'uproot',
        require_exact_match=False
    )
    
    daq_timestamp_branch, _ = find_tbranch_in_root_ttree(
        bulk_data_tree,
        'daq_timestamp',
        'uproot',
        require_exact_match=False
    )

    record_branch, _ = find_tbranch_in_root_ttree(
        bulk_data_tree,
        'record',
        'uproot',
        require_exact_match=False
    )

    # Using a list comprehension here is slightly slower than a for loop
    # (97s vs 102s for 5% of wvfs of a 809 MB file running on lxplus9)

    waveforms = []

    if not set_offset_wrt_daq_window:

        # Code is more extensive this way, but faster than evaluating
        # the conditional at each iteration within the loop.

        for interval in clustered_idcs_to_retrieve:
            # Read the waveforms in contiguous blocks
            branch_start = first_wf_index + interval[0]
            branch_stop = first_wf_index + interval[1]

            # It is slightly faster (~106s vs. 114s,
            # for a 809 MB input file running on lxplus9)
            # to read branch by branch rather than going
            # for bulk_data_tree.arrays()

            current_adcs_array = adcs_branch.array(
                entry_start=branch_start,
                entry_stop=branch_stop)

            current_channel_array = channel_branch.array(
                entry_start=branch_start,
                entry_stop=branch_stop)

            current_timestamp_array = timestamp_branch.array(
                entry_start=branch_start,
                entry_stop=branch_stop)
            
            current_daq_timestamp_array = daq_timestamp_branch.array(
                entry_start=branch_start,
                entry_stop=branch_stop)

            current_record_array = record_branch.array(
                entry_start=branch_start,
                entry_stop=branch_stop)
            for i in range(len(current_adcs_array)):

                endpoint, channel = split_endpoint_and_channel(
                    current_channel_array[i])

                waveforms.append(Waveform(
                    current_timestamp_array[i],
                    16.,    # time_step_ns   ## Hardcoded to 16 ns until the
                    # 'time_to_nsec' value from the
                    # 'metadata' TTree is fixed
                    # meta_data[1],
                    current_daq_timestamp_array[i],
                    np.array(current_adcs_array[i]),
                    meta_data[0],
                    current_record_array[i],
                    endpoint,
                    channel,
                    time_offset=0,
                    # Open task: Implement the truncation of the 
                    # Waveform objects at this level (reading 
                    # from a ROOT file using uproot)
                    starting_tick=0))
    else:

        raw_time_offsets = []

        for interval in clustered_idcs_to_retrieve:

            # Read the waveforms in contiguous blocks
            branch_start = first_wf_index + interval[0]
            branch_stop = first_wf_index + interval[1]

            current_adcs_array = adcs_branch.array(
                entry_start=branch_start,
                entry_stop=branch_stop)

            current_channel_array = channel_branch.array(
                entry_start=branch_start,
                entry_stop=branch_stop)

            current_timestamp_array = timestamp_branch.array(
                entry_start=branch_start,
                entry_stop=branch_stop)
            
            current_daq_timestamp_array = daq_timestamp_branch.array(
                entry_start=branch_start,
                entry_stop=branch_stop)

            current_record_array = record_branch.array(
                entry_start=branch_start,
                entry_stop=branch_stop)

            for i in range(len(current_adcs_array)):

                endpoint, channel = split_endpoint_and_channel(
                    current_channel_array[i])

                waveforms.append(Waveform(
                    current_timestamp_array[i],
                    16.,    # time_step_ns
                    # meta_data[1],
                    current_daq_timestamp_array[i],
                    np.array(current_adcs_array[i]),
                    meta_data[0],
                    current_record_array[i],
                    endpoint,
                    channel,
                    time_offset=0,
                    # Open task: Implement the truncation of the Waveform
                    # objects at this level (reading from a ROOT file
                    # using uproot)
                    starting_tick=0))

                raw_time_offsets.append(
                    int(current_timestamp_array[i]) - int(current_daq_timestamp_array[i]))

        time_offsets = wun.reference_to_minimum(raw_time_offsets)

        for i in range(len(waveforms)):
            waveforms[i]._WaveformAdcs__set_time_offset(time_offsets[i])

    return waveforms


def __build_waveforms_list_from_root_file_using_pyroot(
    idcs_to_retrieve: np.ndarray,
    bulk_data_tree: 'ROOT.TTree',
    meta_data_tree: 'ROOT.TTree',
    set_offset_wrt_daq_window: bool = False,
    first_wf_index: int = 0,
    subsample: int = 1,
    verbose: bool = True
) -> List[Waveform]:
    """This is a helper function which must only be called by
    the WaveformSet_from_ROOT_file() function. This function
    reads a subset of waveforms from the given ROOT.TTree
    and appends them one by one to a list of Waveform objects,
    which is finally returned by this function. When the
    pyroot library is specified, WaveformSet_from_ROOT_file()
    delegates such task to this helper function.

    Parameters
    ----------
    idcs_to_retrieve: np.ndarray
        A numpy array of (strictly) increasingly-ordered
        integers which contains the indices of the waveforms
        to be read from the TTree given to the bulk_data_tree
        parameter. These indices are referred to the
        first_wf_index iterator value of the bulk data tree.
    bulk_data_tree (resp. meta_data_tree): ROOT.TTree
        The tree from which the bulk data (resp. meta data)
        of the waveforms will be read. Branches whose name
        start with 'adcs', 'channel', 'timestamp', 
        'daq_timestamp' and 'record' (resp. 'run' and 
        'ticks_to_nsec') will be required. For more information 
        on the expected data types for these branches, check 
        the WaveformSet_from_ROOT_file()
        function documentation.
    set_offset_wrt_daq_window: bool
        If True, then the time_offset attribute of each 
        Waveform is set to the difference between its 
        value for the 'timestamp' branch and its value
        for the 'daq_timestamp' branch, in such order,
        referenced to the minimum value of such difference
        among all the waveforms. This is useful to align
        waveforms whose time overlap is not null, for
        plotting and analysis purposes.
    first_wf_index: int
        The index of the first Waveform of the chunk in
        the bulk data, which can be potentially read.
        WaveformSet_from_ROOT_file() calculates this value
        based on its 'start_fraction' input parameter.
    subsample: int
        It matches one plus the number of waveforms to be
        skipped between two consecutive waveforms to be read.
        I.e. if subsample is set to N, then the i-th Waveform
        to be read is the one with index equal to
        first_wf_index + idcs_to_retrieve[i*N]. P.e.
        the 0-th Waveform to be read is the one with
        index equal to first_wf_index + idcs_to_retrieve[0],
        the 1-th Waveform to be read is the one with
        index equal to first_wf_index + idcs_to_retrieve[N]
        and so on.
        If True, then functioning-related messages will be
        printed.

    Returns
    ----------
    waveforms: list of Waveform
    """

    meta_data = __read_metadata_from_root_file_using_pyroot(meta_data_tree)

    _, adcs_branch_exact_name = find_tbranch_in_root_ttree(
        bulk_data_tree,
        'adcs',
        'pyroot',
        require_exact_match=False
    )
    adcs_address = ROOT.std.vector('short')()
    bulk_data_tree.SetBranchAddress(adcs_branch_exact_name,
                                    adcs_address)

    _, channel_branch_exact_name = find_tbranch_in_root_ttree(
        bulk_data_tree,
        'channel',
        'pyroot',
        require_exact_match=False
    )

    channel_address = array.array(
        root_to_array_type_code('S'),
        [0])

    bulk_data_tree.SetBranchAddress(
        channel_branch_exact_name,
        channel_address)

    _, timestamp_branch_exact_name = find_tbranch_in_root_ttree(
        bulk_data_tree,
        'timestamp',
        'pyroot',
        require_exact_match=False
    )

    timestamp_address = array.array(
        root_to_array_type_code('l'),
        [0])

    bulk_data_tree.SetBranchAddress(
        timestamp_branch_exact_name,
        timestamp_address)
    
    _, daq_timestamp_branch_exact_name = find_tbranch_in_root_ttree(
        bulk_data_tree,
        'daq_timestamp',
        'pyroot',
        require_exact_match=False
    )
    
    daq_timestamp_address = array.array(
        root_to_array_type_code('l'),
        [0])

    bulk_data_tree.SetBranchAddress(
        daq_timestamp_branch_exact_name,
        daq_timestamp_address)

    _, record_branch_exact_name = find_tbranch_in_root_ttree(
        bulk_data_tree,
        'record',
        'pyroot',
        require_exact_match=False
    )

    record_address = array.array(
        root_to_array_type_code('i'),
        [0])

    bulk_data_tree.SetBranchAddress(
        record_branch_exact_name,
        record_address)

    idcs_to_retrieve_ = first_wf_index + idcs_to_retrieve[::subsample]

    waveforms = []

    if not set_offset_wrt_daq_window:

        # Code is more extensive this way, but faster than evaluating
        # the conditional at each iteration within the loop.

        for idx in idcs_to_retrieve_:

            bulk_data_tree.GetEntry(int(idx))

            endpoint, channel = split_endpoint_and_channel(channel_address[0])

            waveforms.append(Waveform(
                timestamp_address[0],
                16.,    # time_step_ns   ## Hardcoded to 16 ns until the
                # 'time_to_nsec' value from the
                # 'metadata' TTree is fixed
                # meta_data[1],   # time_step_ns
                daq_timestamp_address[0],
                np.array(adcs_address),
                meta_data[0],
                record_address[0],
                endpoint,
                channel,
                time_offset=0,
                # Open task: Implement the truncation of the Waveform
                # objects at this level (reading from a ROOT file
                # using pyroot)
                starting_tick=0))
    else:

        raw_time_offsets = []

        for idx in idcs_to_retrieve_:

            bulk_data_tree.GetEntry(int(idx))

            endpoint, channel = split_endpoint_and_channel(channel_address[0])

            waveforms.append(Waveform(
                timestamp_address[0],
                16.,
                # time_step_ns
                # meta_data[1],
                daq_timestamp_address[0],
                np.array(adcs_address),
                meta_data[0],
                record_address[0],
                endpoint,
                channel,
                time_offset=0,
                # Open task: Implement the truncation of the Waveform
                # objects at this level (reading from a ROOT file
                # using pyroot)
                starting_tick=0))

            raw_time_offsets.append(
                int(timestamp_address[0]) - int(daq_timestamp_address[0]))

        time_offsets = wun.reference_to_minimum(raw_time_offsets)

        for i in range(len(waveforms)):
            waveforms[i]._WaveformAdcs__set_time_offset(time_offsets[i])

    # This is necessary to avoid a segmentation fault.
    #  For more information, check
    # https://root.cern/doc/master/classTTree.html
    
    bulk_data_tree.ResetBranchAddresses()

    return waveforms


def __read_metadata_from_root_file_using_uproot(
    meta_data_tree: uproot.TTree
) -> Tuple[Union[int, float]]:
    """This is a helper function which must only be called by
    the __build_waveforms_list_from_ROOT_file_using_uproot()
    helper function. Such function delegates the task of
    reading the data in the meta-data tree to this function.
    This function reads and packs such data into a tuple,
    which is the returned object.

    Parameters
    ----------
    meta_data_tree: uproot.TTree
        The tree from which the meta data of the waveforms
        will be read. Branches whose names start with 'run'
        and 'ticks_to_nsec' will be required.

    Returns
    ----------
    output: tuple of ( int, float, )
        The first (resp. second) element of the returned
        tuple is the run number (resp. nanoseconds per time
        tick) of the data in the ROOT file.
    """

    run_branch, _ = find_tbranch_in_root_ttree(
        meta_data_tree,
        'run',
        'uproot',
        require_exact_match=False
    )

    ticks_to_nsec_branch, _ = find_tbranch_in_root_ttree(
        meta_data_tree,
        'ticks_to_nsec',
        'uproot',
        require_exact_match=False
    )
        
    run = int(run_branch.array()[0])

    ticks_to_nsec = float(ticks_to_nsec_branch.array()[0])

    return (run, ticks_to_nsec,)


def __read_metadata_from_root_file_using_pyroot(
    meta_data_tree: 'ROOT.TTree'
) -> Tuple[Union[int, float]]:
    """This is a helper function which must only be called by
    the __build_waveforms_list_from_root_file_using_pyroot()
    helper function. Such function delegates the task of
    reading the data in the meta-data tree to this function.
    This function reads and packs such data into a tuple,
    which is the returned object.

    Parameters
    ----------
    meta_data_tree: ROOT.TTree
        The tree from which the meta data of the waveforms
        will be read. Branches whose names start with 'run'
        and 'ticks_to_nsec' will be required.

    Returns
    ----------
    output: tuple of ( int, float, )
        The first (resp. second) element of the returned
        tuple is the run number (resp. nanoseconds per time
        tick) of the data in the ROOT file.
    """

    _, run_branch_exact_name = find_tbranch_in_root_ttree(
        meta_data_tree,
        'run',
        'pyroot',
        require_exact_match=False
    )

    _, ticks_to_nsec_branch_exact_name = find_tbranch_in_root_ttree(
        meta_data_tree,
        'ticks_to_nsec',
        'pyroot',
        require_exact_match=False
    )
    
    run_address = array.array(
        root_to_array_type_code('i'),
        [0])

    ticks_to_nsec_address = array.array(
        root_to_array_type_code('F'),
        [0])

    meta_data_tree.SetBranchAddress(
        run_branch_exact_name,
        run_address)

    meta_data_tree.SetBranchAddress(
        ticks_to_nsec_branch_exact_name,
        ticks_to_nsec_address)

    meta_data_tree.GetEntry(0)

    run = int(run_address[0])

    ticks_to_nsec = float(ticks_to_nsec_address[0])

    meta_data_tree.ResetBranchAddresses()

    return (run, ticks_to_nsec,)


def filepath_is_root_file_candidate(filepath: str) -> bool:
    """This function returns True if the given file path points
    to a file which exists and whose extension is '.root'. It
    returns False if else.

    Parameters
    ----------
    filepath: str
        The file path to be checked.

    Returns
    ----------
    bool
    """

    path = Path(filepath)

    if path.is_file() and path.suffix == '.root':
        return True

    return False


def write_permission(directory_path: str) -> bool:
    """This function gets the path to a directory, and checks
    whether the running process has write permissions in such
    directory.

    Parameters
    ----------
    directory_path: str
        Path to an existing directory. If the directory does
        not exist, an exception is raised.

    Returns
    ----------
    bool
        True if the the running process has write permissions
        in the specified directory. False if else."""

    try:
        with tempfile.TemporaryFile(
            dir=directory_path
        ):
            pass

        return True

    except FileNotFoundError:
        raise we.NonExistentDirectory(
            we.GenerateExceptionMessage(
                1,
                'write_permission()',
                f"The specified directory ({directory_path}) does not exist."
            )
        )

    except (OSError, PermissionError):
        return False



def __build_beam_list_from_root_file_using_pyroot(
    nentries: int,
    bulk_data_tree: 'ROOT.TTree',
    verbose: bool = True
) -> List[BeamInfo]:
    """This is a helper function which must only be called by
    the BeamInfo_from_root_file() function. This function
    reads a subset of waveforms from theCreate a list of
    BeamInfo objects reading a root file. 
    When the pyroot library is specified, BeamInfo_from_root_file()
    delegates such task to this helper function.

    Parameters
    ----------
    nentries: int
        the number of entries of the root tree
    bulk_data_tree: ROOT.TTree
        The root tree containing the beam information
    verbose: bool
        Currently not used
    
    Returns
    ----------
    List[BeamInfo]: list of BeamInfo objects
    """

    evt_address     = get_branch_address(bulk_data_tree, 'Event', 'I')        
    run_address     = get_branch_address(bulk_data_tree, 'Run', 'I')    
    tof_address     = get_branch_address(bulk_data_tree, 'TOF', 'D')
    C0_address      = get_branch_address(bulk_data_tree, 'C0',  'I')
    C1_address      = get_branch_address(bulk_data_tree, 'C1',  'I')
    P_address       = get_branch_address(bulk_data_tree, 'P',   'D')
    acqTime_address = get_branch_address(bulk_data_tree, 'acqTime',   'D')
    Time_address    = get_branch_address(bulk_data_tree, 'Time',   'l')
    RDTS_address    = get_branch_address(bulk_data_tree, 'RDTS',   'l')    

    beam_infos = []

    for idx in range(nentries):
        bulk_data_tree.GetEntry(int(idx+1))
        beam_infos.append(BeamInfo(run_address[0],
                                   evt_address[0],
                                   RDTS_address[0],
                                   P_address[0],
                                   tof_address[0],
                                   C0_address[0],
                                   C1_address[0]))
        
    bulk_data_tree.ResetBranchAddresses()
    
    return beam_infos


def __build_beam_list_from_root_file_using_uproot(
    nentries: int,
    bulk_data_tree: uproot.TTree,
    verbose: bool = True
) -> List[BeamInfo]:
    """This is a helper function which must only be called by
    the BeamInfo_from_root_file() function. This function
    reads a subset of waveforms from theCreate a list of
    BeamInfo objects reading a root file. 
    When the uproot library is specified, BeamInfo_from_root_file()
    delegates such task to this helper function.

    Parameters
    ----------
    nentries: int
        the number of entries of the root tree
    bulk_data_tree: uproot.TTree
        The root tree containing the beam information
    verbose: bool
        Currently not used
    
    Returns
    ----------
    List[BeamInfo]: list of BeamInfo objects
    """

    evt_array     = get_branch_array_uproot(bulk_data_tree, nentries, 'Event') 
    run_array     = get_branch_array_uproot(bulk_data_tree, nentries, 'Run')
    tof_array     = get_branch_array_uproot(bulk_data_tree, nentries, 'TOF') 
    C0_array      = get_branch_array_uproot(bulk_data_tree, nentries, 'C0')  
    C1_array      = get_branch_array_uproot(bulk_data_tree, nentries, 'C1')  
    P_array       = get_branch_array_uproot(bulk_data_tree, nentries, 'P')   
    acqTime_array = get_branch_array_uproot(bulk_data_tree, nentries, 'acqTime')
    Time_array    = get_branch_array_uproot(bulk_data_tree, nentries, 'Time')
    RDTS_array    = get_branch_array_uproot(bulk_data_tree, nentries, 'RDTS')

    beam_infos = []

    for idx in range(len(evt_array)):
        beam_infos.append(BeamInfo(run_array[idx],
                                   evt_array[idx],
                                   RDTS_array[idx],
                                   P_array[idx],
                                   tof_array[idx],
                                   C0_array[idx],
                                   C1_array[idx]))

    return beam_infos

def get_branch_array_uproot(
        bulk_data_tree: uproot.TTree,
        nentries: int,
        branch_name: str
):

    """This is a helper function gets the branch array
    with a given name for a uproot.Tree

    Parameters
    ----------
    bulk_data_tree: uproot.TTree
        The root tree 
    branch_name: str
        The name of the root tree branch
    
    Returns
    ----------
    
    """
    
    branch, branch_exact_name = find_tbranch_in_root_ttree(bulk_data_tree, branch_name, 'uproot')
    branch_array = branch.array(entry_start=0,entry_stop=nentries)

    return branch_array


def get_branch_address(
    bulk_data_tree: 'ROOT.TTree',
    branch_name: str,
    branch_type: str
):
    """This is a helper function gets the branch address
    with a given name and type for a ROOT.Tree

    Parameters
    ----------
    bulk_data_tree: ROOT.TTree
        The root tree 
    branch_name: str
        The name of the root tree branch
    branch_type: str
        The type of the root tree branch    
    
    Returns
    ----------
    
    """

    
    _, branch_exact_name = find_tbranch_in_root_ttree(bulk_data_tree, branch_name, 'pyroot')
    branch_address = array.array(root_to_array_type_code(branch_type), [0])
    bulk_data_tree.SetBranchAddress(branch_exact_name, branch_address)

    
    return branch_address


def __truncate_waveforms_to_minimum_length_in_WaveformSet(
        waveform_list: List[Waveform]
) -> None:
    """This helper function truncates the adcs object of every
    Waveform object in the given list, to the length of the
    shortest adcs found in such list.

    Parameters
    ----------
    waveform_list: List[Waveform]
        The input waveforms list whose waveforms may be
        truncated

    Returns
    ----------
    None
    """

    minimum_length = np.array(
        [len(wf.adcs) for wf in waveform_list]
    ).min()

    for wf in waveform_list:
        wf._WaveformAdcs__slice_adcs(
            0,
            minimum_length
        )

    return