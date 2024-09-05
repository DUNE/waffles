import math
import inspect

from typing import List, Callable, Optional

from waffles.data_classes.Map import Map
from waffles.data_classes.ChannelMap import ChannelMap
from waffles.data_classes.WaveformSet import WaveformSet

import waffles.utils.filtering_utils as wuf

from waffles.Exceptions import GenerateExceptionMessage


def get_contiguous_indices_map(
    indices_per_slot: int,
    nrows: int = 1,
    ncols: int = 1
) -> Map:
    """This function creates and returns a Map object whose 
    Type attribute is list. Namely, each entry of the output 
    Map is a list of integers. Such Map contains nrows rows 
    and ncols columns. The resulting Map, say output, contains 
    contiguous positive integers in 
    [0, nrows*ncols*indices_per_slot - 1]. I.e.
    output.data[0][0] contains 0, 1, ... , indices_per_slot - 1,
    output.data[0][1] contains indices_per_slot, 
    indices_per_slot + 1, ...  , 2*indices_per_slot - 1, 
    and so on. 

    Parameters
    ----------
    indices_per_slot: int
        The number of indices contained within each 
        entry of the returned Map object
    nrows (resp. ncols): int
        Number of rows (resp. columns) of the returned 
        Map object

    Returns
    ----------
    Map
        A Map object with nrows (resp. ncols) rows 
        (resp. columns). Each entry is a list containing 
        indices_per_slot integers.
    """

    if nrows < 1 or ncols < 1:
        raise Exception(
            GenerateExceptionMessage(
                1,
                'get_contiguous_indices_map()',
                f"The given number of rows ({nrows}) and columns ({ncols}) must be positive."))
    if indices_per_slot < 1:
        raise Exception(
            GenerateExceptionMessage(
                2,
                'get_contiguous_indices_map()',
                f"The given number of indices per slot ({indices_per_slot}) must be positive."))

    aux = [[[k + indices_per_slot*(j + (ncols*i)) for k in range(indices_per_slot)]
            for j in range(ncols)] for i in range(nrows)]

    return Map(
        nrows,
        ncols,
        list,
        data=aux)


def __get_map_of_wf_idcs_by_run(
    waveform_set: WaveformSet,
    blank_map: Map,
    filter_args: Map,
    fMaxIsSet: bool,
    max_wfs_per_axes: Optional[int] = 5
) -> Map:
    """This function should only be called by the
    get_map_of_wf_idcs() function, where the 
    well-formedness checks of the input have
    already been performed. This function generates 
    an output as described in such function 
    docstring, for the case when wf_filter is 
    wuf.match_run. Refer to the get_map_of_wf_idcs() 
    function documentation for more information.

    Parameters
    ----------
    waveform_set: WaveformSet
    blank_map: Map
    filter_args: Map
    fMaxIsSet: bool
    max_wfs_per_axes: int

    Returns
    ----------
    Map
    """

    for i in range(blank_map.rows):
        for j in range(blank_map.columns):

            if filter_args.data[i][j][0] not in waveform_set.runs:
                continue

            # blank_map should not be very big (visualization purposes)
            # so we can afford evaluating the fMaxIsSet conditional here
            # instead of at the beginning of the function (which would
            # be more efficient but would entail a more extensive code)

            if fMaxIsSet:

                counter = 0
                for k in range(len(waveform_set.waveforms)):
                    if wuf.match_run(waveform_set.waveforms[k],
                                     *filter_args.data[i][j]):

                        blank_map.data[i][j].append(k)
                        counter += 1
                        if counter == max_wfs_per_axes:
                            break
            else:
                for k in range(len(waveform_set.waveforms)):
                    if wuf.match_run(waveform_set.waveforms[k],
                                     *filter_args.data[i][j]):

                        blank_map.data[i][j].append(k)
    return blank_map


def __get_map_of_wf_idcs_by_endpoint_and_channel(
    waveform_set: WaveformSet,
    blank_map: Map,
    filter_args: ChannelMap,
    fMaxIsSet: bool,
    max_wfs_per_axes: Optional[int] = 5
) -> Map:
    """This function should only be called by the 
    get_map_of_wf_idcs() function, where the 
    well-formedness checks of the input have 
    already been performed. This function 
    generates an output as described in such 
    function docstring, for the case when 
    wf_filter is wuf.match_endpoint_and_channel. 
    Refer to the get_map_of_wf_idcs() function
    documentation for more information.

    Parameters
    ----------
    waveform_set: WaveformSet
    blank_map: Map
    filter_args: ChannelMap
    fMaxIsSet: bool
    max_wfs_per_axes: int

    Returns
    ----------
    Map
    """

    aux = waveform_set.get_run_collapsed_available_channels()

    for i in range(blank_map.rows):
        for j in range(blank_map.columns):

            if filter_args.data[i][j].endpoint not in aux.keys():
                continue

            elif filter_args.data[i][j].channel not in aux[filter_args.data[i][j].endpoint]:
                continue

            # blank_map should not be very big (visualization purposes)
            # so we can afford evaluating the fMaxIsSet conditional here
            # instead of at the beginning of the function (which would
            # be more efficient but would entail a more extensive code)

            if fMaxIsSet:
                counter = 0
                for k in range(len(waveform_set.waveforms)):
                    if wuf.match_endpoint_and_channel(
                            waveform_set.waveforms[k],
                            filter_args.data[i][j].endpoint,
                            filter_args.data[i][j].channel):
                        blank_map.data[i][j].append(k)
                        counter += 1
                        if counter == max_wfs_per_axes:
                            break
            else:
                for k in range(len(waveform_set.waveforms)):
                    if wuf.match_endpoint_and_channel(
                            waveform_set.waveforms[k],
                            filter_args.data[i][j].endpoint,
                            filter_args.data[i][j].channel):
                        blank_map.data[i][j].append(k)
    return blank_map


def __get_map_of_wf_idcs_general(
    waveform_set: WaveformSet,
    blank_map: Map,
    wf_filter: Callable[..., bool],
    filter_args: Map,
    fMaxIsSet: bool,
    max_wfs_per_axes: Optional[int] = 5
) -> List[List[List[int]]]:
    """This function should only be called by the 
    get_map_of_wf_idcs() function, where the 
    well-formedness checks of the input have 
    already been performed. This function generates 
    an output as described in such function 
    docstring, for the case when wf_filter is 
    neither wuf.match_run nor 
    wuf.match_endpoint_and_channel. Refer to the 
    get_map_of_wf_idcs() function documentation 
    for more information.

    Parameters
    ----------
    waveform_set: WaveformSet
    blank_map: Map
    wf_filter: callable
    filter_args: Map
    fMaxIsSet: bool
    max_wfs_per_axes: int

    Returns
    ----------
    list of list of list of int
    """

    for i in range(blank_map.rows):
        for j in range(blank_map.columns):

            if fMaxIsSet:
                counter = 0
                for k in range(len(waveform_set.waveforms)):
                    if wf_filter(waveform_set.waveforms[k],
                                 *filter_args.data[i][j]):

                        blank_map.data[i][j].append(k)
                        counter += 1
                        if counter == max_wfs_per_axes:
                            break
            else:
                for k in range(len(waveform_set.waveforms)):
                    if wf_filter(waveform_set.waveforms[k],
                                 *filter_args.data[i][j]):
                        blank_map.data[i][j].append(k)
    return blank_map


def get_map_of_wf_idcs(
    waveform_set: WaveformSet,
    nrows: int,
    ncols: int,
    wfs_per_axes: Optional[int] = None,
    wf_filter: Optional[Callable[..., bool]] = None,
    filter_args: Optional[Map] = None,
    max_wfs_per_axes: Optional[int] = 5
) -> Map:
    """This function returns a Map of lists of integers,
    i.e. a Map object whose Type attribute equals 
    list. The contained integers should be interpreted 
    as iterator values for waveforms in the given
    WaveformSet object, waveform_set.

    Parameters
    ----------
    waveform_set: WaveformSet
        The WaveformSet object whose waveforms will be
        iterated through to fill the output Map object
    nrows: int
        The number of rows of the returned Map object
    ncols: int
        The number of columns of the returned Map object
    wfs_per_axes: int
        If it is not None, then it must be a positive
        integer which is smaller or equal to
        math.floor(len(waveform_set.waveforms) / (nrows * ncols)),
        so that the iterator values contained 
        in the output Map are contiguous in
        [0, nrows*ncols*wfs_per_axes - 1]. I.e.
        output.data[0][0] contains 0, 1, ... , wfs_per_axes - 1,
        output.data[0][1] contains wfs_per_axes, wfs_per_axes + 1,
        ... , 2*wfs_per_axes - 1, and so on. 
    wf_filter: callable
        This parameter only makes a difference if
        the 'wfs_per_axes' parameter is None. In such
        case, this one must be a callable object whose 
        first parameter must be called 'Waveform' and 
        must be hinted as a Waveform object. Also, the
        return type of such callable must be annotated
        as a boolean. If wf_filter is 
            - wuf.match_run or
            - wuf.match_endpoint_and_channel,
        this function can benefit from the information 
        in waveform_set.runs and 
        waveform_set.available_channels and its execution 
        time may be reduced with respect to the case 
        where an arbitrary (but compliant) callable 
        is passed to wf_filter.
    filter_args: Map
        This parameter only makes a difference if 
        the 'wfs_per_axes' parameter is None. In such
        case, this parameter must be defined and
        it must be a Map object whose rows (resp.
        columns) attribute match nrows (resp. ncols).
        Its Type attribute must be list. 
        filter_args.data[i][j], for all i and j, is 
        interpreted as a list of arguments which will 
        be given to wf_filter at some point. The user 
        is responsible for giving a set of arguments 
        which comply with the signature of the 
        specified wf_filter. For more information 
        check the return value documentation.
    max_wfs_per_axes: int
        This parameter only makes a difference if           ## If max_wfs_per_axes applies and 
        the 'wfs_per_axes' parameter is None. In such       ## is a positive integer, it is never
        case, and if 'max_wfs_per_axes' is not None,        ## checked that there are enough waveforms
        then output.data[i][j] will contain the indices     ## in the WaveformSet to fill the map.
        for the first max_wfs_per_axes waveforms in the     ## This is an open issue.
        given WaveformSet object, waveform_set, which 
        passed the filter. If it is None, then this 
        function iterates through the whole WaveformSet 
        for every i,j pair. Note that setting this 
        parameter to None may result in a long 
        execution time for big Waveform sets.

    Returns
    ----------
    output: Map
        It is a Map object whose Type attribute is list.
        Namely, output.data[i][j] is a list of integers.
        If the 'wfs_per_axes' parameter is defined, then
        the iterator values contained in the output Map 
        are contiguous in [0, nrows*ncols*wfs_per_axes - 1].
        For more information, check the 'wfs_per_axes'
        parameter documentation. If the 'wfs_per_axes'
        is not defined, then the 'wf_filter' and 'filter_args'
        parameters must be defined and output.data[i][j] 
        gives the indices of the waveforms in the given 
        WaveformSet, say wf, for which 
        wf_filter(wf, *filter_args.data[i][j]) returns True.
        In this last case, the number of indices in each
        entry may be limited, up to the value given to the 
        'max_wfs_per_axes' parameter.
    """

    if nrows < 1 or ncols < 1:
        raise Exception(GenerateExceptionMessage(1,
                                                 'get_map_of_wf_idcs()',
                                                 'The number of rows and columns must be positive.'))
    fFilteringMode = True
    if wfs_per_axes is not None:
        if wfs_per_axes < 1 or wfs_per_axes > math.floor(len(waveform_set.waveforms) / (nrows * ncols)):
            raise Exception(GenerateExceptionMessage(2,
                                                     'get_map_of_wf_idcs()',
                                                     f"The given wfs_per_axes ({wfs_per_axes}) must belong to the range [1, math.floor(len(waveform_set.waveforms) / (nrows * ncols))] ( = {[1, math.floor(len(waveform_set.waveforms) / (nrows * ncols))]})."))
        fFilteringMode = False

    # This one should only be defined as
    # a boolean if fFilteringMode is True
    
    fMaxIsSet = None    
    
    if fFilteringMode:

        try:
            signature = inspect.signature(wf_filter)
        except TypeError:
            raise Exception(GenerateExceptionMessage(3,
                                                     'get_map_of_wf_idcs()',
                                                     "The given wf_filter is not defined or is not callable. It must be suitably defined because the 'wfs_per_axes' parameter is not. At least one of them must be suitably defined."))

        wuf.check_well_formedness_of_generic_waveform_function(signature)

        if filter_args is None:
            raise Exception(GenerateExceptionMessage(4,
                                                     'get_map_of_wf_idcs()',
                                                     "The 'filter_args' parameter must be defined if the 'wfs_per_axes' parameter is not."))

        elif not Map.list_of_lists_is_well_formed(filter_args.data,
                                                  nrows,
                                                  ncols):

            raise Exception(GenerateExceptionMessage(5,
                                                     'get_map_of_wf_idcs()',
                                                     f"The shape of the given filter_args list is not nrows ({nrows}) x ncols ({ncols})."))
        fMaxIsSet = False
        if max_wfs_per_axes is not None:
            if max_wfs_per_axes < 1:
                raise Exception(GenerateExceptionMessage(6,
                                                         'get_map_of_wf_idcs()',
                                                         f"The given max_wfs_per_axes ({max_wfs_per_axes}) must be positive."))
            fMaxIsSet = True

    if not fFilteringMode:

        return get_contiguous_indices_map(wfs_per_axes,
                                          nrows=nrows,
                                          ncols=ncols)

    # fFilteringMode is True and so, wf_filter,
    # filter_args and fMaxIsSet are defined

    else:   
        
        mode_map = {wuf.match_run: 0,
                    wuf.match_endpoint_and_channel: 1}
        try:
            fMode = mode_map[wf_filter]
        except KeyError:
            fMode = 2

        output = Map.from_unique_value(nrows,
                                       ncols,
                                       list,
                                       [],
                                       independent_copies=True)
        if fMode == 0:
            return __get_map_of_wf_idcs_by_run(waveform_set,
                                               output,
                                               filter_args,
                                               fMaxIsSet,
                                               max_wfs_per_axes)
        elif fMode == 1:
            return __get_map_of_wf_idcs_by_endpoint_and_channel(waveform_set,
                                                                output,
                                                                filter_args,
                                                                fMaxIsSet,
                                                                max_wfs_per_axes)
        else:
            return __get_map_of_wf_idcs_general(waveform_set,
                                                output,
                                                wf_filter,
                                                filter_args,
                                                fMaxIsSet,
                                                max_wfs_per_axes)