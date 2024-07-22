from typing import Optional

from waffles.data_classes.Map import Map
from waffles.data_classes.WaveformSet import WaveformSet

import waffles.utils.filtering_utils as wuf

from waffles.Exceptions import generate_exception_message

def get_contiguous_indices_map( indices_per_slot : int,
                                nrows : int = 1,
                                ncols : int = 1) -> Map:
    
    """
    This function creates and returns a Map object whose 
    Type attribute is list. Namely, each entry of the output 
    Map is a list of integers. Such Map contains nrows rows 
    and ncols columns. The resulting Map, say output, contains 
    contiguous positive integers in 
    [0, nrows*ncols*indices_per_slot - 1]. I.e.
    output.Data[0][0] contains 0, 1, ... , indices_per_slot - 1,
    output.Data[0][1] contains indices_per_slot, 
    indices_per_slot + 1, ...  , 2*indices_per_slot - 1, 
    and so on. 
    
    Parameters
    ----------
    indices_per_slot : int
        The number of indices contained within each 
        entry of the returned Map object
    nrows (resp. ncols) : int
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
        raise Exception(generate_exception_message( 1,
                                                    'get_contiguous_indices_map()',
                                                    f"The given number of rows ({nrows}) and columns ({ncols}) must be positive."))
    if indices_per_slot < 1:
        raise Exception(generate_exception_message( 2,
                                                    'get_contiguous_indices_map()',
                                                    f"The given number of indices per slot ({indices_per_slot}) must be positive."))
    
    aux = [[[k + indices_per_slot*(j + (ncols*i)) for k in range(indices_per_slot)] for j in range(ncols)] for i in range(nrows)]

    return Map( nrows,
                ncols,
                list,
                data = aux)

def __get_map_of_wf_idcs_by_run(waveform_set : WaveformSet,   
                                blank_map : Map,
                                filter_args : Map,
                                fMaxIsSet : bool,
                                max_wfs_per_axes : Optional[int] = 5) -> Map:
    
    """
    This function should only be called by the
    get_map_of_wf_idcs() function, where the 
    well-formedness checks of the input have
    already been performed. This function generates 
    an output as described in such function 
    docstring, for the case when wf_filter is 
    wuf.match_run. Refer to the get_map_of_wf_idcs() 
    function documentation for more information.

    Parameters
    ----------
    waveform_set : WaveformSet
    blank_map : Map
    filter_args : Map
    fMaxIsSet : bool
    max_wfs_per_axes : int

    Returns
    ----------
    Map
    """

    for i in range(blank_map.Rows):
        for j in range(blank_map.Columns):

            if filter_args.Data[i][j][0] not in waveform_set.Runs:
                continue

            if fMaxIsSet:   # blank_map should not be very big (visualization purposes)
                            # so we can afford evaluating the fMaxIsSet conditional here
                            # instead of at the beginning of the function (which would
                            # be more efficient but would entail a more extensive code)

                counter = 0
                for k in range(len(waveform_set.Waveforms)):
                    if wuf.match_run(   waveform_set.Waveforms[k],
                                        *filter_args.Data[i][j]):
                        
                        blank_map.Data[i][j].append(k)
                        counter += 1
                        if counter == max_wfs_per_axes:
                            break
            else:
                for k in range(len(waveform_set.Waveforms)):
                    if wuf.match_run(   waveform_set.Waveforms[k],
                                        *filter_args.Data[i][j]):
                        
                        blank_map.Data[i][j].append(k)
    return blank_map