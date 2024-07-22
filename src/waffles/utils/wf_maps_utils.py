from waffles.data_classes.Map import Map

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