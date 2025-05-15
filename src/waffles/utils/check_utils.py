from typing import List
from waffles.utils.baseline.baseline import SBaseline

def fraction_is_well_formed(lower_limit : float = 0.0,
                            upper_limit : float = 1.0) -> bool:
    
    """
    This function returns True if 

        0.0 <= lower_limit < upper_limit <= 1.0,

    and False if else.

    Parameters
    ----------
    lower_limit : float
    upper_limit : float

    Returns
    ----------
    bool
    """

    if lower_limit < 0.0:
        return False
    elif upper_limit <= lower_limit:
        return False
    elif upper_limit > 1.0:
        return False
    
    return True

def subinterval_is_well_formed( i_low : int, 
                                i_up : int,
                                points_no : int) -> bool:
    
    """
    This method returns True if 0 <= i_low < i_up <= points_no - 1,
    and False if else.

    Parameters
    ----------
    i_low : int
    i_up : int
    points_no : int
        Stands for number of points

    Returns
    ----------
    bool
    """

    if i_low < 0:
        return False
    elif i_up <= i_low:
        return False
    elif i_up > points_no - 1:
        return False
    
    return True

def baseline_limits_are_well_formed(baseline_limits : List[int],
                                    points_no : int) -> bool:

    """
    This method returns True if len(baseline_limits) is an even
    positive number and
    0 <= baseline_limits[0] < baseline_limits[1] < ... < baseline_limits[-1] <= points_no - 1.
    It returns False if else.

    Parameters
    ----------
    baseline_limits : list of int
    points_no : int
        Stands for number of points

    Returns
    ----------
    bool
    """

    if len(baseline_limits) == 0 or len(baseline_limits)%2 != 0:
        return False

    if baseline_limits[0] < 0:
        return False
        
    for i in range(0, len(baseline_limits) - 1):
        if baseline_limits[i] >= baseline_limits[i + 1]:
            return False
            
    if baseline_limits[-1] > points_no - 1:
        return False
    
    return True

def baseliner_class_is_given(baseliner: SBaseline) -> bool:
    """
    This method returns True if baseliner is an instance of SBaseline and False if else.

    Parameters
    ----------
    baseliner : SBaseline

    Returns
    ----------
    bool
    """

    if not isinstance(baseliner, SBaseline):
        return False
    
    return True

def baseliner_class_has_filtering_set(baseliner: SBaseline) -> bool:
    """
    This method returns True if baseliner has filtering set and False if else.

    Parameters
    ----------
    baseliner : SBaseline

    Returns
    ----------
    bool
    """

    if baseliner.filtering is None:
        return False
    
    return True
