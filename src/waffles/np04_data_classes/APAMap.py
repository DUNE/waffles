from typing import List

from waffles.data_classes.UniqueChannel import UniqueChannel
from waffles.data_classes.ChannelMap import ChannelMap

class APAMap(ChannelMap):

    """
    This class implements a channel map for an APA. I.e.
    it is a ChannelMap whose Rows (resp. columns) attribute 
    is fixed to 10 (resp. 4).

    Attributes
    ----------
    Rows : int (inherited from ChannelMap)
    Columns : int (inherited from ChannelMap)
    Type : type (inherited from ChannelMap)
    Data : list of lists (inherited from ChannelMap)
    
    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    def __init__(self, data : List[List[UniqueChannel]]):
        
        """
        APAMap class initializer
        
        Parameters
        ----------
        data : list of lists of UniqueChannel objects
            The length of data must be equal to 10 
            and the length of each one of its lists 
            must be equal to 4.
        """

        # All of the checks are performed 
        # by the base class initializer

        super().__init__(   10,
                            4,                           
                            data)
