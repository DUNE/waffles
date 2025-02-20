from typing import List

from waffles.data_classes.UniqueChannel import UniqueChannel
from waffles.data_classes.ChannelMap import ChannelMap


class MEMMap_geo(ChannelMap):
    """This class implements a channel map for a membrane. I.e.
    it is a ChannelMap whose rows (resp. columns) attribute
    is fixed to 4 (resp. 2).

    Attributes
    ----------
    rows: int (inherited from ChannelMap)
    columns: int (inherited from ChannelMap)
    type: type (inherited from ChannelMap)
    data: list of lists (inherited from ChannelMap)pip 

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    def __init__(self, data: List[List[UniqueChannel]], r: int = 4, c: int = 2):
        """MEMMap_geo class initializer

        Parameters
        ----------
        data: list of lists of UniqueChannel objects
            The length of data must be equal to 4
            and the length of each one of its lists
            must be equal to 2.
        """

        # All of the checks are performed
        # by the base class initializer

        super().__init__(r, c, data)

class MEMMap_ind(ChannelMap):
    """This class implements a channel map for an membrane. I.e.
    it is a ChannelMap whose rows (resp. columns) attribute
    is fixed to 4 (resp. 4).

    Attributes
    ----------
    rows: int (inherited from ChannelMap)
    columns: int (inherited from ChannelMap)
    type: type (inherited from ChannelMap)
    data: list of lists (inherited from ChannelMap)pip 

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    def __init__(self, data: List[List[UniqueChannel]], r: int = 4, c: int = 4):
        """MEMMap_ind class initializer

        Parameters
        ----------
        data: list of lists of UniqueChannel objects
            The length of data must be equal to 4
            and the length of each one of its lists
            must be equal to 4.
        """

        # All of the checks are performed
        # by the base class initializer

        super().__init__(r, c, data)