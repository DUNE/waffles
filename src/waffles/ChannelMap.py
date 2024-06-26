from typing import List

from .UniqueChannel import UniqueChannel
from .Map import Map
from .Exceptions import generate_exception_message

class ChannelMap(Map):

    """
    This class implements a Map whose type is UniqueChannel.

    Attributes
    ----------
    Rows : int (inherited from Map)
    Columns : int (inherited from Map)
    Type : type (inherited from Map)
    Data : list of lists (inherited from Map)
    
    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    def __init__(self,  rows : int,
                        columns : int,
                        data : List[List[UniqueChannel]]):
        
        """
        ChannelMap class initializer
        
        Parameters
        ----------
        rows : int
            It must be a positive integer
        columns : int
            It must be a positive integer
        data : list of lists of UniqueChannel objects
            The length of data must be equal to rows 
            and the length of each one of its lists 
            must be equal to columns.
        """

        ## Shall we add type checks here?

        if data is None:
            raise Exception(generate_exception_message( 1,
                                                        'ChannelMap.__init__()',
                                                        "The data parameter must be defined."))
        # The rest of the checks are performed 
        # by the base class initializer

        super().__init__(   rows,
                            columns,
                            UniqueChannel,                               
                            data = data)