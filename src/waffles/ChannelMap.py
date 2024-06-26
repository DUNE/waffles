from typing import List, Tuple

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
        
    def find_channel(self,  unique_channel : UniqueChannel) -> Tuple[bool, Tuple[int, int]]:
        
        """
        This method gets an UniqueChannel object 
        and returns a tuple with a boolean and a
        tuple with two integers. If the given 
        channel is spotted in this ChannelMap object,
        the boolean is True and the tuple contains
        the position of the channel in the map,
        i.e. the iterator values i, j so that
        self.Data[i][j] is an UniqueChannel object
        with the same endpoint and channel values as
        the given unique_channel. If the given 
        channel is not found, the boolean is False 
        and the tuple is (-1, -1).
        
        Parameters
        ----------
        unique_channel : UniqueChannel
            Unique channel to look for within this
            ChannelMap object

        Returns
        -------
        output : tuple of ( bool and tuple of two integers )
        """

        for i in range(self.Rows):
            for j in range(self.Columns):
                aux = self.Data[i][j]
                if aux.Endpoint == unique_channel.Endpoint:
                    if aux.Channel == unique_channel.Channel:
                        return (True, (i, j))
                
        return (False, (-1, -1))