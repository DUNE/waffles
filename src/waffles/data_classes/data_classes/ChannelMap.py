from typing import List, Tuple

from waffles.data_classes.UniqueChannel import UniqueChannel
from waffles.data_classes.Map import Map
from waffles.Exceptions import GenerateExceptionMessage


class ChannelMap(Map):
    """This class implements a Map whose type is UniqueChannel.

    Attributes
    ----------
    rows: int (inherited from Map)
    columns: int (inherited from Map)
    type: type (inherited from Map)
    data: list of lists (inherited from Map)

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    def __init__(
        self, 
        rows: int,
        columns: int,
        data: List[List[UniqueChannel]]
    ):
        """ChannelMap class initializer

        Parameters
        ----------
        rows: int
            It must be a positive integer
        columns: int
            It must be a positive integer
        data: list of lists of UniqueChannel objects
            The length of data must be equal to rows
            and the length of each one of its lists
            must be equal to columns.
        """

        # Shall we add type checks here?

        if data is None:
            raise Exception(GenerateExceptionMessage(
                1,
                'ChannelMap.__init__()',
                "The data parameter must be defined."))
        # The rest of the checks are performed
        # by the base class initializer

        super().__init__(
            rows,
            columns,
            UniqueChannel,
            data=data)

    def find_channel(
        self, 
        unique_channel: UniqueChannel
    ) -> Tuple[bool, Tuple[int, int]]:
        """This method gets an UniqueChannel object
        and returns a tuple with a boolean and a
        tuple with two integers. If the given
        channel is spotted in this ChannelMap object,
        the boolean is True and the tuple contains
        the position of the channel in the map,
        i.e. the iterator values i, j so that
        self.data[i][j] is an UniqueChannel object
        with the same endpoint and channel values as
        the given UniqueChannel. If the given
        channel is not found, the boolean is False
        and the tuple is (-1, -1).

        Parameters
        ----------
        unique_channel: UniqueChannel
            Unique channel to look for within this
            ChannelMap object

        Returns
        -------
        output: tuple of ( bool, tuple of (int, int, ), )
        """

        for i in range(self.rows):
            for j in range(self.columns):
                aux = self.data[i][j]
                if aux.endpoint == unique_channel.endpoint:
                    if aux.channel == unique_channel.channel:
                        return (True, (i, j))

        return (False, (-1, -1))
