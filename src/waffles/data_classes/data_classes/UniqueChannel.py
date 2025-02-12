class UniqueChannel:
    """This class implements a unique channel, in the sense
    that its endpoint and channel number is identified.

    Attributes
    ----------
    endpoint: int
        An endpoint value
    channel: int
        A channel value

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    def __init__(self, endpoint, channel):
        """UniqueChannel class initializer

        Parameters
        ----------
        endpoint: int
        channel: bool
        """

        # Shall we add type checks here?

        self.__endpoint = endpoint
        self.__channel = channel

    # Getters
    @property
    def endpoint(self):
        return self.__endpoint

    @property
    def channel(self):
        return self.__channel

    def __repr__(self) -> str:
        """Returns a string representation of the UniqueChannel
        object. P.e. for a UniqueChannel object with endpoint
        105 and channel 3, the string representation would be
        "105-3".
        """

        return f"{self.endpoint}-{self.channel}"
