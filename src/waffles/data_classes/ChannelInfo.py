class ChannelInfo:
    """This class implements an object that holds PD channel info, specifically location information like position.

    Attributes
    ----------
    APA : int
        APA value
    TPC : int
        TPC value
    X   : float
        x position of channel in centimeters
    Y   : float
        y position of channel in centimeters
    Z   : float
        z position of channel in centimeters

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    def __init__(self, APA, TPC, X, Y, Z):
        """ChannelInfo class initializer

        Parameters
        ----------
        APA: int
        TPC: int
        X:   float
        Y:   float
        Z:   float
        """

        self.__APA = APA
        self.__TPC = TPC
        self.__X = X
        self.__Y = Y
        self.__Z = Z

    # Getters
    @property
    def APA(self):
        return self.__APA

    @property
    def TPC(self):
        return self.__TPC

    @property
    def X(self):
        return self.__X

    @property
    def Y(self):
        return self.__Y

    @property
    def Z(self):
        return self.__Z

    def __repr__(self) -> str:
        """Returns a string representation of the ChannelInfo
        object.
        """

        return f"{self.APA}-{self.TPC}-{self.X}-{self.Y}-{self.Z}"
