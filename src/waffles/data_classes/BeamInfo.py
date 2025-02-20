import numpy as np
from typing import List

class BeamInfo():
    """This class implements a BeamInfo, containing information
    about the beam particle, such as time of flight, momentum, cherenkov
    bits, etc

    Attributes
    ----------
    run: int
        run number

    evt: 
        event number

    t: int
        timestamp of the event, it coincides with the daq_window_timestamp

    p: float
        beam particle momentum

    tof: float
        beam particle time of flight

    c0: float
        low pressure cerenkov bit

    c0: float
        hight pressure cerenkov bit
    
    
    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    def __init__(
        self,
            run: int = 0,
            evt: int = 0,            
            t:   int = 0,
            p:   float = 0,
            tof: float = 0,
            c0:  float = 0,
            c1:  float = 0):
            

        # Shall we add add type checks here?

        self.__run = run
        self.__event = evt
        self.__t   = t
        self.__p   = p
        self.__tof = tof
        self.__c0  = c0
        self.__c1  = c1
        

    # Getters
    @property
    def run(self):
        return self.__run

    @property
    def event(self):
        return self.__event
    
    @property
    def t(self):
        return self.__t

    @property
    def p(self):
        return self.__p
    
    @property
    def tof(self):
        return self.__tof
    
    @property
    def c0(self):
        return self.__c0
    
    @property
    def c1(self):
        return self.__c1
    

    
    
