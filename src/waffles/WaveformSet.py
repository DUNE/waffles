import math
from typing import Tuple

import uproot
import numpy as np

from src.waffles.NWaveform import Waveform
from src.waffles.Exceptions import generate_exception_message

class WaveformSet:

    """
    This class implements a set of waveforms.

    Attributes
    ----------
    Waveforms : list of Waveform objects
        Waveforms[i] gives the i-th waveform in the set.

    Runs : list of int                                          ## Shall we keep this attribute?
        It contains the run number of any run for which
        there is at least one waveform in the set.

    AvailableChannels : dictionary                              ## Shall we keep this attribute?
        It is a dictionary whose keys are endpoints (int) 
        and its values are lists of channels (list of int).
        If there is at least one Waveform object within
        this WaveformSet which comes from endpoint n, then
        n belongs to AvailableChannels.keys(). 
        AvailableChannels[n] is a list of channels for 
        endpoint n. If there is at least one waveform for
        endpoint n and channel m, then m belongs to 
        AvailableChannels[n].

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    def __init__(self,  *waveforms):
        
        """WaveformSet class initializer
        
        Parameters
        ----------
        waveforms : unpacked list of Waveform objects
            The waveforms that will be added to the set
        """

        ## Shall we add type checks here?
        
        self.__waveforms = list(waveforms)

        # self.__runs = []                  ## Implement filling
        # self.__available_channels = {}    ## of these attributes


    #Getters
    @property
    def Waveforms(self):
        return self.__waveforms
    
    #Getters
    @property
    def Runs(self):
        return self.__runs
    
    #Getters
    @property
    def AvailableChannels(self):
        return self.__available_channels

    @classmethod
    def from_ROOT_file(cls, filepath : str,
                            tree_to_look_for : str ='raw_waveforms',
                            start_fraction : float = 0.0,
                            stop_fraction : float = 1.0) -> 'WaveformSet':

        """
        Alternative initializer for a WaveformSet object out of the
        waveforms stored in a ROOT file

        Parameters
        ----------
        filepath : str
            Path to the ROOT file to be read. Such ROOT file should 
            have a defined TTree object whose name matches tree_to_look_for.
            Such TTree should have at least three branches, with names
            'channel', 'timestamp', 'adcs', from which the values for           ## For the moment, the timestamp branch may
            the Waveform objects attributes Channel, Timestamp and Adcs         ## be called 'timestamps'
            will be taken respectively.
        tree_to_look_for : str
            Name of the tree which will be extracted from the given
            ROOT file
        start_fraction (resp. stop_fraction) : float
            Gives the iterator value for the first (resp. last) waveform
            which will be loaded into this WaveformSet object. P.e. 
            setting start_fraction to 0.5 and stop_fraction to 0.75 
            will result in loading the waveforms that belong to the 
            third quarter of the input file.
        """

        if not WaveformSet.fraction_is_well_formed(start_fraction, stop_fraction):
            raise Exception(generate_exception_message( 1,
                                                        'WaveformSet.from_ROOT_file()',
                                                        f"Fraction limits are not well-formed"))
        input_file = uproot.open(filepath)

        try:
            aux = input_file[tree_to_look_for+';1']     # Assuming that ROOT appends
        except KeyError:                                # ';1' to its trees names
            raise Exception(generate_exception_message( 2,
                                                        'WaveformSet.from_ROOT_file()',
                                                        f"TTree {tree_to_look_for} not found in {filepath}"))
        if 'channel' not in aux.keys():
            raise Exception(generate_exception_message( 3,
                                                        'WaveformSet.from_ROOT_file()',
                                                        f"Branch 'channel' not found in the given TTree"))
        if 'timestamp' not in aux.keys() and 'timestamps' not in aux.keys():    ## Temporal
            raise Exception(generate_exception_message( 4,
                                                        'WaveformSet.from_ROOT_file()',
                                                        f"Branch 'timestamp' not found in the given TTree"))
        if 'adcs' not in aux.keys():
            raise Exception(generate_exception_message( 5,
                                                        'WaveformSet.from_ROOT_file()',
                                                        f"Branch 'adcs' not found in the given TTree"))
        
        adcs = aux['adcs']  # adcs is an uproot.TBranch object

        wf_start = math.floor(start_fraction*adcs.num_entries)
        wf_stop = math.ceil(stop_fraction*adcs.num_entries)

        channels = aux['channel'].array(entry_start=wf_start, 
                                        entry_stop=wf_stop)         # It is slightly faster (~106s vs. 114s, for a
                                                                    # 809 MB input file running on lxplus9) to read
        adcs = aux['adcs'].array(   entry_start=wf_start,           # branch by branch rather than going for aux.arrays()
                                    entry_stop=wf_stop)          
        try:
            timestamps = aux['timestamp'].array(entry_start=wf_start,
                                                entry_stop=wf_stop)   
        except uproot.exceptions.KeyInFileError:    
            timestamps = aux['timestamps'].array(   entry_start=wf_start,
                                                    entry_stop=wf_stop) ## Temporal

        waveforms = []                  # Using a list comprehension here is slightly slower than a for loop
        for i in range(len(adcs)):      # (97s vs 102s for 5% of wvfs of a 809 MB file running on lxplus9)

            endpoint, channel = WaveformSet.get_endpoint_and_channel(channels[i])

            waveforms.append(Waveform(  timestamps[i],
                                        0,      # TimeStep_ns   ## To be implemented from the new
                                                                ## 'metadata' TTree in the ROOT file
                                        np.array(adcs[i]),
                                        0,      #RunNumber      ## To be implemented from the new
                                                                ## 'metadata' TTree in the ROOT file
                                        endpoint,
                                        channel))      
        return cls(*waveforms)

    @staticmethod
    def get_endpoint_and_channel(input : int) -> Tuple[int, int]:
    
        """
        Parameters
        ----------
        input : str
            len(input) must be 5. Such input is interpreted as the
            concatenation of the endpoint int(input[0:3]) and the 
            channel int(input[3:5]).

        Returns
        ----------
        int
            The endpoint value
        int
            The channel value
        """

        return int(str(input)[0:3]), int(str(input)[3:5])
    
    @staticmethod
    def fraction_is_well_formed(lower_limit : float = 0.0,
                                upper_limit : float = 1.0) -> bool:
        
        """
        This method returns True if 0.0<=lower_limit<upper_limit<=1.0,
        and False if else.

        Parameters
        ----------
        lower_limit : float
        upper_limit : float

        Returns
        ----------
        bool
        """

        if lower_limit<0.0:
            return False
        elif upper_limit<=lower_limit:
            return False
        elif upper_limit>1.0:
            return False
        
        return True