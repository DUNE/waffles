import math
import uproot
import numpy as np

from src.waffles.Exceptions import generate_exception_message

from src.waffles.NWaveform import Waveform

class WaveformSet:

    """
    This class implements a set of waveforms.

    Attributes
    ----------
    Waveforms : list of Waveform objects
        Waveforms[i] gives the i-th waveform in the set

    Runs : list of int
        It contains the run number of any run for which
        there is at least one waveform in the set.

    Available_channels : dictionary
        It is a dictionary whose keys are endpoints (int) 
        and its values are lists of channels (list of int).
        If there is at least one Waveform object within
        this WaveformSet which comes from endpoint n, then
        n belongs to Available_channels.keys(). 
        Available_channels[n] is a list of channels for 
        endpoint n. If there is at least one waveform for
        endpoint n and channel m, then m belongs to 
        Available_channels[n].
    """

    def __init__(self,  *waveforms):
        
        """Waveform class initializer
        
        Parameters
        ----------
        waveforms : unpacked list of Waveform objects
            The waveforms that will be added to the set
        """

        # Do we want to add type checks here?
        
        self.__waveforms = list(waveforms)

        # self.__runs = []                  # Implement filling
        # self.__available_channels = {}    # of these attributes


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
    def Available_channels(self):
        return self.__available_channels

    @classmethod
    def from_ROOT_file(cls, filepath,
                            tree_to_look_for='raw_waveforms',
                            fraction_to_load=1.0):

        """
        Alternative initializer for a WaveformSet object out of the
        waveforms stored in a ROOT file.

        Parameters
        ----------
        filepath : str
            Path to the ROOT file to be read. Such ROOT file should 
            have a defined TTree object whose name matches tree_to_look_for.
            Such TTree should have at least three branches, with names
            'channel', 'timestamps', 'adcs', from which the values for
            the Waveform objects attributes Channel, Timestamp and Adcs
            will be taken respectively.
        tree_to_look_for : str
            Name of the tree which will be extracted from the given
            ROOT file.
        fraction_to_load : float
            Fraction of the total number of waveforms which will be
            loaded to this WaveformSet object.
        """

        fraction_to_load_ = fraction_to_load
        if fraction_to_load<0.0:
            fraction_to_load_ = 0.0
        elif fraction_to_load>1.0:
            fraction_to_load_ = 1.0


        input_file = uproot.open(filepath)

        try:
            aux = input_file[tree_to_look_for+';1']     # Assuming that ROOT appends
        except KeyError:                                # ';1' to its trees names
            raise Exception(generate_exception_message( 1,
                                                        'WaveformSet.from_ROOT_file()',
                                                        f"TTree {tree_to_look_for} not found in {filepath}"))
        if 'channel' not in aux.keys():
            raise Exception(generate_exception_message( 2,
                                                        'WaveformSet.from_ROOT_file()',
                                                        f"Branch 'channel' not found in the given TTree"))
        if 'timestamps' not in aux.keys():
            raise Exception(generate_exception_message( 3,
                                                        'WaveformSet.from_ROOT_file()',
                                                        f"Branch 'timestamps' not found in the given TTree"))
        if 'adcs' not in aux.keys():
            raise Exception(generate_exception_message( 4,
                                                        'WaveformSet.from_ROOT_file()',
                                                        f"Branch 'adcs' not found in the given TTree"))
        
        channels = aux['channel'].array()       # It is slightly faster (~106s vs. 114s, for a
        adcs = aux['adcs'].array()              # 809 MB input file running on lxplus9) to read
        timestamps = aux['timestamps'].array()  # branch by branch rather than going for aux.arrays()

        wvfs_no_to_load = math.ceil(fraction_to_load_*len(channels))

        waveforms = []                      # Using a list comprehension here is slightly slower than a for loop
        for i in range(wvfs_no_to_load):    # (97s vs 102s for 5% of wvfs of a 809 MB file running on lxplus9)

            endpoint, channel = WaveformSet.get_endpoint_and_channel(channels[i])

            waveforms.append(Waveform(  timestamps[i],
                                        np.array(adcs[i]),
                                        0,      # To be implemented from 
                                                # new TTree in the ROOT file
                                        endpoint,
                                        channel))      
        return cls(*waveforms)

    @staticmethod
    def get_endpoint_and_channel(input):
    
        """
        Parameters
        ----------
        input : str
            len(input) must be 5. Such input is interpreted as the
            concatenation of the endpoint int(input[0:3]) and the 
            channel int(input[3:5]).

        Returns
        ----------
        int, int
            The endpoint and the channel, in such order.
        """

        return int(str(input)[0:3]), int(str(input)[3:5])