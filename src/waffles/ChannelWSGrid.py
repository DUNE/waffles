import numpy as np
from typing import List, Dict, Optional

from .UniqueChannel import UniqueChannel
from .WaveformSet import WaveformSet
from .ChannelWS import ChannelWS
from .ChannelMap import ChannelMap

class ChannelWSGrid:

    """
    Stands for Channel Waveform Set Grid. This class 
    implements a set of ChannelWS which are ordered 
    according to some ChannelMap object. 

    Attributes
    ----------
    ChMap : ChannelMap
        A ChannelMap object which is used to physically 
        order the ChannelWS objects
    ChWfSets : dict of dict of ChannelWS
        A dictionary whose keys are endpoint values
        for which there is at least one ChannelWS object
        in this ChannelWSGrid object. The values of such
        dictionary are dictionaries, whose keys are
        channel values for which there is at least one
        ChannelWS object in this ChannelWSGrid object.
        The values for the deeper-level dictionaries are
        ChannelWS objects.

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    def __init__(self,  ch_map : ChannelMap,
                        input_waveformset : WaveformSet,
                        compute_calib_histo : bool = False,
                        bins_number : Optional[int] = None,
                        domain : Optional[np.ndarray] = None,
                        variable : str = 'integral',
                        analysis_label : Optional[str] = None):
                        
        """
        ChannelWSGrid class initializer. This initializer
        takes a WaveformSet object as an input, and creates
        a ChannelWSGrid object by partitioning the given
        WaveformSet object using the Endpoint and Channel
        attribute of its constituent Waveform objects.
        To do so, this initializer delegates
        the ChannelWSGrid.clusterize_WaveformSet() static
        method. After having partitioned the given
        WaveformSet object, the initializer purges the
        ChannelWS objects which come from channels which
        are not present in the given ChannelMap object.
        
        Parameters
        ----------
        ch_map : ChannelMap
            The waveforms, within input_waveformset, which
            come from unique channels (endpoint and channel)
            which do not belong to this ChannelMap will not
            be added to this ChannelWSGrid object.
        input_waveformset : WaveformSet
            The WaveformSet object which will be partitioned
            into ChannelWS objects and ordered according
            to the given ChannelMap object. This parameter
            is given to the 'waveform_set' parameter of the
            'clusterize_WaveformSet' static method.
        compute_calib_histo : bool
            If True, then the calibration histogram for each
            resulting ChannelWS object will be computed. 
            It is given to the 'compute_calib_histo' 
            parameter of the 'clusterize_WaveformSet' static
            method.
        bins_number : int
            This parameter only makes a difference if
            'compute_calib_histo' is set to True.
            If so, this parameter must be defined.
            It is given to the 'bins_number' parameter 
            of the 'clusterize_WaveformSet' static method.
            Check its docstring for more information.
        domain : np.ndarray
            This parameter only makes a difference if
            'compute_calib_histo' is set to True. It 
            is given to the 'domain' parameter of the 
            'clusterize_WaveformSet' static method. 
            Check its docstring for more information.
        variable : str
            This parameter only makes a difference if
            'compute_calib_histo' is set to True.
            It is given to the 'variable' parameter of 
            the 'clusterize_WaveformSet' static method. 
            Check its docstring for more information.
        analysis_label : str
            This parameter only makes a difference if
            'compute_calib_histo' is set to True
            and 'variable' is set to 'integral'.
            It is given to the 'analysis_label' parameter 
            of the 'clusterize_WaveformSet' static 
            method. Check its docstring for more 
            information.
        """

        ## Shall we add type checks here?

        self.__ch_map = ch_map

        self.__ch_wf_sets = ChannelWSGrid.clusterize_WaveformSet(   input_waveformset,
                                                                    channel_map = ch_map,
                                                                    compute_calib_histo = compute_calib_histo,
                                                                    bins_number = bins_number,
                                                                    domain = domain,
                                                                    variable = variable,
                                                                    analysis_label = analysis_label)
                
    #Getters
    @property
    def ChMap(self):
        return self.__ch_map
    
    @property
    def ChWfSets(self):
        return self.__ch_wf_sets
    
    @staticmethod
    def clusterize_WaveformSet( waveform_set : WaveformSet,
                                channel_map : Optional[ChannelMap] = None,
                                compute_calib_histo : bool = False,
                                bins_number : Optional[int] = None,
                                domain : Optional[np.ndarray] = None,
                                variable : str = 'integral',
                                analysis_label : Optional[str] = None) -> Dict[int, Dict[int, ChannelWS]]:

        """
        This function returns a dictionary, say output, 
        whose keys are endpoint values. The values of
        of such dictionary are dictionaries, whose keys
        are channel values. The values for the deeper-level
        dictionaries are ChannelWS objects, which are
        initialized by this static method, in a way that
        output[i][j] is the ChannelWS object which contains
        all of the Waveform objects within the given
        WaveformSet object which come from endpoint i and
        channel j.

        This method is useful to partition the given 
        WaveformSet object into WaveformSet objects 
        (actually ChannelWS objects, which inherit from 
        the WaveformSet class but require the Endpoint 
        and the Channel attribute of its constituent 
        Waveform objects to be homogeneous) which are 
        subsets of the given WaveformSet object, and 
        whose Waveform objects have homogeneous endpoint 
        and channel values.

        Parameters
        ----------
        waveform_set : WaveformSet
            The WaveformSet object which will be partitioned
            into ChannelWS objects.
        channel_map : ChannelMap
            If it is not given, then all of the waveforms
            in this WaveformSet object will be considered
            for partitioning. If it is given, then only
            the waveforms which come from channels which
            are present in this ChannelMap object will be
            considered for partitioning.
        compute_calib_histo : bool
            If True, then the calibration histogram for each
            ChannelWS object will be computed. It is given
            to the 'compute_calib_histo' parameter of the
            ChannelWS initializer. Check its docstring for
            more information.
        bins_number : int
            This parameter only makes a difference if
            'compute_calib_histo' is set to True.
            If so, this parameter must be defined.
            It is the number of bins that the calibration 
            histogram will have.
        domain : np.ndarray
            This parameter only makes a difference if
            'compute_calib_histo' is set to True. It 
            is given to the 'domain' parameter of the
            ChannelWS initializer. Check its docstring 
            for more information.
        variable : str
            This parameter only makes a difference if
            'compute_calib_histo' is set to True.
            It is given to the 'variable' parameter of 
            the ChannelWS initializer. Check its docstring 
            for more information.
        analysis_label : str
            This parameter only makes a difference if
            'compute_calib_histo' is set to True
            and 'variable' is set to 'integral'.
            It is given to the 'analysis_label' parameter 
            of the ChannelWS initializer. Check its 
            docstring for more information.

        Returns
        ----------
        output : dict of dict of ChannelWS
        """

        if channel_map is None:
            idcs = {}

            for idx in range(len(waveform_set.Waveforms)):
                try:
                    aux = idcs[waveform_set.Waveforms[idx].Endpoint]

                except KeyError:
                    idcs[waveform_set.Waveforms[idx].Endpoint] = {}
                    aux = idcs[waveform_set.Waveforms[idx].Endpoint]

                try:
                    aux[waveform_set.Waveforms[idx].Channel].append(idx)

                except KeyError:
                    aux[waveform_set.Waveforms[idx].Channel] = [idx]

        else:
            idcs = ChannelWSGrid.get_nested_dictionary_template(channel_map)    # idcs contain the endpoints and channels for 
                                                                                # which we can potentially save waveforms
            for idx in range(len(waveform_set.Waveforms)):
                try:
                    aux = idcs[waveform_set.Waveforms[idx].Endpoint]

                except KeyError:
                    continue

                try:
                    aux[waveform_set.Waveforms[idx].Channel].append(idx)

                except KeyError:
                    continue
                    
        output = {}

        for endpoint in idcs.keys():
            output[endpoint] = {}

            for channel in idcs[endpoint].keys():
                aux = [waveform_set.Waveforms[idx] for idx in idcs[endpoint][channel]]

                output[endpoint][channel] = ChannelWS(  *aux,
                                                        compute_calib_histo = compute_calib_histo,
                                                        bins_number = bins_number,
                                                        domain = domain,
                                                        variable = variable,
                                                        analysis_label = analysis_label)
        return output
    
    @staticmethod
    def get_nested_dictionary_template(channel_map : ChannelMap) -> Dict[int, Dict[int, List]]:

        """
        This method returns a dictionary which has the same
        structure as the ChWfSets attribute of ChannelWSGrid,
        but whose values are emtpy lists instead of ChannelWS 
        objects. The endpoints and channels that are considered
        for such output are those which are present in the
        input ChannelMap object.

        Parameters
        ----------
        channel_map : ChannelMap
            The ChannelMap object which contains the endpoints
            and channels which will end up in the ouput of
            this method.
    
        Returns
        ----------
        output : dict of dict of list
        """

        output = {}

        for i in range(channel_map.Rows):
            for j in range(channel_map.Columns):
                
                try:
                    aux = output[channel_map.Data[i][j].Endpoint]

                except KeyError:
                    output[channel_map.Data[i][j].Endpoint] = {}
                    aux = output[channel_map.Data[i][j].Endpoint]

                aux[channel_map.Data[i][j].Channel] = []
        
        return output

    def purge(self) -> None:    # Before 2024/06/27, this method was used in
                                # ChannelWSGrid.__init___, because the output
                                # of ChannelWSGrid.clusterize_WaveformSet()
                                # contained channels which were present in its
                                # WaveformSet input, but were not present in the
                                # self.__ch_map attribute. As of such date, 
                                # ChannelWSGrid.clusterize_WaveformSet() is 
                                # fixed and this method is not used anymore, but 
                                # it is kept here in case we need this 
                                # functionality in the future.

        """
        Removes the ChannelWS objects from self.__ch_wf_sets 
        which come from unique channels which are not present
        in self.__ch_map.
        """

        unique_channels_to_remove = {}

        for endpoint in self.__ch_wf_sets.keys():
            for channel in self.__ch_wf_sets[endpoint].keys():

                aux = UniqueChannel(endpoint, channel)
                
                if not self.__ch_map.find_channel(aux)[0]:
                    try:
                        unique_channels_to_remove[aux.Endpoint].append(aux.Channel)     # Keep note of the channel to remove, 
                    except KeyError:                                                    # but not remove it yet, since we are
                        unique_channels_to_remove[aux.Endpoint] = [aux.Channel]         # iterating over the dictionary keys

        for endpoint in unique_channels_to_remove.keys():
            for channel in unique_channels_to_remove[endpoint]:
                del self.__ch_wf_sets[endpoint][channel]

        endpoints_to_remove = []    # Second scan to remove endpoints 
                                    # which have no channels left

        for endpoint in self.__ch_wf_sets.keys():
            if len(self.__ch_wf_sets[endpoint]) == 0:
                endpoints_to_remove.append(endpoint)

        for endpoint in endpoints_to_remove:
            del self.__ch_wf_sets[endpoint]

        return