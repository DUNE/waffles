import numpy as np
from typing import List, Dict, Optional

from waffles.data_classes.UniqueChannel import unique_channel
from waffles.data_classes.WaveformSet import waveform_set
from waffles.data_classes.ChannelWS import channel_ws
from waffles.data_classes.ChannelMap import channel_map


class channel_ws_grid:

    """
    Stands for channel Waveform Set Grid. This class
    implements a set of channel_ws which are ordered
    according to some channel_map object.

    Attributes
    ----------
    ch_map : channel_map
        A channel_map object which is used to physically
        order the channel_ws objects
    ch_wf_sets : dict of dict of channel_ws
        A dictionary whose keys are endpoint values
        for which there is at least one channel_ws object
        in this channel_ws_grid object. The values of such
        dictionary are dictionaries, whose keys are
        channel values for which there is a channel_ws
        object in this channel_ws_grid object. The values
        for the deeper-level dictionaries are channel_ws
        objects. Note that there might be a certain
        UniqueChannel object which is present in the
        channel_map, but for which there is no channel_ws
        object in this attribute (ch_wf_ets). I.e.
        appearance of a certain UniqueChannel object
        in the channel_map does not guarantee that there
        will be a channel_ws object in this attribute
        which comes from such unique channel. Hence,
        one should always handle a KeyError exceptions
        when trying to subscribe ch_wf_sets with the endpoint
        and channel coming from an UniqueChannel object
        within the ch_map attribute.

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    def __init__(
            self, ch_map: channel_map,
            input_waveformset: waveform_set,
            compute_calib_histo: bool = False,
            bins_number: Optional[int] = None,
            domain: Optional[np.ndarray] = None,
            variable: Optional[str] = None,
            analysis_label: Optional[str] = None):
        """
        channel_ws_grid class initializer. This initializer
        takes a waveform_set object as an input, and creates
        a channel_ws_grid object by partitioning the given
        waveform_set object using the endpoint and channel
        attributes of the UniqueChannel objects which are
        present in the channel_map object given to the
        'ch_map' input parameter. To do so, this initializer
        delegates the channel_ws_grid.clusterize_WaveformSet()
        static method.

        Parameters
        ----------
        ch_map : channel_map
            The waveforms, within input_waveformset, which
            come from unique channels (endpoint and channel)
            which do not belong to this channel_map will not
            be added to this channel_ws_grid object.
        input_waveformset : waveform_set
            The waveform_set object which will be partitioned
            into channel_ws objects and ordered according
            to the given channel_map object. This parameter
            is given to the 'waveform_set' parameter of the
            'clusterize_WaveformSet' static method.
        compute_calib_histo : bool
            If True, then the calibration histogram for each
            resulting channel_ws object will be computed.
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
            If so, this parameter must be defined,
            and it is given to the 'variable' parameter of
            the 'clusterize_WaveformSet' static method.
            Check its docstring for more information.
        analysis_label : str
            This parameter only makes a difference if
            'compute_calib_histo' is set to True.
            It is given to the 'analysis_label' parameter
            of the 'clusterize_WaveformSet' static
            method. Check its docstring for more
            information.
        """

        # Shall we add type checks here?

        self.__ch_map = ch_map

        self.__ch_wf_sets = channel_ws_grid.clusterize_waveform_set(
            input_waveformset,
            channel_map=ch_map,
            compute_calib_histo=compute_calib_histo,
            bins_number=bins_number,
            domain=domain,
            variable=variable,
            analysis_label=analysis_label)

    # Getters
    @property
    def ch_map(self):
        return self.__ch_map

    @property
    def ch_wf_sets(self):
        return self.__ch_wf_sets

    def get_channel_ws_by_ij_position_in_map(
            self, i: int,
            j: int) -> Optional[channel_ws]:
        """
        This method returns the channel_ws object whose
        endpoint (resp. channel) attribute matches the
        endpoint (resp. channel) attribute of the unique_channel
        object which is placed the i-th row and j-th column
        of the self.__ch_map channel_map, if any. If there is
        no such channel_ws object, then this method returns
        None.
        """

        try:
            output = self.__ch_wf_sets[
                self.__ch_map.data[i][j].endpoint][
                    self.__ch_map.data[i][j].channel]
        except KeyError:
            output = None

        return output

    @staticmethod
    def clusterize_waveform_set(
            waveform_set: waveform_set,
            channel_map: Optional[channel_map] = None,
            compute_calib_histo: bool = False,
            bins_number: Optional[int] = None,
            domain: Optional[np.ndarray] = None,
            variable: Optional[str] = None,
            analysis_label: Optional[str] = None
    ) -> Dict[int, Dict[int, channel_ws]]:
        """
        This function returns a dictionary, say output,
        whose keys are endpoint values. The values of
        of such dictionary are dictionaries, whose keys
        are channel values. The values for the deeper-level
        dictionaries are channel_ws objects, which are
        initialized by this static method, in a way that
        output[i][j] is the channel_ws object which contains
        all of the Waveform objects within the given
        waveform_set object which come from endpoint i and
        channel j.

        This method is useful to partition the given
        waveform_set object into waveform_set objects
        (actually channel_ws objects, which inherit from
        the waveform_set class but require the endpoint
        and the channel attribute of its constituent
        Waveform objects to be homogeneous) which are
        subsets of the given waveform_set object, and
        whose Waveform objects have homogeneous endpoint
        and channel values.

        Parameters
        ----------
        waveform_set : waveform_set
            The waveform_set object which will be partitioned
            into channel_ws objects.
        channel_map : channel_map
            If it is not given, then all of the waveforms
            in this waveform_set object will be considered
            for partitioning. If it is given, then only
            the waveforms which come from channels which
            are present in this channel_map object will be
            considered for partitioning.
        compute_calib_histo : bool
            If True, then the calibration histogram for each
            channel_ws object will be computed. It is given
            to the 'compute_calib_histo' parameter of the
            channel_ws initializer. Check its docstring for
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
            channel_ws initializer. Check its docstring
            for more information.
        variable : str
            This parameter only makes a difference if
            'compute_calib_histo' is set to True.
            If so, this parameter must be defined,
            and it is given to the 'variable' parameter
            of the channel_ws initializer. Check its
            docstring for more information.
        analysis_label : str
            This parameter only makes a difference if
            'compute_calib_histo' is set to True.
            It is given to the 'analysis_label' parameter
            of the channel_ws initializer. Check its
            docstring for more information.

        Returns
        ----------
        output : dict of dict of channel_ws
        """

        if channel_map is None:
            idcs = {}

            for idx in range(len(waveform_set.waveforms)):
                try:
                    aux = idcs[waveform_set.waveforms[idx].endpoint]

                except KeyError:
                    idcs[waveform_set.waveforms[idx].endpoint] = {}
                    aux = idcs[waveform_set.waveforms[idx].endpoint]

                try:
                    aux[waveform_set.waveforms[idx].channel].append(idx)

                except KeyError:
                    aux[waveform_set.waveforms[idx].channel] = [idx]

        else:
            idcs = channel_ws_grid.get_nested_dictionary_template(
                channel_map)    # idcs contains the endpoints and channels for
            # which we can potentially save waveforms.
            # Contrary to the channel_map == None case,
            # in this case some of the idcs entries may
            # never be filled not even with a single waveform.
            # We will need to remove those after.
            for idx in range(len(waveform_set.waveforms)):
                try:
                    aux = idcs[waveform_set.waveforms[idx].endpoint]

                except KeyError:
                    continue

                try:
                    aux[waveform_set.waveforms[idx].channel].append(idx)

                except KeyError:
                    continue

            # Now let's remove the channels that are empty.
            empty_channels = {}
            for endpoint in idcs.keys():
                # To do so, find those first.
                for channel in idcs[endpoint].keys():
                    if len(idcs[endpoint][channel]) == 0:
                        try:
                            empty_channels[endpoint].append(channel)
                        except KeyError:
                            empty_channels[endpoint] = [channel]

            for endpoint in empty_channels.keys():
                # Then remove them. This process is staged to
                # prevent a 'RuntimeError: dictionary changed
                for channel in empty_channels[endpoint]:
                    # size during iteration' error
                    del idcs[endpoint][channel]

        output = {}

        for endpoint in idcs.keys():
            output[endpoint] = {}

            for channel in idcs[endpoint].keys():
                aux = [
                    waveform_set.waveforms[idx]
                    for idx in idcs[endpoint][channel]]

                output[endpoint][channel] = channel_ws(
                    *aux,
                    compute_calib_histo=compute_calib_histo,
                    bins_number=bins_number,
                    domain=domain,
                    variable=variable,
                    analysis_label=analysis_label)
        return output

    @staticmethod
    def get_nested_dictionary_template(
            channel_map: channel_map) -> Dict[int, Dict[int, List]]:
        """
        This method returns a dictionary which has the same
        structure as the ch_wf_sets attribute of channel_ws_grid,
        but whose values are emtpy lists instead of channel_ws
        objects. The endpoints and channels that are considered
        for such output are those which are present in the
        input channel_map object.

        Parameters
        ----------
        channel_map : channel_map
            The channel_map object which contains the endpoints
            and channels which will end up in the ouput of
            this method.

        Returns
        ----------
        output : dict of dict of list
        """

        output = {}

        for i in range(channel_map.rows):
            for j in range(channel_map.columns):

                try:
                    aux = output[channel_map.data[i][j].endpoint]

                except KeyError:
                    output[channel_map.data[i][j].endpoint] = {}
                    aux = output[channel_map.data[i][j].endpoint]

                aux[channel_map.data[i][j].channel] = []

        return output

    def purge(self) -> None:    # Before 2024/06/27, this method was used in
        # channel_ws_grid.__init___, because the output
        # of channel_ws_grid.clusterize_WaveformSet()
        # contained channels which were present in its
        # waveform_set input, but were not present in the
        # self.__ch_map attribute. As of such date,
        # channel_ws_grid.clusterize_WaveformSet() is
        # fixed and this method is not used anymore, but
        # it is kept here in case we need this
        # functionality in the future.
        """
        Removes the channel_ws objects from self.__ch_wf_sets
        which come from unique channels which are not present
        in self.__ch_map.
        """

        unique_channels_to_remove = {}

        for endpoint in self.__ch_wf_sets.keys():
            for channel in self.__ch_wf_sets[endpoint].keys():

                aux = unique_channel(endpoint, channel)

                if not self.__ch_map.find_channel(aux)[0]:
                    try:
                        unique_channels_to_remove[aux.endpoint].append(
                            aux.channel)
                        # Keep note of the channel to remove,
                    except KeyError:
                        # but not remove it yet, since we are
                        unique_channels_to_remove[aux.endpoint] = [
                            aux.channel]
                        # iterating over the dictionary keys

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
