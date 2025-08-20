import numpy as np
from typing import List, Dict, Optional

from waffles.data_classes.UniqueChannel import UniqueChannel
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.ChannelWs import ChannelWs
from waffles.data_classes.ChannelMap import ChannelMap


class ChannelWsGrid:
    """Stands for Channel Waveform Set Grid. This class
    implements a set of ChannelWs which are ordered
    according to some ChannelMap object.

    Attributes
    ----------
    ch_map: ChannelMap
        A ChannelMap object which is used to physically
        order the ChannelWs objects
    ch_wf_sets: dict of dict of ChannelWs
        A dictionary whose keys are endpoint values
        for which there is at least one ChannelWs object
        in this ChannelWsGrid object. The values of such
        dictionary are dictionaries, whose keys are
        channel values for which there is a ChannelWs
        object in this ChannelWsGrid object. The values
        for the deeper-level dictionaries are ChannelWs
        objects. Note that there might be a certain
        UniqueChannel object which is present in the
        ChannelMap, but for which there is no ChannelWs
        object in this attribute (ch_wf_sets). I.e.
        appearance of a certain UniqueChannel object
        in the ChannelMap does not guarantee that there
        will be a ChannelWs object in this attribute
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
        self, 
        ch_map: ChannelMap,
        input_waveformset: WaveformSet,
        compute_calib_histo: bool = False,
        bins_number: Optional[int] = None,
        domain: Optional[np.ndarray] = None,
        variable: Optional[str] = None,
        analysis_label: Optional[str] = None
    ):
        """ChannelWsGrid class initializer. This initializer
        takes a WaveformSet object as an input, and creates
        a ChannelWsGrid object by partitioning the given
        WaveformSet object using the endpoint and channel
        attributes of the UniqueChannel objects which are
        present in the ChannelMap object given to the
        'ch_map' input parameter. To do so, this initializer
        delegates the ChannelWsGrid.clusterize_waveform_set()
        static method.

        Parameters
        ----------
        ch_map: ChannelMap
            The waveforms, within input_waveformset, which
            come from unique channels (endpoint and channel)
            which do not belong to this ChannelMap will not
            be added to this ChannelWsGrid object.
        input_waveformset: WaveformSet
            The WaveformSet object which will be partitioned
            into ChannelWs objects and ordered according
            to the given ChannelMap object. This parameter
            is given to the 'WaveformSet' parameter of the
            'clusterize_waveform_set' static method.
        compute_calib_histo: bool
            If True, then the calibration histogram for each
            resulting ChannelWs object will be computed.
            It is given to the 'compute_calib_histo'
            parameter of the 'clusterize_waveform_set' static
            method.
        bins_number: int
            This parameter only makes a difference if
            'compute_calib_histo' is set to True.
            If so, this parameter must be defined.
            It is given to the 'bins_number' parameter
            of the 'clusterize_waveform_set' static method.
            Check its docstring for more information.
        domain: np.ndarray
            This parameter only makes a difference if
            'compute_calib_histo' is set to True. It
            is given to the 'domain' parameter of the
            'clusterize_waveform_set' static method.
            Check its docstring for more information.
        variable: str
            This parameter only makes a difference if
            'compute_calib_histo' is set to True.
            If so, this parameter must be defined,
            and it is given to the 'variable' parameter of
            the 'clusterize_waveform_set' static method.
            Check its docstring for more information.
        analysis_label: str
            This parameter only makes a difference if
            'compute_calib_histo' is set to True.
            It is given to the 'analysis_label' parameter
            of the 'clusterize_waveform_set' static
            method. Check its docstring for more
            information.
        """

        # Shall we add type checks here?

        self.__ch_map = ch_map

        self.__ch_wf_sets = ChannelWsGrid.clusterize_waveform_set(
            input_waveformset,
            channel_map=ch_map,
            compute_calib_histo=compute_calib_histo,
            bins_number=bins_number,
            domain=domain,
            variable=variable,
            analysis_label=analysis_label)

        flat_chs = [ch for row in ch_map.data for ch in row]
        self.titles = getattr(ch_map, "titles", [f"{ch.endpoint},{ch.channel}" for ch in flat_chs])

    # Getters
    @property
    def ch_map(self):
        return self.__ch_map

    @property
    def ch_wf_sets(self):
        return self.__ch_wf_sets

    def get_channel_ws_by_ij_position_in_map(
        self, 
        i: int,
        j: int
    ) -> Optional[ChannelWs]:
        """This method returns the ChannelWs object whose
        endpoint (resp. channel) attribute matches the
        endpoint (resp. channel) attribute of the UniqueChannel
        object which is placed in the i-th row and j-th column
        of the self.__ch_map ChannelMap, if any. If there is
        no such ChannelWs object, then this method returns
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
        waveform_set: WaveformSet,
        channel_map: Optional[ChannelMap] = None,
        compute_calib_histo: bool = False,
        bins_number: Optional[int] = None,
        domain: Optional[np.ndarray] = None,
        variable: Optional[str] = None,
        analysis_label: Optional[str] = None
    ) -> Dict[int, Dict[int, ChannelWs]]:
        """This function returns a dictionary, say output,
        whose keys are endpoint values. The values of
        of such dictionary are dictionaries, whose keys
        are channel values. The values for the deeper-level
        dictionaries are ChannelWs objects, which are
        initialized by this static method, in a way that
        output[i][j] is the ChannelWs object which contains
        all of the Waveform objects within the given
        WaveformSet object which come from endpoint i and
        channel j.

        This method is useful to partition the given
        WaveformSet object into WaveformSet objects
        (actually ChannelWs objects, which inherit from
        the WaveformSet class but require the endpoint
        and the channel attribute of its constituent
        Waveform objects to be homogeneous) which are
        subsets of the given WaveformSet object, and
        whose Waveform objects have homogeneous endpoint
        and channel values.

        Parameters
        ----------
        waveform_set: WaveformSet
            The WaveformSet object which will be partitioned
            into ChannelWs objects.
        channel_map: ChannelMap
            If it is not given, then all of the waveforms
            in this WaveformSet object will be considered
            for partitioning. If it is given, then only
            the waveforms which come from channels which
            are present in this ChannelMap object will be
            considered for partitioning.
        compute_calib_histo: bool
            If True, then the calibration histogram for each
            ChannelWs object will be computed. It is given
            to the 'compute_calib_histo' parameter of the
            ChannelWs initializer. Check its docstring for
            more information.
        bins_number: int
            This parameter only makes a difference if
            'compute_calib_histo' is set to True.
            If so, this parameter must be defined.
            It is the number of bins that the calibration
            histogram will have.
        domain: np.ndarray
            This parameter only makes a difference if
            'compute_calib_histo' is set to True. It
            is given to the 'domain' parameter of the
            ChannelWs initializer. Check its docstring
            for more information.
        variable: str
            This parameter only makes a difference if
            'compute_calib_histo' is set to True.
            If so, this parameter must be defined,
            and it is given to the 'variable' parameter
            of the ChannelWs initializer. Check its
            docstring for more information.
        analysis_label: str
            This parameter only makes a difference if
            'compute_calib_histo' is set to True.
            It is given to the 'analysis_label' parameter
            of the ChannelWs initializer. Check its
            docstring for more information.

        Returns
        ----------
        output: dict of dict of ChannelWs
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
            idcs = ChannelWsGrid.get_nested_dictionary_template(
                channel_map)    
            
            # idcs contains the endpoints and channels for
            # which we can potentially save waveforms.
            # Contrary to the channel_map == None case,
            # in this case some of the idcs entries may
            # never be filled not even with a single Waveform.
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
                # size during iteration' error

                for channel in empty_channels[endpoint]:    
                    del idcs[endpoint][channel]

        output = {}

        for endpoint in idcs.keys():
            output[endpoint] = {}

            for channel in idcs[endpoint].keys():
                aux = [
                    waveform_set.waveforms[idx]
                    for idx in idcs[endpoint][channel]]

                output[endpoint][channel] = ChannelWs(
                    *aux,
                    compute_calib_histo=compute_calib_histo,
                    bins_number=bins_number,
                    domain=domain,
                    variable=variable,
                    analysis_label=analysis_label)
        return output

    @staticmethod
    def get_nested_dictionary_template(
        channel_map: ChannelMap
    ) -> Dict[int, Dict[int, List]]:
        """This method returns a dictionary which has the same
        structure as the ch_wf_sets attribute of ChannelWsGrid,
        but whose values are emtpy lists instead of ChannelWs
        objects. The endpoints and channels that are considered
        for such output are those which are present in the
        input ChannelMap object.

        Parameters
        ----------
        channel_map: ChannelMap
            The ChannelMap object which contains the endpoints
            and channels which will end up in the ouput of
            this method.

        Returns
        ----------
        output: dict of dict of list
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
    
    def compute_calib_histos(
        self,
        bins_number: int,
        domain: np.ndarray,
        variable: str,
        analysis_label: Optional[str] = None
    ) -> None:
        """This method iterates through all of the endpoint and
        channels in the self.__ch_wf_sets attribute and, for
        every ChannelWs object, say x, it computes its associated
        calibration histogram by calling x.compute_calib_histo().

        Parameters
        ----------
        bins_number: int
            The number of bins that the calibration histograms
            will have. It must be greater than 1.
        domain: np.ndarray
            A 2x1 numpy array where (domain[0], domain[1])
            gives the range to consider for the
            calibration histograms. Any sample which falls
            outside this range is ignored.
        variable: str
            It is eventually given to the 'variable'
            positional argument of the
            CalibrationHistogram.from_WaveformSet class
            method. For each Waveform object within
            each ChannelWs, this parameter gives the key
            for the considered WfAna object (up to the
            analysis_label input parameter) from where
            to take the sample to add to the computed
            calibration histogram. Namely, for a WfAna
            object x, x.result[variable] is the considered
            sample. It is the caller's responsibility to
            ensure that the values for the given variable
            (key) are scalars, i.e. that they are valid
            samples for a 1D histogram.
        analysis_label: str
            For each Waveform object in each ChannelWs,
            this parameter gives the key for the WfAna
            object within the analyses attribute from
            where to take the sample to add to the
            calibration histogram. If 'analysis_label'
            is None, then the last analysis added to the
            analyses attribute will be the used one. If
            there is not even one analysis, then an
            exception will be raised.

        Returns
        ----------
        None
        """
        
        for endpoint in self.__ch_wf_sets.keys():
            for channel in self.__ch_wf_sets[endpoint].keys():
                self.__ch_wf_sets[endpoint][channel].compute_calib_histo(
                    bins_number,
                    domain,
                    variable,
                    analysis_label=analysis_label
                )

        return

    # Before 2024/06/27, this method was used in 
    # ChannelWsGrid.__init___, because the output
    # of ChannelWsGrid.clusterize_waveform_set()
    # contained channels which were present in its
    # WaveformSet input, but were not present in
    # the self.__ch_map attribute. As of such date,
    # ChannelWsGrid.clusterize_waveform_set() is
    # fixed and this method is not used anymore,
    # but it is kept here in case we need this
    # functionality in the future.

    def purge(self) -> None:    
        """Removes the ChannelWs objects from self.__ch_wf_sets
        which come from unique channels which are not present
        in self.__ch_map.
        """

        UniqueChannels_to_remove = {}

        for endpoint in self.__ch_wf_sets.keys():
            for channel in self.__ch_wf_sets[endpoint].keys():

                aux = UniqueChannel(endpoint, channel)

                if not self.__ch_map.find_channel(aux)[0]:

                    # Keep note of the channel to remove,
                    # but not remove it yet, since we are
                    # iterating over the dictionary keys

                    try:
                        UniqueChannels_to_remove[aux.endpoint].append(
                            aux.channel)

                    except KeyError:
                        UniqueChannels_to_remove[aux.endpoint] = [
                            aux.channel]
                        

        for endpoint in UniqueChannels_to_remove.keys():
            for channel in UniqueChannels_to_remove[endpoint]:
                del self.__ch_wf_sets[endpoint][channel]

        # Second scan to remove endpoints
        # which have no channels left
        
        endpoints_to_remove = []

        for endpoint in self.__ch_wf_sets.keys():
            if len(self.__ch_wf_sets[endpoint]) == 0:
                endpoints_to_remove.append(endpoint)

        for endpoint in endpoints_to_remove:
            del self.__ch_wf_sets[endpoint]

        return