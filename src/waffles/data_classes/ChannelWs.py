import numpy as np
from typing import Optional
from waffles.data_classes.CalibrationHistogram import CalibrationHistogram
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.Exceptions import GenerateExceptionMessage


class ChannelWs(WaveformSet):
    """Stands for Channel Waveform Set. This class inherits
    from the WaveformSet class. It implements a set of
    Waveform objects for which its endpoint attribute is
    the same accross the whole set, and their channel
    attribute is also homogeneous.

    Attributes
    ----------
    waveforms: list of Waveform objects (inherited from WaveformSet)
    points_per_wf: int (inherited from WaveformSet)
    runs: set of int (inherited from WaveformSet)
    record_numbers: dictionary of sets (inherited from WaveformSet)
    available_channels: dictionary of dictionaries of sets
                        (inherited from WaveformSet)
    mean_adcs: WaveformAdcs (inherited from WaveformSet)
    mean_adcs_idcs: tuple of int (inherited from WaveformSet)
    endpoint: int
        Endpoint number for this set of waveforms
    channel: int
        Channel number for this set of waveforms
    calib_histo: CalibrationHistogram
        A calibration histogram for this set of waveforms.
        It is not computed by default. I.e. if
        self.calib_histo equals to None, it should be
        interpreted as unavailable data.

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    def __init__(
        self, 
        *waveforms,
        compute_calib_histo: bool = False,
        bins_number: Optional[int] = None,
        domain: Optional[np.ndarray] = None,
        variable: Optional[str] = None,
        analysis_label: Optional[str] = None
    ):
        """ChannelWs class initializer

        Parameters
        ----------
        waveforms: unpacked list of Waveform objects
            The waveforms that will be added to the set.
            Their endpoint and channel attributes must be
            homogeneous. Otherwise, an exception will be
            raised.
        compute_calib_histo: bool
            If True, then the calibration histogram for
            this ChannelWs object will be computed, up to
            the input given to the 'variable' parameter.
            If False, then the calibration histogram for
            this ChannelWs object will be set to None.
        bins_number: int
            This parameter only makes a difference if
            'compute_calib_histo' is set to True.
            If so, this parameter must be defined.
            In that case, it gives the number of bins
            that the calibration histogram will have.
            It must be greater than 1.
        domain: np.ndarray
            This parameter only makes a difference if
            'compute_calib_histo' is set to True.
            If so, this parameter must be defined.
            A 2x1 numpy array where (domain[0], domain[1])
            gives the range to consider for the
            calibration histogram. Any sample which falls
            outside this range is ignored.
        variable: str
            This parameter only makes a difference if
            'compute_calib_histo' is set to True.
            If so, this parameter must be defined,
            and it is eventually given to the 'variable'
            positional argument of the
            CalibrationHistogram.from_WaveformSet class
            method. For each Waveform object within
            this ChannelWs, this parameter gives the key
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
            This parameter only makes a difference if
            'compute_calib_histo' is set to True. For
            each Waveform object in this ChannelWs,
            this parameter gives the key for the WfAna
            object within the analyses attribute from
            where to take the sample to add to the
            calibration histogram. If 'analysis_label'
            is None, then the last analysis added to the
            analyses attribute will be the used one. If
            there is not even one analysis, then an
            exception will be raised.
        """

        # Shall we add type checks here?

        super().__init__(*waveforms)

        self.__endpoint = None
        self.__channel = None
        self.update_endpoint_and_channel()

        self.__calib_histo = None

        if compute_calib_histo:

            if bins_number is None:
                raise Exception(GenerateExceptionMessage(
                    1,
                    'ChannelWs.__init__()',
                    'The bins number must be provided if the'
                    ' calibration histogram is to be computed.'))
            if domain is None:
                raise Exception(GenerateExceptionMessage(
                    2,
                    'ChannelWs.__init__()',
                    'The domain must be provided if the '
                    'calibration histogram is to be computed.'))

            self.compute_calib_histo(
                bins_number,
                domain,
                variable,
                analysis_label=analysis_label
            )

    # Getters
    @property
    def endpoint(self):
        return self.__endpoint

    @property
    def channel(self):
        return self.__channel

    @property
    def calib_histo(self):
        return self.__calib_histo

    def update_endpoint_and_channel(self) -> None:
        """This method checks the information returned by
        self.get_run_collapsed_available_channels(), to ensure 
        that the endpoint and the channel attributes of every 
        Waveform object within this set is homogeneous. If it 
        is not, then it raises an exception. If they are, then 
        the endpoint and channel attributes of this ChannelWs 
        object are updated accordingly.

        Returns
        ----------
        None
        """

        aux = self.get_run_collapsed_available_channels()

        if (len(aux) != 1):
            raise Exception(GenerateExceptionMessage(
                1,
                'ChannelWs.update_endpoint_and_channel()',
                'Every Waveform object within this set must'
                ' have the same endpoint attribute.'))
        else:
            endpoint = next(iter(aux.keys()))
            if len(aux[endpoint]) != 1:
                raise Exception(GenerateExceptionMessage(
                    2,
                    'ChannelWs.update_endpoint_and_channel()',
                    'Every Waveform object within this set must'
                    ' have the same channel attribute.'))
            else:
                channel = next(iter(aux[endpoint]))

        self.__endpoint = endpoint
        self.__channel = channel

        return
    
    def compute_calib_histo(
        self,
        bins_number: int,
        domain: np.ndarray,
        variable: str,
        analysis_label: Optional[str] = None
    ) -> None:
        """This method computes the calibration histogram for
        this ChannelWs object.
        
        Parameters
        ----------
        bins_number: int
            The number of bins that the calibration histogram
            will have. It must be greater than 1.
        domain: np.ndarray
            A 2x1 numpy array where (domain[0], domain[1])
            gives the range to consider for the
            calibration histogram. Any sample which falls
            outside this range is ignored.
        variable: str
            It is given to the 'variable' positional
            argument of the CalibrationHistogram.from_WaveformSet
            class method. For each Waveform object within
            this ChannelWs, this parameter gives the key
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
            For each Waveform object in this ChannelWs,
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

        # One could have the code below called directly from the
        # __init__() method. However, it is convenient to have this
        # wrapper method (and optionally call it from the __init__()
        # method) for cases when one needs to recompute the calibration
        # histogram, or when the calibration histogram cannot be
        # computed at the moment of the ChannelWs object creation.

        self.__calib_histo = CalibrationHistogram.from_WaveformSet(
            self,
            bins_number,
            domain,
            variable,
            analysis_label=analysis_label)
        
        return