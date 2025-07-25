import numpy as np
from typing import Optional
from waffles.data_classes.WaveformAdcs import WaveformAdcs
from enum import IntEnum

# Fallback enum, in case the real TriggerCandidateData.Type isn't usable directly
class TriggerType(IntEnum):
    kUnknown = 0
    kTiming = 1
    kTPCLowE = 2
    kSupernova = 3
    kRandom = 4
    kPrescale = 5
    kADCSimpleWindow = 6
    kHorizontalMuon = 7
    kMichelElectron = 8
    kPlaneCoincidence = 9
    kDBSCAN = 10
    kChannelDistance = 11
    kBundle = 12
    kCTBFakeTrigger = 13
    kCTBBeam = 14
    kCTBBeamChkvHL = 15
    kCTBCustomD = 16
    kCTBCustomE = 17
    kCTBCustomF = 18
    kCTBCustomG = 19
    kCTBBeamChkvHLx = 20
    kCTBBeamChkvHxL = 21
    kCTBBeamChkvHxLx = 22
    kNeutronSourceCalib = 23
    kChannelAdjacency = 24
    kCIBFakeTrigger = 25
    kCIBLaserTriggerP1 = 26
    kCIBLaserTriggerP2 = 27
    kCIBLaserTriggerP3 = 28
    kCTBOffSpillSnapshot = 29
    kCTBOffSpillCosmicJura = 30
    kCTBOffSpillCRTCosmic = 31
    kCTBCustomA = 32
    kCTBCustomB = 33
    kCTBCustomC = 34
    kCTBCustomPulseTrain = 35
    kDTSPulser = 36
    kDTSCosmic = 37
    kSSPLEDCalibration = 38

class Waveform(WaveformAdcs):

    """
    This class implements a Waveform which includes
    information which is relative to detector readout.
    It inherits from the WaveformAdcs class.

    Attributes
    ----------
    timestamp: int
        The timestamp value for this Waveform
    time_step_ns: float (inherited from WaveformAdcs)
    daq_window_timestamp: int
        The timestamp value for the DAQ window in which
        this Waveform was acquired
    adcs: unidimensional numpy array of integers
    (inherited from WaveformAdcs)
    run_number: int
        Number of the run from which this Waveform was
        acquired
    record_number: int
        Number of the record within which this Waveform
        was acquired
    endpoint: int
        Endpoint number from which this Waveform was
        acquired
    channel: int
        Channel number for this Waveform
    time_offset: int (inherited from WaveformAdcs)
    starting_tick: int
        The iterator value (zero-indexed) for the
        first point of this Waveform, with respect
        to the full acquired waveform. P.e. if this
        Waveform is the result of eliminating the
        first two points of a certain waveform,
        then this attribute equals 2. If no initial
        points have been truncated, then this
        attribute equals 0.
    analyses: OrderedDict of WfAna objects
    (inherited from WaveformAdcs)

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    def __init__(
        self,
        timestamp: int,
        time_step_ns: float,
        daq_window_timestamp: int,
        adcs: np.ndarray,
        run_number: int,
        record_number: int,
        endpoint: int,
        channel: int,
        time_offset: int = 0,
        starting_tick: int = 0,
        trigger_type: Optional[int] = None):
        """
        Waveform class initializer

        Parameters
        ----------
        timestamp: int
        time_step_ns: float
            It is given to the 'time_step_ns' parameter of
            the base class initializer.
        daq_window_timestamp: int
        adcs: unidimensional numpy array of integers
            It is given to the 'adcs' parameter of the base
            class initializer.
        run_number: int
        record_number: int
        endpoint: int
        channel: int
        time_offset: int
            It is given to the 'time_offset' parameter of the
            base class initializer. It must be semipositive
            and smaller than len(self.__adcs)-1. Its default
            value is 0.
        starting_tick: int
        """

        # Shall we add add type checks here?

        self.__timestamp = timestamp
        self.__daq_window_timestamp = daq_window_timestamp
        self.__run_number = run_number
        self.__record_number = record_number
        self.__endpoint = endpoint
        self.__channel = channel
        self.__starting_tick = starting_tick
        self.__trigger_type = trigger_type

        # Do we need to add trigger primitives as attributes?

        super().__init__(
            time_step_ns,
            adcs,
            time_offset=time_offset)

    # Getters
    @property
    def timestamp(self):
        return self.__timestamp

    @property
    def daq_window_timestamp(self):
        return self.__daq_window_timestamp

    @property
    def run_number(self):
        return self.__run_number

    @property
    def record_number(self):
        return self.__record_number

    @property
    def endpoint(self):
        return self.__endpoint

    @property
    def channel(self):
        return self.__channel

    @property
    def starting_tick(self):
        return self.__starting_tick
    @property
    def trigger_type(self):
        return getattr(self, "_Waveform__trigger_type", None)

    @property
    def trigger_type_bits(self):
        return [
            TriggerType(i)
            for i in range(64)
            if self.trigger_type is not None and (self.trigger_type & (1 << i)) != 0 and i in TriggerType.__members__.values()
        ]

    @property
    def trigger_type_names(self):
        return [t.name for t in self.trigger_type_bits]

    # For the moment there are no setters for
    # the attributes of Waveform. I.e. you can
    # only set the value of its attributes
    # through Waveform.__init__. Here's an example
    # of what a setter would look like, though.

    # Overrides WaveformAdcs.__slice_adcs()
    def __slice_adcs(
        self,
        start: int,
        end: int
    ) -> None:
        """This method is not intended for user usage.
        No well-formedness checks are performed here.
        This method slices the self.__adcs attribute
        array to self.__adcs[start:end] and sets
        the self.__starting_tick attribute to start.
        This method applies the change in place.

        Parameters
        ----------
        start: int
            Iterator value for the (inclusive) start of
            the slice
        end: int
            Iterator value for the (exclusive) end of
            the slice

        Returns
        ----------
        None
        """

        self.__starting_tick = start
        super()._WaveformAdcs__slice_adcs(start, end)
        return

    def get_global_channel(self):
        """
        Returns
        ----------
        int
        An integer value for the readout channel with respect to a numbering
        scheme which identifies the endpoint and the APA channel at the same
        time
        """

        pass

    def __repr__(self):
        return (f"Waveform: \n"
            f"run_number: {self.__run_number}, \n"
            f"endpoint: {self.__endpoint}, \n"
            f"channel: {self.__channel}, \n"
            f"record_number: {self.__record_number}, \n"
            f"time_step_ns: {self.time_step_ns}, \n"
            f"timestamp: {self.__timestamp} [ticks], \n"
            f"daq_window_timestamp: {self.__daq_window_timestamp} [ticks], \n"
            f"starting_tick: {self.__starting_tick}, \n"
            f"time_offset: {self.time_offset}\n"
                )

