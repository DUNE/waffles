# Analysis class to search for peaks and beam event timestamp 

from waffles.data_classes.WaveformAdcs import WaveformAdcs
from waffles.data_classes.IPDict import IPDict
from waffles.data_classes.WfAna import WfAna
from waffles.data_classes.WfAnaResult import WfAnaResult
from waffles.Exceptions import GenerateExceptionMessage


import waffles.Exceptions as we
import numpy as np
from scipy.signal import find_peaks


class MyAnaPeak_NEW(WfAna):
    def __init__(self, input_parameters: IPDict):
        """BasicWfAna class initializer. 

        Parameters
        ----------
        input_parameters: IPDict
            This IPDict must contain the following keys:
                - baseline_limits (list of int) -> limits to compute the baseline as mean [start, stop]
                - n_std (float) -> number of standard deviazione wrt baseline to use to search for peaks
                - peak_distance (int) -> minimun distance (in timeticks) between two peaks
                - beam_timeoffset_limits (list of int) -> limits to search for beam events, looking at timeoffset (peak timetick + start waveforms - daq timestamp), usually (-40,-20)
                - signal_sign (str) -> "negative" or "positive", depending on the signal polarity
        """

        self.__baseline_limits = input_parameters['baseline_limits']
        self.__n_std = input_parameters['n_std']
        self.__peak_distance = input_parameters['peak_distance']
        self.__beam_timeoffset_limits = input_parameters['beam_timeoffset_limits']
        self.__signal_sign = input_parameters.get('signal_sign').lower()


        super().__init__(input_parameters)

    @property
    def baseline_limits(self):
        return self.__baseline_limits

    @property
    def n_std(self):
        return self.__n_std

    @property
    def peak_distance(self):
        return self.__peak_distance
    
    @property
    def beam_timeoffset_limits(self):
        return self.__beam_timeoffset_limits
    
    @property
    def signal_sign(self):
        return self.__signal_sign


    def analyse(self, waveform: WaveformAdcs) -> None:
        """With respect to the given WaveformAdcs object, this 
        analyser method does the following:
          - compute the baseline as simple mean
          - remove the baseline from the signal, and invert it 
          - search for peaks (index, peak_time, peak_absolute_time, peak_amplitude)
          - compute the time_offset (PDS - DAQ) as peak_absolute_time - daq_window_timestamp
          - select beam peaks 

        Parameters
        ----------
        waveform: WaveformAdcs
            The WaveformAdcs object which will be analysed

        Returns
        ----------
        None
        """
    
        # Baseline of first 300 points + removing it and inverting (find_peaks search for positive ones)
        mean_baseline = np.mean(waveform.adcs[self.__baseline_limits[0] : self.__baseline_limits[1]])
        if self.__signal_sign == "negative":
            clean = mean_baseline - waveform.adcs
        elif self.__signal_sign == "positive":
            clean = waveform.adcs - mean_baseline
        else:
            raise we.WafflesException(GenerateExceptionMessage(
                f"MyAnaPeak_NEW:analyse: invalid signal_sign '{self.__signal_sign}'. It must be 'negative' or 'positive'."
            ))
        std = np.std(clean[self.__baseline_limits[0] : self.__baseline_limits[1]])

        # Peak finding 
        peak_index, properties = find_peaks(
            clean,
            height = self.__n_std * std,        
            prominence = self.__n_std * std,
            distance = self.__peak_distance)
        
        if self.__signal_sign == "negative":
            peak_amplitude = mean_baseline-properties['peak_heights']
        else:  # positive
            peak_amplitude = properties['peak_heights']-mean_baseline
        
        x = np.arange(len(waveform.adcs)) 
        peak_time = x[peak_index]
        peak_absolute_time = waveform.timestamp + waveform.time_offset + peak_time 
        time_offset = peak_absolute_time - waveform.daq_window_timestamp

        # Beam selection
        mask = (time_offset >= self.__beam_timeoffset_limits[0]) & (time_offset <= self.__beam_timeoffset_limits[1])
        beam_peak_index = peak_index[mask]
        beam_peak_time = peak_time[mask]
        beam_peak_amplitude = peak_amplitude[mask]
        beam_peak_absolute_time = peak_absolute_time[mask]
        
        # Results 
        self._WfAna__result = WfAnaResult(
            mean_baseline = mean_baseline,
            peak_index=peak_index,
            peak_time = peak_time,
            peak_amplitude = peak_amplitude,
            peak_absolute_time= peak_absolute_time,
            beam_peak_index =beam_peak_index,
            beam_peak_time = beam_peak_time,
            beam_peak_amplitude = beam_peak_amplitude,
            beam_peak_absolute_time = beam_peak_absolute_time,
            time_offset = time_offset
        )
        return
    

    @staticmethod
    @we.handle_missing_data
    def check_input_parameters(
            input_parameters: IPDict,
            points_no: int
    ) -> None:

        # No checks for now
        return