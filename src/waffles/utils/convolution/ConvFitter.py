import numpy as np
import waffles.utils.time_align_utils as tutils

from iminuit import Minuit, cost
from iminuit.util import describe
from iminuit.util import FMin
from waffles.utils.convolution.ConvUtils import *

from scipy import interpolate

class ConvFitter:
    def __init__(self, 
                 threshold_align_template = 0.27, 
                 threshold_align_response = 0.1, 
                 error = 10,
                 dointerpolation = False, 
                 interpolation_factor = 8,
                 align_waveforms: bool=True,
                 dtime = 16,
                 convtype = 'time'
                 ):
        """ This class is used to fit a template waveform to a response waveform using a convolution model of LAr response.

        Parameters
        ----------
        threshold_align_template: float
            See `align_waveforms` parameter.
        threshold_align_response: float
            See `align_waveforms` parameter.
        error: float
            Uncertainty on the response waveform.
        dointerpolation: bool
            If True, the response and template waveforms will be interpolated to a finer time resolution.
            The new dtime is dtime/interpolation_factor.
        interpolation_factor: int
            See `dointerpolation` parameter.
        align_waveforms: bool
            If True, the response waveform will be shifted to align with the
            template waveform based on the threshold_align_template and
            threshold_align_response. The response waveform will be shifted to
            align with the template waveform at the threshold_align_template
            crossing.
        dtime: float
            The time of each tick: 16 ns by default.
        convtype: str ('fft' or 'time')
            Convolution can be done in time O(n^2) or in frequency O(n log n).
            Frequency is faster but adds weird effect in the baseline.


        Example of usage:
        ```python
        conv_fitter = ConvFitter(...)
        conv_fitter.set_template_waveform(template_waveform) # A numpy array 
        conv_fitter.set_response_waveform(response_waveform)
        conv_fitter.prepare_waveforms() # Prepares the waveforms for fitting
        conv_fitter.fit(scan=2, print_flag=True)
        ```
        """

        self.threshold_align_template = threshold_align_template
        self.threshold_align_response = threshold_align_response
        self.error = error
        self.dointerpolation = dointerpolation
        self.interpolation_factor = interpolation_factor
        self.reduce_offset = False
        self.align_waveforms = align_waveforms

        self.template:np.ndarray
        self.response:np.ndarray
        self.dtime = dtime
        self.convtype = convtype

    #################################################
    def set_template_waveform(self, wvf: np.ndarray):
        self.template = wvf.copy()

    #################################################
    def set_response_waveform(self, wvf: np.ndarray):
        self.response = wvf.copy()

    #################################################
    def prepare_waveforms(self):    

        if self.dointerpolation:                        

            self.response = self.interpolate(self.response, self.interpolation_factor)
            self.template = self.interpolate(self.template, self.interpolation_factor)

        if self.align_waveforms:
            offsettemplate = tutils.find_threshold_crossing(self.template, self.threshold_align_template)
            self.response, offset = tutils.shift_waveform_to_align_threshold(self.response, threshold=self.threshold_align_response, 
                                                                             reduce_offset=self.reduce_offset,target_index=offsettemplate)
            self.response = self.response[offset:]
            self.template = self.template[offset:]

        if self.convtype == 'fft':
            self.templatefft = getFFT(self.template)

    #################################################
    def interpolate(self, wf:np.ndarray, interpolation_factor: int):

        # Create an array of times with 16 ns tick width 
        tick_width = self.dtime
        nticks = len(wf)
        times = np.linspace(0, nticks*tick_width, nticks, endpoint=False)

        # these are the continues functions using x=times and y=adcs
        wf_inter = interpolate.interp1d(times, wf, kind='linear', fill_value="extrapolate")

        # these are the new times at which to compute the value of the function
        tick_width = self.dtime/interpolation_factor
        nticks = len(wf)*interpolation_factor
        newtimes = np.linspace(0, nticks*tick_width, nticks, endpoint=False)

        # create new waveforms using the new time values
        wf = wf_inter(newtimes)

        return wf

    #################################################
    def fit(self, scan: int = 0, print_flag: bool = False):

        # scan over offsets to minimize the chi2 between the response and the template x model
        if scan > 0:

            resp_original = self.response.copy()
            temp_original = self.template.copy()

            chi2s = []
            offsets = np.arange(0, scan)
            print ('    scanning over offsets to minimize chi2 between the response and the template x model')
            for offset in offsets:
                self.response = np.roll(resp_original, offset, axis=0)
                self.response = self.response[offset:]
                self.template = temp_original[offset:]
                self.templatefft = getFFT(self.template)
                params, chi2 = self.minimize(False)
                chi2s.append(chi2)
                if(print_flag): print(offset, params, chi2)

            # recompute the waveforms for the minimum chi2
            idxMinChi2 = np.argmin(chi2s)
            self.response = np.roll(resp_original, offsets[idxMinChi2], axis=0)
            self.response = self.response[offsets[idxMinChi2]:]
            self.template = temp_original[offsets[idxMinChi2]:]
            self.templatefft = getFFT(self.template)

        # recompute parameters for the minimum chi2
        params, chi2 = self.minimize(print_flag)
        if(print_flag): print(params, chi2)

        self.fit_results = params

    #################################################
    def lar_convolution_time(self, t, A, fp, t1, t3):
        self.lar = A*(fp*np.exp(-t/t1)/t1 + (1-fp)*np.exp(-t/t3)/t3)
        return np.convolve(self.lar,self.template,mode='full')[:len(self.lar)]

    #################################################
    def lar_convolution_freq(self, t, A, fp, t1, t3):
        self.lar = A*(fp*np.exp(-t/t1)/t1 + (1-fp)*np.exp(-t/t3)/t3)
        larfreq = getFFT(self.lar)
        res = backFFT(convolveFFT(self.templatefft, larfreq))
        return np.real(res)

    #################################################
    def minimize(self, printresult:bool):

        tick_width = self.dtime if not self.dointerpolation else self.dtime/self.interpolation_factor
        nticks = len(self.response)

        times  = np.linspace(0, tick_width*nticks, nticks,endpoint=False)
        errors = np.ones(nticks)*self.error

        if self.convtype == 'time':
            self.model = self.lar_convolution_time
        elif self.convtype == 'fft':
            self.model = self.lar_convolution_freq
        mcost = cost.LeastSquares(times, self.response, errors, self.model)
        # mcost = self.mycost

        A = 10e3
        fp = 0.3
        t1 = 35.
        t3 = 1600.

        m = Minuit(mcost,A=A,fp=fp,t1=t1,t3=t3)

        m.limits['A'] = (0,None)
        m.limits['fp'] = (0,1)
        m.limits['t1'] = (2,50)
        m.limits['t3'] = (500,2000)


        m.fixed['fp'] =True
        m.migrad()
        m.migrad()
        m.migrad()
        m.fixed['fp'] = False
        m.migrad()
        m.migrad()
        m.migrad()

        pars = describe(self.model)[1:]
        params = [m.values[p] for p in pars]

        self.m = m
        if printresult:
            print(m)

        chi2res: float = 0
        if isinstance(m.fmin, FMin):
            chi2res = m.fmin.reduced_chi2
        return params, chi2res
