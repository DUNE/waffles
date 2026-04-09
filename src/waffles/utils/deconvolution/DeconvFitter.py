import numpy as np
from scipy.special import erfc

from iminuit import Minuit, cost
from iminuit.util import describe
from iminuit.util import FMin
from waffles.utils.fft.fftutils import FFTWaffles
from waffles.utils.time_align_utils import find_threshold_crossing
from waffles.np02_utils.LArXeFitUtils import FitResults, FitParameter, FitInitParams


class DeconvFitter(FFTWaffles):
    def __init__(self, 
                 scinttype:str = 'lar',
                 error:float = 0.1,
                 filter_type:str = 'Gauss',
                 cutoff_MHz:float = 10,
                 dtime:int = 16,
                 ):
        """ This class is used to deconvolve a template waveform from a response waveform using a and fit LAr response.
        """


        self.template:np.ndarray
        self.response:np.ndarray
        self.error = error
        self.dtime = dtime
        self.scinttype = scinttype
        self.chi2 = -1
        self.m: Minuit
        self.parameters_fit = FitResults()
        super().__init__(filter_type=filter_type, cutoff_MHz=cutoff_MHz)

    ##################################################
    def set_template_waveform(self, wvf: np.ndarray):
        self.template = wvf.copy()

    ##################################################
    def set_response_waveform(self, wvf: np.ndarray):
        self.response = wvf.copy()

    ##################################################
    def getFFTs(self):
        self.templatefft, self.fft_len = self.getFFTFull(self.template)
        self.responsefft, self.fft_len = self.getFFTFull(self.response)

    ##################################################
    # MATHEMATICAL MODELS 
    ##################################################
    def expo_conv_gauss(self, x, tau, sigma, t0):
        log_term =  -(x - t0) / tau + sigma**2 / (2 * tau**2) # avoids overflow for large x, tau and sigma
        term_erfc = (sigma / tau - (x - t0) / sigma) / np.sqrt(2)
        return np.exp(np.clip(log_term, -500, 500)) * erfc(term_erfc) / 2.0


    def model_lar(self, x, A, fp, t1, t3, sigma, t0):

        
        term_fast = (fp / t1) * self.expo_conv_gauss(x, t1, sigma, t0)
                    
        term_slow = ((1-fp) / t3) * self.expo_conv_gauss(x, t3, sigma, t0)
                    
        return A * (term_fast + term_slow)

    def model_larxe_reparam(self, t, A, fp, t1, t3, td, fs_frac, sigma, t0):
        fs = fs_frac * (1 - fp)   # guarantees fp + fs < 1 always
        return self.model_larxe(t, A, fp, fs, t1, t3, td, sigma, t0)

    def model_larxe(self, x, A, fp, fs, t1, t3, td, sigma, t0):

        
        term_fast = self.expo_conv_gauss(x, t1, sigma, t0)
        term_slow = self.expo_conv_gauss(x, t3, sigma, t0)
        term_inter = self.expo_conv_gauss(x, td, sigma, t0)

        # return A * ( (fp / t1) * term_fast + ( fs / t3 ) * term_slow - ((1-fp-fs) / td) * term_inter )
        if t3 != td:
            return A * ( (fp / t1) * term_fast + ( fs / t3 ) * term_slow + ((1-fp-fs) / (t3-td)) * (term_slow - term_inter) )
        else:
            return A * ( (fp / t1) * term_fast + (fs / t3) * term_slow )

    def generate_deconvolved_signal(self, response: np.ndarray = np.array([]), template: np.ndarray = np.array([])):
        if response.size > 0:
            self.set_response_waveform(response)
        if template.size > 0:
            self.set_template_waveform(template)

        deconv_fft = self.deconvolve(self.response, self.template)
        self.deconvolved = self.backFFT(deconv_fft)[:len(self.response)]

        cross_template = find_threshold_crossing(self.template, 0.5)
        cross_response = find_threshold_crossing(self.response, 0.5) - 10
        self.shift = int(cross_template)
        self.deconvolved = np.roll(self.deconvolved, self.shift)
        self.deconvolved = self.deconvolved - np.mean(self.deconvolved[:int(cross_response)])


    ##################################################
    # FIT
    ##################################################
    def fit(self, oneexp: bool = False, print_flag: bool = False, **kwargs):
        """
        Fit deconvolved signal
        """

        params, chi2 = self.minimize(self.deconvolved, oneexp, printresult=print_flag, **kwargs)
        
        self.fit_results = params
        self.chi2 = chi2

        if print_flag:
            print(f"Final Fit Params: {params} | Chi2: {chi2}")

        fp_fit = self.m.values['fp']

        for p in describe(self.model)[1:]:
            param_name = 'fs' if p == 'fs_frac' else p
            value = self.m.values['fs_frac'] * (1 - fp_fit) if p == 'fs_frac' else self.m.values[p]
            error = self.m.errors['fs_frac'] * (1 - self.m.values['fp']) if p == 'fs_frac' else self.m.errors[p]
            self.parameters_fit[param_name] = FitParameter(value=value, error=error)

        return params, chi2

    ##################################################
    # MINUIT IMPLEMENTATION
    ##################################################
    def minimize(self,
                 signal_to_fit: np.ndarray,
                 oneexp: bool,
                 printresult: bool,
                 fit_limits_ns: list = [None, 8.e3],
                 force_range: bool = False,
                 tolerance: float = 2e-2,
                 init: FitInitParams = FitInitParams()
                 ):


        nticks = len(signal_to_fit)
        times = np.linspace(0, self.dtime * nticks, nticks, endpoint=False)
        errors = np.ones(nticks) * self.error

        # t0 dynamically initialized at the maximum peak
        maxBin = np.argmax(signal_to_fit)
        max_slow = signal_to_fit[maxBin+15] # Max plus 15 ticks (240 ns)
        t0_init = float(maxBin * self.dtime)

        xlim_min, xlim_max = fit_limits_ns
        if xlim_min is not None:
            xlim_min = xlim_min // self.dtime
        if xlim_max is not None:
            xlim_max = int(xlim_max // self.dtime)
            if not force_range:
                lim_attempt = np.argwhere(signal_to_fit[maxBin:xlim_max]/max_slow < tolerance)
                if len(lim_attempt) > 0:
                    xlim_max = lim_attempt[0][0] + maxBin

        slice_lim = slice(xlim_min, xlim_max)
        times = times[slice_lim]
        self.times = times
        signal_to_fit = signal_to_fit[slice_lim]
        errors = errors[slice_lim]

        if not init.initialized:
            init = FitInitParams.for_lar() if self.scinttype == 'lar' else FitInitParams.for_larxe()
            if oneexp:
                init = FitInitParams.for_lar_oneexp()

        if self.scinttype == 'lar':
            self.model = self.model_lar
            mcost = cost.LeastSquares(times, signal_to_fit, errors, self.model)
            
            A = init.A
            fp = init.fp
            t1 = init.t1
            t3 = init.t3
            sigma = init.sigma

            m = Minuit(mcost,A=A,fp=fp,t1=t1,t3=t3,sigma=sigma,t0=t0_init)
            
            m.limits['A'] = init.A_limits
            m.limits['fp'] = init.fp_limits
            m.limits['t1'] = init.t1_limits
            m.limits['t3'] = init.t3_limits
            m.limits['sigma'] = init.sigma_limits
            m.limits['t0'] = (t0_init-100, nticks * self.dtime)

            m.fixed['fp'] = True
            m.migrad()
            m.migrad()
            m.migrad()
            m.fixed['fp'] = False
            m.migrad()
            m.migrad()
            m.migrad()

        else: # Xenon + Argon
            self.model = self.model_larxe
            self.model = self.model_larxe_reparam
            mcost = cost.LeastSquares(times, signal_to_fit, errors, self.model)

            A = init.A
            fp = init.fp
            fs_frac_init = init.fs_frac
            t1 = init.t1
            t3 = init.t3
            td = init.td
            sigma = init.sigma

            m = Minuit(mcost, A=A, fp=fp, t1=t1, t3=t3, td=td, fs_frac=fs_frac_init, sigma=sigma, t0=t0_init)

            m.limits['A'] = init.A_limits
            m.limits['fp'] = init.fp_limits
            m.limits['fs_frac'] = init.fs_frac_limits
            m.limits['t1'] = init.t1_limits
            m.limits['t3'] = init.t3_limits
            m.limits['td'] = init.td_limits
            m.limits['sigma'] = init.sigma_limits

            m.limits['t0'] = (0, nticks * self.dtime)

            m.fixed['fp'] = True
            m.fixed['fs_frac'] = True
            m.fixed['t1'] = True
            m.fixed['t3'] = True
            m.fixed['td'] = True
            m.migrad()
            m.migrad()
            m.migrad()
            m.fixed['fp'] = False
            m.fixed['fs_frac'] = False
            m.fixed['t1'] = False
            m.fixed['t3'] = False
            m.fixed['td'] = False
            m.migrad()
            m.migrad()
            m.migrad()

        m.hesse()
        
        pars = describe(self.model)[1:]
        params = [m.values[p] for p in pars]

        
        self.m = m
        if printresult:
            print(m)
            
        chi2 = m.fmin.reduced_chi2 if isinstance(m.fmin, FMin) else 0
        return params, chi2
