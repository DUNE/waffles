import numpy as np
from scipy.special import erfc

from iminuit import Minuit, cost
from iminuit.util import describe
from iminuit.util import FMin
from waffles.utils.fft.fftutils import FFTWaffles
from waffles.utils.time_align_utils import find_threshold_crossing


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
        super().__init__(filter_type=filter_type, cutoff_MHz=cutoff_MHz)

    ##################################################
    def set_template_waveform(self, wvf: np.ndarray):
        self.template = wvf.copy()

    ##################################################
    def set_response_waveform(self, wvf: np.ndarray):
        self.response = wvf.copy()

    ##################################################
    def getFFTs(self):
        self.templatefft = self.getFFT(self.template)
        self.responsefft = self.getFFT(self.response)

    ##################################################
    # MATHEMATICAL MODELS 
    ##################################################
    def expo_conv_gauss(self, x, A, tau, sigma, t0):
        return ((A / tau) * np.exp(-(x - t0) / tau) * np.exp(sigma**2 / (2 * tau**2))) * \
                    erfc(((t0 - x) / sigma + sigma / tau) / np.sqrt(2)) / 2.0


    def model_lar(self, x, A, fp, t1, t3, sigma, t0):

        
        term_fast = self.expo_conv_gauss(x, fp, t1, sigma, t0)
                    
        term_slow = self.expo_conv_gauss(x, 1 - fp, t3, sigma, t0)
                    
        return A * (term_fast + term_slow)

    def model_larxe(self, x, A, fp, fs, t1, t3, td, sigma, t0):

        
        term_fast = self.expo_conv_gauss(x, fp, t1, sigma, t0)
        term_slow = self.expo_conv_gauss(x, fs, t3, sigma, t0)
                    
        term_inter = self.expo_conv_gauss(x, 1 - fp- fs, td, sigma, t0)
                     
        return A * (term_fast + term_slow - term_inter)

    def generate_deconvolved_signal(self, response:np.ndarray = np.array([]), template:np.ndarray = np.array([])):
        if response.size > 0:
            self.set_response_waveform(response)
        if template.size > 0:
            self.set_template_waveform(template)

        self.deconvolved = self.backFFT(self.deconvolve(self.response, self.template))
        cross_template = find_threshold_crossing(self.template, 0.5)
        self.shift = cross_template
        self.deconvolved = np.roll(self.deconvolved, int(self.shift))

    ##################################################
    # FIT
    ##################################################
    def fit(self, oneexp: bool = False, print_flag: bool = False):
        """
        Fit deconvolved signal
        """

        params, chi2 = self.minimize(self.deconvolved, oneexp, printresult=print_flag)
        
        self.fit_results = params
        self.chi2 = chi2

        if print_flag:
            print(f"Final Fit Params: {params} | Chi2: {chi2}")

        return params, chi2

    ##################################################
    # MINUIT IMPLEMENTATION
    ##################################################
    def minimize(self, signal_to_fit: np.ndarray, oneexp: bool, printresult: bool):


        nticks = len(signal_to_fit)
        times = np.linspace(0, self.dtime * nticks, nticks, endpoint=False)
        errors = np.ones(nticks) * self.error

        # t0 dynamically initialized at the maximum peak
        maxBin = np.argmax(signal_to_fit)
        t0_init = float(maxBin * self.dtime)
        xlim = 10000//self.dtime
        lim_attempt = np.argwhere(signal_to_fit[maxBin:xlim] < 10**-3)
        if len(lim_attempt) > 0:
            xlim = lim_attempt[0][0] + maxBin

        times = times[:xlim]
        self.times = times
        signal_to_fit = signal_to_fit[:xlim]
        errors = errors[:xlim]

        if self.scinttype == 'lar':
            self.model = self.model_lar
            mcost = cost.LeastSquares(times, signal_to_fit, errors, self.model)
            
            A = 10e3
            fp = 0.3 if not oneexp else 1
            t1 = 25.
            t3 = 1600.
            if oneexp:
                t3 = 35
                fp = 0.95
            sigma = 40.0
            m = Minuit(mcost,A=A,fp=fp,t1=t1,t3=t3,sigma=sigma,t0=t0_init)
            
            m.limits['A'] = (0, None)
            m.limits['t1'] = (2, 50)
            m.limits['fp'] = (0, 1) 
            m.limits['t3'] = (500, 2000)
            if oneexp:
                m.limits['t3'] = (0,100)
            m.limits['sigma'] = (5, 100)
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
            mcost = cost.LeastSquares(times, signal_to_fit, errors, self.model)
            
            A = 10e3
            fp = 0.3
            fs = 1-fp-0.1

            t1 = 35.
            t3 = 1200.
            td = 50.
            sigma = 20.0
            m = Minuit(mcost, A=1e5, t1=10.0, fp=0.3, t3=1400.0, sigma=20.0, t0=t0_init, fs=0.6, td=200.0)
            m = Minuit(mcost,A=A,fp=fp,t1=t1,t3=t3,td=td, fs=fs, sigma=sigma, t0=t0_init)
            
            m.limits['A'] = (0, None)
            m.limits['t1'] = (2, 50)
            m.limits['fp'] = (0, 1) 
            m.limits['t3'] = (500, 2000)
            m.limits['sigma'] = (5, 100)
            m.limits['t0'] = (0, nticks * self.dtime)
            m.limits['fs'] = (0, 1) 
            m.limits['td'] = (50, 500)

            m.fixed['fp'] = True
            m.fixed['fs'] = True
            m.migrad()
            m.migrad()
            m.migrad()
            m.fixed['fp'] = False
            m.fixed['fs'] = False
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
