import numpy as np

from iminuit import Minuit, cost
from iminuit.util import describe
from iminuit.util import FMin
from waffles.utils.fft.fftutils import FFTWaffles
from waffles.utils.numerical_utils import lar_response, lar_xe_response


class DeconvFitter(FFTWaffles):
    def __init__(self, 
                 scinttype:str = 'lar',
                 error:float = 10,
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
    def model_lar(self):
        pass

    ##################################################
    def model_larxe(self):
        pass

    ##################################################
    def minimize(self, printresult:bool, oneexp:bool=False):

        self.deconvolved = self.backFFT(self.deconvolve(self.response, self.template))

        self.model = self.model_lar if self.scinttype == 'lar' else self.model_larxe
        nticks = len(self.response)
        times  = np.linspace(0, self.dtime*nticks, nticks,endpoint=False)
        errors = np.ones(nticks)*self.error

        # SOME EXAMPLE...
        # mcost = cost.LeastSquares(times, self.deconvolved, errors, self.model)
        # # mcost = self.mycost

        # A = 10e3
        # fp = 0.3 if not oneexp else 1
        # t1 = 35.
        # t3 = 1600.
        # if oneexp:
        #     t3 = 35 fp = 0.95

        # m = Minuit(mcost,A=A,fp=fp,t1=t1,t3=t3)

        # m.limits['A'] = (0,None)
        # m.limits['fp'] = (0,1)
        # m.limits['t1'] = (2,50)
        # m.limits['t3'] = (500,2000)
        # if oneexp:
        #     m.limits['t3'] = (0,100)
        # m.fixed['fp'] =True
        # m.migrad()
        # m.migrad()
        # m.migrad()
        # m.fixed['fp'] = False 
        # m.migrad()
        # m.migrad()
        # m.migrad()
        # m.hesse()

        # pars = describe(self.model)[1:]
        # params = [m.values[p] for p in pars]

        # self.m: Minuit = m
        # if printresult:
        #     print(m)

        # chi2res: float = 0
        # if isinstance(m.fmin, FMin):
        #     chi2res = m.fmin.reduced_chi2
        # self.chi2 = chi2res
        # return params, chi2res
