import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import waffles.utils.time_align_utils as tutils

from iminuit import Minuit, cost
from iminuit.util import describe

from scipy import interpolate

class ConvFitter:
    def __init__(self, 
                threshold_align_template = 0.27, 
                threshold_align_response = 0.1, 
                error=10, usemplhep=True, 
                dointerpolation=False, 
                interpolation_fraction = 8,
                align_waveforms: bool=True):

        self.threshold_align_template = threshold_align_template
        self.threshold_align_response = threshold_align_response
        self.error = error
        self.usemplhep = usemplhep
        self.dointerpolation = dointerpolation
        self.interpolation_fraction = interpolation_fraction
        self.dosave = False
        self.reduce_offset = False
        self.align_waveforms = align_waveforms


        self.first_time:int = 0
        self.template:np.ndarray = None
        self.response:np.ndarray = None

    #################################################
    def read_waveforms(self, file_template, file_response):

        self.wf_response = self.parse_input(file_response)
        self.wf_template = self.parse_input(file_template)

        self.template = self.wf_template["avgwvf"].copy()
        self.response = self.wf_response["avgwvf"].copy()    

    #################################################
    def parse_input(self,file):
        if os.path.isfile(file) is not True:
            print("No response from", file)
            exit(0)        
        try:
            with open(file, 'rb') as f:
                finput = pickle.load(f)
        except Exception as error:
            print(error)
            print("Could not load file", file)
            exit(0)

        keys = ["avgwvf", "firsttime", "nselected"]

        output = {}
        try:
            for k, v in zip(keys, finput):
                output[k] = v
        except Exception as error:
            print(error)
            exit(0)
        return output

    #################################################
    def prepare_waveforms(self):    
        
        if self.dointerpolation:                        
            self.threshold_align_response = 0.1

            self.response = self.interpolate(self.response, self.interpolation_fraction)
            self.template = self.interpolate(self.template, self.interpolation_fraction)

        if self.align_waveforms:
            offsettemplate = tutils.find_threshold_crossing(self.template, self.threshold_align_template)
            self.response, offset = tutils.shift_waveform_to_align_threshold(self.response, threshold=self.threshold_align_response, 
                                                                            reduce_offset=self.reduce_offset,target_index=offsettemplate)
            self.response = self.response[offset:]
            self.template = self.template[offset:]

    #################################################
    def interpolate(self, wf, interpolation_fraction: float):

        # Create an array of times with 16 ns tick width 
        tick_width = 16
        nticks = len(wf)
        times = np.linspace(0, nticks*tick_width, nticks, endpoint=False)

        # these are the continues functions using x=times and y=adcs
        wf_inter = interpolate.interp1d(times, wf, kind='linear', fill_value="extrapolate")
       
        # these are the new times at which to compute the value of the function
        tick_width = 16/interpolation_fraction
        nticks = len(wf)*interpolation_fraction
        newtimes = np.linspace(0, nticks*tick_width, nticks, endpoint=False)

        # create new waveforms using the new time values
        wf = wf_inter(newtimes)

        return wf

    #################################################
    def fit(self, scan: bool = False, print_flag: bool = False):

        # scan over offsets to minimize the chi2 between the response and the template x model
        if scan > 0:
            
            self.dosave = False
            resp_original = self.response.copy()
            temp_original = self.template.copy()

            chi2s = []
            offsets = np.arange(0, scan)
            print ('    scanning over offsets to minimize chi2 between the response and the template x model')
            for offset in offsets:
                self.response = np.roll(resp_original, offset, axis=0)
                self.response = self.response[offset:]
                self.template = temp_original[offset:]
                params, chi2 = self.minimize(False)
                chi2s.append(chi2)
                if(print_flag): print(offset, params, chi2)

            # recompute the waveforms for the minimum chi2
            idxMinChi2 = np.argmin(chi2s)
            self.response = np.roll(resp_original, offsets[idxMinChi2], axis=0)
            self.response = self.response[offsets[idxMinChi2]:]
            self.template = temp_original[offsets[idxMinChi2]:]
            # self.dosave = not self.no_save

        # recompute parameters for the minimum chi2
        params, chi2 = self.minimize(print_flag)
        if(print_flag): print(params, chi2)

        self.fit_results = params

    #################################################
    def model(self, t, A, fp, t1, t3):
        self.lar = A*(fp*np.exp(-t/t1)/t1 + (1-fp)*np.exp(-t/t3)/t3)
        return np.convolve(self.lar,self.template,mode='full')[:len(self.lar)]

    #################################################
    def minimize(self, printresult:bool):
        
        tick_width = 16 if not self.dointerpolation else 16/self.interpolation_fraction
        nticks = len(self.response)

        times  = np.linspace(0, tick_width*nticks, nticks,endpoint=False)
        errors = np.ones(nticks)*self.error

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

        return params, m.fmin.reduced_chi2

#        if self.dosave:
#            self.saveresults()

    #################################################
    def plot(self):

        """ ---------- do the convolution and fit plot ----------- """

        # root ploting style
        if self.usemplhep:
            import mplhep
            mplhep.style.use(mplhep.style.ROOT)
            plt.rcParams.update({'font.size': 20,
                                 'grid.linestyle': '--',
                                 'axes.grid': True,
                                 'figure.autolayout': True,
                                 'figure.figsize': [14,6]
                                 })

        

        # Create an array of times with 16 ns tick width 
        tick_width = 16 if not self.dointerpolation else 16/self.interpolation_fraction
        nticks = len(self.response)
        times  = np.linspace(0, tick_width*nticks, nticks,endpoint=False)

        # create new figure
        plt.figure()

        # do the plot
        plt.plot(times, self.response,'-', lw=2 ,color='k', label='data')
        plt.plot(times, self.model(times,*self.fit_results), color='r', zorder=100, label='fit')
        plt.xlim(times[0], times[-1])
        
        """ ---------- add legend to the plot ----------- """

        fit_info = [
            f"$\\chi^2$/$n_\\mathrm{{dof}}$ = {self.m.fval:.2f} / {self.m.ndof:.0f} = {self.m.fmin.reduced_chi2:.2f}",
        ]

        mapnames = {
            'A': 'Norm.',
            'fp': 'A_\\text{fast}',
            't1': '\\tau_\\text{fast}',
            't3': '\\tau_\\text{slow}',
        }

        for p, v, e in zip(self.m.parameters, self.m.values, self.m.errors):
            fit_info.append(f"${mapnames[p]}$ = ${v:.3f} \\pm {e:.3f}$")
        plt.plot([],[], ' ', label='\n'.join(fit_info))
        plt.xlabel('Time [ns]')
        plt.ylabel('Amplitude [ADC]')

        #get handles and labels
        handles, labels = plt.gca().get_legend_handles_labels()

        return plt
    
