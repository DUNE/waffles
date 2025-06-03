import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import waffles.utils.time_align_utils as tutils

from waffles.utils.convolution.ConvFitter import ConvFitter



class ConvFitterHDWrapper(ConvFitter):
    def __init__(self, 
                 threshold_align_template = 0.27, 
                 threshold_align_response = 0.1, 
                 error=10,
                 dointerpolation=False, 
                 interpolation_factor = 8,
                 align_waveforms: bool=True,
                 dtime=16,
                 convtype = 'time',
                 usemplhep=True, 
                 ):
        super().__init__(
            threshold_align_template = threshold_align_template, 
            threshold_align_response = threshold_align_response, 
            error = error,
            dointerpolation = dointerpolation, 
            interpolation_factor = interpolation_factor,
            align_waveforms = align_waveforms,
            dtime = dtime,
            convtype = convtype
        )

        self.usemplhep = usemplhep

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
        tick_width = self.dtime if not self.dointerpolation else self.dtime/self.interpolation_factor
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

