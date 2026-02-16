
import numpy as np
import pickle
import os

from waffles.utils.convolution.ConvFitter import ConvFitter
import matplotlib.pyplot as plt



class ConvFitterVDWrapper(ConvFitter):
    def __init__(self, 
                 threshold_align_template:float = 0.27, 
                 threshold_align_response:float = 0.1, 
                 error:float = 10,
                 dointerpolation:bool = False, 
                 interpolation_factor:int = 8,
                 align_waveforms:bool=True,
                 dtime:int = 16,
                 convtype:str = 'time',
                 usemplhep:bool = True, 
                 scinttype:str = 'lar'
                 ):
        super().__init__(
            threshold_align_template = threshold_align_template, 
            threshold_align_response = threshold_align_response, 
            error = error,
            dointerpolation = dointerpolation, 
            interpolation_factor = interpolation_factor,
            align_waveforms = align_waveforms,
            dtime = dtime,
            convtype = convtype,
            scinttype = scinttype
        )

        self.usemplhep = usemplhep

    #################################################
    def set_waveforms(self, wfresponse: np.ndarray, wftemplate: np.ndarray):
        self.template = wftemplate.astype(np.float32)
        self.response = wfresponse.astype(np.float32)

    #################################################
    def plot(self, newplot=False):


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

        if newplot:
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
            'fs': 'A_\\text{slow}',
            't1': '\\tau_\\text{fast}',
            't3': '\\tau_\\text{slow}',
            'td': '\\tau_\\text{delay}',
        }

        for p, v, e in zip(self.m.parameters, self.m.values, self.m.errors):
            if p in mapnames:
                fit_info.append(f"${mapnames[p]}$ = ${v:.3f} \\pm {e:.3f}$")
        plt.plot([],[], ' ', label='\n'.join(fit_info))
        plt.xlabel('Time [ns]')
        plt.ylabel('Amplitude [ADC]')

        #get handles and labels
        handles, labels = plt.gca().get_legend_handles_labels()

        return plt

