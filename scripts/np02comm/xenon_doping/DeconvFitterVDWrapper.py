import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import mplhep

from waffles.utils.deconvolution.DeconvFitter import DeconvFitter
import waffles.utils.time_align_utils as tutils 

class DeconvFitterVDWrapper(DeconvFitter):
    def __init__(self, 
                 dtime:int = 16,
                 filter_type:str = 'Gauss',
                 scinttype:str = 'lar',
                 cutoff_MHz:float = 10,
                 usemplhep:bool = True,
                 error:float = 10
                 ):    
        super().__init__(
            scinttype=scinttype,
            filter_type=filter_type,
            cutoff_MHz=cutoff_MHz,
            dtime=dtime,
            error=error
        )
        
        self.usemplhep = usemplhep
        self.cutoff_MHz = cutoff_MHz  

    def set_waveforms(self, wfresponse: np.ndarray, wftemplate: np.ndarray):
        self.set_response_waveform(wfresponse.astype(np.float32))
        self.set_template_waveform(wftemplate.astype(np.float32))



    def plot(self, newplot=False, **modelparams):

        if self.usemplhep:
            mplhep.style.use(mplhep.style.ROOT)
            plt.rcParams.update({'font.size': 20,
                                 'grid.linestyle': '--',
                                 'axes.grid': True,
                                 'figure.autolayout': True,
                                 'figure.figsize': [14,6]
                                 })

        if newplot:
            plt.figure()

        # Plot the deconvolved signal
        if hasattr(self, 'deconvolved') and self.deconvolved is not None:
            pass
        else:
            raise ValueError("Deconvolved signal not found. Please run the fit method before plotting.")

        nticks_deconv = len(self.deconvolved)
        times = np.linspace(0, self.dtime*nticks_deconv, nticks_deconv, endpoint=False)
        plt.plot(times, self.deconvolved, '-', lw=2, color='k', label='data')



        # 3. Plot the fit
        if not modelparams:
            if hasattr(self, 'fit_results') and len(self.fit_results) > 0:
                plt.plot(self.times, self.model(self.times, *self.fit_results), color='r', zorder=100, label='Fit Model')
                
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
                    't0': 't_\\text{0}',
                    'sigma': '\\sigma',
                }

                for p, v, e in zip(self.m.parameters, self.m.values, self.m.errors):
                    if p in mapnames:
                        fit_info.append(f"${mapnames[p]}$ = ${v:.3f} \\pm {e:.3f}$")
                plt.plot([], [], ' ', label='\n'.join(fit_info))
        else:
            plt.plot(times, self.model(times, *modelparams.values()), color='r', zorder=100, label='Model')

        # plt.xlim(times[0], times[-1]-self.shift*self.dtime)
        plt.ylim(1e-3, np.max(self.deconvolved)*1.5)
        plt.yscale('log')
        plt.xlabel('Time [ns]')
        plt.ylabel('Amplitude [ADC]')
        plt.legend()

        return plt
