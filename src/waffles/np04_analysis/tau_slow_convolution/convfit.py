import matplotlib.pyplot as plt
import numpy as np
from waffles.np04_data.tau_slow_runs.load_runs_csv import ReaderCSV

from iminuit import Minuit, cost
from iminuit.util import describe

import argparse
import os
from scipy import interpolate
import pickle
from numba import njit

kwd = {"fastmath": {"reassoc", "contract", "arcp"}}


class ConvFitter:
    def __init__(self, extra="", thre_align_template = 0.27, thre_align_response = 0.1, error=10, usemplhep=True, dointerpolation=False):

        self.extra = extra
        self.thre_align_template = thre_align_template
        self.thre_align_response = thre_align_response
        self.error = error
        self.usemplhep = usemplhep
        self.dointerpolation = dointerpolation
        self.dosave = False
        self.reduce_offset = False


        self.firsttime:int = 0
        self.template:np.ndarray = None
        self.data:np.ndarray = None
        self.all_results = []

    def model(self, t, A, fp, t1, t3):
        self.lar = A*(fp*np.exp(-t/t1)/t1 + (1-fp)*np.exp(-t/t3)/t3)
        return np.convolve(self.lar,self.template,mode='full')[:len(self.lar)]


    def process_waveforms(self, wfsetresponse:dict, wfsettemplate:dict):
        self.firsttime = wfsetresponse["firsttime"]

        self.template = wfsettemplate["avgwvf"].copy()
        self.data = wfsetresponse["avgwvf"].copy()
        self.nselected = wfsetresponse["nselected"]
        self.times = np.linspace(0,len(self.data)*16,len(self.data),endpoint=False)

        if self.dointerpolation:
            datainter = interpolate.interp1d(self.times, self.data, kind='linear', fill_value="extrapolate")
            templateinter = interpolate.interp1d(self.times, self.template, kind='linear', fill_value="extrapolate")
            newtimes = np.linspace(0,(8*len(self.data))*2,8*len(self.data),endpoint=False)
            self.data = datainter(newtimes)
            self.template = templateinter(newtimes)
            self.thre_align_response = 0.1


        offsettemplate = self.find_threshold_crossing(self.template, self.thre_align_template)
        self.data, offset = self.shift_waveform_to_align_threshold(self.data, threshold=self.thre_align_response, target_index=offsettemplate)
        self.data = self.data[offset:]
        self.template = self.template[offset:]

    def minimize(self, runresponse:int, runtemplate:int, ch:int, printresult:bool):
        self.runresponse = runresponse
        self.runtemplate = runtemplate
        ticks = 16 if not self.dointerpolation else 2

        self.times = np.linspace(0,len(self.data)*ticks,len(self.data),endpoint=False)
        self.errors = np.ones(len(self.data))*self.error

        mcost = cost.LeastSquares(self.times, self.data, self.errors, self.model)
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
        vals = [m.values[p] for p in pars]

        self.fit_results = vals
        self.m = m
        self.all_results.append(vals)
        if printresult:
            print(m)
            return

        if self.dosave:
            self.saveresults()
    
    def saveresults(self):

        if self.usemplhep:
            import mplhep
            mplhep.style.use(mplhep.style.ROOT)
            plt.rcParams.update({'font.size': 20,
                                 'grid.linestyle': '--',
                                 'axes.grid': True,
                                 'figure.autolayout': True,
                                 'figure.figsize': [14,6]
                                 })

        plt.plot(self.times,self.data,'-', lw=2 ,color='k', label='data')
        plt.plot(self.times,self.model(self.times,*self.fit_results),color='r',zorder=100,label='fit')
        plt.xlim(self.times[0], self.times[-1])
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

        #add legend to plot
        plt.legend(title=f'run {self.runresponse}')
        dirout = f'results{self.extra}/run0{self.runresponse}'
        os.makedirs(dirout, exist_ok=True)

        with open(f"{dirout}/convolution_output_{self.runresponse}_{self.runtemplate}_ch{ch}.txt", "w") as fout:
            fout.write(f"{self.firsttime} {self.m.values['fp']} {self.m.values['t1']} {self.m.values['t3']} {self.m.fmin.reduced_chi2} {self.nselected} \n")

        with open(f"{dirout}/run_output_{self.runresponse}_{self.runtemplate}_ch{ch}.txt", "w") as fout:
            print(self.m, file=fout)

        plt.savefig(f'{dirout}/convfit_data_{self.runresponse}_template_{self.runtemplate}_ch{ch}.png')



    def shift_waveform_continuous_fowards(self, waveform, shift_amount):

        if (shift_amount%1 == 0):
            return np.roll(waveform, shift_amount, axis=0)

        # raise Exception("Nope...")
        # Create an array of indices corresponding to the original waveform
        original_indices = np.arange(len(waveform))

        # The new indices we want to interpolate at, shifted by the shift_amount
        new_indices = original_indices - shift_amount

        # Use scipy's interp1d to create a linear interpolator
        interpolator = interpolate.interp1d(original_indices, waveform, kind='linear', fill_value="extrapolate")

        # Interpolate the waveform at the new shifted indices
        shifted_waveform = interpolator(new_indices)

        return shifted_waveform

    def find_threshold_crossing(self, waveform, threshold_per_cent:float):
        threshold = threshold_per_cent*np.max(waveform)
        # Find indices where the waveform crosses the threshold (positive slope)
        crossings = np.where(np.diff(np.sign(waveform - threshold)) > 0)[0]
        
        if len(crossings) == 0:
            raise ValueError("No threshold crossing found in the waveform.")
        
        # For simplicity, let's take the first crossing point
        # Interpolate between the two points that straddle the threshold crossing
        idx_before = crossings[0]
        y1, y2 = waveform[idx_before], waveform[idx_before + 1]
        x1, x2 = idx_before, idx_before + 1
        
        # Linear interpolation to find the exact fractional index of crossing
        fractional_crossing = x1 + (threshold - y1) / (y2 - y1)
        
        return fractional_crossing


    def shift_waveform_to_align_threshold(self, waveform, threshold, target_index=-1):
        # Find the crossing point (fractional index)
        crossing_point = self.find_threshold_crossing(waveform, threshold)
        
        # Compute how much we need to shift the waveform to align the crossing point to `target_index`
        if (target_index == -1):
            shift_amount = crossing_point % 1
        shift_amount = round(target_index - crossing_point)
        if (self.reduce_offset):
            shift_amount-=2
        
        # Shift the waveform by this amount using linear interpolation
        return self.shift_waveform_continuous_fowards(waveform, shift_amount), round(shift_amount)

    def parse_inputs(self, file):
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


if __name__ == "__main__":
    
    parse = argparse.ArgumentParser()
    parse.add_argument('-runs','--runs', type=int, nargs="+", help="Keep empty for all, or put the runs you want to be processed")
    parse.add_argument('-ch','--channel', type=int, help="Which channel to analyze", default=11225)
    parse.add_argument('-ft','--fix-template', action="store_true", help="Fix template to run 26261 (or thetemplate)")
    parse.add_argument('-tt', '--thetemplate', type=int, help="If fix-template is set, use this to tell which template to use", default=0)
    parse.add_argument('-ns', '--namespace', type=str, help="Name space in case different folder", default="")
    parse.add_argument('-rl','--runlist', type=str, help="What run list to be used (purity or beam)", default="purity")
    parse.add_argument('-fr','--folder-responses', type=str, help="Directory of responses (just the name, default: responses)", default="responses")
    parse.add_argument('-p','--print', action="store_true", help="If you want you can print result and not save")
    parse.add_argument('-i','--interpolate', action="store_true", help="If you want 16 ns to be linear interpolated to 2 ns")
    parse.add_argument('--no-save', action="store_true", help="If you want the output to be saved")
    parse.add_argument('-scan','--scan', type=int, help="Set maximum offset if you want to scan different offsets and get minimum. Scan is done around the default offset applied (-2, -(scan-2)). Set 0 to not scan.", default=0)
    args = vars(parse.parse_args())
    

    if args['runs'] is None:
        print('Please give a run')
        exit(0)
    runnumbers = [ r for r in args['runs'] ]
    ch = args['channel']
    use_fix_template = args['fix_template']
    
    dfcsv = ReaderCSV()
    df = dfcsv.dataframes[args['runlist']]
    runs = df['Run'].to_numpy()
    ledruns = df['Run LED'].to_numpy()

    ledruntemplate = args['thetemplate']
    
    extra=""
    if args['runlist'] != "purity":
        extra += f"_{args['runlist']}"
    if args['namespace'] != "":
        extra += f"_{args['namespace']}"
    if use_fix_template:
        extra += "_fixtemplate"
    
    runpairs = { r:lr for r, lr in zip(runs, ledruns) }

    for run in runnumbers:
        print(f"Processing run {run}")
        if run not in runpairs:
            print('Run not found in runlist, check it')
            exit(0)
        runled = runpairs[run]

        # changes template in case it is fixed at 0 for endpoint 112
        if ledruntemplate == 0 and run > 27901:# and ch//100 == 112:
            ledruntemplate = 1
        
        if use_fix_template:
            runled = ledruntemplate

        fileresponse = f"/eos/home-h/hvieirad/waffles/analysis/{args['folder_responses']}/response_run0{run}_ch{ch}_avg.pkl"


        filetemplate = f'/eos/home-h/hvieirad/waffles/analysis/templates/template_run0{runled}_ch{ch}_avg.pkl'
        if os.path.isfile(filetemplate) is not True:
                print(f"No match of LED run {runled}.. using \'thetemplate\' instead: {ledruntemplate} ")
                runled = ledruntemplate
                filetemplate = f'/eos/home-h/hvieirad/waffles/analysis/templates/template_run0{runled}_ch{ch}_avg.pkl'


        

            
        cfit = ConvFitter(extra=extra, dointerpolation=args['interpolate'])
        cfit.dosave = not args['no_save']
        wfsetresponse = cfit.parse_inputs(fileresponse)
        wfsettemplate = cfit.parse_inputs(filetemplate)

        doscan = False
        if args['scan'] > 0:
            doscan = True
            cfit.reduce_offset = True

        cfit.process_waveforms(wfsetresponse, wfsettemplate)

        if doscan:
            cfit.dosave = False
            dataoriginal = cfit.data.copy()
            templateoriginal = cfit.template.copy()

            chi2s = []
            offsets = np.arange(0, args['scan'])
            for offset in offsets:
                cfit.data = np.roll(dataoriginal, offset, axis=0)
                cfit.data = cfit.data[offset:]
                cfit.template = templateoriginal[offset:]
                cfit.minimize(run, runled, ch, False)
                chi2s.append(cfit.m.fmin.reduced_chi2)
                if(args['print']): print(offset, cfit.all_results[-1], chi2s[-1])

            
            idxMinChi2 = np.argmin(chi2s)
            cfit.data = np.roll(dataoriginal, offsets[idxMinChi2], axis=0)
            cfit.data = cfit.data[offsets[idxMinChi2]:]
            cfit.template = templateoriginal[offsets[idxMinChi2]:]
            cfit.dosave = not args['no_save']

        cfit.minimize(run, runled, ch, args['print'])
        if(args['print']): print(cfit.all_results[-1], chi2s[-1])

    



