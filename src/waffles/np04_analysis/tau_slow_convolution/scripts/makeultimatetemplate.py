# IMPORT ALL THE LIBRARIES USED IN THE NOTEBOOK

from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.Waveform import Waveform
from waffles.input_output.pickle_file_reader import WaveformSet_from_pickle_file
from waffles.np04_data.tau_slow_runs.load_runs_csv import ReaderCSV
from extract_selection import Extractor
import argparse
import numpy as np
import pandas as pd
import pickle



class Filter:
    def __init__(self):
        self.is_first = True

    def get_only_one_wvf(self, waveform: Waveform) -> bool:
        if self.is_first:
            self.is_first = False
            return True
        else:
            return False
        return False
    
if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("-ch","--channels", type=int, nargs="+", help="List of channels", default=[11225])
    args = vars(parse.parse_args())
    channels = args['channels']

    dfcsv = ReaderCSV()
    runnumbers = dfcsv.dataframes['led']['Run'].to_numpy()

    for ch in channels:
        allwvfs =  {0: [], 1:[]}
        ultimatetype = 0
        ft = Filter()
        wfout:WaveformSet = None
        for run in runnumbers:
            if run > 27901:# and ch//100 == 112:
                ultimatetype = 1
            print(f'run {run}', end=' ')
            wfset:WaveformSet
            try:
                wfset = WaveformSet_from_pickle_file(f"/eos/home-h/hvieirad/waffles/analysis/templates/template_run0{run}_ch{ch}.pkl")
            except:
                print('nope...')
                continue

            try:
                wfout = WaveformSet.from_filtered_WaveformSet( wfset, ft.get_only_one_wvf)
            except:
                pass
            first = False
            print(f'total: {len(wfset.waveforms)}')
            allwvfs[ultimatetype] += [(waveform.adcs.astype(np.float32) - waveform.baseline)*-1 for waveform in wfset.waveforms]
        for ultimatetype in range(2):

            allwvfs[ultimatetype] = np.array(allwvfs[ultimatetype])
            print(ultimatetype, 'Total waveforms for master template:', len(allwvfs[ultimatetype]))
            avgwvf = np.mean(allwvfs[ultimatetype], axis=0)
            
            extractor = Extractor("template")
            extractor.baseliner.binsbase = np.linspace(-20,20,500)
            res0, status = extractor.baseliner.compute_baseline(avgwvf)
            avgwvf-=res0
            wfout.avgwvf = avgwvf
            wfout.nselected = len(allwvfs[ultimatetype])
            print(f'\t {ch}: {len(allwvfs[ultimatetype])}')
            pickleselecname = f'templates/template_run0{ultimatetype}_ch{ch}.pkl'
            pickleavgname = f'templates/template_run0{ultimatetype}_ch{ch}_avg.pkl'
            print(pickleselecname)
            with open(pickleselecname, "wb") as f:
                pickle.dump(wfout, f)

            output = np.array([wfout.avgwvf, wfout.waveforms[0].timestamp, wfout.nselected], dtype=object)
            with open(pickleavgname, "wb") as f:
                pickle.dump(output, f)
            print('Saved... ')
