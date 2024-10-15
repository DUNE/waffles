from waffles.data_classes.WaveformSet import WaveformSet
from waffles.input.pickle_file_to_WaveformSet import pickle_file_to_WaveformSet
from waffles.np04_analysis.tau_slow_convolution.extractor_waveforms import Extractor
from waffles.np04_data.tau_slow_runs.load_runs_csv import ReaderCSV
import numpy as np
import pickle
import os

import argparse


if __name__ == "__main__":
    """Computes `template` or `response` for a specific list of `runs` and `channels`"""
    parse = argparse.ArgumentParser()
    parse.add_argument('-runs','--runs', type=int, nargs="+", help="Keep empty for all, or put the runs you want to be processed")
    parse.add_argument('-r','--response', action="store_true", help="Set true if response")
    parse.add_argument('-t','--template', action="store_true", help="Set true if template")
    parse.add_argument('-rl','--runlist', type=str, help="What run list to be used (purity or beam)", default="purity")
    parse.add_argument('-ch','--channels', type=int, nargs="+", help="Channels to analyze (format: 11225)", default=[11225])
    parse.add_argument('-p','--showp', action="store_true", help="Show progress bar")
    parse.add_argument('-f','--force', action="store_true", help='Overwrite...')
    parse.add_argument('-n','--dry', action="store_true", help="Dry run")
    args = vars(parse.parse_args())

    channels = args['channels']
    endpoint = channels[0]//100

    safemode = True
    if args['force']:
        safemode = False

    if args['response'] == args['template']:
        print("Please, choose one type -r or -t")
        exit(0)

    if args['response']:
        selectiontype='response'
    elif args['template']:
        selectiontype='template'

    runlist = args['runlist']
    dfcsv = ReaderCSV()
    raw_data_path = "./rawdata/waffles_tau_slow_protoDUNE_HD/"

    # these runs should be analyzed only on the last half
    blacklist = [ 28210, 28211, 28212, 28213, 28215, 28216, 28217, 28218, 28219 ] 
    try: 
        tmptype = 'Run'
        if args['template']:
            tmptype = 'Run LED'
        runnumbers = np.unique(dfcsv.dataframes[runlist][tmptype].to_numpy())
    except Exception as error:
        print(error)
        print('Could not open the csv file...')
        exit(0)


    if args['runs'] is not None:
        for r in args['runs']:
            if r not in runnumbers:
                print(f"Run {r} is not in database... check {runlist}_runs.csv")
        runnumbers = [ r for r in runnumbers if r in args['runs'] ]

    for runnumber in runnumbers:
        file = f"{raw_data_path}/{endpoint}/wfset_run0{runnumber}.pkl"

        if not os.path.isfile(file):
            print("No file for run", runnumber, "endpoint", endpoint)
            continue
        if args['dry']:
            print(runnumber, file)
            continue

        wfset = 0
        try:
            wfset = pickle_file_to_WaveformSet(file)
        except Exception as error:
            print(error)
            print("Could not load the file... of run ", runnumber, file)
            continue

        wfset_ch:WaveformSet = 0
        for ch in channels:
            pickleselecname = f'{selectiontype}s/{selectiontype}_run0{runnumber}_ch{ch}.pkl'
            pickleavgname = f'{selectiontype}s/{selectiontype}_run0{runnumber}_ch{ch}_avg.pkl'
            os.makedirs(f'{selectiontype}s', exist_ok=True)
            if safemode and os.path.isfile(pickleselecname):
                val:str
                val = input('File already there... overwrite? (y/n)\n')
                val = val.lower()
                if val == "y" or val == "yes":
                    pass
                else:
                    continue
            extractor = Extractor(selectiontype, runnumber) #here because I changed the baseline down..

            wch = ch
            if (wfset.waveforms[0].channel).astype(np.int64) - 100 < 0: # the channel stored is the short one
                wch = int(str(ch)[3:])
                extractor.channel_correction = True
            try: 
                wfset_ch = WaveformSet.from_filtered_WaveformSet( wfset, extractor.allow_certain_endpoints_channels, [endpoint] , [wch], show_progress=args['showp'])
            except Exception as error:
                print(error)
                print(f"No waveform for run {runnumber}, channel {ch}")
                continue

            wvf_arrays = np.array([(waveform.adcs.astype(np.float32) - waveform.baseline)*-1 for waveform in wfset_ch.waveforms if waveform.channel == wch])
            if runnumber in blacklist:
                print("Skipping first half...")
                skip = int(0.5*len(wvf_arrays))
                wvf_arrays = wvf_arrays[skip:]



            avgwvf = np.mean(wvf_arrays, axis=0)
            extractor.baseliner.binsbase = np.linspace(-20,20,500)
            res0, status = extractor.baseliner.compute_baseline(avgwvf)
            avgwvf-=res0
            wfset_ch.avgwvf = avgwvf
            wfset_ch.nselected = len(wvf_arrays)
            print(f'{runnumber} total: {len(wfset.waveforms)}\t {ch}: {len(wvf_arrays)}')

            with open(pickleselecname, "wb") as f:
                pickle.dump(wfset_ch, f)

            output = np.array([wfset_ch.avgwvf, wfset_ch.waveforms[0].timestamp, wfset_ch.nselected], dtype=object)

            with open(pickleavgname, "wb") as f:
                pickle.dump(output, f)
            print('Saved... ')
