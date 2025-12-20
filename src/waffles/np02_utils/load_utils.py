from glob import glob
from importlib import resources
import copy
import yaml
import pandas as pd
import pathlib
from waffles.input_output.hdf5_structured import load_structured_waveformset

def open_processed(run, dettype, datadir, channels = None, endpoints=None, nwaveforms=None, mergefiles = True, verbose=True):
    """
    Open the processed waveform set for a given run and detector type.
    """
    try: 
        wfset = load_structured_waveformset(
            f"{datadir}/processed/run{run:0d}_{dettype}/processed_merged_run{run:06d}_structured_{dettype}.hdf5",
            max_to_load=nwaveforms,
            channels_filter=channels,
            endpoint_filter=endpoints
        )
    except:
        files = glob(f"{datadir}/processed/run{run:06d}_{dettype}/processed_*_run{run:06d}_*_{dettype}.hdf5")
        if verbose:
            print("List of files found:")
            print(files)
        if not mergefiles or len(files)==1:
            files = files[0]
            wfset = load_structured_waveformset(files, max_to_load=nwaveforms, channels_filter=channels, endpoint_filter=endpoints, verbose=verbose)
        elif len(files)>1: 
            wfset = load_structured_waveformset(files[0], max_to_load=nwaveforms, channels_filter= channels, endpoint_filter=endpoints, verbose=verbose)
            for f in files[1:]:
                tmpwf = load_structured_waveformset(f, max_to_load=nwaveforms, channels_filter= channels, endpoint_filter=endpoints, verbose=verbose)
                wfset.merge(copy.deepcopy(tmpwf))
        else:
            raise FileNotFoundError(f"No processed files found for run {run} and dettype {dettype} in {datadir}/processed/")
    return wfset

def ch_read_params(filename:str = 'ch_snr_parameters.yaml') -> dict:
    thefile = resources.files('waffles.np02_utils.data').joinpath(filename)  # type: ignore
    if thefile.is_file() is False:
        # Trying to load file locally...
        if pathlib.Path(filename).is_file():
            thefile = pathlib.Path(filename)
        else:
            raise FileNotFoundError(
                f"Could not find the {filename} file in the waffles.np02_utils.PlotUtils.data package or locally.\nWaffles should be installed with -e option to access this file.\n"
            )
    try:
        with thefile.open('r') as f:
            return yaml.safe_load(f)
    except Exception as error:
        print(error)
        print("\n\n")
        raise FileNotFoundError(
            f"Could not load the {filename} file ..."
        )

def ch_read_calib(filename: str = 'calibration_results_file.csv') -> dict:
    try:
        with resources.files('waffles.np02_utils.data.calibration_data').joinpath(filename).open('r') as f:
            df = pd.read_csv(f)
            result = ( df.set_index(['endpoint', 'channel'])[['Gain', 'SpeAmpl']].to_dict(orient='index'))
            # now regroup by endpoint
            nested_dict = {}
            for (ep, ch), values in result.items():
                nested_dict.setdefault(ep, {})[ch] = values
            return nested_dict
    except Exception as error:
        print(error)
        print("\n\n")
        raise FileNotFoundError(
            f"Could not find the {filename} file in the waffles.np02_utils.PlotUtils.data package.\nWaffles should be installed with -e option to access this file.\n"
        )
