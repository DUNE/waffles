from glob import glob
from importlib import resources
import copy
import yaml
import pandas as pd
import pathlib
import re
from waffles.input_output.hdf5_structured import load_structured_waveformset
from waffles.np02_utils.AutoMap import dict_module_to_uniqch

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

def ch_show_avaliable_calib_files():
    data_path = resources.files('waffles.np02_data.calibration_data')
    files = [f.name for f in data_path.iterdir() if f.is_file() and f.name.endswith('.csv')]
    print("Available calibration files:")
    for f in files:
        print(f"- {f}")

def ch_read_calib(filename: str = 'np02-config-v3.0.0.csv', attributes = ['Gain', 'SpeAmpl']) -> dict:

    thefile = resources.files('waffles.np02_data.calibration_data').joinpath(filename)
    if thefile.is_file() is False:
        ch_show_avaliable_calib_files()
        raise FileNotFoundError(
            f"Could not find the {filename} file in the waffles.np02_utils.PlotUtils.data package.\nWaffles should be installed with -e option to access this file.\n"
            )

    with thefile.open('r') as f:
        df = pd.read_csv(f, skipinitialspace=True)
        df = df.set_index(['endpoint', 'channel'])[attributes]
        # now regroup by endpoint
        nested_dict = {}
        for values in df.itertuples():
            ep, ch = getattr(values, 'Index')
            nested_dict.setdefault(ep, {})[ch] = { k: getattr(values, k) for k in attributes }
        return nested_dict

def ch_read_template(template_folder:str = 'templates_multi_pe_normalized'):
    thepath = resources.files('waffles.np02_data.templates').joinpath(template_folder)
    template_files = [f for f in thepath.iterdir() if f.is_file() and f.name.endswith('.txt')]
    ret = {}
    # file name is template_{reference_run}_{module}_{channel}.txt
    # getting module and channel from file name to create dictionary
    for tfile in template_files:
        match = re.match(r'template_\d+_([CM][0-9])_([1,2])\.txt', tfile.name)
        if not match:
            raise ValueError(f"Template file name {tfile} does not match expected pattern.")

        modulename = f"{match.group(1)}({match.group(2)})"
        uch = dict_module_to_uniqch[modulename]
        endpoint, channel = uch.endpoint, uch.channel
        with tfile.open('r') as f:
            template_waveform = pd.read_csv(f, header=None).to_numpy().flatten()
        ret.setdefault(endpoint, {})[channel] = template_waveform

    return ret
