#!/usr/bin/env python


import waffles
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from pathlib import Path
import argparse
import re

from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.UniqueChannel import UniqueChannel
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
from waffles.utils.numerical_utils import average_wf_ch, compute_mpv_waveforms
from waffles.utils.selector_waveforms import WaveformSelector
from waffles.np02_utils.AutoMap import dict_uniqch_to_module, dict_module_to_uniqch, strUch, ordered_channels_membrane, ordered_channels_cathode, getModuleName
from waffles.np02_utils.PlotUtils import runBasicWfAnaNP02, plot_detectors, wfset_remove_bad_baselines
from waffles.np02_utils.load_utils import open_processed, remove_extra_channels_membrane
from waffles.utils.utils import print_colored

from utils import DEFAULT_RESPONSE, PATH_XE_AVERAGES
from utils import make_standard_analysis_name, list_of_ints


from enum import Enum, auto
from collections import defaultdict

class ProcessStatus(Enum):
    COMPLETED  = auto()  # ran and processed successfully
    SKIPPED    = auto()  # already done, nothing to do
    DRYRUN    = auto()  # would have run, but dryrun=True
    FAILED     = auto()  # attempted to run but failed with an error


def __print_summary(results: dict[int, ProcessStatus], dryrun: bool) -> None:

    grouped = defaultdict(list)
    for run, status in results.items():
        grouped[status].append(run)

    if dryrun:
        # if grouped[ProcessStatus.DRYRUN]:
        print_colored(
            "Runs that will be processed: " + ",".join( str(i) for i in grouped[ProcessStatus.DRYRUN] ),
            color="INFO")

        if grouped[ProcessStatus.SKIPPED]:
            print_colored(
                "Runs already there: " + ",".join( str(i) for i in grouped[ProcessStatus.SKIPPED] ),
                color="WARNING")
    else:
        if grouped[ProcessStatus.COMPLETED]:
            print_colored(
                "Runs processed: " + ",".join( str(i) for i in grouped[ProcessStatus.COMPLETED] ),
                color="SUCCESS")
        if grouped[ProcessStatus.SKIPPED]:
            print_colored(
                "Runs already there: " + ",".join( str(i) for i in grouped[ProcessStatus.SKIPPED] ),
                color="WARNING")
    if grouped[ProcessStatus.FAILED]:
        print_colored(
            "Runs failed: " + ",".join( str(i) for i in grouped[ProcessStatus.FAILED] ),
            color="ERROR")

def __write_outputs(averages, channels_info, run, dettype, cutyaml, wfset_clean, endpoint, saveplots, averages_dir):

    # Create a README file with some information about the templates
    # Keep permisson restricted, so there is no overlap with other users
    # Saves the number of averages waveforms in each channel
    with open(averages_dir / "README.md", "w", ) as f:
        f.write(f"Run: {run}\n")
        f.write(f"Detector type: {dettype}\n")
        f.write(f"Number of waveforms averaged in each channel:\n")
        for line in channels_info:
            f.write(line)

    # Saves the yaml cuts used to select the waveforms as cuts_used.yaml in the same directory
    with open(averages_dir / "cuts_used.yaml", "w") as f:
        with open(cutyaml, "r") as fcuts:
            f.write(fcuts.read())

    # Save the average waveforms as numpy arrays
    # The filename is "average_endpoint{endpoint}_ch{ch}.txt"
    for ch, avg in averages.items():
        np.savetxt(averages_dir / f"average-endpoint-{endpoint}-ch-{ch}.txt", avg)


    if saveplots:
        print("Saving persistence plots...")
        groupalldict = { k: v for k, v in sorted(dict_module_to_uniqch.items(), key=lambda x: x[0]) if v.endpoint == endpoint }
        groupall = list(groupalldict.keys())
        argsheat = dict(
            mode="heatmap",
            analysis_label="std",
            adc_range_above_baseline=8000,
            adc_range_below_baseline=-1250,
            adc_bins=250,
            time_bins=wfset_clean.points_per_wf//2,
            filtering=0,
            share_y_scale=False,
            share_x_scale=True,
            wfs_per_axes=None,
            zlog=True,
            width=1300,
            height=650*4,
            showplots=False,
            return_fig=True,
            cols=2,
        )

        htmlname = averages_dir / f"persistence_selected.html"
        detector = groupall
        figs = plot_detectors(wfset_clean, detector=detector, html=htmlname, **argsheat) # type: ignore
        fig, rows, cols, title, g = figs[0]
        del fig


def __generate_readme(wfsetch, endpoint, timestamps, averages_dir):

    readmefile = averages_dir / "README.md"
    ret = []
    channels_info = {}

    for ch, wfs in wfsetch.items():
        if dict_uniqch_to_module.get(strUch(endpoint, ch), None) is not None:
            channels_info[ch] = f"{dict_uniqch_to_module[strUch(endpoint, ch)]} {endpoint}-{ch}: nwaveforms {len(wfs.waveforms)} timestamp {timestamps[ch]} ticks\n"

    if readmefile.exists():
        pattern = r'(\S+)\s+(\d+)-(\d+):\s+nwaveforms\s+(\d+)\s+timestamp\s+(\d+)'
        with open(readmefile, 'r') as f:
            lines = f.readlines()
            lines = [ l.strip() for l in lines if l[0] in ["C", "M", "P"]]
            for line in lines:
                match = re.match(pattern, line)
                if match:
                    ep = int(match.group(2))
                    if ep != endpoint:
                        raise ValueError(f"Endpoint in README {ep} does not match the current endpoint {endpoint}.")
                    ch = int(match.group(3))
                    if ch not in wfsetch:
                        channels_info[ch] = line +"\n"

    channels_info = { k: v for k, v in sorted(channels_info.items(), key=lambda x: getModuleName(endpoint, x[0])) }
    for ch, info in channels_info.items():
        ret.append(info)
    return ret

def create_response(wfsetch, avgmethod="mean", maxfev=100000, specific_tick=None):
    averages = {}
    timestamps = {}
    for ch, wfs in wfsetch.items():
        if avgmethod == "mean":
            averages[ch] = average_wf_ch(wfs,show_progress=True)
        elif avgmethod == "mpv":
            fit_info = compute_mpv_waveforms(wfs, maxfev=maxfev, specific_tick=specific_tick)
            averages[ch] = fit_info["mpv"]

        timestamps[ch] = sorted( [ wf.timestamp for wf in wfs.waveforms ] )[0]
    return averages, timestamps


def main(run, dettype, datadir, analysisname:str, nwaveforms=None, outputdir:Path=Path("./"), cutyaml="cuts.yaml", saveplots=False, nfiles=None, channels=None, avgmethod="mean", force = False, dryrun=False):

    if channels is not None:
        saveplots = False 
    endpoint = 106 if dettype == "cathode" else 107 if dettype == "membrane" else 110
            
    if outputdir.absolute().as_posix() == Path(PATH_XE_AVERAGES).as_posix():
        analysisname = make_standard_analysis_name(outputdir, analysisname)


    outputdir = outputdir / analysisname
    averages_dir = Path(outputdir) / f"run{run:06d}_{dettype}_response"



    if averages_dir.exists() and not force:
        print(f"Directory {averages_dir.absolute().as_posix()} already exists. Skipping run {run}. Use --force to overwrite.")
        return ProcessStatus.SKIPPED

    if dryrun:
        print(f"Run: {run}, will be processed and saved in {averages_dir.absolute().as_posix()}")
        return ProcessStatus.DRYRUN

    try:
        wfset_full = open_processed(run, dettype, datadir, nwaveforms=nwaveforms, mergefiles=True, file_slice=slice(None,nfiles), channels=channels, endpoints=endpoint)
    except Exception as e:
        print_colored(e, color="ERROR")
        return ProcessStatus.FAILED

    wfset_full = WaveformSet.from_filtered_WaveformSet(wfset_full, remove_extra_channels_membrane)


    runBasicWfAnaNP02(wfset_full, onlyoptimal=True, baselinefinish=60, int_ll=64, int_ul=100, amp_ll=64, amp_ul=150, configyaml="")
    wfset_full = wfset_remove_bad_baselines(wfset_full)

    extractor = WaveformSelector(yamlfile=cutyaml)

    extractor.loadcuts()
    if endpoint not in extractor.cutsdata:
        raise ValueError(f"No cuts defined for endpoint {endpoint} in {cutyaml}. Please check the YAML file and make sure it contains cuts for the correct endpoint.")
    print("Applying cuts to waveforms...")
    wfset_clean = WaveformSet.from_filtered_WaveformSet(wfset_full, extractor.applycuts, show_progress=True)
    print(f"Original waveforms: {len(wfset_full.waveforms)}, after cut: {len(wfset_clean.waveforms)}")


    wfsetch = ChannelWsGrid.clusterize_waveform_set(wfset_clean)[endpoint]

    wfsetch = { k: v for k, v in sorted(wfsetch.items(), key=lambda x: getModuleName(endpoint, x[0])) }

    averages, timestamps = create_response(wfsetch, avgmethod=avgmethod)

    # Create with permission 775 to allow group members to read and write the files
    averages_dir.mkdir(exist_ok=True, parents=True)
    averages_dir.chmod(0o775) # apparently it needs to be done after
    averages_dir.parent.chmod(0o775)

    channels_info = __generate_readme(wfsetch, endpoint, timestamps, averages_dir)

    __write_outputs(averages=averages, channels_info=channels_info, run=run,
                    dettype=dettype, cutyaml=cutyaml, wfset_clean=wfset_clean,
                    endpoint=endpoint, saveplots=saveplots,
                    averages_dir=averages_dir)

    return ProcessStatus.COMPLETED


if __name__ == "__main__":
    argp = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Generate average waveforms for a given run and detector type.")
    argp.add_argument("--runs", type=list_of_ints, default=39510, help="List of runs to process.")
    argp.add_argument("--dettype", type=str, default="m", choices=["m", "c", "p"], help="Detector type to process.")
    argp.add_argument("--datadir", type=str, default="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/", help="Directory where the processed waveform files are located.")
    argp.add_argument("--nwaveforms", type=int, default=None, help="Maximum number of waveforms to load for each channel.")
    argp.add_argument("--outputdir", type=str, default=PATH_XE_AVERAGES, help="Directory where the average waveforms will be saved.")
    argp.add_argument("--cutyaml", type=str, default="cuts.yaml", help="YAML file containing the cuts to apply to the waveforms.")
    argp.add_argument("--saveplots", action="store_true", help="Whether to save the persistence plots of the waveforms.")
    argp.add_argument("--analysisname", type=str, default=DEFAULT_RESPONSE, help="Name of the analysis, used to create a subdirectory in the output directory.")
    argp.add_argument("--average-method", type=str, default="mean", choices=["mean", "mpv"], help="Method to use for averaging the waveforms.")
    argp.add_argument("--nfiles", type=int, default=None, help="Maximum number of files to load for each channel.")
    argp.add_argument("--channels", nargs="+", type=int, default=[], help="List of channels to process. If not specified, all channels will be processed.\
        \n\tFormat: endpoint+channel, e.g. 10600 for endpoint 106 channel 0.")
    argp.add_argument("--force", action="store_true", help="If set, the script will overwrite existing output directories. Use with caution.")
    argp.add_argument("--dryrun", action="store_true", help="If set, the script will only print the parameters and not execute the main function.")

    args = argp.parse_args()
    runs = args.runs
    dettype = args.dettype
    datadir = args.datadir
    nwaveforms = args.nwaveforms
    outputdir = Path(args.outputdir)
    cutyaml = args.cutyaml
    saveplots = args.saveplots
    analysisname = args.analysisname
    channels = args.channels
    avgmethod = args.average_method
    nfiles = args.nfiles
    force = args.force

    dettype = "membrane" if dettype == "m" else "cathode" if dettype == "c" else "pmt"
    endpoint = 106 if dettype == "cathode" else 107 if dettype == "membrane" else 110
    if channels:
        outch = []
        for epch in channels:
            ep = epch // 100
            ch = epch % 100
            if ep != endpoint:
                raise ValueError(f"Invalid channel format: {ep}. The endpoint must be {endpoint} for detector type {dettype}. Please check the --channels argument.")
            outch.append(ch)
        channels = outch
    else:
        channels = None
        
    results: dict[int, ProcessStatus] = {}
    for run in runs:
         results[run] = main(run, dettype, datadir, analysisname, nwaveforms,
             outputdir=outputdir, cutyaml=cutyaml, saveplots=saveplots,
             nfiles=nfiles, channels=channels, avgmethod=avgmethod,
             force=force, dryrun=args.dryrun)

    __print_summary(results, dryrun=args.dryrun)
