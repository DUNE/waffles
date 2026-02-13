#!/usr/bin/env python


import waffles
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from pathlib import Path
import argparse

from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.UniqueChannel import UniqueChannel
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
from waffles.utils.numerical_utils import average_wf_ch
from waffles.utils.selector_waveforms import WaveformSelector
from waffles.np02_utils.AutoMap import dict_uniqch_to_module, dict_module_to_uniqch, strUch, ordered_channels_membrane, ordered_channels_cathode
from waffles.np02_utils.PlotUtils import runBasicWfAnaNP02, plot_detectors
from waffles.np02_utils.load_utils import open_processed


def main(run, dettype, datadir, nwaveforms=None, outputdir="./", cutyaml="cuts.yaml", saveplots=False):

    endpoint = 106 if dettype == "cathode" else 107
    if outputdir == "/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/xenon_averages/":
        outputdir = f"/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/xenon_averages/{os.getlogin()}/"


    wfset_full = open_processed(run, dettype, datadir, nwaveforms=nwaveforms, mergefiles=True, file_slice=slice(None,None))


    runBasicWfAnaNP02(wfset_full, onlyoptimal=False, baselinefinish=60, int_ll=64, int_ul=100, amp_ll=64, amp_ul=150, configyaml="")

    extractor = WaveformSelector(yamlfile=cutyaml)

    extractor.loadcuts()
    print("Applying cuts to waveforms...")
    wfset_clean = WaveformSet.from_filtered_WaveformSet(wfset_full, extractor.applycuts, show_progress=True)
    print(f"Original waveforms: {len(wfset_full.waveforms)}, after cut: {len(wfset_clean.waveforms)}")


    wfsetch = ChannelWsGrid.clusterize_waveform_set(wfset_clean)[endpoint]

    wfsetch = { k: v for k, v in sorted(wfsetch.items(), key=lambda x: dict_uniqch_to_module[strUch(endpoint, x[0])]) }

    averages = {}
    for ch, wfs in wfsetch.items():
        averages[ch] = average_wf_ch(wfs,show_progress=True)

    template_dir = Path(outputdir) / f"run{run:06d}_{dettype}_response"
    # Create with permission 775 to allow group members to read and write the files
    template_dir.mkdir(exist_ok=True, parents=True)
    template_dir.chmod(0o775) # apparently it needs to be done after

    # Create a README file with some information about the templates
    # Keep permisson restricted, so there is no overlap with other users
    # Saves the number of averages waveforms in each channel
    with open(template_dir / "README.md", "w", ) as f:
        f.write(f"Run: {run}\n")
        f.write(f"Detector type: {dettype}\n")
        f.write(f"Number of waveforms averaged in each channel:\n")
        for ch, wfs in wfsetch.items():
            f.write(f"{dict_uniqch_to_module[strUch(endpoint, ch)]} {endpoint}-{ch}: {len(wfs.waveforms)} waveforms\n")

    # Saves the yaml cuts used to select the waveforms as cuts_used.yaml in the same directory
    with open(template_dir / "cuts_used.yaml", "w") as f:
        with open(cutyaml, "r") as fcuts:
            f.write(fcuts.read())

    # Save the average waveforms as numpy arrays
    # The filename is "average_endpoint{endpoint}_ch{ch}.txt"
    for ch, avg in averages.items():
        np.savetxt(template_dir / f"average-endpoint-{endpoint}-ch-{ch}.txt", avg)


    if saveplots:
        dletter = dettype.upper()[0] # C or M...
        groupall = [ f"{dletter}{detnum}({chnum})" for detnum in range(1, 9) for chnum in range(1,3) ]
        argsheat = dict(
            mode="heatmap",
            analysis_label="std",
            adc_range_above_baseline=8000,
            adc_range_below_baseline=-1250,
            adc_bins=250,
            time_bins=wfset_full.points_per_wf//2,
            filtering=8,
            share_y_scale=False,
            share_x_scale=True,
            wfs_per_axes=5000,
            zlog=True,
            width=1300,
            height=650*4,
            showplots=False,
            cols=2,
        )

        htmlname = template_dir / f"persistence_selected.html"
        detector = groupall
        plot_detectors(wfset_clean, detector, html=htmlname, **argsheat)

if __name__ == "__main__":
    argp = argparse.ArgumentParser(description="Generate average waveforms for a given run and detector type.")
    argp.add_argument("--runs", nargs="+", type=int, default=[39510], help="List of runs to process.")
    argp.add_argument("--dettype", type=str, default="m", choices=["m", "c"], help="Detector type to process.")
    argp.add_argument("--datadir", type=str, default="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/", help="Directory where the processed waveform files are located.")
    argp.add_argument("--nwaveforms", type=int, default=None, help="Maximum number of waveforms to load for each channel.")
    argp.add_argument("--outputdir", type=str, default="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/xenon_averages/", help="Directory where the average waveforms will be saved.")
    argp.add_argument("--cutyaml", type=str, default="cuts.yaml", help="YAML file containing the cuts to apply to the waveforms.")
    argp.add_argument("--saveplots", action="store_true", help="Whether to save the persistence plots of the waveforms.")

    args = argp.parse_args()
    runs = args.runs
    dettype = args.dettype
    datadir = args.datadir
    nwaveforms = args.nwaveforms
    outputdir = args.outputdir
    cutyaml = args.cutyaml
    saveplots = args.saveplots

    dettype = "membrane" if dettype == "m" else "cathode"

    for run in runs:
        main(run, dettype, datadir, nwaveforms, outputdir=outputdir, cutyaml=cutyaml, saveplots=saveplots)
