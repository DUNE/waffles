import numpy as np
from pathlib import Path
from typing import List
import os
from glob import glob
from importlib.resources import files

from waffles.data_classes.ChannelWs import ChannelWs
from waffles.np02_utils.AutoMap import getModuleName, expand_modules 


def __template_module_name(ep, ch):
    # Build reverse lookup: channel integer -> module name formatted as C6_1
    module_name = getModuleName(ep, ch)
    label = module_name.replace("(", "_").replace(")", "")
    return label 

def __remove_existing_templates(wfsetch: dict[int, dict[int, ChannelWs]], template_outputdir: str, detector: List[str], dry_run: bool = False):

    # Delete stale templates only if a DIFFERENT run exists for the same channel
    removed = []

    ep = list(wfsetch.keys())[0]

    for ch in wfsetch[ep].keys():
        run_num = list(wfsetch[ep][ch].runs)[0]
        module = getModuleName(ep, ch)
        if module in detector:
            ch_label = __template_module_name(ep, ch)
            stale = glob(template_outputdir + f"template_*_{ch_label}.txt")
            for old_file in stale:
                # Extract run number from filename e.g. "template_40807_C6_1.txt" -> 40807
                old_run = int(os.path.basename(old_file).split("_")[1])
                if old_run != run_num: # only delete if different run
                    if not dry_run:
                        os.remove(old_file)
                        if os.path.exists(old_file):
                            raise Exception(f"Failed to remove stale template: {old_file}")


                    removed.append(os.path.basename(old_file))
    for f in sorted(removed, key=lambda x: x.split("_", 2)[2]):
        if dry_run:
            print(f"Would remove stale template: {f}")
        else:
            print(f"Removed stale template: {f}")

def __generate_templates_info(wfsetch: dict[int, dict[int, ChannelWs]],
                              info_file:str,
                              detector: List[str],
                              peaks: dict[tuple[int,int], float]
                              ) -> dict[tuple[str,int], tuple[str,int,float]]:

    data = {}

    ep = list(wfsetch.keys())[0]
    epchannels_new = {}
    for ch in wfsetch[ep].keys():
        run_num = list(wfsetch[ep][ch].runs)[0]
        module = getModuleName(ep, ch)
        if module in detector:
            wfs = wfsetch[ep][ch]
            endpoint_ch = f"{ep}-{ch}"
            peak_ch = peaks[(ep,ch)]
            data[(endpoint_ch, run_num)] = (module, len(wfs.waveforms), peak_ch)
            epchannels_new[endpoint_ch] = 1

    if not os.path.exists(info_file):
        return data

    with open(info_file, "r") as f:
        lines = f.readlines()
        lines = [ line for line in lines if line.strip() ] 
        for line in lines[1:]:  # Skip header line
            # Split by comma to get the main parts
            parts = line.split(", ")
            if len(parts) >= 4:
                # The first part contains "Module endpoint-channel"
                module_and_endpointch = parts[0].strip()
                # Split to separate module from endpoint-channel
                split_parts = module_and_endpointch.rsplit(" ", 1)
                split_parts = [ p.strip() for p in split_parts ]
                if len(split_parts) == 2:
                    module = split_parts[0]
                    endpoint_ch = split_parts[1]
                    if endpoint_ch in epchannels_new:
                        continue
                    run_num = int(parts[1])
                    waveforms = int(parts[2])
                    peak_ch = float(parts[3])
                    if (endpoint_ch, run_num) not in data:
                        data[(endpoint_ch, run_num)] = (module, waveforms, peak_ch)

    return data


def __yieldTemplateInfo(existing_data: dict[tuple[str,int], tuple[str,int,float]]):
    for (endpoint_ch, run_num), (module, waveforms, peak_ch) in sorted(existing_data.items(), key=lambda x: x[1][0]):
        yield f"{module} {endpoint_ch}, {run_num}, {waveforms}, {peak_ch:.1f}"


def __savewaveforms(wfsetch: dict[int, dict[int, ChannelWs]],
                    template_outputdir: str,
                    detector: List[str],
                    templates: dict[int, dict[int, np.ndarray]],
                    dry_run: bool = False
                    ):
    ep = list(wfsetch.keys())[0]
    for ch in wfsetch[ep].keys():
        module_name = getModuleName(ep, ch)
        if module_name not in detector:
            continue
        run = list(wfsetch[ep][ch].runs)[0]
        # C1(1)
        if module_name.startswith ("P"):
            module_for_title = module_name[0]
            channel_for_title = module_name[1:]
        else:
            module_for_title = module_name[:2]
            channel_for_title = module_name[3]
        
        filename = f"template_{run}_{module_for_title}_{channel_for_title}.txt"
        
        filepath = os.path.join(template_outputdir, filename)
        
        if dry_run:
            print(f"Dry run: Would save template for module {module_name}: {ep}-{ch} to {filepath}")
        else:
            np.savetxt(
                filepath,
                templates[ep][ch],
                fmt="%.9e"
            )

            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                print(f"File saved: {filepath}")
            else:
                print(f"Failed to save: {filepath}")

def save_templates(wfsetch: dict[int, dict[int, ChannelWs]],
                   template_name:str,
                   cutyaml:str,
                   detector: List[str],
                   templates: dict[int, dict[int, np.ndarray]],
                   peaks: dict[tuple[int,int], float],
                   dry_run: bool = False
                   ):
    if not template_name:
        print("No template name provided, skipping template saving.")
        return

    if len(wfsetch.keys()) > 1:
        print(f"Multiple endpoints found in the waveform set. Expected only one. Found endpoints: {list(wfsetch.keys())}")
        return
    ep = list(wfsetch.keys())[0]

    wafflespath = str(files("waffles"))
    template_outputdir = wafflespath + f"/np02_data/templates/{template_name}/"

    available_modules = [ getModuleName(ep, ch) for ch in wfsetch[ep].keys() ]
    detector = expand_modules(detector, available_modules)


    __remove_existing_templates(wfsetch, template_outputdir, detector, dry_run=dry_run)

    dettype = "membrane" if ep == 107 else "cathode" if ep == 106 else "pmt"

    info_file = template_outputdir + f"wfdetails_{dettype}.info"

    existing_data = __generate_templates_info(wfsetch, info_file, detector, peaks)

    if dry_run and existing_data:
        print("Dry run mode: Existing templates info")
        for template_info in __yieldTemplateInfo(existing_data):
            print(template_info)

    if not dry_run:
        Path(template_outputdir).mkdir(parents=True, exist_ok=True)
        # Write back to file
        with open(info_file, "w") as f:
            f.write("Module, endpoint-channel, run number, # of waveforms, amplitude(ADC)\n")
            for template_info in __yieldTemplateInfo(existing_data):
                f.write(f"{template_info}\n")

        with open(template_outputdir+"cuts_used.yaml", "w") as f:
                with open(cutyaml, "r") as fcuts:
                    f.write(fcuts.read())

    __savewaveforms(wfsetch, template_outputdir, detector, templates, dry_run=dry_run)
