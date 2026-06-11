from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np
from enum import Enum, auto
from collections import defaultdict
from glob import glob
import argparse

import matplotlib.pyplot as plt
import dunestyle.matplotlib as dunestyle
from cycler import cycler

plt.rcParams.update( { "axes.prop_cycle":cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']) } )

from waffles.data_classes.UniqueChannel import UniqueChannel
from waffles.np02_utils.AutoMap import expand_modules, dict_module_to_uniqch, getModuleName
from waffles.utils.utils import print_colored

from utils import PATH_XE_AVERAGES, PATH_XE_OUTPUTS, list_of_ints, make_standard_analysis_name
from GlobalFitter import GlobalFitter
from monitoring.utils_monitoring import add_ppm_per_run, load_database
from larfitter import check_processed_modules, ProcessStatus


def genkeys(detectors:list):
    if len(detectors) == 0:
        return []
    modules = expand_modules(detectors, list(dict_module_to_uniqch.keys()))
    keys = []
    for m in modules:
        if m[:2] in ["M1", "M2", "M3", "M4" ]:
            continue
        uch:UniqueChannel = dict_module_to_uniqch[m]
        ep, ch = uch.endpoint, uch.channel
        keys.append((ep,ch))

    return keys

def save_plot(run, gFit, keys, suffix, label, outputdir):
    if not keys:
        return
    fig, axs = gFit.plot_results(keys=keys, ncols=2, logscale='log' in suffix)
    for ax in np.ravel(axs):
        if 'log' in suffix:
            ax.set_ylim(1e-3, None)
        # ax.set_xlim(55*16,120*16)
        # ax.set_ylim(-0.5,0.5)
    plt.savefig(outputdir / f'globalfit_plot_run{run:06d}_{label}{suffix}.png')
    plt.close(fig)
    
def write_output(gFit, results, run:int, list_of_modules:list, outputdir:Path, savefig:bool, chinfo: dict[int, dict[int, dict]]):
    outputdir.mkdir(parents=True, exist_ok=True)
    outputdir.chmod(0o755)
    
    logscale=True
    suffix = "_log" if logscale else ""
    if savefig:
        print("Writing fit result plots...")
        save_plot(run, gFit, genkeys([ v for v in list_of_modules if v.startswith("M") ]), suffix, "membrane", outputdir=outputdir)
        save_plot(run, gFit, genkeys([ v for v in list_of_modules if v.startswith("C") ]), suffix, "cathode", outputdir=outputdir)
        save_plot(run, gFit, genkeys([ v for v in list_of_modules if v.startswith("P") ]), suffix, "pmt", outputdir=outputdir)

    tf = results['tf'].value*1e-3
    errtf = results['tf'].error*1e-3
    tta = results['tta'].value*1e-3
    errtta = results['tta'].error*1e-3
    ttx = results['ttx'].value*1e-3
    errttx = results['ttx'].error*1e-3
    chi2 = gFit.minuit.fmin.reduced_chi2 if gFit.minuit and gFit.minuit.fmin else np.nan
    firsttime = min( chinfo.get(ep, {}).get(ch, {}).get('first_time', np.inf) for (ep, ch) in results.modules.keys() )


    with open( outputdir / f"global_fit_output_run{run:06d}.csv", "w") as f:
        f.write(f"timestamp[ticks],run,ep,ch,A,errA,B,errB,C,errC,t0,errt0,sigma,errsigma,tf,errtf,tta,errtta,ttx,errttx,nselected,chi2,nmodules\n")
        for (ep, ch) in results.modules.keys():
            key = (ep, ch)
            A = results.modules[key]['A'].value
            errA = results.modules[key]['A'].error
            B = results.modules[key]['B'].value
            errB = results.modules[key]['B'].error
            C = results.modules[key]['C'].value
            errC = results.modules[key]['C'].error
            t0 = results.modules[key]['t0'].value
            errt0 = results.modules[key]['t0'].error
            sigma = results.modules[key]['sigma'].value
            errsigma = results.modules[key]['sigma'].error
            nselected = chinfo.get(ep, {}).get(ch, {}).get('counts', np.nan)
            nmodules = len(results.modules)
            f.write(f"{firsttime},{run},{ep},{ch},{A},{errA},{B},{errB},{C},{errC},{t0},{errt0},{sigma},{errsigma},{tf},{errtf},{tta},{errtta},{ttx},{errttx},{nselected},{chi2},{nmodules}\n")
            

def __print_summary(results: dict[int, ProcessStatus], dryrun: bool) -> None:

    grouped = defaultdict(list)
    for run, status in results.items():
        grouped[status].append(run)

    if dryrun:
        print_colored(
            "Runs that will be processed: " + ",".join( str(i) for i in grouped[ProcessStatus.COMPLETED] ),
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

def main(run: int,
         data_folder:Path = Path(PATH_XE_AVERAGES),
         deconvolved_name:str = 'average_waveforms_few_cuts_deconvolved-hvieirad',
         list_of_modules:list[str] = ["M", "C"],
         outputdir:Union[str,Path] = PATH_XE_OUTPUTS,
         analysisname: str = "global_analysis_results",
         force:bool = False,
         savefig:bool = True,
         ppm: float = 0.0,
         dryrun:bool = False
        ):


    outputdir = Path(outputdir)
    analysisname = make_standard_analysis_name(outputdir, analysisname)
    outputdir = outputdir / analysisname

    if glob(str(outputdir / f"global_fit_output_run{run:06d}.csv") ) and not force:
        print_colored(f"Output for run {run} already exists. Skipping. Use --force to overwrite.", "WARNING")
        return ProcessStatus.SKIPPED

    dettypes = []
    dettypes.append('membrane') if any(m.startswith("M") for m in list_of_modules) else None
    dettypes.append('cathode') if any(m.startswith("C") for m in list_of_modules) else None
    dettypes.append('pmt') if any(m.startswith("P") for m in list_of_modules) else None

    responses = {}
    chinfo = {}
    for dettype in dettypes:
        response_folder = data_folder / deconvolved_name / f"run{run:06d}_{dettype}_response"
        status = check_processed_modules(dettype, run, outputdir, force=True, response_folder=response_folder, responses=responses, chinfo=chinfo, loadfile=not dryrun) # Force true because I am not actually creaing it
        if status != ProcessStatus.COMPLETED:
            print_colored(f"Failed to process {dettype} for run {run}.", "ERROR")
            return status

    datasetfull = {}
    for ep, rch in responses.items():
        for ch, wf in rch.items():
            if not dryrun:
                datasetfull[(ep, ch)] = wf - np.mean(wf[:40])
            else:
                datasetfull[(ep, ch)] = wf

    keys_fit = genkeys(list_of_modules)
    dataset_fit = {k: w for k, w in sorted(datasetfull.items(), key=lambda x: getModuleName(x[0][0], x[0][1])) if k in keys_fit}

    missing_modules = [ getModuleName(ep, ch) for (ep, ch) in keys_fit if (ep, ch) not in dataset_fit ]
    if "M7(1)" in missing_modules and "M7(2)" in missing_modules:
        print_colored(f"Error: Both M7(1) and M7(2) are required for the fit. Missing modules: {', '.join(missing_modules)}", "ERROR")
        return ProcessStatus.FAILED

    if dryrun:
        print_colored(f"run: {run}", "INFO")
        print_colored(f"list_of_modules: {list_of_modules}", "INFO")
        print_colored(f"analysisname: {analysisname}", "INFO")
        print_colored(f"outputdir: {outputdir}", "INFO")
        if missing_modules:
            print_colored(f"Modules requested, but missing from dataset: {', '.join(missing_modules) }", "WARNING")
        return ProcessStatus.COMPLETED



    oneexp = True if run < 43418 else False
    gFit = GlobalFitter(datasets=dataset_fit, offset_t0=320, penalty_strength=100, error=1)


    print("Performing global fit...")
    results = gFit.minimize(fit_limit_ns=12500, oneexp=oneexp, ppm=ppm)
    if not gFit.minuit or gFit.minuit.fmin is None:
        print_colored(f"Fit failed for run {run}.", "ERROR")
        return ProcessStatus.FAILED

    write_output(gFit, results, run, list_of_modules, outputdir, savefig, chinfo)

    return ProcessStatus.COMPLETED
    

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="Run global fit on specified run and modules.")
    argument_parser.add_argument("--runs", type=list_of_ints, required=True, help="Run number(s) to process. Can be a single integer or a comma-separated list of integers.")
    argument_parser.add_argument("--modules", type=str, default="M,C", help="Comma-separated list of module prefixes to include (e.g., 'M,C' for membrane and cathode).")
    argument_parser.add_argument("--force", action="store_true", help="Force reprocessing even if output already exists.")
    argument_parser.add_argument("--analysisname", type=str, default="global_analysis_results", help="Name of the analysis for output directory.")
    argument_parser.add_argument("--outputdir", type=str, default=PATH_XE_OUTPUTS, help="Base directory for output. A subdirectory with the analysis name will be created inside this directory.")
    argument_parser.add_argument("--dryrun", action="store_true", help="If set, the script will only print the parameters and not execute the main function.")
    argument_parser.add_argument("--savefig", action="store_true", help="If set, the script will save the fit result plots.")
    args = argument_parser.parse_args()

    runs = args.runs
    list_of_modules = args.modules.split(",")
    force = args.force
    analysisname = args.analysisname

    df = load_database()
    df = add_ppm_per_run(df.drop(columns=["ppm"]))
    df = df.set_index("run")


    results: dict[int, ProcessStatus] = {}
    for run in runs:
        if run not in df.index:
            print_colored(f"Run {run} not found in database. Skipping.", "ERROR")
            results[run] = ProcessStatus.FAILED
            continue
        ppm = df.loc[run, 'ppm']
        print("----------------------------------------")
        print("Prossessing run:", run)
        status = main(run=run, list_of_modules=list_of_modules, analysisname=analysisname, force=force, outputdir=args.outputdir, savefig=args.savefig, dryrun=args.dryrun, ppm=ppm)
        results[run] = status

    __print_summary(results, dryrun=args.dryrun)
