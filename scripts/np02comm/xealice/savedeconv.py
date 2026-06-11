import numpy as np
from pathlib import Path
from typing import Union
import argparse


from waffles.np02_utils.AutoMap import dict_endpoints_channels_list, dict_module_to_uniqch, dict_uniqch_to_module, strUch, getModuleName
from waffles.np02_utils.load_utils import ch_read_template, ch_show_avaliable_template_folders
from waffles.utils.utils import print_colored


from DeconvFitterVDWrapper import DeconvFitterVDWrapper
from utils import DEFAULT_RESPONSE, DEFAULT_TEMPLATE
from utils import list_of_ints
from utils_deconv import DeconvFitParams, process_deconvfit
from larfitter import ProcessStatus, check_processed_modules, __print_summary

def __write_outputs(averages, readmefile, cutyaml, outputdir):

    outputdir.mkdir(parents=True, exist_ok=True)
    outputdir.chmod(0o755)
    outputdir.parent.chmod(0o755)

    with open(outputdir / "README.md", "w", ) as f:
        with open(readmefile, "r") as freadme:
            f.write(freadme.read())

    # Saves the yaml cuts used to select the waveforms as cuts_used.yaml in the same directory
    with open(outputdir / "cuts_used.yaml", "w") as f:
        with open(cutyaml, "r") as fcuts:
            f.write(fcuts.read())

    # Save the average waveforms as numpy arrays
    # The filename is "average_endpoint{endpoint}_ch{ch}.txt"
    for ep, chs in averages.items():
        for ch, avg in chs.items():
            np.savetxt(outputdir / f"average-endpoint-{ep}-ch-{ch}.txt", avg)
    
def main(run:int = 39510,
         rootdir:Union[Path,str] = "./data/",
         response:str = DEFAULT_RESPONSE,
         template:str = DEFAULT_TEMPLATE,
         analysisname:str = "",
         analysisparams:str = "params_deconv.yaml",
         dettype:str = "cathode",
         force:bool = False,
         dryrun:bool = False
         ) -> ProcessStatus:


    rootdir = Path(rootdir)
    outputdir = rootdir 


    if analysisname == "":
        analysisname = response.split("-")[0] + '_deconvolved'



    fileparams = Path(analysisparams)
    allparamsClass = DeconvFitParams(response, template)
    if not fileparams.is_file():
        print_colored(f"Warning: {fileparams.as_posix()} is not a valid file. Using default parameters for the convolution fit.", 'WARNING')
    else:
        allparamsClass.update_values_from_yaml(fileparams, response, template)

    allparams:dict = allparamsClass.__dict__

    templates = ch_read_template(template_folder=template)


    responses = {}
    chinfo = {}


    response_folder = rootdir / response / f'run{run:06d}_{dettype}_response'
    outputdir = outputdir / analysisname / f'run{run:06d}_{dettype}_response'

    if outputdir.exists() and not force:
        print(f"Directory {outputdir.absolute().as_posix()} already exists. Skipping run {run}. Use --force to overwrite.")
        return ProcessStatus.SKIPPED

    status = ProcessStatus.NOT_ASKED
    
    status = check_processed_modules(dettype, run, outputdir, force, response_folder, responses, chinfo)
    if status != ProcessStatus.COMPLETED:
        return status


    dict_ep_ch_user_want = {}
    for ep, rch in responses.items():
        for ch in rch.keys():
            dict_ep_ch_user_want[ep] = dict_ep_ch_user_want.get(ep, []) + [ch]


    # I can only analyze channels for which I have both the response and the
    # template, so I create a dictionary with the channels that have both
    # signals, and then I create a ConvFitterVDWrapper | DeconvFitterVDWrapper for each of them
    cfit = {}
    dict_ep_ch_with_both_signals = {}
    for ep, ch_wvf in responses.items():
        for ch, _ in ch_wvf.items():
            if ep in templates.keys():
                cfit[ep] = cfit.get(ep, {})
                if ch in templates[ep].keys():
                    if ep not in dict_ep_ch_user_want.keys():
                        continue
                    if ch not in dict_ep_ch_user_want[ep]:
                        continue
                    dict_ep_ch_with_both_signals[ep] = dict_ep_ch_with_both_signals.get(ep, []) + [ch]
                    cfit[ep][ch] = DeconvFitterVDWrapper(**allparams['deconvparams'])


    if len(dict_ep_ch_with_both_signals) == 0:
        dryrun = True
    if dryrun:
        print("Dry run mode. Parameters:")
        print(f"Run: {runs}")
        print(f"Current run: {run}")
        print(f"Root directory: {rootdir}")
        print(f"Response folder name: {response}")
        print(f"Template folder name: {template}")
        print(f"Output directory: {outputdir}")
        print(f"Parameters: {allparams}")
        ch_show_avaliable_template_folders()

        if len(dict_ep_ch_with_both_signals) != 0:
            print("The following channels will be processed:")
            for ep, chs in dict_ep_ch_with_both_signals.items():
                for ch in chs:
                    print(f"Endpoint {ep} Channel {ch}: {getModuleName(ep, ch)}")
        else:
            print_colored("No channels will be processed. Please check the parameters and the availability of the response and template files.", 'ERROR')
       
        return status


    averages = {}
    for ep, chs in dict_ep_ch_with_both_signals.items():
        averages[ep] = averages.get(ep, {})
        for ch in chs:
            process_deconvfit(
                ep,
                ch,
                responses[ep][ch],
                templates[ep][ch],
                cfit[ep][ch],
                print_flag=allparams['print_flag'],
                slice_template=allparams['slice_template'],
                slice_response=allparams['slice_response'],
                dofit=False
            )
            averages[ep][ch] = cfit[ep][ch].deconvolved

    print("Saving output...")
    readmefile = response_folder / "README.md"
    cutyaml = response_folder / "cuts_used.yaml"
    __write_outputs(averages, readmefile, cutyaml, outputdir)


    # Warning the user about the were requested but there is no response or
    # template for them, so they will be skipped in the analysis
    for ep, chs in dict_ep_ch_user_want.items():
        for ch in chs:
            if ep not in dict_ep_ch_with_both_signals.keys() or ch not in dict_ep_ch_with_both_signals[ep]:
                print_colored(f"Warning: {getModuleName(ep, ch)} {ep}-{ch} was requested but there is no response or template for it. It was skipped in the analysis.", 'WARNING')
                if ep not in responses.keys() or ch not in responses[ep].keys():
                    print_colored(f"\tReason: response not found.", 'WARNING')
                if ep not in templates.keys() or ch not in templates[ep].keys():
                    print_colored(f"\tReason: template not found.", 'WARNING')

    
    return status

if __name__ == "__main__":
    argp = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Fit a template waveform to a response waveform using a convolution model of LAr/Xe response.")
    argp.add_argument("--runs", type=list_of_ints, default="39510", help="List of runs to process. Example: --runs 39510,39511,39512")
    argp.add_argument("--rootdir", type=str, default="./data/", help="Directory where the processed waveform files are located.")
    argp.add_argument("-r", "--response", type=str, default=DEFAULT_RESPONSE, help=f"Name of the analysis containing the response waveforms to fit.") 
    argp.add_argument("-t", "--template", type=str, default=DEFAULT_TEMPLATE, help=f"Name of the analysis containing the template waveforms to fit.")
    argp.add_argument("-a", "--analysisname", type=str, default="", help="Name of the analysis, used to create a subdirectory in the output directory.")
    argp.add_argument("--analysisparams", type=str, default="params_deconv.yaml", help="Parameters to be loaded from a yaml file")
    argp.add_argument("-c", "--cathode", action="store_true", help="If set, cathode channels will be processed. If both -c and -m are not set, all channels will be processed.")
    argp.add_argument("-m", "--membrane", action="store_true", help="If set, membrane channels will be processed. If both -c and -m are not set, all channels will be processed.")
    argp.add_argument("-p", "--pmt", action="store_true", help="If set, pmt channels will be processed. If -p is set, -c and -m will be ignored.")
    argp.add_argument("--force", action="store_true", help="If set, the script will overwrite existing output directories. Use with caution.")
    argp.add_argument("--dryrun", action="store_true", help="If set, the script will only print the parameters and not execute the main function.")

    args = argp.parse_args()

    runs = args.runs
    rootdir = args.rootdir
    response = args.response
    template = args.template
    analysisname = args.analysisname
    analysisparams = args.analysisparams
    cathode = args.cathode
    membrane = args.membrane
    pmt = args.pmt
    force = args.force
    dryrun = args.dryrun

    if pmt:
        cathode = False
        membrane = False

    if not cathode and not membrane and not pmt:
        cathode = True
        membrane = True

    detectors = []
    if cathode:
        detectors.append("cathode")
    if membrane:
        detectors.append("membrane")
    if pmt:
        detectors.append("pmt")
        
    results: dict[int, dict[str, ProcessStatus]] = {}

    for run in runs:
        if not dryrun:
            print(f"Processing run {run} ...")
        for dettype in detectors:
            ret = main(run, rootdir, response, template, analysisname, analysisparams, dettype , force, dryrun)
            results[run] = results.get(run, {})
            results[run][dettype] = ret
    __print_summary(results, dryrun=args.dryrun)

