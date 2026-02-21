    
from copy import deepcopy
from dataclasses import asdict
from time import sleep
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from pathlib import Path
from typing import Dict, List, Union
import yaml
from glob import glob
import argparse
import os
import re


from waffles.data_classes.EasyWaveformCreator import EasyWaveformCreator
from waffles.np02_utils.PlotUtils import matplotlib_plot_WaveformSetGrid
from waffles.np02_utils.AutoMap import dict_endpoints_channels_list, dict_module_to_uniqch, dict_uniqch_to_module, strUch, getModuleName
from waffles.np02_utils.AutoMap import ordered_channels_cathode, ordered_channels_membrane, ordered_modules_cathode, ordered_modules_membrane
from waffles.np02_utils.load_utils import ch_read_template, ch_show_avaliable_template_folders
from waffles.utils.utils import print_colored
from waffles.utils.numerical_utils import error_propagation


from ConvFitterVDWrapper import ConvFitterVDWrapper
from DeconvFitterVDWrapper import DeconvFitterVDWrapper
from utils import DEFAULT_RESPONSE, DEFAULT_TEMPLATE, PATH_XE_AVERAGES, PATH_XE_OUTPUTS, DEFAULT_ANA_NAME
from utils import make_standard_analysis_name, list_of_ints
from utils_plot import plot_fit
from utils_conv import ConvFitParams, process_convfit
from utils_deconv import DeconvFitParams, process_deconvfit

import mplhep
mplhep.style.use(mplhep.style.ROOT)
plt.rcParams.update({'font.size': 20,
                     'grid.linestyle': '--',
                     'axes.grid': True,
                     'figure.autolayout': True,
                     'figure.figsize': [14,6]
                     })

def retrieve_responses(response_folder:Path, output={}, chinfo={}):

    if not response_folder.is_dir():
        # Trying to add user name to the path, in case the response folder was created with the -user tag
        userslogin = os.getlogin()
        therunfolder = response_folder.name
        theresponsefolder = response_folder.parent 
        response_folder_user = theresponsefolder.parent / f"{theresponsefolder.name}-{userslogin}" / therunfolder
        if not response_folder_user.is_dir():
            raise NotADirectoryError(f"{response_folder.as_posix()} and {response_folder_user.as_posix()} are not valid directories.")
        else:
            print_colored(f"\n\nWarning: {response_folder.as_posix()} is not a valid directory.\nUsing {response_folder_user.as_posix()} instead.\n\n", 'WARNING')
            response_folder = response_folder_user
            sleep(0)

    # file format is like this: average-endpoint-107-ch-0.txt
    files_averages = [ Path(x) for x in glob(f"{response_folder.as_posix()}/*.txt")]

    endpoint = 0
    list_ordered_chs = []
    tmpoutput = {}
    for file in files_averages:
        fname_fragments = file.name.split('-')
        ep = int(fname_fragments[2])
        ch = int(fname_fragments[4].split('.')[0])
        tmpoutput[ep] = tmpoutput.get(ep, {})
        tmpoutput[ep][ch] = np.loadtxt(file, dtype=np.float32)

        chinfo[ep] = chinfo.get(ep, {})
        chinfo[ep][ch] = {
            'counts': 0,
            'first_time': 0,
        }
        endpoint = ep
        if ep == 106:
            list_ordered_chs = ordered_channels_cathode
        else:
            list_ordered_chs = ordered_channels_membrane
    
    output[endpoint] = { k: tmpoutput[endpoint][k] for k in list_ordered_chs if k in tmpoutput[endpoint].keys() }


    readmefile = response_folder / "README.md"
    if not readmefile.exists():
        raise FileNotFoundError(f"README.md file not found in {response_folder.as_posix()}.\n\
                                This file is required to retrieve the number of waveforms averaged for each channel.\n\
                                If neeeded, you can create an empty README.md file, and then number of waveofrms will be set to 1 for all channels.")

    # pattern to match:
    # M1(1) 107-47: nwaveforms 46961 timestamp 109892117694173128 ticks
    pattern = r'(\S+)\s+(\d+)-(\d+):\s+nwaveforms\s+(\d+)\s+timestamp\s+(\d+)'
    with open(readmefile, 'r') as f:
        lines = f.readlines()
        lines = [ l.strip() for l in lines if l[0] in ["C", "M"]]
        for line in lines:
            match = re.match(pattern, line)
            if match:
                ep = int(match.group(2))
                ch = int(match.group(3))
                nwaveforms = int(match.group(4))
                timestamp = int(match.group(5))
                if ep in chinfo and ch in chinfo[ep]:
                    chinfo[ep][ch]['counts'] = nwaveforms
                    chinfo[ep][ch]['first_time'] = timestamp


    # Checking if all counts are zero, if all are zero... ok
    # If some are zero and some are not, print a warning that the README.md file might be incomplete or not properly formatted
    all_zero = True
    for ep, chs in chinfo.items():
        for ch, info in chs.items():
            if info['counts'] != 0:
                all_zero = False
                break
    if not all_zero:
        haszeros = False
        for ep, chs in chinfo.items():
            for ch, info in chs.items():
                if info['counts'] == 0:
                    haszeros = True
                    print(f"Warning: Count for endpoint {ep} channel {ch} is zero. This might indicate that the README.md file is incomplete or not properly formatted.")
        if haszeros:
            raise ValueError("Some counts are zero. Please check the README.md file in the response folder.")

def write_output(outputdir:Path, cfit:dict[int, dict[int,ConvFitterVDWrapper]], chinfo:dict[int, dict[int,dict]], response:str, template:str, allparams:dict, run:int, method:str, gridfigs:dict[str, Figure]):
    outputdir.mkdir(parents=True, exist_ok=True)
    outputdir.chmod(0o775)
    outputdir.parent.chmod(0o775)

    for ep, chs in cfit.items():
        for ch, cfitch in chs.items():
            # Retrieving all results
            modulename = getModuleName(ep, ch)
            submodulename = modulename[3]
            moduletype = modulename[:2]
            nselected = chinfo[ep][ch]['counts']
            first_time = chinfo[ep][ch]['first_time']
            params_name_to_save = ['A', 'fp', 'fs', 't1', 't3', 'td']
            params_to_save = {}
            errors = {}
            for param_name in params_name_to_save:
                if param_name in cfitch.m.parameters:
                    params_to_save[param_name] = cfitch.m.params[param_name].value
                    errors[param_name] = cfitch.m.params[param_name].error

            params_to_save['fs'] = params_to_save['fs'] if 'fs' in cfitch.m.parameters else 1-params_to_save['fp']
            params_to_save['td'] = params_to_save['td'] if 'td' in cfitch.m.parameters else 0
            errors['fs'] = errors['fs'] if 'fs' in cfitch.m.parameters else error_propagation(1,0, params_to_save['fp'], errors['fp'],"sub")
            errors['td'] = errors['td'] if 'td' in cfitch.m.parameters else 0
            
            # Saving results
            with open( outputdir / f"{method}fit_output_run{run:06}_{moduletype}_{submodulename}_{ep}_{ch}.txt", 'w') as f:
                f.write(f"# timestamp[ticks] run ep ch A errA fp errfp fs errfs t1[ns] errt1[ns] t3[ns] errt3[ns] td[ns] errtd[ns] nselected chi2\n")
                f.write(f"{first_time} ")
                f.write(f"{run} {ep} {ch} ")
                for pname in params_name_to_save:
                    f.write(f"{params_to_save[pname]} {errors[pname]} ")
                f.write(f"{nselected} {cfitch.chi2}\n")
            
            # Saving plots
            figplt = cfitch.plot(newplot=True)
            figplt.legend()
            figplt.title(f"{modulename}: {ep}-{ch}")
            figplt.savefig( outputdir / f"{method}fit_plot_run{run:06d}_{moduletype}_{submodulename}_ep{ep}_ch{ch}.png" )
            figplt.close()

            # Saving parameters used:
            with open( outputdir / f"{method}fit_params_run{run:06d}_{moduletype}_{submodulename}_ep{ep}_ch{ch}.yaml", 'w') as f:

                allparams_clean = allparams.copy()
                slice_template = allparams_clean.pop('slice_template')
                slice_response = allparams_clean.pop('slice_response')
                allparams_clean['slice_template'] = {
                    't0': slice_template.start,
                    'tf': slice_template.stop
                }
                allparams_clean['slice_response'] = {
                    't0': slice_response.start,
                    'tf': slice_response.stop
                }

                yaml.dump(allparams_clean, f, sort_keys=False)
    for detector, fig in gridfigs.items():
        fig.savefig( outputdir / f"{method}fit_grid_run{run:06d}_{detector}.png" )
        plt.close()




def main(run:int = 39510,
         rootdir:Union[Path,str] = PATH_XE_AVERAGES,
         response:str = DEFAULT_RESPONSE,
         template:str = DEFAULT_TEMPLATE,
         outputdir:Union[Path,str] = "./",
         analysisname:str = DEFAULT_ANA_NAME,
         analysisparams:str = "params_conv.yaml",
         channels:list = [],
         blacklist:list = [],
         cathode:bool = True,
         membrane:bool = True,
         method:str = "conv",
         dryrun:bool = False
         ):

    outputdir = Path(outputdir)
    rootdir = Path(rootdir)

    if outputdir.absolute().as_posix() == Path(PATH_XE_AVERAGES).as_posix():
        raise Exception(f"Output directory cannot be the same as the response directory ({PATH_XE_AVERAGES}).")

    extra_folder = "deconvolution" if method == "deconv" else "convolution"

    if outputdir.absolute().as_posix() == Path(PATH_XE_OUTPUTS).as_posix():
        analysisname = make_standard_analysis_name(outputdir, analysisname)

    outputdir = outputdir / analysisname / extra_folder 

    response_folder_cathode = rootdir / response / f'run{run:06d}_cathode_response'
    response_folder_membrane = rootdir / response / f'run{run:06d}_membrane_response'

    fileparams = Path(analysisparams)

    
    allparamsClass = ConvFitParams(response, template) if method == "conv" else DeconvFitParams(response, template)
    if not fileparams.is_file():
        print_colored(f"Warning: {fileparams.as_posix()} is not a valid file. Using default parameters for the convolution fit.", 'WARNING')
    else:
        allparamsClass.update_values_from_yaml(fileparams, response, template)

    allparams:dict = allparamsClass.__dict__


    templates = ch_read_template(template_folder=template)

    endpoints_to_process = []
    if cathode: 
        endpoints_to_process += [ 106 ]
    if membrane:
        endpoints_to_process += [ 107 ]

    gridcathode = cathode and len(channels) == 0
    gridmembrane = membrane and len(channels) == 0

    dict_ep_ch_user_want = {}
    if len(channels) == 0: # Creates a list with all channels, excluding the ones in the blacklist
        for ep in endpoints_to_process:
            chlist = dict_endpoints_channels_list[ep]
            if ep not in endpoints_to_process:
                continue
            epchcombos = [ ep*100 + ch for ch in chlist ]
            channels += [ epc for epc in epchcombos if epc not in blacklist ]

    for ch in channels: 
        ep = ch // 100
        chnum = ch % 100
        if ep*100 + chnum in blacklist:
            continue
        dict_ep_ch_user_want[ep] = dict_ep_ch_user_want.get(ep, []) + [chnum]

    cathode = cathode and 106 in dict_ep_ch_user_want.keys()
    membrane = membrane and 107 in dict_ep_ch_user_want.keys()

    responses = {}
    chinfo = {}
    ep_to_analyze = []
    if cathode:
        retrieve_responses(response_folder_cathode, responses, chinfo)
        ep_to_analyze += [ 106 ] 
    if membrane:
        retrieve_responses(response_folder_membrane, responses, chinfo)
        ep_to_analyze += [ 107 ]


    # I can only analyze channels for which I have both the response and the
    # template, so I create a dictionary with the channels that have both
    # signals, and then I create a ConvFitterVDWrapper | DeconvFitterVDWrapper for each of them
    cfit = {}
    dict_ep_ch_with_both_signals = {}
    for ep, ch_wvf in responses.items():
        if ep not in ep_to_analyze:
            continue
        for ch, _ in ch_wvf.items():
            if ep in templates.keys():
                cfit[ep] = cfit.get(ep, {})
                if ch in templates[ep].keys():
                    if ep not in dict_ep_ch_user_want.keys():
                        continue
                    if ch not in dict_ep_ch_user_want[ep]:
                        continue
                    dict_ep_ch_with_both_signals[ep] = dict_ep_ch_with_both_signals.get(ep, []) + [ch]
                    if method == "conv":
                        cfit[ep][ch] = ConvFitterVDWrapper(**allparams['cfitparams']) # type: ignore
                    elif method == "deconv":
                        cfit[ep][ch] = DeconvFitterVDWrapper(**allparams['deconvparams'])
                    else:
                        raise ValueError(f"Invalid method {method}. Valid methods are 'conv' and 'deconv'.")


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
        print(f"Channels: {channels}")
        print(f"Blacklist: {blacklist}")
        print(f"Cathode: {cathode}")
        print(f"Membrane: {membrane}")
        print(f"Parameters: {allparams}")
        ch_show_avaliable_template_folders()

        if len(dict_ep_ch_with_both_signals) != 0:
            print("The following channels will be processed:")
            for ep, chs in dict_ep_ch_with_both_signals.items():
                for ch in chs:
                    print(f"Endpoint {ep} Channel {ch}: {getModuleName(ep, ch)}")
        else:
            print_colored("No channels will be processed. Please check the parameters and the availability of the response and template files.", 'ERROR')
        return

    for ep, chs in dict_ep_ch_with_both_signals.items():
        for ch in chs:
            if method == "conv":
                process_convfit(
                    ep,
                    ch,
                    responses[ep][ch],
                    templates[ep][ch],
                    cfit[ep][ch],
                    scan=allparams['scan'],
                    print_flag=allparams['print_flag'],
                    slice_template=allparams['slice_template'],
                    slice_response=allparams['slice_response']
                )
            elif method == "deconv":
                process_deconvfit(
                    ep,
                    ch,
                    responses[ep][ch],
                    templates[ep][ch],
                    cfit[ep][ch],
                    print_flag=allparams['print_flag'],
                    slice_template=allparams['slice_template'],
                    slice_response=allparams['slice_response']
                )
    
    # Create empty waveform
    wfset = EasyWaveformCreator.create_WaveformSet_dictEndpointCh(dict_endpoint_ch=dict_endpoints_channels_list)
    funcparams = {
        'cfit':cfit,
        'dofit':False,
        'responses': responses,
        'templates': templates,
        'verbose': False,
    }
    gridfigs = {}

    if gridcathode:
        fig, _ = matplotlib_plot_WaveformSetGrid(wfset, detector=ordered_modules_cathode, plot_function=plot_fit, func_params=funcparams, cols=4, figsize=(32,32))
        gridfigs['cathode'] = deepcopy(fig)
    if gridmembrane:
        fig, _ = matplotlib_plot_WaveformSetGrid(wfset, detector=ordered_modules_membrane, plot_function=plot_fit, func_params=funcparams, cols=4, figsize=(32,32))
        gridfigs['membrane'] = deepcopy(fig)



    write_output(outputdir, cfit, chinfo, response, template, allparams, run, method, gridfigs)


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



if __name__ == "__main__":
    argp = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Fit a template waveform to a response waveform using a convolution model of LAr/Xe response.")
    argp.add_argument("--method", required=True, type=str, help="Method to use for fitting, either 'conv' for convolution fit or 'deconv' for deconvolution fit.")
    argp.add_argument("--runs", type=list_of_ints, default="39510", help="List of runs to process. Example: --runs 39510,39511,39512")
    argp.add_argument("--rootdir", type=str, default=PATH_XE_AVERAGES, help="Directory where the processed waveform files are located.")
    argp.add_argument("-r", "--response", type=str, default=DEFAULT_RESPONSE, help=f"Name of the analysis containing the response waveforms to fit.") 
    argp.add_argument("-t", "--template", type=str, default=DEFAULT_TEMPLATE, help=f"Name of the analysis containing the template waveforms to fit.")
    argp.add_argument("-o", "--outputdir", type=str, default="./", help="Directory where the average waveforms will be saved.")
    argp.add_argument("-a", "--analysisname", type=str, default=DEFAULT_ANA_NAME, help="Name of the analysis, used to create a subdirectory in the output directory.")
    argp.add_argument("--analysisparams", type=str, default="params_conv.yaml", help="Parameters to be loaded from a yaml file")
    argp.add_argument("--channels", nargs="+", type=int, default=[], help="List of channels to process. If not specified, all channels will be processed.\
        \n\tFormat: endpoint+channel, e.g. 1060 for endpoint 106 channel 0.")
    argp.add_argument("--blacklist", nargs="+", type=int, default=[], help="List of channels to exclude from processing. Format: endpoint+channel, e.g. 10600 for endpoint 106 channel 0.")
    argp.add_argument("-c", "--cathode", action="store_true", help="If set, cathode channels will be processed. If both -c and -m are not set, all channels will be processed.")
    argp.add_argument("-m", "--membrane", action="store_true", help="If set, membrane channels will be processed. If both -c and -m are not set, all channels will be processed.")
    argp.add_argument("--dryrun", action="store_true", help="If set, the script will only print the parameters and not execute the main function.")

    args = argp.parse_args()

    runs = args.runs
    rootdir = args.rootdir
    response = args.response
    template = args.template
    outputdir = args.outputdir
    analysisname = args.analysisname
    analysisparams = args.analysisparams
    channels = args.channels
    blacklist = args.blacklist
    cathode = args.cathode
    membrane = args.membrane
    method = args.method
    dryrun = args.dryrun

    if method == "deconv" and analysisparams == "params_conv.yaml":
        raise ValueError("Default parameters file for convolution fit is 'params_conv.yaml'. Please specify it with --analysisparams or choose a different file.")

    if method not in ["conv", "deconv"]:
        print_colored(f"Error: invalid method {method}. Valid methods are 'conv' and 'deconv'.", 'ERROR')
        exit(0)

    if not cathode and not membrane:
        cathode = True
        membrane = True
    for run in runs:
        if not dryrun:
            print(f"Prossessing run {run} ...")
        main(run, rootdir, response, template, outputdir, analysisname, analysisparams, channels, blacklist, cathode, membrane, method, dryrun)


