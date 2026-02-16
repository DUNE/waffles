    
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Union
from glob import glob
import argparse
import os

from ConvFitterVDWrapper import ConvFitterVDWrapper
from waffles.data_classes.EasyWaveformCreator import EasyWaveformCreator
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.UniqueChannel import UniqueChannel
from waffles.np02_utils.AutoMap import dict_endpoints_channels_list, dict_module_to_uniqch, dict_uniqch_to_module, strUch
from waffles.np02_utils.AutoMap import ordered_channels_cathode, ordered_channels_membrane, ordered_modules_cathode, ordered_modules_membrane
from waffles.np02_utils.PlotUtils import matplotlib_plot_WaveformSetGrid
from waffles.np02_utils.load_utils import ch_read_template, ch_show_avaliable_template_folders

from utils import DEFAULT_RESPONSE, DEFAULT_TEMPLATE, PATH_XE_AVERAGES, DEFAULT_CONV_NAME, process_convfit
from waffles.utils.utils import print_colored

import mplhep
mplhep.style.use(mplhep.style.ROOT)
plt.rcParams.update({'font.size': 20,
                     'grid.linestyle': '--',
                     'axes.grid': True,
                     'figure.autolayout': True,
                     'figure.figsize': [14,6]
                     })

def retrieve_responses(response_folder:Path, output={}, counts={}):
    if not response_folder.is_dir():
        # Trying to add user name to the path, in case the response folder was created with the -user tag
        userslogin = os.getlogin()
        therunfolder = response_folder.name
        theresponsefolder = response_folder.parent 
        response_folder_user = theresponsefolder.parent / f"{theresponsefolder.name}-{userslogin}" / therunfolder
        if not response_folder_user.is_dir():
            raise NotADirectoryError(f"{response_folder.as_posix()} and {response_folder_user.as_posix()} are not valid directories.")
        else:
            response_folder = response_folder_user
            print_colored(f"\n\nWarning: {response_folder.as_posix()} is not a valid directory.\nUsing {response_folder_user.as_posix()} instead.\n\n", 'WARNING')
            sleep(2)

    # file format is like this: average-endpoint-107-ch-0.txt
    files_averages = [ Path(x) for x in glob(f"{response_folder.as_posix()}/*.txt")]
    for file in files_averages:
        fname_fragments = file.name.split('-')
        ep = int(fname_fragments[2])
        ch = int(fname_fragments[4].split('.')[0])
        output[ep] = output.get(ep, {})
        output[ep][ch] = np.loadtxt(file, dtype=np.float32)

        counts[ep] = counts.get(ep, {})
        counts[ep][ch] = np.loadtxt(file, dtype=np.float32)
    readmefile = response_folder / "README.md"
    if not readmefile.exists():
        raise FileNotFoundError(f"README.md file not found in {response_folder.as_posix()}.\n\
                                This file is required to retrieve the number of waveforms averaged for each channel.\n\
                                If neeeded, you can create an empty README.md file, and then number of waveofrms will be set to 1 for all channels.")

    with open(readmefile, 'r') as f:
        lines = f.readlines()
        lines = [ l.strip() for l in lines if l[0] in ["C", "M"]]
        for line in lines:
            data = line.split()
            chinfo = data[1].replace(":", "").split("-")
            ep = int(chinfo[0])
            ch = int(chinfo[1])
            if ep in counts and ch in counts[ep]:
                counts[ep][ch] = int(data[2])

    # Checking if all counts are zero, if all are zero... ok
    # If some are zero and some are not, print a warning that the README.md file might be incomplete or not properly formatted
    all_zero = True
    for ep, chs in counts.items():
        for ch, count in chs.items():
            if count != 0:
                all_zero = False
                break
    if not all_zero:
        haszeros = False
        for ep, chs in counts.items():
            for ch, count in chs.items():
                if count == 0:
                    haszeros = True
                    print(f"Warning: Count for endpoint {ep} channel {ch} is zero. This might indicate that the README.md file is incomplete or not properly formatted.")
        if haszeros:
            raise ValueError("Some counts are zero. Please check the README.md file in the response folder.")



def main(run:int = 39510,
         rootdir:Union[Path,str] = PATH_XE_AVERAGES,
         response:str = DEFAULT_RESPONSE,
         template:str = DEFAULT_TEMPLATE,
         outputdir:Union[Path,str] = "./",
         analysisname:str = DEFAULT_CONV_NAME,
         channels:list = [],
         blacklist:list = [],
         cathode:bool = True,
         membrane:bool = True,
         dryrun:bool = False
         ):

    outputdir = Path(outputdir)
    rootdir = Path(rootdir)
    # Ensures same standard for the analysis name, so it can be easily identified and sorted in the output directory
    analysisname = analysisname.replace("-", "_")
    if outputdir.absolute().as_posix() == Path(PATH_XE_AVERAGES).as_posix():
        analysisname = analysisname + '-' + os.getlogin()

    outputdir = outputdir / analysisname

    response_folder_cathode = rootdir / response / f'run{run:06d}_cathode_response'
    response_folder_membrane = rootdir / response / f'run{run:06d}_membrane_response'

    cfitparams = dict(
        threshold_align_template=0.27,
        threshold_align_response=0.2,
        error=10,
        dointerpolation=True,
        interpolation_factor=8,
        align_waveforms = True,
        dtime=16,
        convtype='fft',
        usemplhep=True,
    )

    templates = ch_read_template(template_folder=template)

    endpoints_to_process = []
    if cathode: 
        endpoints_to_process += [ 106 ]
    if membrane:
        endpoints_to_process += [ 107 ]

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
    counts = {}
    ep_to_analyze = []
    if cathode:
        retrieve_responses(response_folder_cathode, responses, counts)
        ep_to_analyze += [ 106 ] 
    if membrane:
        retrieve_responses(response_folder_membrane, responses, counts)
        ep_to_analyze += [ 107 ]


    # I can only analyze channels for which I have both the response and the
    # template, so I create a dictionary with the channels that have both
    # signals, and then I create a ConvFitterVDWrapper for each of them
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
                    cfit[ep][ch] = ConvFitterVDWrapper(**cfitparams) # type: ignore


    if len(dict_ep_ch_with_both_signals) == 0:
        dryrun = True
    if dryrun:
        print("Dry run mode. Parameters:")
        print(f"Runs: {runs}")
        print(f"Root directory: {rootdir}")
        print(f"Response folder name: {response}")
        print(f"Template folder name: {template}")
        print(f"Output directory: {outputdir}")
        print(f"Analysis name: {analysisname}")
        print(f"Channels: {channels}")
        print(f"Blacklist: {blacklist}")
        print(f"Cathode: {cathode}")
        print(f"Membrane: {membrane}")
        ch_show_avaliable_template_folders()

        if len(dict_ep_ch_with_both_signals) != 0:
            print("The following channels will be processed:")
            for ep, chs in dict_ep_ch_with_both_signals.items():
                for ch in chs:
                    print(f"Endpoint {ep} Channel {ch}")
        else:
            print_colored("No channels will be processed. Please check the parameters and the availability of the response and template files.", 'ERROR')
        return

    for ep, chs in dict_ep_ch_with_both_signals.items():
        for ch in chs:
            modulename = dict_uniqch_to_module[strUch(ep,ch)]
            print(f"Processing {ep}-{ch}: {modulename}")
            process_convfit(
                responses[ep][ch],
                templates[ep][ch],
                cfit[ep][ch],
                scan=8,
                print_flag=False,
                dofit=True,
                verbose=False
            )




    # Warning the user about the were requested but there is no response or
    # template for them, so they will be skipped in the analysis
    for ep, chs in dict_ep_ch_user_want.items():
        for ch in chs:
            if ep not in dict_ep_ch_with_both_signals.keys() or ch not in dict_ep_ch_with_both_signals[ep]:
                print_colored(f"Warning: channel {ep}-{ch} was requested but there is no response or template for it. It was skipped in the analysis.", 'WARNING')



if __name__ == "__main__":
    argp = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Fit a template waveform to a response waveform using a convolution model of LAr/Xe response.")
    argp.add_argument("--runs", nargs="+", type=int, default=[39510], help="List of runs to process.")
    argp.add_argument("--rootdir", type=str, default=PATH_XE_AVERAGES, help="Directory where the processed waveform files are located.")
    argp.add_argument("-r", "--response", type=str, default=DEFAULT_RESPONSE, help=f"Name of the analysis containing the response waveforms to fit.") 
    argp.add_argument("-t", "--template", type=str, default=DEFAULT_TEMPLATE, help=f"Name of the analysis containing the template waveforms to fit.")
    argp.add_argument("-o", "--outputdir", type=str, default="./", help="Directory where the average waveforms will be saved.")
    argp.add_argument("-a", "--analysisname", type=str, default=DEFAULT_CONV_NAME, help="Name of the analysis, used to create a subdirectory in the output directory.")
    argp.add_argument("--channels", nargs="+", type=int, default=[], help="List of channels to process. If not specified, all channels will be processed.\
        \n\tFormat: endpoint+channel, e.g. 1060 for endpoint 106 channel 0.")
    argp.add_argument("--blacklist", nargs="+", type=int, default=[], help="List of channels to exclude from processing. Format: endpoint+channel, e.g. 1060 for endpoint 106 channel 0.")
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
    channels = args.channels
    blacklist = args.blacklist
    cathode = args.cathode
    membrane = args.membrane
    dryrun = args.dryrun

    if not cathode and not membrane:
        cathode = True
        membrane = True
    for run in runs:
        main(run, rootdir, response, template, outputdir, analysisname, channels, blacklist, cathode, membrane, dryrun)





