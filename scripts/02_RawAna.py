import click, pickle, gc, json, inquirer
from rich import print
import numpy as np
import plotly.subplots as psu

from waffles.utils.utils import print_colored
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.np04_analysis.np04_ana import comes_from_channel
from waffles.data_classes.IPDict import IPDict
from waffles.data_classes.BeamWfAna import BeamWfAna
from waffles.np04_data_classes.APAMap import APAMap
from waffles.data_classes.UniqueChannel import UniqueChannel
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
from waffles.plotting.plot import plot_ChannelWsGrid
from waffles.plotting.plot_utils import save_plot

@click.command(help=f"\033[34mSave the WaveformSet object in a pickle file for easier loading.\n\033[0m")
@click.option("--run",   default = None, help="Run number to process", type=str)
@click.option("--chs",   default = None, help="Channel number to process (i.e. 44)", type=int)
@click.option("--eps",   default = None, help="Endpoint number to process (i.e. 109)", type=int) #TODO: allow simulation channels map?
@click.option("--debug", default = True, help="Debug flag")
def main(run,chs, eps, debug):
    '''
    Script to process peak/pedestal variables and save the WaveformSet object + unidimensional variables in a pickle file.

    Args:
        - run (int): Run number to be analysed. I can also be a list of runs separated by commas.
        - chs (int): Channel number to be analysed. I can also be a list of channels separated by commas.
        - eps (int): Endpoint number to be analysed.
        - debug (bool): Debug flag.
    Example: python 02RawAna.py --run 28602 --chs 31 --eps 109 --debug True
    '''
    
    if run is None: 
        print_colored("Please provide a run(s) number(s) to be analysed, separated by commas:)", color="yellow", styles=["bold"])
        run = [int(input("Run(s) number(s): "))]
    if type(run)==list and len(run)!=1: runs_list = list(map(int, list(run.split(","))))
    else: runs_list = [run]
    
    if chs is None: 
        print_colored("Please provide the channel(s) number(s) to be analysed, separated by commas:)", color="yellow", styles=["bold"])
        chs = [int(input("Channel(s) number(s): "))]
    if type(chs)==list and len(chs)!=1: chs_list = list(map(int, list(chs.split(","))))
    else: chs_list = [chs]
    
    if eps is None:
        print_colored("Please provide the endpoint number to be analysed, separated by commas:)", color="yellow", styles=["bold"])
        eps = int(input("Endpoint number: "))
    
    
    for r in runs_list:
        # Read from output JSON file
        json_file_path = f"conf/{str(r).zfill(6)}.json"
        try:
            with open(json_file_path, 'r') as json_file:
                analysis_conf = json.load(json_file)
        except FileNotFoundError:
            print_colored(f"\nFile {json_file_path} not found.", color="ERROR")
            continue
        
        # Inquirer question to ask which analysis label among the keys inside the JSON
        analysis_labels = list(analysis_conf.keys())
        if len(analysis_labels) == 1:
            print_colored(f"\nAnalysis label: {analysis_labels[0]}", color="INFO")
            analysis_label = analysis_labels[0]
        else:
            question = [inquirer.List('analysis_label', message="Choose the analysis label", choices=analysis_labels)]
            answer = inquirer.prompt(question)
            analysis_label = answer['analysis_label']
        
        
        #Define the analysis parameters
        baseline_limits  = analysis_conf[analysis_label]['base_lim'] #[0, 100, 900, 1000]
        input_parameters = IPDict(baseline_limits = baseline_limits)
        input_parameters['int_ll'] =  analysis_conf[analysis_label]['int_ll']
        input_parameters['int_ul'] =  analysis_conf[analysis_label]['int_ul']
        input_parameters['amp_ll'] =  analysis_conf[analysis_label]['amp_ll']
        input_parameters['amp_ul'] =  analysis_conf[analysis_label]['amp_ul']
        checks_kwargs = IPDict()
        
        print(input_parameters)
        
        try:
            with open(f"data/{str(r).zfill(6)}_full_wfset_raw.pkl", 'rb') as file:
                wfset = pickle.load(file)
        except FileNotFoundError:
            print_colored(f"\nFile {str(r).zfill(6)}_full_wfset_raw.pkl not found. Please run 01Process.py first.", color="ERROR")
            return
            
        if debug: 
            print_colored(f"\nProcessing run {str(r).zfill(6)}. Loaded WfSet with {len(wfset.waveforms)} wvfs.", color="DEBUG")
        
        for ch in chs_list:
            filter_wfset = WaveformSet.from_filtered_WaveformSet( wfset, comes_from_channel, eps, [ch])
            checks_kwargs['points_no'] = filter_wfset.points_per_wf
            
            #Analysis saved in the Waveform object
            print_colored(f"Processing channel {ch} ({len(filter_wfset.waveforms)} wvfs) with analysis label {analysis_label}.", color="INFO")
            _ = filter_wfset.analyse(  analysis_label,
                                       BeamWfAna,
                                       input_parameters,
                                       *[], # *args,
                                       analysis_kwargs = {},
                                       checks_kwargs = checks_kwargs,
                                       overwrite = True)
            
            #Prepare for the plotting
            my_map = APAMap ([[UniqueChannel(eps, ch)]], 1, 1 )
            grid_apa = ChannelWsGrid( my_map, filter_wfset, compute_calib_histo = False )
            
            
            print(f"Plotting a sample of waveforms with the analysis parameters as in conf/{r}.json ...")
            figure = psu.make_subplots( rows = 1, cols = 1 )
            plot_ChannelWsGrid( grid_apa,
                                figure = figure,
                                share_x_scale = False,
                                share_y_scale = False,
                                mode = 'overlay',
                                wfs_per_axes = 10,
                                analysis_label = analysis_label,
                                plot_analysis_markers = True,
                                show_baseline_limits = True, 
                                show_baseline = True,
                                show_general_integration_limits = True,
                                show_general_amplitude_limits = True,
                                show_spotted_peaks = False,
                                # show_peaks_integration_limits = True,    
                                # plot_peaks_fits = True,
                                detailed_label = False,
                                verbose = True)
            figure.update_layout(template="presentation", title = "", xaxis_title="Time [ticks]", yaxis_title="Amplitude [ADC]")
            figure.show()
            
            confirmation = input("Do you want to save the analysed WvfSet? (y/n): ")
            if confirmation.lower() == 'y':
                with open(f"data/{str(r).zfill(6)}_wfset_ana_ep{eps}_ch{ch}.pkl", 'wb') as file:
                    pickle.dump(filter_wfset, file)
                print_colored(f"\nWaveformSet saved in {str(r).zfill(6)}_wfset.pkl", color="SUCCESS")
            else:
                print_colored(f"\nWaveformSet not saved. Exiting...", color="INFO")
            
            # for wf in filter_wfset.waveforms:
            #     plot_WaveformAdcs(  wf, figure = figure,
            #                         plot_analysis_markers = True,
            #                         show_baseline_limits = True, 
            #                         show_baseline = True,
            #                         show_general_integration_limits = True,
            #                         show_general_amplitude_limits = True,
            #                         show_spotted_peaks = True,
            #                         show_peaks_integration_limits = False,)
            
            # save_plot(figure, f"eps_{eps}_ch_{ch}_wvfs.png")
            
            # custom_plotly_layout(figure, xaxis_title="Time [ticks]", yaxis_title="Amplitude [ADC]", title=f"Waveform ADCs for EP {eps} - CH {ch}")
            
            #Opcion1: bulce and plot with plotly
            #Opcion2: ChannelWSGrid con Unique channel grid de tu canal creado en la linea anterior. lo bueno de esto es que luego se puede pintar el analisis
            
            #Cleaning memory
            # del filter_wfset
            # gc.collect()
            

if __name__ == "__main__":
    main()