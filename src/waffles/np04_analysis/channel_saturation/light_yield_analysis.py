# python light_yield_analysis.py --set_name A --input_filename set_A_self_100files109.pkl --beam_data no --overwrite yes --endpoint_list '109' --save_file no --searching_beam_timeoffset yes --searching_integration_range yes

# Sistemare l'output a schermo
# sistemare il salvataggio del file pdf ( cartella, overwrite, when ecc)

# implementare # fotoelettroni

from tools_analysis import *


@click.command()
@click.option("--set_name", 
              type=str, 
              required=True, 
              callback=validate_set_list_or,
              help="Which set do you want to analyze (A or B)")
@click.option("--input_filename", 
              required=True,
              help="Input filename (no folder path)")
@click.option("--additional_output_filename", 
              default=None,
              help="Output  filename (no folder path)")
@click.option("--full_streaming", 
              default='no',
              callback=validate_choice,
              type=click.Choice(['yes', 'no'], case_sensitive=False),
              help="Are you reading full streaming data? (yes/no, by default no)")
@click.option("--pickles_folder", 
              default='/afs/cern.ch/work/a/anbalbon/public/np04_beam_pickles',
              help="Folder where all pickle files are saved")
@click.option("--output_folder", 
              default='/afs/cern.ch/user/a/anbalbon/waffles/src/waffles/np04_analysis/channel_saturation/output',
              help="Folder where yo save results")
@click.option("--beam_data", 
              default='no',
              callback=validate_choice,
              type=click.Choice(['yes', 'no'], case_sensitive=False),
              help="Are you using beam data (already selected)? (yes/no, by default no)")
@click.option("--searching_beam_timeoffset", 
              default='no',
              callback=validate_choice,
              type=click.Choice(['yes', 'no'], case_sensitive=False),
              help="Do yu want to look for beam timeoffset range? (yes/no, by default no to use default value from -120 to -90)")
@click.option("--searching_integration_range", 
              default='no',
              callback=validate_choice,
              type=click.Choice(['yes', 'no'], case_sensitive=False),
              help="Do yu want to look for integration range? (yes/no, by default no to use default value from 55 to 115)")
@click.option("--overwrite", 
              default='no',
              callback=validate_choice,
              type=click.Choice(['yes', 'no'], case_sensitive=False),
              help="If the filename exists, do you want to overwrite the file? (yes/no, by default no)")
@click.option("--apa_list", 
              default='all', 
              help="List of APA numbers (1, 2, 3, 4) or 'all'. Default is 'all'.",
              callback=lambda ctx, param, value: [int(x) for x in value.split(',')] if value != 'all' else [1, 2, 3, 4])
@click.option("--endpoint_list", 
              default='all', 
              help="List of endpoint numbers (104, 105, 107, 109, 111, 112, 113) or 'all'. Default is 'all'.",
              callback=lambda ctx, param, value: [int(x) for x in value.split(',')] if value != 'all' else [104, 105, 107, 109, 111, 112, 113])
@click.option("--save_file", 
              default='yes',
              callback=validate_choice,
              type=click.Choice(['yes', 'no'], case_sensitive=False),
              help="Do you want to save the pickles file? (yes/no, by default yes)")



def main(set_name, input_filename, additional_output_filename, full_streaming, pickles_folder, output_folder,  beam_data, searching_beam_timeoffset, searching_integration_range, overwrite, apa_list, endpoint_list, save_file):
    print()
    run_set = run_set_dict[set_name]
    
    valid_endpoints = {104, 105, 107, 109, 111, 112, 113}
    
    if any(x not in valid_endpoints for x in endpoint_list):
        raise click.BadParameter("Invalid value for --endpoint_list. Must be 'all' or a comma-separated list of valid endpoint numbers (104, 105, 107, 109, 111, 112, 113).\n")

    valid_endpoints_across_apas = set()
    excluded_apa_list = []

    for apa in apa_list:
        valid_endpoints_for_apa = set(which_endpoints_in_the_APA(apa))
        common_endpoints = valid_endpoints_for_apa & set(endpoint_list)

        if common_endpoints:
            valid_endpoints_across_apas.update(common_endpoints)
        else:
            excluded_apa_list.append(apa)

    excluded_endpoints = [x for x in endpoint_list if x not in valid_endpoints_across_apas]
    endpoint_list = list(valid_endpoints_across_apas)

    if excluded_endpoints:
        print(f"Warning: The following endpoints were excluded because they are not compatible with the selected APAs {apa_list}: {excluded_endpoints}\n")

    if excluded_apa_list:
        apa_list = [apa for apa in apa_list if apa not in excluded_apa_list]
        print(f"Warning: The following APAs were excluded because they had no compatible endpoints: {excluded_apa_list}\n")

    if not endpoint_list:
        raise click.BadParameter(f"No valid endpoints in the provided list are available for the selected APAs {apa_list}.\n")
    
    
    
    if additional_output_filename is None:
        output_filepath = f"{output_folder}/{input_filename.split('.')[0]}_lightyield.json"
    else: 
        output_filepath = f"{output_folder}/{input_filename.split('.')[0]}_{additional_output_filename}_lightyield.json" 
    if save_file:        
        if overwrite:
            print('\nAttention: if the output filename already exists, it will be overwritten!!\n')
        else:
            while os.path.exists(output_filepath):
                print(f"The filename {output_filepath.split('/')[-1]} exists, please write the new additional output filename (that will replace the previous one if presents):")
                output_filepath = f"{output_folder}/{input_filename.split('.')[0]}_{input()}_lightyield.json" 
        print(f"\nValid output filename: {output_filepath.split('/')[-1]}\n")
        

      
    print(f"\nSUMMARY: \nFull-streaming data: {full_streaming}\nBeam data: {beam_data}\nSearching beam timeoffset: {searching_beam_timeoffset}\nSearching integration range: {searching_integration_range}\nSet name: {set_name}\nAPA: {apa_list}\nENDPOINT: {endpoint_list}\n\n")
  
    ######################### READING #########################
    print('Reading...\n')
    input_filepath = f"{pickles_folder}/set_{run_set['Name']}/{input_filename}"
    if Path(input_filepath).exists():
        with open(input_filepath, 'rb') as f:
            wfset= pickle.load(f) 
                        
        available_run = set(wfset.runs)
        run_set_values = set(run_set['Runs'].values())

        # Verify missing or incorrect runs in the wfset
        if available_run != run_set_values:
            if missing := run_set_values - available_run:
                print(f"Missing run: {missing} \n")
            if missing := available_run - run_set_values:
                print(f"ERROR, your set has different runs from the requested: {missing} should not be present - please update your pickle file or change the map!!\n")
                sys.exit()      
                
        ######################### BEAM EVENT OFFSET RANGE ######################### 
        if not beam_data:  
            print('\nSelecting beam data event... \n')
            if searching_beam_timeoffset:
                print('Searching for beam time-offset range')
                answer = 'yes'
                while(answer == 'yes'):
                    searching_for_beam_events_interactive(wfset, show=True)
                    beam_min = int(input('Minimum time-offset: '))  
                    beam_max = int(input('Maximum time-offset: ')) 
                    searching_for_beam_events_interactive(wfset, show=True, beam_min=beam_min, beam_max=beam_max, x_min = -10000, x_max = 10000)
                    answer = input('Do you want to CHANGE time-offset range? (yes/no) ').strip().lower()   
            else:
                beam_min = -120
                beam_max = -90     
        
            beam_wfset = WaveformSet.from_filtered_WaveformSet(wfset, beam_self_trigger_filter, timeoffset_min = beam_min, timeoffset_max = beam_max)
        else:
            beam_wfset = wfset

        print('done\n\n')
        
        ######################### INTEGRAL REGION SELECTION #########################  --> inutile per la channel saturation
        if searching_integration_range:
            print('Searching for integration range')
            answer = 'yes'
            while(answer == 'yes'):
                plotting_overlap_wf(beam_wfset, show=True)
                int_ll = int(input('Lower integration limit: '))  
                int_ul = int(input('Upper integration limit: ')) 
                plotting_overlap_wf(beam_wfset, show=True, int_ll=int_ll, int_ul=int_ul)
                answer = input('Do you want to CHANGE integration limits? (yes/no) ').strip().lower()   
        else:
            int_ll = 55
            int_ul = 115
    
        ######################### WAVEFORM ANALYSIS ######################### 
        print('\nAnalysis... ')
        analysis_label = 'standard'
        bl = [0, 40, 900, 1000]
        peak_finding_kwargs = dict( prominence = 20,rel_height=0.5,width=[0,75])
            
        ip = IPDict(baseline_limits=bl,
                    int_ll=int_ll,int_ul=int_ul,amp_ll=int_ll,amp_ul=int_ul,
                    points_no=10,
                    peak_finding_kwargs=peak_finding_kwargs)
        analysis_kwargs = dict(  return_peaks_properties = False)
        checks_kwargs   = dict( points_no = beam_wfset.points_per_wf )
        a = beam_wfset.analyse(label=analysis_label ,analysis_class=BasicWfAna, input_parameters=ip, checks_kwargs = checks_kwargs, overwrite=True)

        baseline = beam_wfset.waveforms[0].analyses[analysis_label].result['baseline']
        #plotting_overlap_wf(beam_wfset, int_ll=int_ll, int_ul=int_ul, baseline = baseline)
        print('done\n\n')
    
        ######################### SAVING INFORMATION #########################
        results_info=[]
        

        ######################### COMPUTING LY vs ENERGY BEAM X CHANNEL ######################### 
        for APA in apa_list:
            print(f'\n\n ------------------------------------\n \t       APA {APA} \n ------------------------------------\n')
            APA_pdf_file = PdfPages(output_filepath.replace('.json',f'_APA{APA}.pdf')) # NEW
            for endpoint in endpoint_list: 
                print(f'\n --- \t ENDPOINT {endpoint} \t ---')
                for channel in sorted(which_channels_in_the_ENDPOINT(endpoint)):
                    print(f"\n\nLet's study APA {APA} - Endpoint {endpoint} - Channel {channel}")
                    ID_String = f"APA{APA}_endpoint{endpoint}_channel{channel}"
                    ch_info = {'ID_String': ID_String, 'APA': APA, 'endpoint' : endpoint, 'channel' : channel, 'Runs' : run_set['Runs']}
                    ly_data_dic = {} 
                    ly_result_dic = {} 
     
                                        
                    if (APA == 1) and full_streaming:
                        try:
                            ch_wfset = WaveformSet.from_filtered_WaveformSet(beam_wfset, channel_filter, end = endpoint, ch = channel)
                        except Exception as e:  #Exception: WaveformSet.__init__() raised exception #1: There must be at least one Waveform in the set.
                            print(f"Error: {e}\nSkipped channel")
                        
                                  
                    if ((APA == 2) or (APA == 3) or (APA == 4) ) and not full_streaming: #or (APA == 3) or (APA == 4)
                        
                        try:
                            ch_beam_wfset = WaveformSet.from_filtered_WaveformSet(beam_wfset, channel_filter, end=endpoint, ch=channel)
                            ly_data_dic, ly_result_dic = LightYield_SelfTrigger_channel_analysis(wfset = ch_beam_wfset, end = endpoint, ch = channel, run_set = run_set, pdf_file = APA_pdf_file, analysis_label = analysis_label)
                                    
                        except Exception as e:
                            if "There must be at least one Waveform in the set" in str(e):
                                print(f"No beam events --> Skipped channel")
                            else:
                                print(f"Error: {e} --> Skipped channel")    
                                
                    
                    ch_info['LY data'] = ly_data_dic
                    ch_info['LY result'] = ly_result_dic
                    results_info.append(ch_info)            
                                      
            APA_pdf_file.close()                                
        print('\n\nAnalysis... done')
        print(results_info)
        
        if save_file: 
            print('\n\nSaving...', end='')
            with open(output_filepath, "w") as file:
                json.dump(results_info, file, indent=4)
                print(f' done: {output_filepath}')
        else:
            print(f'\n\nThe file was not saved (save_file = {save_file})')
        

        print(f"\n\nSUMMARY: \nFull-streaming data: {full_streaming}\nBeam data: {beam_data}\nSearching beam timeoffset: {searching_beam_timeoffset}\nSearching integration range: {searching_integration_range}\nSet name: {set_name}\nAPA: {apa_list}\nENDPOINT: {endpoint_list}\n# analyzed beam wavforms: {len(beam_wfset.waveforms)}\n\n")

        
    else:
        print(f"The input_file ({input_filename}) doesn't exist --> skipped!!\n\n")    
    
    
    print('\n\t ------------------------ \n\t\t DONE \n\t ------------------------ \n')  
         
######################################################################################

if __name__ == "__main__":
    main()
    
    