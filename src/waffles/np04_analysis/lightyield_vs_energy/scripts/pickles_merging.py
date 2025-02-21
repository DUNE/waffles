from imports_scripts import *



@click.command()
@click.option("--full_streaming", 
              default='no',
              callback=validate_choice,
              type=click.Choice(['yes', 'no'], case_sensitive=False),
              help="Are you reading full streaming data? (yes/no, by default no)")
@click.option("--pickles_folder", 
              default='/afs/cern.ch/work/a/anbalbon/public/np04_beam_pickles',
              help="Folder where all pickle files are saved")
@click.option("--output_filename", 
              required=True,
              help="Output filename")
@click.option("--overwrite", 
              default='no',
              callback=validate_choice,
              type=click.Choice(['yes', 'no'], case_sensitive=False),
              help="If the filename exists, do you want to overwrite the file? (yes/no, by default no)")
@click.option("--merging_mode", 
              default='from-to',  
              type=click.Choice(['list_index', 'from-to', 'readNfiles'], case_sensitive=False),
              help="Which merging mode do you want use: 'list_index', 'from-to' or 'readNfiles' (default: 'from-to')")
@click.option("--n_files", 
              type=int,
              default=None,
              help="Number of files to read (used only if merging_mode='readNfiles')")
@click.option("--index_list", 
              default=None,
              help="Comma-separated list of indices (used only if merging_mode='list_index')",
              callback=lambda ctx, param, value: [int(i) for i in value.split(',')] if value else None)
@click.option("--index_start", 
              type=str,  
              default=None, 
              help="Start index (used only if merging_mode='from-to'). Can be 'min' or an integer.")
@click.option("--index_stop", 
              type=str,  
              default=None, 
              help="Stop index (used only if merging_mode='from-to'). Can be 'max' or an integer.")
@click.option("--beam_data", 
              default='no',
              callback=validate_choice,
              type=click.Choice(['yes', 'no'], case_sensitive=False),
              help="Do you want to save just beam events? (yes/no, by default no)")
@click.option("--beam_selection", 
              default='no',
              callback=validate_choice,
              type=click.Choice(['yes', 'no'], case_sensitive=False),
              help="Do you want to apply beam selection? (yes/no, by default no)",
              required=False)
@click.option("--timeoffset_min", 
              type=int, 
              default=-120, 
              help="Minimum time offset (used only if beam_selection is yes, by default -120)")
@click.option("--timeoffset_max", 
              type=int, 
              default=-90, 
              help="Maximum time offset (used only if beam_selection is yes,  by default -90)")
@click.option("--set_name", 
              required=True,
              type=click.Choice(['A', 'B'], case_sensitive=False),
              help="Which set do you want to analyze? (A or B)")
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

def main(full_streaming, pickles_folder, output_filename, overwrite, merging_mode, n_files, index_list, index_start, index_stop, beam_data, beam_selection, timeoffset_min, timeoffset_max, set_name, apa_list, endpoint_list, save_file):
    with open('/afs/cern.ch/user/a/anbalbon/waffles/src/waffles/np04_analysis/lightyield_vs_energy/data/beam_run_info.json', "r") as file:
        run_set_list = json.load(file)
        
    if merging_mode == 'from-to':
        if index_start is None or index_stop is None:
            raise click.BadParameter("Both index_start and index_stop are required when merging_mode is 'from-to'.\n")
        index_start = validate_start_stop(index_start, 'start')
        index_stop = validate_start_stop(index_stop, 'stop')
    
    if merging_mode == 'list_index' and not index_list:
        raise click.BadParameter("index_list is required when merging_mode is 'list_index'.\n")
    
    if merging_mode == 'readNfiles' and (n_files is None or n_files <= 0):
        raise click.BadParameter("n_files must be a positive integer when merging_mode is 'readNfiles'.\n")
    
    if beam_data and beam_selection:
        raise click.BadParameter("If 'beam_data' is 'yes', 'beam_selection' cannot be 'yes'.\n")
    
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
    
    print(f"\nSUMMARY: \nFull-streaming data: {full_streaming}\nMerging mode: {merging_mode}\nBeam data: {beam_data}\nBeam selection: {beam_selection}\nSet name: {set_name}\nAPA: {apa_list}\nENDPOINT: {endpoint_list}\n\n")

    
    pickles_folder = os.path.join(pickles_folder, f'set_{set_name}')
    
    if beam_data or beam_selection:
        output_filename = f'{pickles_folder}/set_{set_name}_{trigger_string(full_streaming)}_beam_{output_filename}.pkl'
    else:
        output_filename = f'{pickles_folder}/set_{set_name}_{trigger_string(full_streaming)}_{output_filename}.pkl'
    
    if overwrite:
        print('\nAttention: if the output filename already exists, it will be overwritten!!\n')
    else:
        while os.path.exists(output_filename):
            print(f"The filename {output_filename.split('/')[-1].split('_')[-1].split('.')[0]} exists, please write the new filename:")
            output_filename = f'{pickles_folder}/set_{set_name}_{trigger_string(full_streaming)}_{input()}.pkl'
        print(f"\nValid output filename {output_filename.split('/')[-1]}\n")
 
    print('... starting creating the merged pickles file ...')
    i_index = 0 
    originale_index_list = index_list
    for energy, run in run_set_list[set_name]['Runs'].items(): #for run in [d for d in os.listdir(pickles_folder) if os.path.isdir(os.path.join(pickles_folder, d))]:  #for energy, run in 
        print(f"\n\n--- Reading run {run} ---")
        run_folder = os.path.join(pickles_folder, f'run_{run}')
        if os.path.isdir(run_folder):
            os.chdir(run_folder)  
            all_files_indices = sorted([int(re.search(r'\d+', f).group()) for f in os.listdir() if f.endswith('.pkl') and (("full" in f) if full_streaming else ("self" in f)) and (("beam" in f) if beam_data else ("beam" not in f))])
            
            if len(all_files_indices) == 0:
                print('No files for this run\n')
                continue
            
            if merging_mode == 'from-to':
                if (index_start == '-inf') or (index_start < min(all_files_indices)):
                    index_start = min(all_files_indices)
                if (index_stop == 'inf') or (index_stop > max(all_files_indices)):
                    index_start = max(all_files_indices)
                index_list =  list(set(all_files_indices) & set(range(index_start, index_stop)))
                missing_elements = list(set(range(index_start, index_stop)) - set(index_list))
                print(f'Reading from {index_start} to {index_stop} ({len(index_list)} files): {index_list}\nAttention, {len(missing_elements)} files are missing: {missing_elements}\n')    
            elif merging_mode == 'readNfiles':
                if n_files > len(all_files_indices):
                    n_files = len(all_files_indices)
                index_list =  all_files_indices[0:n_files]
                print(f'Reading {n_files} files with index: {index_list}')
            elif merging_mode == 'list_index':
                index_list =  list(set(all_files_indices) & set(index_list))
                missing_elements = list(set(originale_index_list) - set(index_list))
                print(f'Reading files of the index list {originale_index_list} present in the folder too ({len(index_list)} files): {index_list} \nAttention, {len(missing_elements)} files are missing: {missing_elements}\n')    
            
            
            for i in tqdm(range(0, len(index_list)), desc="Merging files", unit="file"):
                with open(f'{run_folder}/{trigger_string(full_streaming)}_{index_list[i]}.pkl', 'rb') as f:
                    if i_index == 0:
                        if beam_selection:
                            wfset = WaveformSet.from_filtered_WaveformSet(WaveformSet.from_filtered_WaveformSet(pickle.load(f), endpoint_list_filter, endpoint_list = endpoint_list), beam_self_trigger_filter, timeoffset_min = timeoffset_min, timeoffset_max = timeoffset_max)
                            
                        else:
                            wfset = WaveformSet.from_filtered_WaveformSet(pickle.load(f), endpoint_list_filter, endpoint_list = endpoint_list)
                        i_index += 1
                    else: 
                        if beam_selection:
                            wfset.merge(WaveformSet.from_filtered_WaveformSet(WaveformSet.from_filtered_WaveformSet(pickle.load(f), endpoint_list_filter, endpoint_list = endpoint_list), beam_self_trigger_filter, timeoffset_min = timeoffset_min, timeoffset_max = timeoffset_max))
                        else:
                            wfset.merge(WaveformSet.from_filtered_WaveformSet(pickle.load(f), endpoint_list_filter, endpoint_list = endpoint_list))
                        i_index += 1

    print(f"\nSUMMARY: \nFull-streaming data: {full_streaming}\nMerging mode: {merging_mode}\nBeam data: {beam_data}\nBeam selection: {beam_selection}\nSet name: {set_name}\nAPA: {apa_list}\nENDPOINT: {endpoint_list} \n# read files: {i_index}")
        
    if i_index == 0:
        print(f"# waveforms: 0\n\n\n --- NO DATA - NO FILE SAVED ---\n\n")
    else:
        print(f"# waveforms: {len(wfset.waveforms)}\n\n")
        if save_file:
            print(f'Saving the merged pickles file: \n{output_filename}')
            with open(output_filename, "wb") as f:
                pickle.dump(wfset, f)
            

            summary_txt = f"{output_filename.replace(output_filename.split('/')[-1], 'summary.txt')}"
            new_summary_entry = f"Output filename: {output_filename}\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} \n" \
                                f"Full-streaming data: {full_streaming}\nPickles folder: {pickles_folder}\n" \
                                f"Overwrite: {overwrite}\nMerging mode: {merging_mode}\nNumber of files: {n_files}\nIndex list: {index_list}\n" \
                                f"Index start: {index_start}\nIndex stop: {index_stop}\nBeam data: {beam_data}\nBeam selection: {beam_selection}\n" \
                                f"Time offset min: {timeoffset_min}\nTime offset max: {timeoffset_max}\nSet name: {set_name}\nAPA: {apa_list}\n" \
                                f"ENDPOINT: {endpoint_list}\n# read files: {i_index}\n# waveforms: {len(wfset.waveforms)}\n\n"

            if os.path.exists(summary_txt):
                with open(summary_txt, "r") as file:
                    lines = file.readlines()
                new_lines = []
                inside_old_entry = False
                for line in lines:
                    if line.startswith(f"Output filename: {output_filename}"):
                        inside_old_entry = True  
                    elif inside_old_entry and line.strip() == "":  
                        inside_old_entry = False 
                    if not inside_old_entry:
                        new_lines.append(line)
                with open(summary_txt, "w") as file:
                    file.writelines(new_lines)  
                    file.write(new_summary_entry)  
            else:
                with open(summary_txt, "w") as file:
                    file.write(new_summary_entry)
        else:
            print('File not saved!!')
        
  
    print('\n\t ------------------------ \n\t\t DONE \n\t ------------------------ \n')

    

#####################################################################

if __name__ == "__main__":
    
    main()