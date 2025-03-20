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
              default='all',  
              type=click.Choice([ 'from-to', 'readNfiles', 'all'], case_sensitive=False),
              help="Which merging mode do you want use:  'from-to' 'all' or 'readNfiles' (default: 'all')")
@click.option("--n_files", 
              type=int,
              default=None,
              help="Number of files to read (used only if merging_mode='readNfiles')")
@click.option("--index_start", 
              type=str,  
              default=None, 
              help="Start index (used only if merging_mode='from-to'). Can be 'min' or an integer.")
@click.option("--index_stop", 
              type=str,  
              default=None, 
              help="Stop index (used only if merging_mode='from-to'). Can be 'max' or an integer.")
@click.option("--run", 
              required=True,
              type=int, 
              default=None, 
              help="Run to read") 
@click.option("--endpoint", 
              type=int, 
              default=None, 
              help="Endpoint to read") 
@click.option("--channel", 
              type=int, 
              default=None, 
              help="Channel to read") 

def main(full_streaming, pickles_folder, output_filename, overwrite, merging_mode, n_files, index_start, index_stop, run, endpoint, channel):
   
    if merging_mode == 'from-to':
        if index_start is None or index_stop is None:
            raise click.BadParameter("Both index_start and index_stop are required when merging_mode is 'from-to'.\n")
        index_start = validate_start_stop(index_start, 'start')
        index_stop = validate_start_stop(index_stop, 'stop')
    elif merging_mode == 'readNfiles' and (n_files is None or n_files <= 0):
        raise click.BadParameter("n_files must be a positive integer when merging_mode is 'readNfiles'.\n")
    i_index = 0 
    
    run_folder = f"{pickles_folder}/led_calibration_runs/run_{run}"
    
    if not os.path.isdir(run_folder):
        print('\nRun folder doesnt exist \n')
        sys.exit()
    
    all_files_indices = sorted([int(re.search(r'\d+', f).group()) for f in os.listdir(run_folder) if f.endswith('.pkl') and (("full" in f) if full_streaming else ("self" in f))])
    print(all_files_indices)
    if len(all_files_indices) == 0:
        print('\nNo files for this run\n')
        sys.exit()
    

    output_filename = f'{pickles_folder}/led_calibration_runs/run_{run}_{trigger_string(full_streaming)}_{output_filename}.pkl'
    if overwrite:
        print('\nAttention: if the output filename already exists, it will be overwritten!!\n')
    else:
        if os.path.exists(output_filename):
            print(f"The filename {output_filename.split('/')[-1]} exists, please change")
            sys.exit()
        print(f"\nValid output filename {output_filename.split('/')[-1]}\n")
    
    
    print('... starting creating the merged pickles file ...')
        
    if merging_mode == 'from-to':
        if (index_start == '-inf') or (index_start < min(all_files_indices)):
            index_start = min(all_files_indices)
        if (index_stop == 'inf') or (index_stop > max(all_files_indices)):
            index_stop = max(all_files_indices)
        index_list =  list(set(all_files_indices) & set(range(index_start, index_stop)))
        missing_elements = list(set(range(index_start, index_stop)) - set(index_list))
        print(f'Reading from {index_start} to {index_stop} ({len(index_list)} files): {index_list}\nAttention, {len(missing_elements)} files are missing: {missing_elements}\n')    
    elif merging_mode == 'readNfiles':
        original_n_files = n_files
        if original_n_files > len(all_files_indices):
            n_files = len(all_files_indices)
        else:
            n_files = original_n_files
        index_list =  all_files_indices[0:n_files]
        print(f'Reading {n_files} files with index: {index_list}')
    elif merging_mode == 'all':
        index_start = min(all_files_indices)
        index_stop = max(all_files_indices)
        index_list =  list(set(all_files_indices) & set(range(index_start, index_stop+1)))

    
    for i in tqdm(range(0, len(index_list)), desc="Merging files", unit="file"):
        with open(f'{run_folder}/{trigger_string(full_streaming)}_{index_list[i]}.pkl', 'rb') as f:
            try:
                if i_index == 0:
                    if endpoint is not None and channel is not None:
                        wfset = WaveformSet.from_filtered_WaveformSet(pickle.load(f), channel_filter, end=endpoint, ch=channel)
                    else:
                        wfset = pickle.load(f)
                else:
                    if endpoint is not None and channel is not None:
                        wfset.merge(WaveformSet.from_filtered_WaveformSet(pickle.load(f), channel_filter, end=endpoint, ch=channel))
                    else:
                        wfset.merge(pickle.load(f))
            except Exception as e:
                print(f"Errore durante il caricamento del file {index_list[i]}: {e}")
                sys.exit()
    
        i_index += 1

    print(f"\nSUMMARY: \nFull-streaming data: {full_streaming}\nMerging mode: {merging_mode}\nRun: {run}\nEndpoint: {endpoint}\nChannel: {channel}\n# read files: {i_index}")
        
    if i_index == 0:
        print(f"# waveforms: 0\n\n\n --- NO DATA - NO FILE SAVED ---\n\n")
    else:
        print(f"# waveforms: {len(wfset.waveforms)}\n\n")
        print(f'Saving the merged pickles file: \n{output_filename}')
        with open(output_filename, "wb") as f:
            pickle.dump(wfset, f)
        
  
    print('\n\t ------------------------ \n\t\t DONE \n\t ------------------------ \n')

    

#####################################################################

if __name__ == "__main__":
    
    main()