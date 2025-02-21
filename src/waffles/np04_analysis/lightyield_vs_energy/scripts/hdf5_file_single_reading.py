from imports_scripts import *




@click.command()
@click.option("--full_streaming", 
              default='no',
              callback=validate_full_streaming,
              help="Are you reading full streaming data? (yes/no)") 
@click.option("--rucio_folder", 
              default='/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths',
              callback=validate_folder,
              help="Folder where rucio path txt files are saved") 
@click.option("--pickles_folder", 
              default='/afs/cern.ch/work/a/anbalbon/public/np04_beam_pickles',
              callback=validate_folder,
              help="Folder where all pickle files are saved")
@click.option("--reading_mode", 
              default='from - to',
              type=click.Choice(['list of index', 'from - to'], case_sensitive=False),
              help="Which reading mode to use: 'list of index', 'from - to'") 
@click.option("--index_list", 
              default=None,
              help="Comma-separated list of indices (used only if reading_mode='list of index')",
              callback=lambda ctx, param, value: [int(i) for i in value.split(',')] if value else None)  
@click.option("--start", 
              type=int, 
              default=None, 
              help="Start index included (used only if reading_mode='from - to')") 
@click.option("--stop", 
              type=int, 
              default=None, 
              help="Stop index excluded (used only if reading_mode='from - to')") 
@click.option("--set_list", 
              type=str, 
              default=None, 
              callback=validate_set_list_all,
              help="Which set do you want to analyze (e.g., A, B or both (all), by default all)") 
@click.option("--run_list", 
              type=str, 
              default=None, 
              callback=validate_run_list,
              help="Comma-separated list of runs to analyze (must be valid for the selected set)") 
@click.option("--beam_selection", 
              default='no',
              callback=validate_beam_selection,
              help="Do you want to save just beam events? (yes/no)") 
@click.option("--timeoffset_min", 
              type=int, 
              default=-120, 
              help="Minimum time offset (used only if beam_selection='yes')",
              required=False)
@click.option("--timeoffset_max", 
              type=int, 
              default=-90, 
              help="Maximum time offset (used only if beam_selection='yes')",
              required=False)


def main(full_streaming, rucio_folder, pickles_folder, reading_mode, index_list, start, stop, set_list, run_list, beam_selection, timeoffset_min, timeoffset_max):
    if reading_mode == 'list of index':
        if index_list is None:
            raise click.BadParameter("If reading_mode='list of index', you must provide --index_list")
        print(f"Processing list of indices: {index_list}")
    elif reading_mode == 'from - to':
        if start is None or stop is None:
            raise click.BadParameter("If reading_mode='from - to', you must provide --start and --stop")
        print(f"Processing range from {start} to {stop}")
        index_list = list(range(start, stop))
    
    with open('/afs/cern.ch/user/a/anbalbon/waffles/src/waffles/np04_analysis/lightyield_vs_energy/data/beam_run_info.json', "r") as file:
        run_set_list = json.load(file)
    
    print(f"Full streaming: {full_streaming}")
    print(f"Rucio folder: {rucio_folder}")
    print(f"Pickles folder: {pickles_folder}")
    print(f"Reading mode: {reading_mode}")
    print(f"Set selected: {set_list}")
    print(f"Run list: {run_list}")

    if isinstance(set_list, list):
        run_set_dic_list = {key: value for key, value in run_set_list.items() if key in set_list}
    else:
        run_set_dic_list = {set_list:run_set_list[set_list]}
 
    for set, run_set in run_set_dic_list.items():
        print(f"\n------------------\n------------------\n\nSET {run_set['Name']}")
        set_folder_name = f"{pickles_folder}/set_{run_set['Name']}"
        if not os.path.exists(set_folder_name):
            os.makedirs(set_folder_name)  
        
        if run_list is None:
            run_list_iteration = run_set['Runs'].values()
        else: 
            run_list_iteration =run_list
        
        
        for run in run_list_iteration:
            print(f'\n--------- RUN {run} ---------')
            
            run_folder_name = f'{set_folder_name}/run_{run}'
            if not os.path.exists(run_folder_name):
                os.makedirs(run_folder_name)           
    
            try:
                run_hdf5_filepaths = hdf5_reader.get_filepaths_from_rucio(f"{rucio_folder}/0{run}.txt")
            except Exception as e:
                print(f"{e}\n--> Skipped\n")
                continue
            
            if min(index_list) > len(run_hdf5_filepaths):
                print('Start > available hdf5 files --> skipped')
                continue

            if max(index_list) > len(run_hdf5_filepaths):
                index_list = list(range(start, len(run_hdf5_filepaths)))
                print(f'Stop > available hdf5 files --> go to the max file (i.e. {len(run_hdf5_filepaths)})')
                
            for i in index_list:
                if beam_selection:
                    pickle_name = f"{run_folder_name}/{trigger_string(full_streaming)}_beam_{i}.pkl"
                else: 
                    pickle_name = f"{run_folder_name}/{trigger_string(full_streaming)}_{i}.pkl"
                    
                try:
                    if not os.path.isfile(pickle_name):
                        print(f'Reading: run_{run}_{i}')
                        wfset = hdf5_reader.WaveformSet_from_hdf5_file(run_hdf5_filepaths[i], read_full_streaming_data = full_streaming, erase_filepath = True) ## modified
                        if beam_selection:
                            wfset = WaveformSet.from_filtered_WaveformSet(wfset, beam_self_trigger_filter, timeoffset_min = timeoffset_min, timeoffset_max = timeoffset_max)
                        with open(pickle_name, "wb") as f:
                            pickle.dump(wfset, f)
                    else:
                        print(f"File already read: run {run} file {pickle_name.split('/')[-1]}")
                    
                except Exception as e:
                    keywords = ["no space left on device", "file exists", "unable to open file", "truncated file"]
                    if any(keyword in str(e).lower() for keyword in keywords):
                        print("\n\nCritical error detected: No space, file exists, or file corruption.\n\nSTOPPING EXECUTION.\n\n")
                        sys.exit(1)  
            
######################################################################################

if __name__ == "__main__":
    main()