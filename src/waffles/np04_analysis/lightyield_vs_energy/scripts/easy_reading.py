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
              default='all',
              type=click.Choice(['all', 'from - to'], case_sensitive=False),
              help="Which reading mode to use: 'list of index', 'from - to'")   
@click.option("--start", 
              type=int, 
              default=None, 
              help="Start index included (used only if reading_mode='from - to')") 
@click.option("--stop", 
              type=int, 
              default=None, 
              help="Stop index excluded (used only if reading_mode='from - to')") 
@click.option("--run_list", 
              type=str, 
              default=None, 
              help="Comma-separated list of runs to analyze (must be valid for the selected set)") 



def main(full_streaming,rucio_folder, pickles_folder, reading_mode, start, stop, run_list):
    run_list = [int(n) for n in re.split(r'[,\s]+', run_list.strip()) if n]
    if reading_mode == 'from - to':
        if start is None or stop is None:
            raise click.BadParameter("If reading_mode='from - to', you must provide --start and --stop")
        print(f"Processing range from {start} to {stop}")
        index_list = list(range(start, stop))
    elif reading_mode == 'all':
        index_list = list(range(0, 1))
        
        
    for run in run_list:
        print(f'\n--------- RUN {run} ---------')
        
        run_folder_name = f'{pickles_folder}/led_calibration_runs/run_{run}'
        if not os.path.exists(run_folder_name):
            os.makedirs(run_folder_name)           

        try:
            run_hdf5_filepaths = hdf5_reader.get_filepaths_from_rucio(f"{rucio_folder}/0{run}.txt")
        except Exception as e:
            print(f"{e}\n--> Skipped\n")
            continue
        
        if reading_mode == 'all':
            index_list = list(range(0, len(run_hdf5_filepaths)))
        
        if min(index_list) > len(run_hdf5_filepaths):
            print('Start > available hdf5 files --> skipped')
            continue

        if max(index_list) > len(run_hdf5_filepaths):
            index_list = list(range(start, len(run_hdf5_filepaths)))
            print(f'Stop > available hdf5 files --> go to the max file (i.e. {len(run_hdf5_filepaths)})')
        
        for i in index_list:
            pickle_name = f"{run_folder_name}/{trigger_string(full_streaming)}_{i}.pkl"
            
            print(f'Reading: run_{run}_{i}')    
            try:
                if not os.path.isfile(pickle_name):
                    wfset = hdf5_reader.WaveformSet_from_hdf5_file(run_hdf5_filepaths[i], read_full_streaming_data = full_streaming) ## modified
                    print(len(wfset.waveforms))
                    with open(pickle_name, "wb") as f:
                        pickle.dump(wfset, f)
                else:
                    print(f"File already read: run {run} file {pickle_name.split('/')[-1]}")
                
            except Exception as e:
                keywords = ["no space left on device", "file exists", "unable to open file", "truncated file"]
                if any(keyword in str(e).lower() for keyword in keywords):
                    print("\n\nCritical error detected: No space, file exists, or file corruption.\n\nSTOPPING EXECUTION.\n\n")
                    sys.exit(1)  
                else:
                    print(f'Error: {e}')
            
######################################################################################

if __name__ == "__main__":
    main()