# il nome del file deve contenere il numero del run, "self" o "full", il numero di hdf5 files letti 
# Crea un pickles file per ogni hdf5 file
# eseguire poi il programma: ......... per fare il merging

from tools_analysis import *

full_streaming = False
hdf5_per_run_start = 40 # Included
hdf5_per_run_stop = 45 # Not included
hdf5_per_run = hdf5_per_run_stop - hdf5_per_run_start

rucio_folder = '/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths'
pickles_folder = '/afs/cern.ch/work/a/anbalbon/public/np04_beam_pickles'

for run_set in run_set_list:
  set_folder_name = f"{pickles_folder}/set_{run_set['Name']}"
  if not os.path.exists(set_folder_name):
    os.makedirs(set_folder_name)  
    
  for run in run_set['Runs'].values():
    print(f'\n--------- RUN {run} ---------')
    
    run_folder_name = f'{set_folder_name}/run_{run}'
    if not os.path.exists(run_folder_name):
        os.makedirs(run_folder_name)           

    try:
      run_hdf5_filepaths = hdf5_reader.get_filepaths_from_rucio(f"{rucio_folder}/0{run}.txt")
    except Exception as e:
      print(f"{e}\n--> Skipped\n")
      continue
    
    if hdf5_per_run_start > len(run_hdf5_filepaths):
      print('Start > available hdf5 files')
      continue

    if hdf5_per_run_stop > len(run_hdf5_filepaths):
      hdf5_per_run_stop = len(run_hdf5_filepaths)
      print('Stop > available hdf5 files --> go to the max file')
        
    for i in range(hdf5_per_run_start, hdf5_per_run_stop):
      pickle_name = f"{run_folder_name}/{trigger_string(full_streaming)}_{i}.pkl"

      try:
        if not os.path.isfile(pickle_name):
          print(f'Reading: run_{run}_{i}')
          wfset = hdf5_reader.WaveformSet_from_hdf5_file(run_hdf5_filepaths[i], read_full_streaming_data = full_streaming)
          with open(pickle_name, "wb") as f:
            pickle.dump(wfset, f)
        else:
          print(f'File already read: run_{run}_{i}')
          
      except Exception as e:
        keywords = ["no space left on device", "file exists", "unable to open file", "truncated file"]
        if any(keyword in str(e).lower() for keyword in keywords):
            print("\n\nCritical error detected: No space, file exists, or file corruption.\n\nSTOPPING EXECUTION.\n\n")
            sys.exit(1)  