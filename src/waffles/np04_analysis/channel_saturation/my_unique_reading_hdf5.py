# il nome del file deve contenere il numero del run, "self" o "full", il numero di hdf5 files letti 
# Creare pickles file per ogni set di run (5 run con stesse condizioni ma diversa energia)

from tools_analysis import *

full_streaming = False
n_hdf5_files = 2

rucio_folder = '/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths'
pickles_folder = '/afs/cern.ch/work/a/anbalbon/public/np04_beam_pickles'



for run_set in run_set_list_B:
  i_hdf5_files = 0
  for run in run_set['Runs'].values():
    print(f'\n--------- RUN {run} ---------')
    try:
      run_hdf5_filepaths = hdf5_reader.get_filepaths_from_rucio(f"{rucio_folder}/0{run}.txt")
    except Exception as e:
      print(f"{e}\n--> Skipped\n")
      continue
  
    if n_hdf5_files > len(run_hdf5_filepaths):
      n_hdf5_files = len(run_hdf5_filepaths)
    
    if i_hdf5_files == 0:
      print('First file ever read -> creating wfset object')
      wfset = hdf5_reader.WaveformSet_from_hdf5_file(run_hdf5_filepaths[0], read_full_streaming_data = full_streaming)
      i_hdf5_files += 1
      start_hdf5_files = 1
    else: 
      print('First run file')
      start_hdf5_files = 0
      
    for i in range(start_hdf5_files, n_hdf5_files): 
      print('Reading...')
      wfset.merge(hdf5_reader.WaveformSet_from_hdf5_file(run_hdf5_filepaths[i], read_full_streaming_data = full_streaming))
      i_hdf5_files += 1


  print('Reading finished!')
  pickle_file = f"{pickles_folder}/{run_set['Name']}_{trigger_string(full_streaming)}_{i_hdf5_files}hdf5.pkl"

  file_counter = 0
  while os.path.exists(pickle_file):
    file_counter += 1
    pickle_file = f"{pickles_folder}/{run_set['Name']}_{trigger_string(full_streaming)}_{i_hdf5_files}hdf5_v{file_counter}.pkl"

  print(f"\n\n{i_hdf5_files} hdf5 files read, Full_streaming = {full_streaming} --> Creting a pickle file: {pickle_file}")
 
  with open(pickle_file, "wb") as f:
    pickle.dump(wfset, f)

  print(f"\nFINISH: {len(wfset.waveforms)} wf saved in the pickles file!!")
