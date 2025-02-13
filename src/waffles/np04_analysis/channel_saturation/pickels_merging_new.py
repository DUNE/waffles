import gc
from tools_analysis import *

pickles_folder = '/afs/cern.ch/work/a/anbalbon/public/np04_beam_pickles'
set_name = 'A'
filename_notes = ''
full_streaming = False

set_folder = os.path.join(pickles_folder, f'set_{set_name}')
output_filename = f'{set_folder}/set_{set_name}_{trigger_string(full_streaming)}.pkl'

if os.path.exists(output_filename):
    print('The default filename exists, please write the new filename:')
    output_filename = f'{set_folder}/set_{set_name}_{trigger_string(full_streaming)}_' + input() + '.pkl'
    print(output_filename)

print('Creating the merged pickles')

# Controllo che ci siano file validi
all_runs = [r for r in os.listdir(set_folder) if os.path.isdir(os.path.join(set_folder, r))]
if not all_runs:
    print("No valid runs found!")
    exit(1)

file_index = 0
batch_size = 50  # Numero di file da processare per volta

for run in all_runs:
    print(f"\n\nReading run {run}")
    run_folder = os.path.join(set_folder, run)
    os.chdir(run_folder)

    file_list = [f for f in os.listdir() if f.endswith('.pkl') and ("full" in f if full_streaming else "self" in f)]
    num_files = len(file_list)
    
    if num_files == 0:
        continue

    print(f'Found {num_files} files')

    # Inizializziamo il dataset con il primo file
    with open(os.path.join(run_folder, file_list[0]), 'rb') as f:
        wfset = pickle.load(f)
        file_index += 1

    # Processiamo i file in batch
    for batch_start in tqdm(range(1, num_files, batch_size), desc="Merging files in batches", unit="batch"):
        batch_end = min(batch_start + batch_size, num_files)
        
        for i in range(batch_start, batch_end):
            with open(os.path.join(run_folder, file_list[i]), 'rb') as f:
                temp_wfset = pickle.load(f)
                wfset.merge(temp_wfset)
                del temp_wfset  # Libera la memoria
            gc.collect()  # Forza il garbage collector

print(f"# saved waveforms: {len(wfset.waveforms)}")

# Salvataggio progressivo per evitare di perdere tutto in caso di errore
with open(output_filename, "wb") as f:
    pickle.dump(wfset, f, protocol=pickle.HIGHEST_PROTOCOL)

print('\n\nDONE\n\n')
