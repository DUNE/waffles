from tools_analysis import *
from tqdm import tqdm  # Importiamo tqdm per la barra di avanzamento

pickles_folder = '/afs/cern.ch/work/a/anbalbon/public/np04_beam_pickles'
set_name = 'A'
filename_notes = ''
full_streaming = False

set_folder = os.path.join(pickles_folder, f'set_{set_name}')
output_filename = f'{set_folder}/set_{set_name}_{trigger_string(full_streaming)}.pkl'
if os.path.exists(output_filename):
    print('The default filename exists, please write the new filename:')
    output_filename = f'{set_folder}/set_{set_name}_{trigger_string(full_streaming)}_' + input() +'.pkl'
    print(output_filename)


print('Creating the merged pickles')
file_index = 0 

for run in os.listdir(set_folder):
    print(f"\n\nReading run {run}")
    run_folder = os.path.join(set_folder, run)
    if os.path.isdir(run_folder):
        os.chdir(run_folder)
        file_list = [f for f in os.listdir() if (f.endswith('.pkl')) and ("full" in f if full_streaming else "self" in f)] 
        print(f'{len(file_list)} files')

        if len(file_list) > 0: 
            with open(f'{run_folder}/{file_list[0]}', 'rb') as f:
                if file_index == 0:
                    wfset = pickle.load(f)
                    file_index += 1
                else: 
                    wfset.merge(pickle.load(f))

            # Creiamo la barra di avanzamento
            for i in tqdm(range(1, len(file_list)), desc="Merging files", unit="file"):
                with open(f'{run_folder}/{file_list[i]}', 'rb') as f:
                    wfset.merge(pickle.load(f))
        else:
            continue
        
print(f"# saved waveforms: {len(wfset.waveforms)}")


with open(output_filename, "wb") as f:
    pickle.dump(wfset, f)

print('\n\DONE \n\n')