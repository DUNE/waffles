from imports import *
import h5py
import os, re, pickle, click
from datetime import datetime
from tqdm import tqdm
import glob
from waffles.input_output.hdf5_structured import load_structured_waveformset

@click.command()
@click.option("--set_name", 
              default='A',
              type=click.Choice(['A', 'B'], case_sensitive=False),
              help="Which set do you want to analyze? (A or B)")
@click.option("--endpoint", 
              default='109', 
              help="Endpoint number to use: 104, 105, 107, 109, 111, 112, 113. Default is 109.")
@click.option("--input_folder",
              default = "/afs/cern.ch/work/a/anbalbon/public/reading_beamrun_NEW", # /afs/cern.ch/work/a/anbalbon/public/reading_beamrun_NEW  /eos/user/a/anbalbon/reading_beamrun_NEW
              help="Input folder (where processed hdf5 files are, without run number)")
@click.option("--output_folder",
              default = "/afs/cern.ch/work/a/anbalbon/public/reading_beamrun_NEW", # /afs/cern.ch/work/a/anbalbon/public/reading_beamrun_NEW  /eos/user/a/anbalbon/reading_beamrun_NEW
              help="Output folder (without run number)")
@click.option("--output_name",
              required=True,
              help="Output name (without extension)")
@click.option("--overwrite/--no-overwrite",
              default=True,
              help="If the output file exists, overwrite it (default: overwrite)")
@click.option("--save_file",
              default='yes',
              type=click.Choice(['yes', 'no'], case_sensitive=False),
              help="Save the output file? (yes/no, default yes)")


def main(set_name, endpoint, input_folder, output_folder, output_name, overwrite, save_file):

    # Output file
    output_filename = f"{output_folder}/set_{set_name}_beam_{output_name}.pkl"
    if os.path.exists(output_filename) and not overwrite:
        raise click.BadParameter(f"The file {output_filename} already exists. Use --overwrite to overwrite.")

    # Set info
    with open('/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/data/beam_run_info.json', "r") as file:
        run_set_list = json.load(file)

    print('... searching for hdf5 processed files to merge ...\n')
    all_files = []
    for energy, run in run_set_list[set_name]['Runs'].items():
        print(f"\n\n--- Reading run {run} ---")

        if 'eos' in input_folder :
            run_folder = f"{input_folder}/run0{run}/processed_hdf5"
        else:
            run_folder = f"{input_folder}/run0{run}"

        if os.path.isdir(run_folder):
            run_files = glob.glob(os.path.join(run_folder, "processed_np04hd_raw*hdf5_structured.hdf5"))
            for f in run_files:
                all_files.append(os.path.abspath(f))
            if len(run_files) == 0:
                print('No files for this run\n')
                continue

    print(f'... starting merging of {len(all_files)} ...\n')        
    wfset = None
    i_index = 0
    i_index_error = 0
    for file in tqdm(all_files, desc="Merging files", unit="file"):
        try:
            current_wfset = WaveformSet.from_filtered_WaveformSet(load_structured_waveformset(file), endpoint_list_filter, endpoint_list = [int(endpoint)]) # load HDF5 structured waveform set + filter to select the endpoint
            if i_index == 0:
                wfset = current_wfset
            else:
                wfset.merge(current_wfset)
            i_index += 1
        except Exception as e:
            print(f"Error loading {file}: {e}")
            i_index_error += 1
            continue

    counts = {}
    for run in run_set_list[set_name]['Runs'].values():
        run_str = str(run)
        counts[run_str] = sum(1 for f in all_files if run_str in os.path.basename(f))
    counts_str = "\n".join([f"  Run {r}: {c} file" for r, c in counts.items()])


    print(f"\n# files read: {i_index}")
    print(f"# files with errors: {i_index_error}")
    print(f"# waveforms: {len(wfset.waveforms)}\n")
    print(f"Files per run:\n{counts_str}\n\n")

    if i_index == 0:
        print(" --- NO DATA FOUND - NOTHING SAVED ---\n")
        return

    if save_file == "yes" and i_index != 0:
        print(f"\nSaving merged waveform set to {output_filename} ...")
        with open(output_filename, "wb") as f:
            pickle.dump(wfset, f)

        summary_txt = os.path.join(output_folder, "summary.txt")
        new_summary_entry = (
            f"Output filename: {output_name}\n"
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"# files read: {i_index}\n"
            f"# files with errors: {i_index_error}\n"
            f"# waveforms: {len(wfset.waveforms)}\n"
            f"Files per run:\n{counts_str}\n\n"
        )

        with open(summary_txt, "a") as file:
            file.write(new_summary_entry)

        print(f"Summary updated")
    else:
        print("File not saved!")

    print("\n\t------------------------\n\t\tDONE ✅\n\t------------------------\n")

    
    


if __name__ == "__main__":
    main()