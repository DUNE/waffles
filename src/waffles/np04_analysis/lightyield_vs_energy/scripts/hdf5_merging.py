from imports import *
import h5py
import os, re, pickle, click
from datetime import datetime
from tqdm import tqdm
from waffles.input_output.hdf5_structured import load_structured_waveformset

@click.command()
@click.option("--run_number",
              required=True,
              type=int,
              help="Run number to read (e.g. 27343)")
@click.option("--output_filename",
              required=True,
              help="Output pickle filename (without extension)")
@click.option("--overwrite",
              is_flag=True,
              help="If the output file exists, overwrite it")
@click.option("--save_file",
              default='yes',
              type=click.Choice(['yes', 'no'], case_sensitive=False),
              help="Save the pickle file? (yes/no, default yes)")

def main(run_number, output_filename, overwrite, save_file):
    # Folder where the processed HDF5 files are
    run_folder = f"/afs/cern.ch/work/a/anbalbon/public/reading_beamrun_NEW/run0{run_number}"

    if not os.path.isdir(run_folder):
        raise click.BadParameter(f"The folder {run_folder} does not exist")

    # Output file
    output_filename = f"{run_folder}/{output_filename}.pkl"

    if os.path.exists(output_filename) and not overwrite:
        raise click.BadParameter(f"The file {output_filename} already exists. Use --overwrite to overwrite.")

    # List all processed HDF5 files
    all_files = sorted([
        f for f in os.listdir(run_folder)
        if f.endswith(".hdf5") and f.startswith("processed_")
    ])

    if not all_files:
        print(f"No processed HDF5 files found in {run_folder}")
        return

    print(f"Found {len(all_files)} files to merge in {run_folder}\n")

    wfset = None
    i_index = 0

    for fname in tqdm(all_files, desc="Merging files", unit="file"):
        filepath = os.path.join(run_folder, fname)
        current_wfset = load_structured_waveformset(filepath)  # load HDF5 structured waveform set

        if i_index == 0:
            wfset = current_wfset
        else:
            wfset.merge(current_wfset)
        i_index += 1

    print(f"\n# files read: {i_index}")
    print(f"# waveforms: {len(wfset.waveforms)}\n")

    if i_index == 0:
        print(" --- NO DATA FOUND - NOTHING SAVED ---\n")
        return

    if save_file == "yes":
        print(f"Saving merged waveform set to {output_filename} ...")
        with open(output_filename, "wb") as f:
            pickle.dump(wfset, f)

        summary_txt = os.path.join(run_folder, "summary.txt")
        new_summary_entry = (
            f"Output filename: {output_filename}\n"
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Run number: {run_number}\n"
            f"# files read: {i_index}\n"
            f"# waveforms: {len(wfset.waveforms)}\n\n"
        )

        with open(summary_txt, "a") as file:
            file.write(new_summary_entry)

        print(f"Summary updated in {summary_txt}")
    else:
        print("File not saved!")

    print("\n\t------------------------\n\t\tDONE ✅\n\t------------------------\n")


#####################################################################

if __name__ == "__main__":
    main()
