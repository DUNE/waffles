import click
import inquirer
from pathlib import Path
from waffles.utils.utils import print_colored
import waffles.input_output.raw_hdf5_reader as reader
from waffles.input_output.persistence_utils import WaveformSet_to_file
from waffles.input_output.pickle_hdf5_reader import WaveformSet_from_hdf5_pickle

class WaveformProcessor:
    """Handles waveform data processing: reading and saving waveform sets."""

    def __init__(self, rucio_paths_directory: str, output_path: str, run_number: int, debug: bool = True):
        self.rucio_paths_directory = rucio_paths_directory
        self.output_path = output_path
        self.run_number = run_number
        self.debug = debug
        self.wfset = None  # Placeholder for WaveformSet

    def read_and_save(self) -> bool:
        """Reads waveforms for the current run and saves each file separately if the list is too large."""
        print_colored(f"Reading waveforms for run {self.run_number}...", color="DEBUG")

        try:
            rucio_filepath = f"{self.rucio_paths_directory}/{str(self.run_number).zfill(6)}.txt"
            filepaths = reader.get_filepaths_from_rucio(rucio_filepath)

            if len(filepaths) > 5:
                print_colored(f"This run has {len(filepaths)} HDF5 files. Processing them individually...", color="WARNING")
                file_lim = inquirer.prompt([inquirer.Text("file_lim", message="How many of them do we process?")])["file_lim"]
                filepaths = filepaths[:int(file_lim)]
            else:
                file_lim = len(filepaths)

            for file in filepaths:
                print_colored(f"Processing file: {file}", color="INFO")

                # Load waveforms from a single file
                self.wfset = reader.WaveformSet_from_hdf5_files(
                    [file],  # Process one file at a time
                    read_full_streaming_data=False,
                    truncate_wfs_to_minimum=True,
                    nrecord_start_fraction=0.0,
                    nrecord_stop_fraction=1.0,
                    subsample=1,
                    wvfm_count=1e9,
                    allowed_endpoints=[],
                    det='HD_PDS',
                    temporal_copy_directory='/tmp',
                    erase_temporal_copy=True
                )

                if self.wfset:
                    self.write_output(file)

            print_colored("All files processed successfully.", color="SUCCESS")
            return True

        except FileNotFoundError:
            print_colored(f"Error: Run file not found at {rucio_filepath}.", color="ERROR")
            return False
        except Exception as e:
            print_colored(f"An error occurred while reading input: {e}", color="ERROR")
            return False

    def write_output(self, input_filepath: str) -> bool:
        """Saves the waveform data to an HDF5 file, maintaining the input file naming structure."""
        input_filename = Path(input_filepath).name
        output_filepath = Path(self.output_path) / f"processed_{input_filename}"
        
        print_colored(f"Saving waveform data to {output_filepath}...", color="DEBUG")

        try:
            WaveformSet_to_file(
                waveform_set=self.wfset,
                output_filepath=str(output_filepath),
                overwrite=True,
                format="hdf5",
                compression="gzip",
                compression_opts=5
            )

            # Reload and print the saved file for verification
            new_ws = WaveformSet_from_hdf5_pickle(str(output_filepath))
            print(new_ws)

            print_colored(f"WaveformSet saved successfully at {output_filepath}", color="SUCCESS")
            return True

        except Exception as e:
            print_colored(f"An error occurred while saving the output: {e}", color="ERROR")
            return False


@click.command(help="\033[34mProcess peak/pedestal variables and save the WaveformSet in an HDF5 file.\n\033[0m")
@click.option("--run", default=None, help="Run number(s) to process (comma-separated).", type=str)
@click.option("--debug", default=True, help="Enable debug mode", type=bool)
@click.option("--rucio-dir", default="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths", help="Path to Rucio directory", type=str)
@click.option("--output-dir", default="../data", help="Path to save the processed HDF5 files", type=str)
def main(run, debug, rucio_dir, output_dir):
    """
    CLI tool to process waveform data and save as HDF5 instead of pickle.
    """
    if run is None:
        run_list = inquirer.prompt([inquirer.Text("run", message="Provide the run number(s), separated by commas:")])["run"].split(",")
    else:
        run_list = run.split(",")

    for r in run_list:
        try:
            run_number = int(r.strip())
        except ValueError:
            print_colored(f"Invalid run number: {r}. Skipping...", color="ERROR")
            continue

        processor = WaveformProcessor(rucio_paths_directory=rucio_dir, output_path=output_dir, run_number=run_number, debug=debug)
        
        processor.read_and_save()


if __name__ == "__main__":
    main()