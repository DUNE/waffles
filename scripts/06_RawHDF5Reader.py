import click
import inquirer
from pathlib import Path
from waffles.utils.utils import print_colored
import waffles.input_output.raw_hdf5_reader as reader
from waffles.input_output.persistence_utils import WaveformSet_to_file
from waffles.input_output.pickle_hdf5_reader import WaveformSet_from_hdf5_pickle
from waffles.input_output.waveform import Waveform, WaveformSet  # Ensure correct import

class WaveformProcessor:
    """Handles waveform data processing: reading, filtering, and saving waveform sets."""

    def __init__(self, rucio_paths_directory: str, output_path: str, run_number: int, debug: bool = True, 
                 allowed_endpoints: str = "", allowed_channels: str = "", save_single_file: bool = False, self_trigger: int = None):
        self.rucio_paths_directory = rucio_paths_directory
        self.output_path = output_path
        self.run_number = run_number
        self.debug = debug
        self.save_single_file = save_single_file
        self.self_trigger = self_trigger  # Self-trigger filtering threshold
        self.wfset_list = []  # List to store all waveform sets

        # Convert comma-separated strings to lists
        self.allowed_endpoints = [int(e) for e in allowed_endpoints.split(",") if e.strip().isdigit()] if allowed_endpoints else []
        self.allowed_channels = [int(c) for c in allowed_channels.split(",") if c.strip().isdigit()] if allowed_channels else []

    def beam_self_trigger_filter(self, waveform: Waveform) -> bool:
        """Filters waveforms based on DAQ PDS time offset."""
        if self.self_trigger is None:
            return True  # No filtering applied if self_trigger is not set
        
        timeoffset_min = -self.self_trigger
        timeoffset_max = self.self_trigger
        daq_pds_timeoffset = waveform.timestamp - waveform.daq_window_timestamp
        return timeoffset_min < daq_pds_timeoffset < timeoffset_max

    def read_and_save(self) -> bool:
        """Reads waveforms for the current run, applies filters, and saves them."""
        print_colored(f"Reading waveforms for run {self.run_number}...", color="DEBUG")

        try:
            rucio_filepath = f"{self.rucio_paths_directory}/{str(self.run_number).zfill(6)}.txt"
            filepaths = reader.get_filepaths_from_rucio(rucio_filepath)

            if len(filepaths) > 5:
                print_colored(f"This run has {len(filepaths)} HDF5 files. Processing them individually...", color="WARNING")
                file_lim = inquirer.prompt([inquirer.Text("file_lim", message="How many of them do we process?")])["file_lim"]
                filepaths = filepaths[:int(file_lim)]

            for file in filepaths:
                print_colored(f"Processing file: {file}", color="INFO")

                # Load waveforms from a single file
                wfset = reader.WaveformSet_from_hdf5_files(
                    [file],  # Process one file at a time
                    read_full_streaming_data=False,
                    truncate_wfs_to_minimum=True,
                    nrecord_start_fraction=0.0,
                    nrecord_stop_fraction=1.0,
                    subsample=1,
                    wvfm_count=1e9,
                    allowed_endpoints=self.allowed_endpoints,
                    allowed_channels=self.allowed_channels,
                    det='HD_PDS',
                    temporal_copy_directory='/tmp',
                    erase_temporal_copy=True
                )

                if wfset:
                    # Apply self-trigger filtering if enabled
                    if self.self_trigger is not None:
                        print_colored("Applying beam self-trigger filter...", color="DEBUG")
                        wfset = WaveformSet.from_filtered_WaveformSet(wfset, self.beam_self_trigger_filter)
                        print_colored(f"✔ Number of waveforms after filtering: {len(wfset.waveforms)}", color="SUCCESS")
                    
                    if self.save_single_file:
                        self.wfset_list.append(wfset)  # Store for later merging
                    else:
                        self.write_output(wfset, file)

            # Save all processed waveforms into a single file if the option is enabled
            if self.save_single_file and self.wfset_list:
                self.write_output(self.wfset_list, f"merged_run_{self.run_number}.hdf5")

            print_colored("All files processed successfully.", color="SUCCESS")
            return True

        except FileNotFoundError:
            print_colored(f"Error: Run file not found at {rucio_filepath}.", color="ERROR")
            return False
        except Exception as e:
            print_colored(f"An error occurred while reading input: {e}", color="ERROR")
            return False

    def write_output(self, wfset, input_filepath: str) -> bool:
        """Saves the waveform data to an HDF5 file. Supports both single-file and individual-file modes."""
        if isinstance(wfset, list):  # Handling single-file mode
            output_filename = f"processed_merged_run_{self.run_number}.hdf5"
            print_colored(f"Saving merged waveform data to {output_filename}...", color="DEBUG")

            combined_wfset = sum(wfset)  # Merge all waveform sets
            output_filepath = Path(self.output_path) / output_filename
        else:
            input_filename = Path(input_filepath).name
            output_filepath = Path(self.output_path) / f"processed_{input_filename}"

        print_colored(f"Saving waveform data to {output_filepath}...", color="DEBUG")

        try:
            WaveformSet_to_file(
                waveform_set=combined_wfset if isinstance(wfset, list) else wfset,
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
@click.option("--allowed-endpoints", default="", help="Comma-separated list of allowed endpoints", type=str)
@click.option("--allowed-channels", default="", help="Comma-separated list of allowed channels", type=str)
@click.option("--save-single-file", is_flag=True, help="Save all processed waveforms in a single HDF5 file")
@click.option("--self-trigger", default=None, help="Threshold for beam self-trigger filtering (single int)", type=int)
def main(run, debug, rucio_dir, output_dir, allowed_endpoints, allowed_channels, save_single_file, self_trigger):
    """CLI tool to process waveform data, apply filtering, and save as HDF5."""
    processor = WaveformProcessor(
        rucio_paths_directory=rucio_dir,
        output_path=output_dir,
        run_number=int(run),
        debug=debug,
        allowed_endpoints=allowed_endpoints,
        allowed_channels=allowed_channels,
        save_single_file=save_single_file,
        self_trigger=self_trigger
    )
    processor.read_and_save()


if __name__ == "__main__":
    main()