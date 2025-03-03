import click
import json
import multiprocessing
from pathlib import Path
from waffles.utils.utils import print_colored
import waffles.input_output.raw_hdf5_reader as reader
from waffles.input_output.persistence_utils import WaveformSet_to_file
from waffles.input_output.pickle_hdf5_reader import WaveformSet_from_hdf5_pickle
from waffles.data_classes.WaveformSet import WaveformSet

class WaveformProcessor:
    """Handles waveform data processing: reading and saving waveform sets."""

    def __init__(self, config: dict):
        self.rucio_paths_directory = config.get("rucio_dir")
        self.output_path = config.get("output_dir")
        self.run_number = config.get("run")
        self.debug = config.get("debug", True)
        self.max_files = config.get("max_files", "all")
        self.save_single_file = config.get("save_single_file", False)
        self.ch = self.parse_ch_dict(config.get("ch", {}))
        self.wfset_list = []  # Store waveform sets for merging

    def parse_ch_dict(self, ch):
        """Validates the endpoint-channel dictionary."""
        if not isinstance(ch, dict):
            raise ValueError("Invalid format: 'ch' must be a dictionary {endpoint: [channels]}.")

        parsed_dict = {}
        for endpoint, channels in ch.items():
            if not isinstance(channels, list) or not all(isinstance(ch, int) for ch in channels):
                raise ValueError(f"Invalid channel list for endpoint {endpoint}. Must be a list of integers.")
            parsed_dict[int(endpoint)] = channels  # Ensure keys are integers
        return parsed_dict

    def process_file(self, file):
        """Processes a single waveform file."""
        try:
            print_colored(f"Processing file: {file}", color="INFO")

            wfset = reader.WaveformSet_from_hdf5_files(
                filepath_list=[file],
                read_full_streaming_data=False,
                truncate_wfs_to_minimum=True,
                nrecord_start_fraction=0.0,
                nrecord_stop_fraction=1.0,
                subsample=1,
                wvfm_count=1e9,
                ch=self.ch,
                det='HD_PDS',
                temporal_copy_directory='/tmp',
                erase_temporal_copy=False
            )

            if wfset:
                if self.save_single_file:
                    return wfset
                else:
                    self.write_output(wfset, file)
        except Exception as e:
            print_colored(f"Error processing file {file}: {e}", color="ERROR")
        return None

    def read_and_save(self) -> bool:
        """Reads waveforms in parallel and saves them."""
        print_colored(f"Reading waveforms for run {self.run_number}...", color="DEBUG")

        try:
            rucio_filepath = f"{self.rucio_paths_directory}/{str(self.run_number).zfill(6)}.txt"
            filepaths = reader.get_filepaths_from_rucio(rucio_filepath)

            if self.max_files != "all":
                filepaths = filepaths[:int(self.max_files)]  # Limit files if specified

            print_colored(f"Processing {len(filepaths)} files in parallel...", color="INFO")

            # Process files in parallel
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                results = pool.map(self.process_file, filepaths)

            if self.save_single_file:
                self.wfset_list = [wf for wf in results if wf is not None]
                if self.wfset_list:
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
        """Saves the waveform data to an HDF5 file."""
        if isinstance(wfset, list):
            output_filename = f"processed_merged_run_{self.run_number}.hdf5"
            output_filepath = Path(self.output_path) / output_filename
            combined_wfset = sum(wfset)  # Merge all WaveformSets
        else:
            input_filename = Path(input_filepath).name
            output_filepath = Path(self.output_path) / f"processed_{input_filename}"
            combined_wfset = wfset

        print_colored(f"Saving waveform data to {output_filepath}...", color="DEBUG")

        try:
            WaveformSet_to_file(
                waveform_set=combined_wfset,
                output_filepath=str(output_filepath),
                overwrite=True,
                format="hdf5",
                compression="gzip",
                compression_opts=5
            )

            # Reload for verification
            new_ws = WaveformSet_from_hdf5_pickle(str(output_filepath))
            print(new_ws)

            print_colored(f"WaveformSet saved successfully at {output_filepath}", color="SUCCESS")
            return True

        except Exception as e:
            print_colored(f"An error occurred while saving the output: {e}", color="ERROR")
            return False


@click.command(help="\033[34mProcess waveform data using a JSON configuration file.\033[0m")
@click.option("--config", required=True, help="Path to JSON configuration file.", type=str)
def main(config):
    """CLI tool to process waveform data based on JSON configuration."""
    try:
        with open(config, 'r') as f:
            config_data = json.load(f)

        required_keys = ["run", "rucio_dir", "output_dir", "ch"]
        missing_keys = [key for key in required_keys if key not in config_data]
        if missing_keys:
            raise ValueError(f"Missing required keys in config file: {missing_keys}")

        processor = WaveformProcessor(config_data)
        processor.read_and_save()

    except FileNotFoundError:
        print_colored(f"Error: Config file '{config}' not found.", color="ERROR")
    except json.JSONDecodeError:
        print_colored(f"Error: Invalid JSON format in '{config}'.", color="ERROR")
    except Exception as e:
        print_colored(f"An error occurred: {e}", color="ERROR")


if __name__ == "__main__":
    main()