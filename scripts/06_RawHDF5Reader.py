import click
import json
import inquirer
from pathlib import Path
from waffles.utils.utils import print_colored
import waffles.input_output.raw_hdf5_reader as reader
from waffles.input_output.persistence_utils import WaveformSet_to_file
# from waffles.input_output.pickle_hdf5_reader import WaveformSet_from_hdf5_pickle
# from typing import Optional, List

class WaveformProcessor:
    """Handles waveform data processing: reading and saving waveform sets."""

    def __init__(self, config: dict):
        """Initializes processor using a configuration dictionary."""
        self.rucio_paths_directory = config.get("rucio_dir")
        self.output_path = config.get("output_dir")
        self.run_number = config.get("run")
        self.save_single_file = config.get("save_single_file", False)
        self.self_trigger = config.get("self_trigger")  # Self-trigger filtering threshold
        self.max_files = config.get("max_files", "all")  # Limit file processing
        self.ch = self.parse_ch_dict(config.get("ch", {}))

        print_colored(f"Loaded configuration: {config}", color="INFO")
    
    def parse_ch_dict(self, ch):
        """Validates the endpoint-channel dictionary."""
        if not isinstance(ch, dict):
            raise ValueError("Invalid format: 'ch' must be a dictionary {endpoint: [channels]}.")
        
        parsed_dict = {}
        for endpoint, channels in ch.items():
            if not isinstance(channels, list) or not all(isinstance(ch, int) for ch in channels):
                raise ValueError(f"Invalid channel list for endpoint {endpoint}. Must be a list of integers.")
            parsed_dict[int(endpoint)] = channels  # Ensure endpoint keys are integers
        return parsed_dict

    def read_and_save(self) -> bool:
        """Reads waveforms and saves based on the chosen granularity."""
        print_colored(f"Reading waveforms for run {self.run_number}...", color="DEBUG")

        try:
            rucio_filepath = f"{self.rucio_paths_directory}/{str(self.run_number).zfill(6)}.txt"
            filepaths = reader.get_filepaths_from_rucio(rucio_filepath)

            if self.max_files != "all":
                filepaths = filepaths[:int(self.max_files)]  # Limit file processing

            print_colored(f"Processing {len(filepaths)} files...", color="INFO")

            if self.save_single_file:
                # Read and merge all files into one WaveformSet
                self.wfset = reader.WaveformSet_from_hdf5_files(
                    filepath_list=filepaths,
                    read_full_streaming_data=False,
                    truncate_wfs_to_minimum=False,
                    folderpath=None,
                    nrecord_start_fraction=0.0,
                    nrecord_stop_fraction=1.0,
                    subsample=1,
                    wvfm_count=1e9,
                    ch=self.ch,
                    det='HD_PDS',
                    temporal_copy_directory='/tmp',
                    erase_temporal_copy=False
                )

                if self.wfset:
                    self.write_merged_output()
            
            else:
                # Read and save each file separately
                for file in filepaths:
                    print_colored(f"Processing file: {file}", color="INFO")

                    wfset = reader.WaveformSet_from_hdf5_file(
                        filepath=file,
                        read_full_streaming_data=False,
                        truncate_wfs_to_minimum=False,
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
                        self.write_output(wfset, file)

            print_colored("All files processed successfully.", color="SUCCESS")
            return True

        except FileNotFoundError:
            print_colored(f"Error: Run file not found at {rucio_filepath}.", color="ERROR")
            return False
        except Exception as e:
            print_colored(f"An error occurred while reading input: {e}", color="ERROR")
            return False

    def write_merged_output(self) -> bool:
        """Saves the merged waveform data into a single HDF5 file."""
        output_filename = f"processed_merged_run_{self.run_number}.hdf5"
        output_filepath = Path(self.output_path) / output_filename

        print_colored(f"Saving merged waveform data to {output_filepath}...", color="DEBUG")

        try:
            WaveformSet_to_file(
                waveform_set=self.wfset,
                output_filepath=str(output_filepath),
                overwrite=True,
                format="hdf5",
                compression="gzip",
                compression_opts=5
            )

            print_colored(f"Merged WaveformSet saved successfully at {output_filepath}", color="SUCCESS")
            return True

        except Exception as e:
            print_colored(f"An error occurred while saving the merged output: {e}", color="ERROR")
            return False

    def write_output(self, wfset, input_filepath: str) -> bool:
        """Saves each waveform set separately, preserving file granularity."""
        try:
            input_filename = Path(input_filepath).name
            output_filepath = Path(self.output_path) / f"processed_{input_filename}"

            print_colored(f"Saving waveform data to {output_filepath}...", color="DEBUG")

            WaveformSet_to_file(
                waveform_set=wfset,
                output_filepath=str(output_filepath),
                overwrite=True,
                format="hdf5",
                compression="gzip",
                compression_opts=5
            )

            print_colored(f"WaveformSet saved successfully at {output_filepath}", color="SUCCESS")
            return True

        except Exception as e:
            print_colored(f"An error occurred while saving individual outputs: {e}", color="ERROR")
            return False


@click.command(help="\033[34mProcess waveform data using a JSON configuration file.\033[0m")
@click.option("--config", required=True, help="Path to JSON configuration file.", type=str)
def main(config):
    """
    CLI tool to process waveform data based on JSON configuration.
    """
    try:
        with open(config, 'r') as f:
            config_data = json.load(f)

        required_keys = ["run", "rucio_dir", "output_dir", "ch"]
        missing_keys = [key for key in required_keys if key not in config_data]
        if missing_keys:
            raise ValueError(f"Missing required keys in config file: {missing_keys}")

        processor = WaveformProcessor(config_data)
        processor.read_and_save()
    except Exception as e:
        print_colored(f"An error occurred: {e}", color="ERROR")

if __name__ == "__main__":
    main()