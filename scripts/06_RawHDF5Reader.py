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
        self.combined_wfset=None
        try:
            #rucio_filepath = f"{self.rucio_paths_directory}/{str(self.run_number).zfill(6)}.txt"
            #filepaths = reader.get_filepaths_from_rucio(rucio_filepath)
            '''
            filepaths=["/eos/experiment/neutplatform/protodune/dune/hd-protodune/3f/0b/np04hd_raw_run030201_0000_dataflow0_datawriter_0_20241016T095316.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/1e/89/np04hd_raw_run030201_0001_dataflow0_datawriter_0_20241016T095558.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/0a/1c/np04hd_raw_run030201_0002_dataflow0_datawriter_0_20241016T095731.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/cb/bf/np04hd_raw_run030201_0003_dataflow0_datawriter_0_20241016T100014.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/a7/61/np04hd_raw_run030201_0004_dataflow0_datawriter_0_20241016T100200.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/8d/a9/np04hd_raw_run030201_0005_dataflow0_datawriter_0_20241016T100523.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/20/ce/np04hd_raw_run030201_0006_dataflow0_datawriter_0_20241016T100848.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/49/5c/np04hd_raw_run030201_0007_dataflow0_datawriter_0_20241016T101148.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/0d/66/np04hd_raw_run030201_0008_dataflow0_datawriter_0_20241016T101720.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/d9/51/np04hd_raw_run030201_0009_dataflow0_datawriter_0_20241016T102013.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/ad/5d/np04hd_raw_run030201_0010_dataflow0_datawriter_0_20241016T102336.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/06/3d/np04hd_raw_run030201_0011_dataflow0_datawriter_0_20241016T102736.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/ac/d4/np04hd_raw_run030201_0012_dataflow0_datawriter_0_20241016T104859.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/65/21/np04hd_raw_run030201_0013_dataflow0_datawriter_0_20241016T105337.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/f6/8a/np04hd_raw_run030201_0014_dataflow0_datawriter_0_20241016T105754.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/1c/2b/np04hd_raw_run030201_0015_dataflow0_datawriter_0_20241016T110201.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/1f/eb/np04hd_raw_run030201_0016_dataflow0_datawriter_0_20241016T110742.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/02/59/np04hd_raw_run030201_0017_dataflow0_datawriter_0_20241016T110954.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/42/a2/np04hd_raw_run030201_0018_dataflow0_datawriter_0_20241016T111106.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/d3/b4/np04hd_raw_run030201_0019_dataflow0_datawriter_0_20241016T111318.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/9d/39/np04hd_raw_run030201_0020_dataflow0_datawriter_0_20241016T111441.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/f7/0e/np04hd_raw_run030201_0021_dataflow0_datawriter_0_20241016T112122.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/de/b3/np04hd_raw_run030201_0022_dataflow0_datawriter_0_20241016T113157.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/f1/24/np04hd_raw_run030201_0023_dataflow0_datawriter_0_20241016T113443.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/fc/46/np04hd_raw_run030201_0024_dataflow0_datawriter_0_20241016T113915.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/5a/1b/np04hd_raw_run030201_0025_dataflow0_datawriter_0_20241016T114916.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/dd/6d/np04hd_raw_run030201_0026_dataflow0_datawriter_0_20241016T115353.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/af/ab/np04hd_raw_run030201_0027_dataflow0_datawriter_0_20241016T115654.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/d6/13/np04hd_raw_run030201_0028_dataflow0_datawriter_0_20241016T115908.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/80/d1/np04hd_raw_run030201_0029_dataflow0_datawriter_0_20241016T120059.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/f1/ee/np04hd_raw_run030201_0030_dataflow0_datawriter_0_20241016T120406.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/6d/19/np04hd_raw_run030201_0031_dataflow0_datawriter_0_20241016T120548.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/0d/77/np04hd_raw_run030201_0032_dataflow0_datawriter_0_20241016T121005.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/7b/b5/np04hd_raw_run030201_0033_dataflow0_datawriter_0_20241016T121300.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/2e/21/np04hd_raw_run030201_0034_dataflow0_datawriter_0_20241016T121704.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/9c/27/np04hd_raw_run030201_0035_dataflow0_datawriter_0_20241016T122159.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/a1/2e/np04hd_raw_run030201_0036_dataflow0_datawriter_0_20241016T122404.hdf5"]   
            '''
            
            filepaths=["/eos/experiment/neutplatform/protodune/dune/hd-protodune/b7/6b/np04hd_raw_run030202_0000_dataflow0_datawriter_0_20241016T124143.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/27/fe/np04hd_raw_run030202_0001_dataflow0_datawriter_0_20241016T124604.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/e6/c5/np04hd_tp_run030202_0000_tpwriter_tpswriter_20241016T124115.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/b1/f1/np04hd_tp_run030202_0001_tpwriter_tpswriter_20241016T124157.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/dd/c4/np04hd_tp_run030202_0002_tpwriter_tpswriter_20241016T124236.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/1d/ec/np04hd_tp_run030202_0003_tpwriter_tpswriter_20241016T124317.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/ba/b9/np04hd_tp_run030202_0004_tpwriter_tpswriter_20241016T124356.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/c9/61/np04hd_tp_run030202_0005_tpwriter_tpswriter_20241016T124436.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/e3/2a/np04hd_tp_run030202_0006_tpwriter_tpswriter_20241016T124517.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/74/61/np04hd_tp_run030202_0007_tpwriter_tpswriter_20241016T124556.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/99/64/np04hd_tp_run030202_0008_tpwriter_tpswriter_20241016T124636.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/08/da/np04hd_tp_run030202_0009_tpwriter_tpswriter_20241016T124717.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/3a/d4/np04hd_tp_run030202_0010_tpwriter_tpswriter_20241016T124757.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/ea/4d/np04hd_tp_run030202_0011_tpwriter_tpswriter_20241016T124838.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/f5/f2/np04hd_tp_run030202_0012_tpwriter_tpswriter_20241016T124918.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/83/e3/np04hd_tp_run030202_0013_tpwriter_tpswriter_20241016T124959.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/b7/6b/np04hd_raw_run030202_0000_dataflow0_datawriter_0_20241016T124143.hdf5",
                        "/eos/experiment/neutplatform/protodune/dune/hd-protodune/27/fe/np04hd_raw_run030202_0001_dataflow0_datawriter_0_20241016T124604.hdf5"]
            
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
        output_filename = f"wfset_{self.max_files}_{self.run_number}.hdf5"
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

        required_keys = ["run", "rucio_dir", "output_dir","ch"]
        missing_keys = [key for key in required_keys if key not in config_data]
        if missing_keys:
            raise ValueError(f"Missing required keys in config file: {missing_keys}")

        processor = WaveformProcessor(config_data)
        processor.read_and_save()
    except Exception as e:
        print_colored(f"An error occurred: {e}", color="ERROR")

if __name__ == "__main__":
    main()