# save_structured_from_config.py

import json
import click
from pathlib import Path
import numpy as np
from waffles.utils.utils import print_colored
import waffles.input_output.raw_hdf5_reader as reader
from waffles.input_output.persistence_utils import WaveformSet_to_file
from waffles.input_output.hdf5_structured import load_structured_waveformset
from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.WaveformSet import WaveformSet

class WaveformProcessor:
    """Handles waveform data processing and structured HDF5 saving."""

    def __init__(self, config: dict):
        self.rucio_paths_directory = config.get("rucio_dir")
        self.output_path = config.get("output_dir")
        self.detector = config.get("det")
        self.run_number = config.get("run")
        self.save_single_file = config.get("save_single_file", False)
        self.max_files = config.get("max_files", "all")
        self.ch = self.parse_ch_dict(config.get("ch", {}))

        print_colored(f"Loaded configuration: {config}", color="INFO")

    def parse_ch_dict(self, ch):
        if not isinstance(ch, dict):
            raise ValueError("'ch' must be a dictionary {endpoint: [channels]}.")
        parsed = {}
        for endpoint, chans in ch.items():
            if not isinstance(chans, list) or not all(isinstance(c, int) for c in chans):
                raise ValueError(f"Invalid channel list for endpoint {endpoint}.")
            parsed[int(endpoint)] = chans
        return parsed

    def ensure_waveformset(self, wfset):
        if isinstance(wfset, WaveformSet):
            return wfset
        if isinstance(wfset, list) and all(isinstance(w, Waveform) for w in wfset):
            print_colored("🛠️ Auto-wrapping list of Waveforms into WaveformSet", color="DEBUG")
            return WaveformSet(wfset)
        raise TypeError(f"Expected WaveformSet or list of Waveforms, got {type(wfset)}")

    def read_and_save(self):
        print_colored(f"Reading waveforms for run {self.run_number}...", color="DEBUG")

        try:
            rucio_filepath = f"{self.rucio_paths_directory}/{str(self.run_number).zfill(6)}.txt"
            filepaths = reader.get_filepaths_from_rucio(rucio_filepath)

            if self.max_files != "all":
                filepaths = filepaths[:int(self.max_files)]

            print_colored(f"Processing {len(filepaths)} files...", color="INFO")

            if self.save_single_file:
                self.wfset = reader.WaveformSet_from_hdf5_files(
                    filepath_list=filepaths,
                    read_full_streaming_data=False,
                    truncate_wfs_to_minimum=False,
                    folderpath=None,
                    nrecord_start_fraction=0.0,
                    nrecord_stop_fraction=0.1,
                    subsample=1,
                    wvfm_count=1e9,
                    ch=self.ch,
                    det=self.detector,
                    temporal_copy_directory='/tmp',
                    erase_temporal_copy=False
                )
                self.wfset = self.ensure_waveformset(self.wfset)
                if self.wfset:
                    self.write_merged_output()

            else:
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
                        det=self.detector,
                        temporal_copy_directory='/tmp',
                        erase_temporal_copy=False
                    )
                    wfset = self.ensure_waveformset(wfset)
                    if wfset:
                        self.write_output(wfset, file)

            print_colored("All files processed successfully.", color="SUCCESS")
            return True

        except FileNotFoundError:
            print_colored(f"Run file not found at {rucio_filepath}.", color="ERROR")
            return False
        except Exception as e:
            print_colored(f"An error occurred: {e}", color="ERROR")
            return False

    def write_merged_output(self):
        output_filename = f"processed_merged_run_{self.run_number}_structured.hdf5"
        output_filepath = Path(self.output_path) / output_filename

        print_colored(f"Saving merged waveform data to {output_filepath}...", color="DEBUG")
        try:
            self.wfset = self.ensure_waveformset(self.wfset)
            print_colored(f"📦 About to save WaveformSet with {len(self.wfset.waveforms)} waveforms", color="DEBUG")

            WaveformSet_to_file(
                waveform_set=self.wfset,
                output_filepath=str(output_filepath),
                overwrite=True,
                format="hdf5",
                compression="gzip",
                compression_opts=0,
                structured=True
            )
            print_colored(f"Merged WaveformSet saved to {output_filepath}", color="SUCCESS")

            wfset_loaded = load_structured_waveformset(str(output_filepath))
            self.compare_waveformsets(self.wfset, wfset_loaded)

            return True
        except Exception as e:
            print_colored(f"Error saving merged output: {e}", color="ERROR")
            return False
        
    def compare_waveformsets(self, original, loaded):
        original = self.ensure_waveformset(original)
        loaded = self.ensure_waveformset(loaded)

        print_colored(f"compare_waveformsets: original={type(original)}, loaded={type(loaded)}", color="DEBUG")
        print_colored(f"original.waveforms={type(original.waveforms)}, loaded.waveforms={type(loaded.waveforms)}", color="DEBUG")

        for i, (w1, w2) in enumerate(zip(original.waveforms, loaded.waveforms)):
            # print_colored(f"  wave {i}: type(w1)={type(w1)}, type(w2)={type(w2)}", color="DEBUG")
            # Next line is likely failing:
            if not np.array_equal(w1.adcs, w2.adcs):
                print_colored(f"Waveform {i} ADC mismatch", color="ERROR")
            elif w1.timestamp != w2.timestamp:
                print_colored(f"Waveform {i} timestamp mismatch", color="ERROR")

        print_colored("Comparison finished.", color="DEBUG")

        if not hasattr(original, "waveforms"):
            print_colored(f"❌ Original has no 'waveforms' attribute. It's a {type(original)}", color="ERROR")
        # if not hasattr(loaded, "waveforms"):
        #     print_colored(f"❌ Loaded has no 'waveforms' attribute. It's a {type(loaded)}", color="ERROR")

        if len(original.waveforms) != len(loaded.waveforms):
            print_colored("Mismatch in number of waveforms!", color="ERROR")
            return

        for i, (w1, w2) in enumerate(zip(original.waveforms, loaded.waveforms)):
            if not np.array_equal(w1.adcs, w2.adcs):
                print_colored(f"Waveform {i} ADC mismatch", color="ERROR")
            elif w1.timestamp != w2.timestamp:
                print_colored(f"Waveform {i} timestamp mismatch", color="ERROR")

        print_colored("Comparison finished.", color="DEBUG")

    def write_output(self, wfset, input_filepath):
        input_filename = Path(input_filepath).name
        output_filepath = Path(self.output_path) / f"processed_{input_filename}_structured.hdf5"

        print_colored(f"Saving waveform data to {output_filepath}...", color="DEBUG")
        try:
            # ✅ Make sure we overwrite the input variable with the wrapped one
            wfset = self.ensure_waveformset(wfset)
            print_colored(f"📦 About to save WaveformSet with {len(wfset.waveforms)} waveforms", color="DEBUG")

            WaveformSet_to_file(
                waveform_set=wfset,
                output_filepath=str(output_filepath),
                overwrite=True,
                format="hdf5",
                compression="gzip",
                compression_opts=0,
                structured=True
            )
            print_colored(f"WaveformSet saved to {output_filepath}", color="SUCCESS")

            print_colored("Going to load...")
            wfset_loaded = load_structured_waveformset(str(output_filepath))
            print_colored("Loaded, about to compare...")  # If you see this, the load worked
            print_colored(f"wfset_loaded type={type(wfset_loaded)}")
            self.compare_waveformsets(wfset, wfset_loaded)
            print_colored("Done comparing!")  # If you never see this, an error happens in compare

            return True
        except Exception as e:
            print_colored(f"Error saving output: {e}", color="ERROR")
            return False

    


@click.command(help="\033[34mProcess and save structured waveform data from JSON config.\033[0m")
@click.option("--config", required=True, help="Path to JSON configuration file.", type=str)
def main(config):
    try:
        import multiprocessing as mp
        mp.set_start_method('spawn')
        with open(config, 'r') as f:
            config_data = json.load(f)

        required_keys = ["run", "rucio_dir", "output_dir", "ch"]
        missing = [key for key in required_keys if key not in config_data]
        if missing:
            raise ValueError(f"Missing keys in config: {missing}")

        processor = WaveformProcessor(config_data)
        processor.read_and_save()

    except Exception as e:
        print_colored(f"An error occurred: {e}", color="ERROR")

if __name__ == "__main__":
    main()