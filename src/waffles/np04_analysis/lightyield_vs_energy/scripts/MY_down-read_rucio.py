# Rember to:
## In the same terminal:
# 1. source waffles/scripts/setup_rucio_a9.sh
# 2. source virtual_env/daq_env_102025/env.sh
## In another terminal:
# 1. source waffles/scripts/setup_rucio_a9.sh
## Turn back to the previous one and:
# python3.10 MY_down-read_rucio.py --run 27355 --run_filename 027355.txt

'''
Configuration parameters (from JSON):
# rucio_dir (str): path to the directory containing Rucio files to be read, example: /afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/data/rucio_filepath/what_to_read
# output_dir (str): path where processed/output files will be saved, example: /afs/cern.ch/work/a/anbalbon/public/reading_rucio
# ch(dic): mapping of detector IDs to channel lists to analyze
APA 1:  "104": [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17],
        "105": [0, 1, 2, 3, 4, 5, 6, 7, 10, 12, 15, 17, 21, 23, 24, 26],
        "107": [0, 2, 5, 7, 10, 12, 15, 17]
APA 2:  "109": [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37, 40, 41, 42, 43, 44, 45, 46, 47]
# det (str): detector name or configuration ID, for protodune-HD "HD_PDS"
# self_trigger (bool): if True, processing self-trigger mode
# full_streaming (bool): if True, processing full-streaming mode
# download (bool): if True, download input files using Rucio
# existing_reprocessing (bool): if True, reprocess existing data
# saving_results (bool): if True, save analysis results
# overwrite_processed (bool): if True, overwrite already processed output files
# deleting_downloaded_rucio (bool): if True, delete downloaded Rucio files after processing
# beam_events_selection (bool): if True, process only beam events; otherwise process all data
'''

import json
import click
import os
import subprocess
import shutil
import numpy as np

from waffles.utils.utils import print_colored
import waffles.input_output.raw_hdf5_reader as reader
from waffles.input_output.persistence_utils import WaveformSet_to_file
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.Waveform import Waveform

def rename_if_exists(processed_path):
    if not os.path.exists(processed_path):
        return
    base, ext = os.path.splitext(processed_path)
    counter = 1
    while True:
        new_path = f"{base}_{counter}{ext}"
        if not os.path.exists(new_path):
            os.rename(processed_path, new_path)
            print(f"🔀 Output file already exists - File renamed to: {new_path}")
            break
        counter += 1

def beam_self_trigger_filter(waveform : Waveform, timeoffset_min : int = -120, timeoffset_max : int = -90) -> bool:
    daq_pds_timeoffset = np.float32(np.int64(waveform.timestamp)-np.int64(waveform.daq_window_timestamp))
    if (daq_pds_timeoffset < timeoffset_max) and (daq_pds_timeoffset > timeoffset_min) :
        return True
    else:
        return False

def parse_ch_dict( ch):
        if not isinstance(ch, dict):
            raise ValueError("'ch' must be a dictionary {endpoint: [channels]}.")
        parsed = {}
        for endpoint, chans in ch.items():
            if not isinstance(chans, list) or not all(isinstance(c, int) for c in chans):
                raise ValueError(f"Invalid channel list for endpoint {endpoint}.")
            parsed[int(endpoint)] = chans
        return parsed


class MY_hdf5_processor:
    def __init__(self, config: dict, run: int, rucio_filepath: str):
        self.run_number = run
        self.run_str = f"{self.run_number:06d}"

        self.rucio_filepath = rucio_filepath
        self.rucio_filename = os.path.basename(self.rucio_filepath)

        self.rucio_paths_directory = config.get("rucio_dir", ".")
        self.output_path = config.get("output_dir", ".")
        #self.ch = self.parse_ch_dict(config.get("ch", {}))
        self.detector = config.get("det")
        self.self_trigger = config.get("self_trigger", True)
        self.full_streaming = config.get("full_streaming", False)
        self.download = config.get("download", True)
        self.truncate_wfs_method = config.get("truncate_wfs_method", "") #come si usa?
        self.existing_reprocessing = config.get("existing_reprocessing", False) 
        self.saving_results = config.get("saving_results", True)
        self.overwrite_processed = config.get("overwrite_processed", False) 
        self.deleating_downloaded_rucio = config.get("deleating_downloaded_rucio", False)
        self.beam_events_selection = config.get("beam_events_selection", False)

        # inizialized later
        self.output_filepath_exists = None
        self.download_filepath = ''


        # print_colored(f"Loaded configuration: {config}", color="INFO")
        print_colored(f"\nStarting analysizing: {os.path.basename(rucio_filepath)}", color="INFO")


    def ensure_waveformset(self, wfset: WaveformSet):
        if isinstance(wfset, WaveformSet):
            return wfset
        if isinstance(wfset, list) and all(isinstance(w, Waveform) for w in wfset):
            print_colored("🛠️ Auto-wrapping list of Waveforms into WaveformSet", color="DEBUG")
            return WaveformSet(wfset)
        raise TypeError(f"Expected WaveformSet or list of Waveforms, got {type(wfset)}")


    def write_output(self, wfset: WaveformSet, mode: str):
        mode_output_dir = os.path.join(self.output_path, f"run{self.run_str}", mode, "beam_only" if self.beam_events_selection else "no_selection")
        os.makedirs(mode_output_dir, exist_ok=True)
       
        processed_name = f"processed_{self.rucio_filename}_structured.hdf5"
        processed_path = os.path.join(mode_output_dir, processed_name)

        if not self.overwrite_processed:
            rename_if_exists(processed_path)

        print_colored(f"Saving waveform data to {processed_path}...", color="DEBUG")
        try:
            # ✅ Make sure we overwrite the input variable with the wrapped one
            wfset = self.ensure_waveformset(wfset)
            print_colored(f"📦 About to save WaveformSet", color="DEBUG")

            WaveformSet_to_file(
                waveform_set=wfset,
                output_filepath=str(processed_path),
                overwrite=True,
                format="hdf5",
                compression="gzip",
                compression_opts=0,
                structured=True
            )
            print_colored(f"WaveformSet saved to {processed_path}", color="SUCCESS")

            return True
        except Exception as e:
            print_colored(f"Error saving output: {e}", color="ERROR")
            return False

    def check_not_processed(self, mode: str):
        mode_output_dir = os.path.join(self.output_path, f"run{self.run_str}", mode, "beam_only" if self.beam_events_selection else "no_selection")        
        os.makedirs(mode_output_dir, exist_ok=True)
       
        processed_name = f"processed_{self.rucio_filename}_structured.hdf5"
        processed_path = os.path.join(mode_output_dir, processed_name)

        if os.path.exists(processed_path):
            # print(f"✅ Already processed ({mode})")
            self.output_filepath_exists = True
            return False
        else:
            # print(f"❎ Not processed yet ({mode})")
            self.output_filepath_exists = False
            return True

    def check_not_eos(self):
        if "/eos" in self.rucio_filepath:
            # print(f"🌐 Found on EOS ruciopath - no download required")
            self.download_filepath = self.rucio_filepath
            return False
        else:
            # print(f"🌐 Not found on EOS ruciopath - download required")
            return True
    
    def rucio_download(self):
        local_folder = os.path.join(self.output_path, "hd-protodune")
        os.makedirs(local_folder, exist_ok=True)
        local_file = os.path.join(local_folder, self.rucio_filename)

        if os.path.exists(local_file):
            print(f"📂 Found local copy: {local_file}")
            self.download_filepath = local_file
            return True

        print(f"⬇️  Downloading via rucio: {self.rucio_filepath}")
        try:
            if self.download:
                subprocess.run(["rucio", "download", f"hd-protodune:{self.rucio_filename}", "--dir", f"{self.output_path}"], check=True)
                print(f"✅ Downloaded to: {local_file}")
                self.download_filepath = local_file
                return True
            else: 
                print(f"❌ Download disabled")
                self.download_filepath = ''
                return False
        except subprocess.CalledProcessError as e:
            print_colored(f"❌ Rucio download failed: {e}", color="ERROR")
            self.download_filepath = ''
            return False

    def wfset_reading_and_save(self, mode: str):
        if mode == "full_streaming":
            if self.truncate_wfs_method == "":
                print_colored("Warning: 'truncate_wfs_method' is empty, using default 'minimum'.", color="WARNING")
                self.truncate_wfs_method = "minimum"

        print(f"▶️  Analysis started")
        wfset = reader.WaveformSet_from_hdf5_file(
            filepath=self.download_filepath,
            read_full_streaming_data= (mode == "full_streaming"),
            truncate_wfs_method=self.truncate_wfs_method,
            nrecord_start_fraction=0.0,
            nrecord_stop_fraction=1.0,
            subsample=1,
            wvfm_count=1e9,
            ch=self.ch,
            det=self.detector,
            temporal_copy_directory='/tmp',
            erase_temporal_copy=False,
            repeat_choice=[0]
        )

        wfset = self.ensure_waveformset(wfset)
        print_colored(f"📊 Read {len(wfset.waveforms)} waveforms from file.", color="INFO")

        if self.beam_events_selection and mode == "full_streaming":
            print_colored("🔍 Applying full-streaming beam events selection.", color="DEBUG")
            ### to be implemented
            print_colored(f"📊 Selected {len(wfset.waveforms)} beam waveforms.", color="DEBUG")

        if self.beam_events_selection and mode == "self_trigger":
            print_colored("🔍 Applying self-trigger beam events selection.", color="DEBUG")
            wfset = WaveformSet.from_filtered_WaveformSet(wfset, beam_self_trigger_filter)
            print_colored(f"📊 Selected {len(wfset.waveforms)} beam waveforms.", color="DEBUG")

        wfset = self.ensure_waveformset(wfset)

        if wfset and self.saving_results:
            self.write_output(wfset, mode)
        elif not self.saving_results:
            print_colored("ℹ️  Saving results is disabled.", color="INFO")
        else:
            print_colored("❌  No waveform data to save.", color="ERROR")

        print(f"⏹️  Analysis finished")


    def check_mode_ch_empty(self, original_ch, mode: str):
        full_streaming_keys = {"104", "105", "107"}
        if mode == "self_trigger":
            ch_dic_new = {k: v for k, v in original_ch.items() if str(k).strip() not in full_streaming_keys}
        else:
            ch_dic_new = {k: v for k, v in original_ch.items() if str(k).strip() in full_streaming_keys}
        if ch_dic_new:
            self.ch = ch_dic_new
            return False
        else:
            return True


@click.command(help="\033[34mProcess and save structured waveform data from JSON config.\033[0m")
@click.option("--config", default='MY_config.json', help="Path to JSON configuration file.", type=str)
@click.option("--run", required=True, help="Run you want to analyze (example 27355)", type=int)
@click.option("--run_filename", default=None, help="Filename of rucio file you want to analyze with .txt extension (by default 0{run}.txt)", type=str)
def main(config, run, run_filename):
    try:
        import multiprocessing as mp
        mp.set_start_method('spawn')
        with open(config, 'r') as f:
            config_data = json.load(f)

        required_keys = ["rucio_dir", "output_dir", "ch", "det"]
        missing = [key for key in required_keys if key not in config_data]
        if missing:
            raise ValueError(f"Missing keys in config: {missing}")

        rucio_paths_directory = config_data.get("rucio_dir", ".")
        output_path = config_data.get("output_dir", ".")
        original_ch = parse_ch_dict(config_data.get("ch", {}))
        
        if run_filename is None:
            input_file = os.path.join(rucio_paths_directory, f"{run:06d}.txt")
        else:
            input_file = os.path.join(rucio_paths_directory, run_filename)

        if not os.path.isfile(input_file):
            print(f"❌ Input file not found: {input_file}")
            return
        
        print(f"📘 Reading HDF5 filepaths from: {input_file}")
        with open(input_file, "r") as f:
            filepaths = [line.strip() for line in f if line.strip()]
        if not filepaths:
            print("⚠️ No file paths found in input file.")
            return
        
        for filepath in filepaths:
            processor = MY_hdf5_processor(config_data, run, filepath)

            for mode, enabled in [("self_trigger", processor.self_trigger), ("full_streaming", processor.full_streaming)]:
                if enabled:
                    print_colored(f"Trigger: {mode}", color="INFO")

                    if processor.check_mode_ch_empty(original_ch, mode):
                        print_colored(f"❌ No channels to process for mode {mode}. Skipping.", color="ERROR")
                        continue 
                    else:
                        print_colored(f"✅ Channels available for mode {mode} (Endpoint: {list(processor.ch.keys())}). Proceeding.", color="INFO")

                    if processor.check_not_processed(mode) or processor.existing_reprocessing:
                        print(f"❎ Not processed yet and/or reprocessing enabled ({mode})")
                        
                        if processor.check_not_eos():
                            print(f"🌐 Not found on EOS ruciopath - download required")

                            if processor.rucio_download():
                                print(f"✅ Starting analysis on downloaded file: {processor.download_filepath}")
                                processor.wfset_reading_and_save(mode)
                            else:
                                print(f"❌ Download failed or not enabled, cannot proceed with analysis.")
                                continue 
                            
                        else:
                            print(f"🌐 Found on EOS ruciopath - no download required")
                            print(f"✅ Starting analysis on eos file: {processor.download_filepath}") 
                            processor.wfset_reading_and_save(mode)
                            continue

                    
                    else:
                        print(f"✅ Already processed ({mode})")
                        continue
                        

            if (processor.deleating_downloaded_rucio) and (processor.saving_results) and (processor.check_not_eos()) and (os.path.exists(processor.download_filepath)):
                print(f"🗑️ Deleting downloaded file: {processor.download_filepath}")
                os.remove(processor.download_filepath)
                print(f"✅  Deleted downloaded file.")
            else:
                print(f"ℹ️  Not deleting downloaded file")
        
    except Exception as e:
        print_colored(f"An error occurred: {e}", color="ERROR")

if __name__ == "__main__":
    main()