from pathlib import Path
import multiprocessing
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.Waveform import Waveform
from waffles.input_output.persistence_utils import WaveformSet_to_file
from waffles.input_output.pickle_hdf5_reader import WaveformSet_from_hdf5_pickle

# ANSI Escape Codes for Colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RESET = "\033[0m"

def channel_filter(waveform: Waveform, endpoint: int, channel: int) -> bool:
    """Returns True if the waveform matches the given channel and endpoint."""
    return waveform.channel == channel and waveform.endpoint == endpoint

def beam_self_trigger_filter(waveform: Waveform, timeoffset_min: int = -1024, timeoffset_max: int = 1024) -> bool:
    """Filters waveforms based on DAQ PDS time offset."""
    daq_pds_timeoffset = waveform.timestamp - waveform.daq_window_timestamp
    return timeoffset_min < daq_pds_timeoffset < timeoffset_max

def filter_by_endpoints(waveform: Waveform, allowed_endpoints: list[int]) -> bool:
    """Returns True if the waveform's endpoint is in the allowed list."""
    return waveform.endpoint in allowed_endpoints

def process_waveform_file(file_path: Path, output_dir: Path, allowed_endpoints: list[int]) -> None:
    """Loads, filters, and saves waveform data for a single file."""
    try:
        print(f"{BLUE}Processing file: {file_path}{RESET}")

        # Load waveform set
        waveform_set = WaveformSet_from_hdf5_pickle(file_path)
        print(f"{CYAN}✔ Loaded waveform set from {file_path}{RESET}")
        print(f"{GREEN}Initial number of waveforms: {len(waveform_set.waveforms)}{RESET}")

        # Apply endpoint filtering
        filtered_waveform_set = WaveformSet.from_filtered_WaveformSet(waveform_set, filter_by_endpoints, allowed_endpoints)
        print(f"{GREEN}✔ Number of waveforms after endpoint filtering: {len(filtered_waveform_set.waveforms)}{RESET}")

        # Apply beam self-trigger filtering
        final_waveform_set = WaveformSet.from_filtered_WaveformSet(filtered_waveform_set, beam_self_trigger_filter)
        print(f"{GREEN}✔ Number of waveforms after self-trigger filtering: {len(final_waveform_set.waveforms)}{RESET}")

        # Save filtered waveform set
        output_path = output_dir / f"filtered_{file_path.name}"
        print(f"{BLUE}Saving filtered waveform set to: {output_path}{RESET}")
        
        WaveformSet_to_file(
            waveform_set=final_waveform_set,
            output_filepath=str(output_path),
            overwrite=True,
            format="hdf5",
            compression="gzip",
            compression_opts=5
        )

        print(f"{GREEN}✔ Successfully saved: {output_path}{RESET}")

    except FileNotFoundError:
        print(f"{RED}❌ Error: File '{file_path}' not found.{RESET}")
    except Exception as e:
        print(f"{RED}❌ An error occurred processing {file_path}: {e}{RESET}")

def main():
    """Main function to process all HDF5 files in a directory in parallel."""
    input_dir = Path(".")
    output_dir = Path(".")
    output_dir.mkdir(exist_ok=True)  # Ensure output directory exists

    allowed_endpoints = [109]  # Define allowed endpoints for filtering

    # Find all HDF5 files in the input directory
    hdf5_files = list(input_dir.glob("*.hdf5"))

    if not hdf5_files:
        print(f"{RED}❌ No HDF5 files found in {input_dir}.{RESET}")
        return

    print(f"{YELLOW}Found {len(hdf5_files)} HDF5 files. Processing in parallel...{RESET}")

    # Use multiprocessing to process files in parallel
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(process_waveform_file, [(file, output_dir, allowed_endpoints) for file in hdf5_files])

    print(f"{GREEN}✔ Parallel processing complete.{RESET}")

if __name__ == "__main__":
    main()