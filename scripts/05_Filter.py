from pathlib import Path
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

def process_waveform_data(input_path: Path, output_path: Path, allowed_endpoints: list[int]) -> None:
    """Loads, filters, and saves waveform data."""
    try:
        # Load waveform set
        print(f"{BLUE}Loading waveform set from:{RESET} {input_path}")
        waveform_set = WaveformSet_from_hdf5_pickle(input_path)
        print(f"{CYAN}✔ Loaded waveform set.{RESET}")
        print(f"{GREEN}Number of waveforms: {len(waveform_set.waveforms)}{RESET}")

        # Apply endpoint filtering
        print(f"{BLUE}Filtering waveforms by endpoints: {allowed_endpoints}...{RESET}")
        filtered_waveform_set = WaveformSet.from_filtered_WaveformSet(waveform_set, filter_by_endpoints, allowed_endpoints)
        print(f"{GREEN}Number of waveforms: {len(filtered_waveform_set.waveforms)}{RESET}")
        print(f"{CYAN}✔ Filtering by endpoints complete.{RESET}")

        # Apply beam self-trigger filtering
        print(f"{BLUE}Applying beam self-trigger filter...{RESET}")
        final_waveform_set = WaveformSet.from_filtered_WaveformSet(filtered_waveform_set, beam_self_trigger_filter)
        print(f"{GREEN}✔ Number of waveforms after filtering: {len(final_waveform_set.waveforms)}{RESET}")
        print(f"{CYAN}✔ Beam self-trigger filter applied.{RESET}")

        # Save the processed waveform set
        print(f"{BLUE}Saving the filtered waveform set to:{RESET} {output_path}")
        WaveformSet_to_file(
            waveform_set=final_waveform_set,
            output_filepath=str(output_path),
            overwrite=True,
            format="hdf5",
            compression="gzip",
            compression_opts=5
        )
        
        print(f"{GREEN}✔ Filtered waveform set successfully saved to: {output_path}{RESET}")

    except FileNotFoundError:
        print(f"{RED}❌ Error: Input file '{input_path}' not found.{RESET}")
    except Exception as e:
        print(f"{RED}❌ An error occurred during processing: {e}{RESET}")

def main():
    """Main function to execute waveform processing."""
    input_file = Path("/afs/cern.ch/work/m/marroyav/wfenv/waffles/data/waveformset_run_27343.hdf5")
    output_file = Path("/afs/cern.ch/work/m/marroyav/wfenv/waffles/data/filtered_waveformset_run_27343.hdf5")
    allowed_endpoints = [109]  # Define allowed endpoints for filtering

    print(f"{YELLOW}Starting waveform processing...{RESET}")
    process_waveform_data(input_file, output_file, allowed_endpoints)
    print(f"{GREEN}✔ Processing complete.{RESET}")

if __name__ == "__main__":
    main()