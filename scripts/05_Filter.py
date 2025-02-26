from pathlib import Path
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.Waveform import Waveform
from waffles.input_output.persistence_utils import WaveformSet_to_file
from waffles.input_output.pickle_hdf5_reader import WaveformSet_from_hdf5_pickle

def channel_filter(waveform: Waveform, endpoint: int, channel: int) -> bool:
    """Returns True if the waveform matches the given channel and endpoint."""
    return waveform.channel == channel and waveform.endpoint == endpoint

def beam_self_trigger_filter(waveform: Waveform, timeoffset_min: int = -120, timeoffset_max: int = -90) -> bool:
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
        waveform_set = WaveformSet_from_hdf5_pickle(input_path)
        print(f"Loaded waveform set: {waveform_set}")

        # Apply endpoint filtering
        filtered_waveform_set = WaveformSet.from_filtered_WaveformSet(waveform_set, filter_by_endpoints, allowed_endpoints)
        print(f"Filtered waveforms by endpoints: {allowed_endpoints}")

        # Apply beam self-trigger filtering
        final_waveform_set = WaveformSet.from_filtered_WaveformSet(filtered_waveform_set, beam_self_trigger_filter)
        print("Applied beam self-trigger filter.")

        # Save the processed waveform set
        WaveformSet_to_file(
            waveform_set=final_waveform_set,
            output_filepath=str(output_path),
            overwrite=True,
            format="hdf5",
            compression="gzip",
            compression_opts=5
        )
        print(f"Filtered waveform set saved to: {output_path}")

    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found.")
    except Exception as e:
        print(f"An error occurred during processing: {e}")

def main():
    """Main function to execute waveform processing."""
    input_file = Path("/afs/cern.ch/work/m/marroyav/wfenv/waffles/data/waveformset_run_27343.hdf5")
    output_file = Path("/afs/cern.ch/work/m/marroyav/wfenv/waffles/data/filtered_waveformset_run_27343.hdf5")
    allowed_endpoints = [109]  # Define allowed endpoints for filtering

    process_waveform_data(input_file, output_file, allowed_endpoints)

if __name__ == "__main__":
    main()