import pickle
import h5py
import numpy as np
import time
import os
import sys
import waffles.input_output.raw_hdf5_reader as reader


def save_as_hdf5_comp(obj, filename, compression):
    
    start_time = time.time()
    
    # Serialize object with pickle
    obj_bytes = pickle.dumps(obj)
    obj_np = np.frombuffer(obj_bytes, dtype=np.uint8)  # Convert to NumPy array
    
    with h5py.File(filename, "w") as hdf:
        hdf.create_dataset("wfset", data=obj_np, compression=compression)  # Save as compressed byte array

    elapsed_time = time.time() - start_time
    file_size = os.path.getsize(filename)
    
    return file_size, elapsed_time

def read_wfset_hdf5(filename):

    start_time = time.time()
    
    with h5py.File(filename, 'r')  as f:
        raw_wfset=f['wfset'][:]
    st_wfset = pickle.loads(raw_wfset.tobytes())
    
    elapsed_time = time.time() - start_time
    
    return elapsed_time, st_wfset

def main(run_number):
    
    print("Reading the complete hdf5 file...")
    
    #From a rucio filepath. Important: First execute python get_rucio.py --runs <run_number> in <repos_dir>/waffles/scripts
    #rucio_filepath = f"/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths/{run_number}.txt"
    #filepaths = reader.get_filepaths_from_rucio(rucio_filepath)
    #wfset = reader.WaveformSet_from_hdf5_file(filepaths[0], read_full_streaming_data=False) # Only takes the first filepath 
    
    #From a directly download file in a specific filepath
    filepaths = ["/eos/experiment/neutplatform/protodune/dune/hd-protodune/41/fc/np04hd_raw_run028676_0145_dataflow5_datawriter_0_20240814T044406.hdf5",
                "/eos/experiment/neutplatform/protodune/dune/hd-protodune/06/c7/np04hd_raw_run028676_0145_dataflow6_datawriter_0_20240814T044406.hdf5",
                "/eos/experiment/neutplatform/protodune/dune/hd-protodune/88/57/np04hd_raw_run028676_0145_dataflow7_datawriter_0_20240814T044406.hdf5",
                "/eos/experiment/neutplatform/protodune/dune/hd-protodune/28/2c/np04hd_raw_run028676_0146_dataflow0_datawriter_0_20240814T044505.hdf5",
                "/eos/experiment/neutplatform/protodune/dune/hd-protodune/4b/bc/np04hd_raw_run028676_0146_dataflow1_datawriter_0_20240814T044504.hdf5",
                "/eos/experiment/neutplatform/protodune/dune/hd-protodune/9e/e2/np04hd_raw_run028676_0146_dataflow2_datawriter_0_20240814T044504.hdf5",
                "/eos/experiment/neutplatform/protodune/dune/hd-protodune/ce/89/np04hd_raw_run028676_0146_dataflow3_datawriter_0_20240814T044504.hdf5",
                "/eos/experiment/neutplatform/protodune/dune/hd-protodune/b3/5e/np04hd_raw_run028676_0146_dataflow4_datawriter_0_20240814T044504.hdf5",
                "/eos/experiment/neutplatform/protodune/dune/hd-protodune/1c/db/np04hd_raw_run028676_0146_dataflow5_datawriter_0_20240814T044505.hdf5",
                "/eos/experiment/neutplatform/protodune/dune/hd-protodune/24/d3/np04hd_raw_run028676_0146_dataflow6_datawriter_0_20240814T044505.hdf5",
                "/eos/experiment/neutplatform/protodune/dune/hd-protodune/32/5f/np04hd_raw_run028676_0146_dataflow7_datawriter_0_20240814T044505.hdf5",
                "/eos/experiment/neutplatform/protodune/dune/hd-protodune/6a/02/np04hd_raw_run028676_0147_dataflow0_datawriter_0_20240814T044611.hdf5",
                "/eos/experiment/neutplatform/protodune/dune/hd-protodune/93/0c/np04hd_raw_run028676_0147_dataflow1_datawriter_0_20240814T044606.hdf5",
                "/eos/experiment/neutplatform/protodune/dune/hd-protodune/1e/9c/np04hd_raw_run028676_0147_dataflow2_datawriter_0_20240814T044606.hdf5",
                "/eos/experiment/neutplatform/protodune/dune/hd-protodune/c6/36/np04hd_raw_run028676_0147_dataflow3_datawriter_0_20240814T044606.hdf5",
                "/eos/experiment/neutplatform/protodune/dune/hd-protodune/4d/2b/np04hd_raw_run028676_0147_dataflow4_datawriter_0_20240814T044606.hdf5",
                "/eos/experiment/neutplatform/protodune/dune/hd-protodune/c4/a6/np04hd_raw_run028676_0147_dataflow5_datawriter_0_20240814T044606.hdf5",
                "/eos/experiment/neutplatform/protodune/dune/hd-protodune/ee/ba/np04hd_raw_run028676_0147_dataflow6_datawriter_0_20240814T044608.hdf5",
                "/eos/experiment/neutplatform/protodune/dune/hd-protodune/f6/0f/np04hd_raw_run028676_0147_dataflow7_datawriter_0_20240814T044608.hdf5",
                "/eos/experiment/neutplatform/protodune/dune/hd-protodune/53/2d/np04hd_raw_run028676_0148_dataflow0_datawriter_0_20240814T044716.hdf5"]
    
    det='HD_PDS'
    
    comp="gzip"
    
    allowed_channels=[30,31,32,33,34,35,36,37]
    
    # Initialize an empty WaveformSet to store all the waveforms
    merged_wfset = None

    for i, filepath in enumerate(filepaths):
        
        print(f"Reading file {filepath}...")

        # Read the waveform data from the current file
        wfset = reader.WaveformSet_from_hdf5_file(filepath, det=det, allowed_channels=allowed_channels, read_full_streaming_data=False)

        # If it's the first file, initialize merged_wfset with this one
        if merged_wfset is None:
            merged_wfset = wfset
        else:
            # Merge the new wfset into the existing one
            merged_wfset.merge(wfset)  # You may need to define or implement a `merge` method if it doesn't exist

        print(f"Processed file {filepath}")

    # Saving the merged waveform set
    hdf5_comp_filename = f"wfset_{run_number}_{comp}.hdf5"
    print(f"\nSaving the merged waveform set in a compressed hdf5 format: {hdf5_comp_filename}")

    size_create, time_taken_create = save_as_hdf5_comp(merged_wfset, hdf5_comp_filename, compression=comp)
    print(f"[HDF5-{comp} creation] Size: {size_create} bytes, Time: {time_taken_create:.2f} sec")

    # Reading the merged waveform set from the compressed hdf5 format
    print("\nReading the waveform from a compressed hdf5 format")

    hdf5_comp_filepath = os.path.join(os.getcwd(), f"wfset_{run_number}_{comp}.hdf5")

    time_taken_read, wfset_ready = read_wfset_hdf5(hdf5_comp_filename)
    print(f"[HDF5-{comp} reading] Time: {time_taken_read:.2f} sec")
    print('\nWaveformset ready for analysis', type(wfset_ready))
    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 hdf5_to_hdf5_converter.py <run_number>")
        sys.exit(1)
    
    main(sys.argv[1])