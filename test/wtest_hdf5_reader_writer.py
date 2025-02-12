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
    rucio_filepath = f"/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths/{run_number}.txt"
    filepaths = reader.get_filepaths_from_rucio(rucio_filepath)
    wfset = reader.WaveformSet_from_hdf5_file(filepaths[0], read_full_streaming_data=False) # Only takes the first filepath 
    
    #From a directly download file in a specific filepath
    #filepath = f""
    #wfset = reader.WaveformSet_from_hdf5_file(filepath, read_full_streaming_data=False)

    print("\n Saving the waveform in a compressed hdf5 format")

    comp="gzip"
    hdf5_comp_filename = f"wfset_{run_number}_{comp}.hdf5"
    
    size_create, time_taken_create = save_as_hdf5_comp(wfset, hdf5_comp_filename, compression=comp)
    print(f"[HDF5-{comp} creation] Size: {size_create} bytes, Time: {time_taken_create:.2f} sec")
    
    print("\n Reading the waveform from a compressed hdf5 format")
    
    hdf5_comp_filepath = os.path.join(os.getcwd(), f"wfset_{run_number}_{comp}.hdf5")
    
    time_taken_read,wfset_ready = read_wfset_hdf5(hdf5_comp_filename)
    print(f"[HDF5-{comp} reading] Time: {time_taken_read:.2f} sec")
    print('\n Waveformset ready to analysis', type(wfset_ready))
    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 wtest_hdf5_reader_new.py <run_number>")
        sys.exit(1)
    
    main(sys.argv[1])