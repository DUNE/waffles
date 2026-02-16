import pickle
import hickle as hkl
import h5py
import numpy as np
import time
import os
import sys
import waffles.input_output.raw_hdf5_reader as reader

def save_as_pickle(obj, filename):
    """Save object using pickle and return file size and time taken."""
    start_time = time.time()
    with open(filename, "wb") as f:
        pickle.dump(obj, f)
    elapsed_time = time.time() - start_time
    file_size = os.path.getsize(filename)
    return file_size, elapsed_time

def save_as_hickle(obj, filename):
    """Save object using hickle (HDF5-based pickle) and return file size and time taken."""
    start_time = time.time()
    hkl.dump(obj, filename)
    elapsed_time = time.time() - start_time
    file_size = os.path.getsize(filename)
    return file_size, elapsed_time

def save_as_hdf5_pickle(obj, filename, compression=None):
    """Save an object in HDF5 format using Pickle (as a compressed byte array)."""
    start_time = time.time()
    
    # Serialize object with pickle
    obj_bytes = pickle.dumps(obj)
    obj_np = np.frombuffer(obj_bytes, dtype=np.uint8)  # Convert to NumPy array
    
    with h5py.File(filename, "w") as hdf:
        hdf.create_dataset("wfset", data=obj_np, compression=compression)  # Save as compressed byte array

    elapsed_time = time.time() - start_time
    file_size = os.path.getsize(filename)
    return file_size, elapsed_time

def main(run_number):
    
    print("Reading the complete hdf5 file...")
    
    #From a rucio filepath. Important: First execute python get_rucio.py --runs <run_number> in <repos_dir>/waffles/scripts
    #rucio_filepath = f"/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths/{run_number}.txt"
    #filepaths = reader.get_filepaths_from_rucio(rucio_filepath)
    #wfset = reader.WaveformSet_from_hdf5_file(filepaths[0], det='VD_Membrane_PDS', read_full_streaming_data=False) # Only takes the first filepath 
    
    #From a directly download file in a specific filepath
    filepath = f""
    det='VD_Membrane_PDS'
    wfset = reader.WaveformSet_from_hdf5_file(filepath,  det=det, read_full_streaming_data=False) 

    # File naming
    pkl_filename = f"wfset_{run_number}.pkl"
    hdf5_filename = f"wfset_{run_number}.hdf5"
    hkl_filename = f"wfset_{run_number}.hkl"

    print("\n### Saving in Different Formats ###")

    # Pickle test
    pkl_size, pkl_time = save_as_pickle(wfset, pkl_filename)
    print(f"[Pickle] Size: {pkl_size} bytes, Time: {pkl_time:.2f} sec")
    
    # Hickle test
    hkl_size, hkl_time = save_as_hickle(wfset, hkl_filename)
    print(f"[Hickle] Size: {hkl_size} bytes, Time: {hkl_time:.2f} sec")

    # HDF5 tests with different compression methods
    compressions = [None, "gzip", "lzf"]
    results = []
    for comp in compressions:
        hdf5_comp_filename = f"wfset_{run_number}_{comp or 'no_comp'}.hdf5"
        size, time_taken = save_as_hdf5_pickle(wfset, hdf5_comp_filename, compression=comp)
        results.append((comp, size, time_taken))
        print(f"[HDF5-{comp or 'No Compression'}] Size: {size} bytes, Time: {time_taken:.2f} sec")

    # Results summary
    print("\n### Summary of File Sizes and Times ###")
    print(f"{'Format':<15} {'Compression':<10} {'Size (MB)':<10} {'Time (sec)':<10}")
    print("="*50)
    print(f"{'Pickle':<15} {'-':<10} {pkl_size / 1e6:<10.2f} {pkl_time:<10.2f}")
    print(f"{'Hickle':<15} {'-':<10} {hkl_size / 1e6:<10.2f} {hkl_time:<10.2f}")
    for comp, size, time_taken in results:
        print(f"{'HDF5':<15} {comp or 'None':<10} {size / 1e6:<10.2f} {time_taken:<10.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 wtest_hdf5_formats.py <run_number>")
        sys.exit(1)
    
    main(sys.argv[1])