'''
import pickle
import hickle as hkl
import re
import waffles.input.raw_hdf5_reader as reader
import sys
import os
import h5py
import numpy as np

if len(sys.argv) != 2:
    print("Use: python3 wtest_hdf5_reader.py <run_number>")
    sys.exit(1)

run_number = sys.argv[1] 
rucio_filepath = f"/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths/{run_number}.txt"

filepaths = reader.get_filepaths_from_rucio(rucio_filepath)


# For analyze all the filepaths inside the rucio_filepath
for i in filepaths:
  wfset = reader.WaveformSet_from_hdf5_file(
    filepaths[i],                    
    read_full_streaming_data=False     
  )
  with open(f'wfset_{run_number}_{i}', "wb") as f:
    pickle.dump(wfset, f)


# For analyze the first filepath inside the rucio_filepath
wfset = reader.WaveformSet_from_hdf5_file(filepaths[0],                     # path to the root file
                                           read_full_streaming_data = False, # self-triggered (False) data
                                         )

hdf5_filename = f"wfset_{run_number}.hdf5"
print(f"Saving the wfset object as an HDF5 file: {hdf5_filename}")

# It doesn't allow to save the wvf in a hdf5 file
with h5py.File(hdf5_filename, "w") as hdf:
    hdf.create_dataset("wfset", data=wfset)

# It allows to save the wvf in a hdf5. However, has the same size as the pkl file
#hkl.dump(wfset, hdf5_filename)

hdf5_size = os.path.getsize(hdf5_filename)
print(f"Size: {hdf5_size} bytes")



pkl_filename=f"wfset_{run_number}.pkl"
print(f'Saving the wfset object as a pkl fil: {pkl_filename}')
with open(pkl_filename, "wb") as f:
    pickle.dump(wfset, f)  
    
pkl_size = os.path.getsize(pkl_filename)
print(f"Size: {pkl_size} bytes")


'''

import pickle
import hickle as hkl
import h5py
import numpy as np
import time
import os
import sys
import waffles.input.raw_hdf5_reader as reader

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
    #rucio_filepath = f"/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths/{run_number}.txt"
    #filepaths = reader.get_filepaths_from_rucio(rucio_filepath)
    #wfset = reader.WaveformSet_from_hdf5_file(filepaths[0], read_full_streaming_data=False)
    
    filepath = f"/afs/cern.ch/work/a/arochefe/private/repositories/waffles/test/np02vd_raw_run{run_number}_0000_df-s04-d0_dw_0_20250210T110326.hdf5"

    print("Reading first file...")
    wfset = reader.WaveformSet_from_hdf5_file(filepath, read_full_streaming_data=False)

    # File naming
    pkl_filename = f"wfset_{run_number}.pkl"
    hdf5_filename = f"wfset_{run_number}.hdf5"
    hkl_filename = f"wfset_{run_number}.hkl"

    print("\n### Saving in Different Formats ###")

    # Pickle test
    pkl_size, pkl_time = save_as_pickle(wfset, pkl_filename)
    print(f"[Pickle] Size: {pkl_size} bytes, Time: {pkl_time:.2f} sec")
    
    '''
    # Hickle test
    hkl_size, hkl_time = save_as_hickle(wfset, hkl_filename)
    print(f"[Hickle] Size: {hkl_size} bytes, Time: {hkl_time:.2f} sec")

    # HDF5 tests with different compression methods
    compressions = [None, "gzip", "lzf", "zstd"]
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
    '''
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 wtest_hdf5_reader.py <run_number>")
        sys.exit(1)
    
    main(sys.argv[1])