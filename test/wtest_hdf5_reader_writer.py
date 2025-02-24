import pickle
import h5py
import numpy as np
import time
import os
import sys
import hickle as hkl
import uproot
import waffles.input_output.raw_hdf5_reader as reader
import pandas as pd
import matplotlib.pyplot as plt

def extract_run_number(filepath):
    """Extract the run number from the original HDF5 file metadata."""
    with h5py.File(filepath, 'r') as f:
        if 'run_number' in f.attrs:
            return str(f.attrs['run_number'])
        else:
            print(f"Run number not found in {filepath}. Using default.")
            return "unknown"

def save_as_pickle(obj, filename):
    start_time = time.time()
    with open(filename, "wb") as f:
        pickle.dump(obj, f)
    elapsed_time = time.time() - start_time
    file_size = os.path.getsize(filename)
    return file_size, elapsed_time

def read_pickle(filename):
    start_time = time.time()
    with open(filename, "rb") as f:
        obj = pickle.load(f)
    elapsed_time = time.time() - start_time
    return elapsed_time, obj

def save_as_hickle(obj, filename):
    start_time = time.time()
    hkl.dump(obj, filename)
    elapsed_time = time.time() - start_time
    file_size = os.path.getsize(filename)
    return file_size, elapsed_time

def read_hickle(filename):
    start_time = time.time()
    obj = hkl.load(filename)
    elapsed_time = time.time() - start_time
    return elapsed_time, obj

def save_as_hdf5_pickle(obj, filename):
    start_time = time.time()
    obj_bytes = pickle.dumps(obj)
    obj_np = np.frombuffer(obj_bytes, dtype=np.uint8)
    with h5py.File(filename, "w") as hdf:
        hdf.create_dataset("wfset", data=obj_np)
    elapsed_time = time.time() - start_time
    file_size = os.path.getsize(filename)
    return file_size, elapsed_time

def read_hdf5_pickle(filename):
    start_time = time.time()
    with h5py.File(filename, 'r') as f:
        raw_wfset = f['wfset'][:]
    obj = pickle.loads(raw_wfset.tobytes())
    elapsed_time = time.time() - start_time
    return elapsed_time, obj

def save_as_hdf5_comp(obj, filename, compression):
    start_time = time.time()
    obj_bytes = pickle.dumps(obj)
    obj_np = np.frombuffer(obj_bytes, dtype=np.uint8)
    with h5py.File(filename, "w") as hdf:
        hdf.create_dataset("wfset", data=obj_np, compression=compression)
    elapsed_time = time.time() - start_time
    file_size = os.path.getsize(filename)
    return file_size, elapsed_time

def save_as_root(obj, filename):
    start_time = time.time()
    with uproot.recreate(filename) as root_file:
        root_file["wfset"] = {"data": np.array(pickle.dumps(obj), dtype=np.uint8)}
    elapsed_time = time.time() - start_time
    file_size = os.path.getsize(filename)
    return file_size, elapsed_time

def read_root(filename):
    start_time = time.time()
    with uproot.open(filename) as root_file:
        raw_wfset = root_file["wfset"]["data"].array()
    obj = pickle.loads(raw_wfset.tobytes())
    elapsed_time = time.time() - start_time
    return elapsed_time, obj

def process_hdf5_files(folder_path):
    results = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".hdf5"):
            filepath = os.path.join(folder_path, filename)
            run_number = extract_run_number(filepath)
            print(f"Processing file: {filename} (Run: {run_number})")
            
            wfset = reader.WaveformSet_from_hdf5_file(filepath, det="VD_Membrane_PDS", read_full_streaming_data=False)
            
            formats = {
                # "Pickle": (save_as_pickle, read_pickle),
                # "Hickle": (save_as_hickle, read_hickle),
                # "HDF5-Pickle": (save_as_hdf5_pickle, read_hdf5_pickle),
                "HDF5-Gzip": (lambda obj, fname: save_as_hdf5_comp(obj, fname, compression="gzip"), read_hdf5_pickle),
                # "ROOT": (save_as_root, read_root)
            }
            
            for method, (save_func, read_func) in formats.items():
                save_filename = f"wfset_{run_number}_{method}.root" if method == "ROOT" else f"wfset_{run_number}_{method}.hdf5"
                size, write_time = save_func(wfset, save_filename)
                read_time, _ = read_func(save_filename)
                waveformset=read_func(save_filename)[1]
                print(waveformset.waveforms)
                results.append([method, size, write_time, read_time])
    
    df = pd.DataFrame(results, columns=["Method", "Size (bytes)", "Write Time (s)", "Read Time (s)"])
    
    # Compute mean values
    mean_values = df.groupby("Method").mean()
    
    # Plot readout performance
    plt.figure(figsize=(10, 5))
    mean_values[["Write Time (s)", "Read Time (s)"]].plot(kind='bar', figsize=(10,5))
    plt.xlabel("Serialization Method")
    plt.ylabel("Time (s)")
    plt.title("Serialization and Readout Performance")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    
    plot_filename = os.path.join(folder_path, "serialization_readout_performance.png")
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")
    
    return df

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 wtest_hdf5_reader_new.py <folder_path>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    df_results = process_hdf5_files(folder_path)
    print(df_results)
