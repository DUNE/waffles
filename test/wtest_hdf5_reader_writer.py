import pickle
import h5py
import numpy as np
import time
import os
import sys
import hickle as hkl
import waffles.input_output.raw_hdf5_reader as reader
import pandas as pd
import matplotlib.pyplot as plt
import uproot

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

def save_as_hickle(obj, filename):
    start_time = time.time()
    hkl.dump(obj, filename)
    elapsed_time = time.time() - start_time
    file_size = os.path.getsize(filename)
    return file_size, elapsed_time

def save_as_hdf5_pickle(obj, filename):
    start_time = time.time()
    obj_bytes = pickle.dumps(obj)
    obj_np = np.frombuffer(obj_bytes, dtype=np.uint8)
    with h5py.File(filename, "w") as hdf:
        hdf.create_dataset("wfset", data=obj_np)
    elapsed_time = time.time() - start_time
    file_size = os.path.getsize(filename)
    return file_size, elapsed_time

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

def process_hdf5_files(folder_path):
    results = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".hdf5"):
            filepath = os.path.join(folder_path, filename)
            run_number = extract_run_number(filepath)
            print(f"Processing file: {filename} (Run: {run_number})")
            
            wfset = reader.WaveformSet_from_hdf5_file(filepath, det="VD_Membrane_PDS", read_full_streaming_data=False)
            
            formats = {
                "Pickle": save_as_pickle,
                "Hickle": save_as_hickle,
                "HDF5-Pickle": save_as_hdf5_pickle,
                "HDF5-Gzip": lambda obj, fname: save_as_hdf5_comp(obj, fname, compression="gzip"),
                # "ROOT": save_as_root
            }
            
            for method, save_func in formats.items():
                save_filename = f"wfset_{run_number}_{method}.root" if method == "ROOT" else f"wfset_{run_number}_{method}.hdf5"
                size, write_time = save_func(wfset, save_filename)
                results.append([method, size, write_time])
    
    df = pd.DataFrame(results, columns=["Method", "Size (bytes)", "Write Time (s)"])
    
    # Compute mean values
    mean_values = df.groupby("Method").mean()
    
    # Use HDF5-Gzip as reference
    reference = mean_values.loc["HDF5-Gzip"]
    df["Size % Change"] = (df["Size (bytes)"] - reference["Size (bytes)"]) / reference["Size (bytes)"] * 100
    df["Write Time % Change"] = (df["Write Time (s)"] - reference["Write Time (s)"]) / reference["Write Time (s)"] * 100
    
    # Plot percentage changes
    plt.figure(figsize=(10, 5))
    df.groupby("Method")["Size % Change"].mean().plot(kind='bar', label="Size % Change", alpha=0.7)
    df.groupby("Method")["Write Time % Change"].mean().plot(kind='bar', label="Write Time % Change", alpha=0.7, color='red')
    
    plt.xlabel("Serialization Method")
    plt.ylabel("Percentage Change (%)")
    plt.title("Performance Comparison Relative to HDF5-Gzip")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot_filename = os.path.join(folder_path, "serialization_performance_comparison.png")
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
