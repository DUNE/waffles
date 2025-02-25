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

def save_as_hdf5_comp(obj, filename, compression, compression_opts=None):
    start_time = time.time()
    obj_bytes = pickle.dumps(obj)
    obj_np = np.frombuffer(obj_bytes, dtype=np.uint8)
    with h5py.File(filename, "w") as hdf:
        hdf.create_dataset("wfset", data=obj_np, compression=compression, compression_opts=compression_opts)
    elapsed_time = time.time() - start_time
    file_size = os.path.getsize(filename)
    return file_size, elapsed_time

def process_hdf5_files(folder_path):
    results = []
    compression_methods = [
        ("gzip", 1), ("gzip", 5), ("gzip", 9),
        ("lzf", None),
        ("szip", ("ec", 8))
    ]
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".hdf5"):
            filepath = os.path.join(folder_path, filename)
            run_number = extract_run_number(filepath)
            print(f"Processing file: {filename} (Run: {run_number})")
            
            wfset = reader.WaveformSet_from_hdf5_file(filepath, det="VD_Membrane_PDS", read_full_streaming_data=False)
            
            formats = {
                
                "HDF5-Pickle": (save_as_hdf5_pickle, read_hdf5_pickle),
                "Pickle": (save_as_pickle, read_pickle),
                "Hickle": (save_as_hickle, read_hickle)
            }
            
            for method, (save_func, read_func) in formats.items():
                save_filename = f"wfset_{run_number}_{method}.hdf5"
                size, write_time = save_func(wfset, save_filename)
                read_time, _ = read_func(save_filename)
                results.append([method, size, write_time, read_time])
            
            for comp_method, comp_opts in compression_methods:
                save_filename = f"wfset_{run_number}_HDF5-{comp_method}-{comp_opts}.hdf5"
                size, write_time = save_as_hdf5_comp(wfset, save_filename, compression=comp_method, compression_opts=comp_opts)
                read_time, _ = read_hdf5_pickle(save_filename)
                results.append([f"HDF5-{comp_method}-{comp_opts}", size, write_time, read_time])
    
    df = pd.DataFrame(results, columns=["Method", "Size (bytes)", "Write Time (s)", "Read Time (s)"])
    
    # Compute mean values
    mean_values = df.groupby("Method").mean()
    
    # Create bar plot with secondary axis for file size
    fig, ax1 = plt.subplots(figsize=(10, 6))
    width = 0.2
    
    ax1.set_xlabel("Serialization Method", fontsize=14)
    ax1.set_ylabel("Time (s)", fontsize=14, color='black')
    ax1.set_title("Serialization Performance: Time & File Size", fontsize=16)
    ax1.grid(True, linestyle="--", linewidth=0.5, color='black')
    
    mean_values[["Write Time (s)", "Read Time (s)"]].plot(kind='bar', ax=ax1, position=0.8, width=width, color=['lightgray', 'gray'], legend=False)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("Size (MB)", fontsize=14, color='black')
    (mean_values["Size (bytes)"] / (1024 * 1024)).plot(kind='bar', ax=ax2, position=-0.5, width=width/2, color='black',  legend=False)
    
    ax1.set_xticklabels(mean_values.index, rotation=90, ha="center", fontsize=12)
    
    ax1.legend(["Write Time", "Read Time"], loc="upper left", bbox_to_anchor=(1, 1))
    ax2.legend(["Size (MB)"], loc="upper left", bbox_to_anchor=(1, 0.85))
    
    plt.tight_layout()
    
    plot_filename = os.path.join(folder_path, "serialization_readout_performance_bw_bars.png")
    plt.savefig(plot_filename, dpi=300, format='png', bbox_inches='tight')
    print(f"Plot saved as {plot_filename}")
    
    return df

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 wtest_hdf5_reader_new.py <folder_path>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    df_results = process_hdf5_files(folder_path)
    print(df_results)