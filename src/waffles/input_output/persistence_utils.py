# persistence_utils.py (updated)

import os
import _pickle as pickle    
import time
import numpy as np
import h5py

from waffles.data_classes.WaveformSet import WaveformSet
from waffles.Exceptions import GenerateExceptionMessage

def WaveformSet_to_file(
        waveform_set: WaveformSet,
        output_filepath: str,
        overwrite: bool = False,
        format: str = "hdf5",
        compression: str = "gzip",
        compression_opts: int = 5,
        structured: bool = False,
) -> None:
    """
    Saves a WaveformSet object to a file using either the Pickle or structured HDF5 format.
    """

    if not overwrite and os.path.exists(output_filepath):
        raise Exception(GenerateExceptionMessage(
            1, 'WaveformSet_to_file', 'The given output filepath already exists. It cannot be overwritten.'
        ))

    if format == "pickle":
        with open(output_filepath, 'wb') as file:
            pickle.dump(waveform_set, file)

    elif format == "hdf5" and not structured:
        start_time = time.time()
        obj_bytes = pickle.dumps(waveform_set)
        obj_np = np.frombuffer(obj_bytes, dtype=np.uint8)
        with h5py.File(output_filepath, "w") as hdf:
            hdf.create_dataset("wfset", data=obj_np, compression=compression, compression_opts=compression_opts)
        elapsed_time = time.time() - start_time
        file_size = os.path.getsize(output_filepath)
        print(f"HDF5 file saved: {output_filepath} | Size: {file_size} bytes | Time: {elapsed_time:.4f} sec")

    elif format == "hdf5" and structured:
        from waffles.input_output.hdf5_structured import save_structured_waveformset
        save_structured_waveformset(waveform_set, output_filepath, compression=compression, compression_opts=compression_opts)

    else:
        raise ValueError("Unsupported format. Use 'pickle' or 'hdf5'.")
