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
) -> None:
    """
    Saves a WaveformSet object to a file using pickle or HDF5 format.

    Parameters
    ----------
    waveform_set : WaveformSet
        The WaveformSet object to persist.
    output_filepath : str
        Path to the file where the WaveformSet object will be saved.
    overwrite : bool
        If True, overwrite the file if it exists. If False, raise an exception if the file exists.
    format : str, optional
        The format to save the file in. Options are "pickle" (default) or "hdf5".
    compression : str, optional
        Compression type for HDF5. Default is "gzip".
    
    Returns
    ----------
    None
    """

    if not overwrite and os.path.exists(output_filepath):
        raise Exception(GenerateExceptionMessage(
            1, 'WaveformSet_to_file', 'The given output filepath already exists. It cannot be overwritten.'
        ))

    if format == "pickle":
        with open(output_filepath, 'wb') as file:
            pickle.dump(waveform_set, file)
    
    elif format == "hdf5":
        start_time = time.time()
        obj_bytes = pickle.dumps(waveform_set)
        obj_np = np.frombuffer(obj_bytes, dtype=np.uint8)
        
        with h5py.File(output_filepath, "w") as hdf:
            hdf.create_dataset("wfset", data=obj_np, compression=compression)

        elapsed_time = time.time() - start_time
        file_size = os.path.getsize(output_filepath)

        print(f"HDF5 file saved: {output_filepath} | Size: {file_size} bytes | Time: {elapsed_time:.4f} sec")

    else:
        raise ValueError("Unsupported format. Use 'pickle' or 'hdf5'.")

    return