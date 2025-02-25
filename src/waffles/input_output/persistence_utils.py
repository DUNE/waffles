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
) -> None:
    """
    Saves a WaveformSet object to a file using either the Pickle or HDF5 format.

    Parameters
    ----------
    waveform_set : WaveformSet
        The WaveformSet object to be saved.
    output_filepath : str
        The path to the output file.
    overwrite : bool, optional
        If True, overwrites the file if it already exists. If False, raises an exception
        if the file exists. Default is False.
    format : str, optional
        The format in which to save the file. Supported options are:
        - "pickle": Saves the object as a serialized Pickle file.
        - "hdf5" (default): Stores the object in an HDF5 file.
    compression : str, optional
        The compression method for the HDF5 format. Default is "gzip".
        Ignored if `format="pickle"`.
    compression_opts : int, optional
        The compression level for HDF5 format. Default is 5.
        Ignored if `format="pickle"`.

    Raises
    ------
    Exception
        If `overwrite` is False and the file already exists.
    ValueError
        If an unsupported format is specified.

    Notes
    -----
    - When using HDF5 format, the function serializes the WaveformSet object using Pickle,
      converts it to a NumPy array, and stores it as a dataset in the HDF5 file.
    - The function prints the file size and the time taken to save when using HDF5 format.

    Returns
    -------
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
            hdf.create_dataset("wfset", data=obj_np, compression=compression, compression_opts=compression_opts)

        elapsed_time = time.time() - start_time
        file_size = os.path.getsize(output_filepath)

        print(f"HDF5 file saved: {output_filepath} | Size: {file_size} bytes | Time: {elapsed_time:.4f} sec")

    else:
        raise ValueError("Unsupported format. Use 'pickle' or 'hdf5'.")

    return