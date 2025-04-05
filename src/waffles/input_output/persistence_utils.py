import os
import time
import pickle
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
    Saves a WaveformSet object to a file using either Pickle or HDF5.
    
    If 'structured=True', uses save_structured_waveformset() to store waveforms
    in a native HDF5 layout. Otherwise, defaults to pickled bytes in an HDF5 dataset.
    
    Args:
        waveform_set (WaveformSet): The WaveformSet to save.
        output_filepath (str): Destination file path.
        overwrite (bool): If True, overwrite existing file. Raises an error if False and file exists.
        format (str): 'pickle' or 'hdf5'.
        compression (str): HDF5 compression method, e.g. 'gzip'.
        compression_opts (int): HDF5 compression level.
        structured (bool): If True, use structured HDF5. If False, store pickled WaveformSet in HDF5.

    Raises:
        Exception: if file exists and overwrite=False.
        TypeError: if waveform_set is not a WaveformSet.
        ValueError: if format is unsupported.
    """

    if not overwrite and os.path.exists(output_filepath):
        raise Exception(GenerateExceptionMessage(
            1, "WaveformSet_to_file",
            f"The given output filepath '{output_filepath}' already exists. It cannot be overwritten."
        ))

    if not isinstance(waveform_set, WaveformSet):
        raise TypeError(f"Expected a WaveformSet object, got {type(waveform_set)}")

    # Option A: Pickle format
    if format == "pickle":
        with open(output_filepath, "wb") as file:
            pickle.dump(waveform_set, file)
        return

    # Option B: HDF5
    if format == "hdf5" and not structured:
        # Store pickled bytes in an HDF5 dataset
        start_time = time.time()
        obj_bytes = pickle.dumps(waveform_set)
        obj_np = np.frombuffer(obj_bytes, dtype=np.uint8)
        with h5py.File(output_filepath, "w") as hdf:
            hdf.create_dataset(
                "wfset",
                data=obj_np,
                compression=compression,
                compression_opts=compression_opts
            )

        elapsed_time = time.time() - start_time
        file_size = os.path.getsize(output_filepath)
        print(f"HDF5 file saved: {output_filepath} "
              f"| Size: {file_size} bytes "
              f"| Time: {elapsed_time:.4f} sec")
        return

    if format == "hdf5" and structured:
        # Native structured layout
        print(f"📨 Structured write requested. Type of waveform_set: {type(waveform_set)}")
        from waffles.input_output.hdf5_structured import save_structured_waveformset
        save_structured_waveformset(
            wfset=waveform_set,
            filepath=output_filepath,
            compression=compression,
            compression_opts=compression_opts
        )
        return

    # If none of the above matched, raise
    raise ValueError("Unsupported format. Use 'pickle' or 'hdf5'.")