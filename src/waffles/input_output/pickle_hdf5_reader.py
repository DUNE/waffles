import time
import h5py
import _pickle as pickle
import os
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.Exceptions import GenerateExceptionMessage

def WaveformSet_from_hdf5_pickle(filepath: str) -> WaveformSet:
    """
    Reads a WaveformSet object from an HDF5 file containing a pickled WaveformSet.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file to be read.

    Returns
    -------
    WaveformSet
        The deserialized WaveformSet object.
    """
    
    if not os.path.isfile(filepath):
        raise Exception(GenerateExceptionMessage(
            1, 'WaveformSet_from_hdf5_pickle', f"The file '{filepath}' does not exist."
        ))
    
    start_time = time.time()
    
    try:
        with h5py.File(filepath, 'r') as hdf:
            if 'wfset' not in hdf:
                raise Exception(GenerateExceptionMessage(
                    2, 'WaveformSet_from_hdf5_pickle', "Dataset 'wfset' not found in the HDF5 file."
                ))
            raw_wfset = hdf['wfset'][:]
        
        waveform_set = pickle.loads(raw_wfset.tobytes())
        
    except Exception as e:
        raise Exception(GenerateExceptionMessage(
            3, 'WaveformSet_from_hdf5_pickle', f"Error while reading HDF5 file: {e}"
        ))
    
    elapsed_time = time.time() - start_time
    file_size = os.path.getsize(filepath)
    
    print(f"HDF5 file loaded: {filepath} | Size: {file_size} bytes | Time: {elapsed_time:.4f} sec")
    
    return waveform_set
