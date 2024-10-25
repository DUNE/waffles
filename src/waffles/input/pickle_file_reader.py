import os
import _pickle as pickle    # Making sure that cPickle is used

from waffles.data_classes.WaveformSet import WaveformSet
from waffles.Exceptions import GenerateExceptionMessage

def pickle_file_to_WaveformSet(
        path_to_pickle_file : str,
        ) -> WaveformSet:
                                
    """
    This function gets a path to a file which should be
    a pickle of a WaveformSet object, and loads it using 
    the pickle library. It returns the resulting WaveformSet 
    object.

    Parameters
    ----------
    path_to_pickle_file: str
        Path to the file which will be loaded. Its extension
        must match '.pkl'.

    Returns
    ----------        
    output: WaveformSet
        The WaveformSet object loaded from the given file
    """

    if os.path.isfile(path_to_pickle_file):
        with open(path_to_pickle_file, 'rb') as file:
            output = pickle.load(file)
    else:
        raise Exception(
            we.GenerateExceptionMessage(
                1, 
                'WaveformSet_from_pickle_file()',
                f"The given path ({path_to_pickle_file}) "
                "does not point to an existing file."))
    
    if not isinstance(output, WaveformSet):
        raise Exception(
            we.GenerateExceptionMessage(2,
            'WaveformSet_from_pickle_file()',
            "The object loaded from the given "
            "file is not a WaveformSet object."))
    
    return output
