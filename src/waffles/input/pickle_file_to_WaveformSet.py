import os
import _pickle as pickle    # Making sure that cPickle is used

from waffles.data_classes.WaveformSet import WaveformSet
from waffles.Exceptions import generate_exception_message

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
    path_to_pickle_file : str
        Path to the file which will be loaded. Its extension
        must match '.pkl'.

    Returns
    ----------        
    output : WaveformSet
        The WaveformSet object loaded from the given file
    """

    if os.path.isfile(path_to_pickle_file) and path_to_pickle_file.endswith('.pkl'):
        with open(path_to_pickle_file, 'rb') as file:
            output = pickle.load(file)
    else:
        raise Exception(generate_exception_message(1, 
                                                   'pickle_file_to_WaveformSet()',
                                                   'The given file path is not a valid pickle file.'))
    if not isinstance(output, WaveformSet):
        raise Exception(generate_exception_message(2,
                                                    'pickle_file_to_WaveformSet()',
                                                    'The object loaded from the given file is not a WaveformSet object.'))
    return output