import os
import _pickle as pickle    # Making sure that cPickle is used

from waffles.data_classes.WaveformSet import WaveformSet
from waffles.Exceptions import GenerateExceptionMessage

def WaveformSet_to_file(
        waveform_set : WaveformSet,
        output_filepath : str,
        overwrite : bool = False,
        ) -> None:
                                
    
    """
    This function gets a WaveformSet object and an output
    filepath, and saves the WaveformSet object to the given
    filepath using the pickle library.

    Parameters
    ----------
    waveform_set : WaveformSet
        The WaveformSet object to persist to a pickle file.
    output_filepath : str
        Path to the file where the WaveformSet object will
        be saved.
    overwrite : bool
        If True, then the file at the given output_filepath
        will be overwritten if it already exists. If False,
        then an exception will be raised if the file already
        exists.

    Returns
    ----------        
    None
    """

    if not overwrite and os.path.exists(output_filepath):
        raise Exception(generate_exception_message(1, 
                                                   'WaveformSet_to_file()',
                                                   'The given output filepath already exists. It cannot be overwritten.'))
    else:
        with open(output_filepath, 'wb') as file:
            pickle.dump(waveform_set, file)
    return