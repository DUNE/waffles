INPUT_OUTPUT CLASSES
======================

.. admonition:: **Data conversion to classes**

The following classes are related to the conversion from data into WAFFLES classes. Please note that they are listed in chronological order, from the oldest to the newest files. Therefore, the latest files are probably the most used right now:

   * **raw_root_reader**: functions to read the raw data from the root files (self-proccessed) and convert it to WaveformAdcs/Waveform/WaveformSet.
   * **pickle_file_reader**: reads a number of pickle files which should contain WaveformSet objects, and merges them into a single WaveformSet object.
   * **pickle_hdf5_reader**: reads a WaveformSet object from a HDF5 file containing a pickled WaveformSet.
   * **input_utils**: implements a set of Waveforms.
   * **raw_hdf5_reader**: functions to read the raw data from the hdf5 files and convert it to WaveformAdcs/Waveform/WaveformSet.
   * **waveformset_dataframe_utils**: functions to convert between WaveformSet objects and pandas DataFrames.
   * **persistence_utils**: saves a WaveformSet object to a file using either Pickle or HDF5.
   * **hdf5_reader**: class-based wrapper for reading HDF5 files (DAPHNE data) either sequentially or in parallel.
   * **event_file_reader**: creates Event objects combining the information from a pickle of a WaveformSet object and a root file with beam info.
   * **hdf5_structured**: loads a structured HDF5 file into a WaveformSet, optionally filtering by run number, endpoint, or max waveforms. **The current conversion employed!**


raw_root_reader
-----------------

.. autofunction:: waffles.input_output.raw_root_reader.WaveformSet_from_root_files


pickle_file_reader
-----------------

.. autofunction:: waffles.input_output.pickle_file_reader.WaveformSet_from_pickle_files


pickle_hdf5_reader
-----------------

.. autofunction:: waffles.input_output.pickle_hdf5_reader.WaveformSet_from_hdf5_pickle


input_utils
-----------------

.. autofunction:: waffles.input.input_utils.find_ttree_in_root_tfile

.. autofunction:: waffles.input.input_utils.find_tbranch_in_root_ttree

.. autofunction:: waffles.input.input_utils.root_to_array_type_code

.. autofunction:: waffles.input.input_utils.get_1d_array_from_pyroot_tbranch

.. autofunction:: waffles.input.input_utils.split_endpoint_and_channel


raw_hdf5_reader
-----------------

.. autofunction:: waffles.input.raw_hdf5_reader.get_filepaths_from_rucio

.. autofunction:: waffles.input.raw_hdf5_reader.WaveformSet_from_hdf5_files

.. autofunction:: waffles.input.raw_hdf5_reader.WaveformSet_from_hdf5_file


persistence_utils
-----------------

.. autofunction:: waffles.input_output.persistence_utils.WaveformSet_to_file


waveformset_dataframe_utils
-----------------

.. autofunction:: waffles.input_output.waveformset_dataframe_utils.waveformset_to_dataframe


hdf5_reader
-----------------

.. autofunction:: waffles.input_output.hdf5_reader.WaveformSet_from_hdf5_files


event_file_reader
-----------------

.. autofunction:: waffles.input_output.event_file_reader.events_from_pickle_and_beam_files


hdf5_structured
-----------------

.. autofunction:: waffles.input_output.hdf5_structured.load_structured_waveformset