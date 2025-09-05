INPUT_OUTPUT CLASSES
======================

.. admonition:: **Data conversion to classes**

   The following classes are related to the conversion from data into WAFFLES classes. Please note that they are listed in order, from the oldest files to the newest:

   * raw_hdf5_reader: functions to read the raw data from the hdf5 files and convert it to WaveformAdcs/Waveform/WaveformSet.
   * raw_root_reader: functions to read the raw data from the root files (self-proccessed) and convert it to WaveformAdcs/Waveform/WaveformSet.
   * input_utils: implements a set of Waveforms.
   * persistence_utils: saves a WaveformSet object to a file using either Pickle or HDF5.
   * pickle_file_reader: reads a number of pickle files which should contain WaveformSet objects, and merges them into a single WaveformSet object.
   * hdf5_reader: class-based wrapper for reading HDF5 files (DAPHNE data) either sequentially or in parallel.
   * event_file_reader: creates Event objects combining the information from a pickle of a WaveformSet object and a root file with beam info.
   * hdf5_structured: loads a structured HDF5 file into a WaveformSet, optionally filtering by run number, endpoint, or max waveforms.


raw_hdf5_reader
-----------------

.. autofunction:: waffles.input_output.raw_hdf5_reader.get_filepaths_from_rucio

.. autofunction:: waffles.input_output.raw_hdf5_reader.WaveformSet_from_hdf5_files

.. autofunction:: waffles.input_output.raw_hdf5_reader.WaveformSet_from_hdf5_file


raw_root_reader
-----------------

.. autofunction:: waffles.input_output.raw_root_reader.WaveformSet_from_root_files

.. autofunction:: waffles.input_output.raw_root_reader.WaveformSet_from_root_file


input_utils
-----------------

.. autofunction:: waffles.input_output.input_utils.find_ttree_in_root_tfile

.. autofunction:: waffles.input_output.input_utils.find_tbranch_in_root_ttree

.. autofunction:: waffles.input_outputinput_utils.root_to_array_type_code

.. autofunction:: waffles.input_output.input_utils.get_1d_array_from_pyroot_tbranch

.. autofunction:: waffles.input_output.input_utils.split_endpoint_and_channel


persistence_utils
-----------------

.. autofunction:: waffles.input_output.persistence_utils.WaveformSet_to_file


pickle_file_reader
-----------------

.. autofunction:: waffles.input_output.pickle_file_reader.WaveformSet_from_pickle_files


hdf5_reader
-----------------

.. autofunction:: waffles.input_output.hdf5_reader.WaveformSet_from_hdf5_files


event_file_reader
-----------------

.. autofunction:: waffles.input_output.event_file_reader.events_from_pickle_and_beam_files


hdf5_structured
-----------------

.. autofunction:: waffles.input_output.hdf5_structured.load_structured_waveformset


   