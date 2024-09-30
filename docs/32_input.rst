INPUT CLASSES
======================

.. admonition:: **Data conversion to classes**

   The following classes are related to the conversion from data to WAFFLES classes:

   * raw_hdf5_reader: functions to read the raw data from the hdf5 files and convert it to WaveformAdcs/Waveform/WaveformSet.
   * raw_root_reader: functions to read the raw data from the root files (self-proccessed) and convert it to WaveformAdcs/Waveform/WaveformSet.
   * input_utils: implements a set of Waveforms.

raw_hdf5_reader
-----------------

.. autofunction:: waffles.input.raw_hdf5_reader.get_filepaths_from_rucio

.. autofunction:: waffles.input.raw_hdf5_reader.WaveformSet_from_hdf5_files

.. autofunction:: waffles.input.raw_hdf5_reader.WaveformSet_from_hdf5_file


raw_root_reader
-----------------

.. autofunction:: waffles.input.raw_root_reader.WaveformSet_from_root_files

.. autofunction:: waffles.input.raw_root_reader.WaveformSet_from_root_file


input_utils
-----------------

.. autofunction:: waffles.input.input_utils.find_ttree_in_root_tfile

.. autofunction:: waffles.input.input_utils.find_tbranch_in_root_ttree

.. autofunction:: waffles.input.input_utils.root_to_array_type_code

.. autofunction:: waffles.input.input_utils.get_1d_array_from_pyroot_tbranch

.. autofunction:: waffles.input.input_utils.split_endpoint_and_channel



   

   