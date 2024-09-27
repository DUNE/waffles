INPUT CLASSES
======================

.. admonition:: **Data conversion to classes**

   The following classes are related to the conversion from data to WAFFLES classes:

   * raw_hdf5_reader: functions to read the raw data from the hdf5 files and convert it to WaveformAdcs/Waveform/WaveformSet.
   * raw_root_reader: functions to read the raw data from the root files (self-proccessed) and convert it to WaveformAdcs/Waveform/WaveformSet.
   * input_utils: implements a set of Waveforms.

raw_hdf5_reader
-----------------

.. inheritance-diagram::  waffles.input_classes.raw_hdf5_reader

.. autoclass:: waffles.raw_hdf5_reader
   :members:

raw_root_reader
-----------------

.. inheritance-diagram::  waffles.input_classes.raw_root_reader

.. autoclass:: waffles.raw_root_reader
   :members:

input_utils
-----------------

.. inheritance-diagram::  waffles.input_classes.input_utils

.. autoclass:: waffles.input_utils
   :members:
   