📚 **LIBRARIES**
=================


This document provides a summary of the folder in which the `waffles` classes are stored.

data_classes
-------------

Inside `data_classes` we can find structural classes that are used to store the data. The `WaveformAdcs`, `Waveform` and `WaveformSet` are defined here. 
For more details, see the :doc:`GENERAL DATA CLASSES <31_data_classes>`.

input
-------------
Inside `input` we can find classes that are used to convert the raw data to the `waffles` classes. The `raw_hdf5_reader`, `raw_root_reader` and `input_utils` are defined here.
For more details, see the :doc:`INPUT CLASSES <32_input>`.

np02
-------------
Particular functions and classes for NP02 analysis can be found here. For more details, see the :doc:`NP02 <33_np02>`.

np04
-------------
Particular functions and classes for NP04 analysis can be found here. For more details, see the :doc:`NP04 <34_np04>`.

plotting
-------------
Classes related to the plotting of the data are stored here. The `plot_utils` and `plot` are defined here. Moreover the `np04_display` includes tools for an interactive dash display of the data.
For more details, see the :doc:`PLOTTING <35_plotting>`.

.. toctree::
   :hidden:

   31_data_classes
   32_input
   33_np02
   34_np04
   35_plotting
   36_utils