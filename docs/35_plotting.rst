PLOTTING 
==============

.. admonition:: **General plotting functions**

   The following classes are related to the plotting of the data:

   * plot_utils: implements complements used by the plotting functions
   * plot: implements a set of functions to plot the data (**i.e. plot_WaveformAdcs, plot_ChannelWsGrid**)


plot
-----------------

.. autofunction:: waffles.plotting.plot.plot_WaveformAdcs

.. autofunction:: waffles.plotting.plot.plot_CalibrationHistogram

.. autofunction:: waffles.plotting.plot.plot_ChannelWsGrid



.. admonition:: **Display**

   Tools for an interactive display of the data are implemented in the following classes:

   * np04_display: Dash app implementation for the display of the NP04 data.
   * set_server: functions to set the server for the display of the data.

np04_display
-----------------

.. autoclass:: waffles.plotting.display.Display
   :members:

   



