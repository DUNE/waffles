GENERAL DATA CLASSES
======================

.. admonition:: **Waveforms related classes**

   The following classes are related to the waveforms.
   * WaveformAdcs: implements the adcs array of a Waveform.
   * Waveform: implements a Waveform from WaveformAdcs.
   * WaveformSet: implements a set of Waveforms.
   * WfAna: stands for Waveform Analysis and implements an analysis over an arbitrary WaveformAdcs.
   * WfAnaResult: store the results of an analysis inside the waveform object.

WaveformAdcs
-----------------
.. inheritance-diagram::  waffles.data_classes.WaveformAdcs

.. autoclass:: waffles.data_classes.WaveformAdcs
   :members:

Waveform
-----------------
.. inheritance-diagram::  waffles.data_classes.Waveform

.. autoclass:: waffles.data_classes.Waveform
   :members:

WaveformSet
-----------------
.. inheritance-diagram::  waffles.data_classes.WaveformSet

.. autoclass:: waffles.data_classes.WaveformSet
   :members:

WfAna
-----------------
.. inheritance-diagram::  waffles.data_classes.WfAna

.. autoclass:: waffles.data_classes.WfAna
   :members:

WfAnaResult
-----------------
.. inheritance-diagram::  waffles.data_classes.WfAnaResult

.. autoclass:: waffles.data_classes.WfAnaResult
   :members:

WfPeak
-----------------
.. inheritance-diagram::  waffles.data_classes.WfPeak

.. autoclass:: waffles.data_classes.WfPeak
   :members:


.. admonition:: **Analysis related classes**
   
      The following classes are related to the analysis of the waveforms.
      * BasicWfAna: implements a basic analysis over a waveform.
      **Each analyser can create a particular class for their analysis (i.e. BeamWfAna.py) from BasicWfAna as template.**
      * CalibrationHistogram: implements a histogram for the calibration of the waveforms.
      * PeakFindingWfAna: implements a peak finding analysis over a waveform.
      * TrackedHistogram: implements a histogram for the tracking of the waveforms. 
      
BasicWfAna
-----------------
.. inheritance-diagram::  waffles.data_classes.BasicWfAna

.. autoclass:: waffles.data_classes.BasicWfAna
   :members:

CalibrationHistogram
----------------------
.. inheritance-diagram::  waffles.data_classes.CalibrationHistogram

.. autoclass:: waffles.data_classes.CalibrationHistogram
   :members:

PeakFindingWfAna
------------------
.. inheritance-diagram::  waffles.data_classes.PeakFindingWfAna

.. autoclass:: waffles.data_classes.PeakFindingWfAna
   :members:

TrackedHistogram
------------------

.. inheritance-diagram::  waffles.data_classes.TrackedHistogram

.. autoclass:: waffles.data_classes.TrackedHistogram
   :members:



.. admonition:: **Structural classes**
   
      The following classes are related to the structure of the data.
      * Map: implements a map of the channels.
      * UniqueChannel: implements a unique channel.
      * ChannelMap: implements a map of the channels.
      * ChannelWs: implements a map of the channels with waveforms.
      * ChannelWsGrid: implements a grid of the channels with waveforms. TUsed for plotting the waveforms of all channels, displayed in a grid as in their physical position inside the detector.
      * IODict: implements a dictionary with input/output data.
      * IPDict: implements a dictionary with input data.
      * ORDict: implements a dictionary with output data.

Map
-----------------
.. inheritance-diagram::  waffles.data_classes.Map

.. autoclass:: waffles.data_classes.Map
   :members:

UniqueChannel
------------------
.. inheritance-diagram::  waffles.data_classes.UniqueChannel

.. autoclass:: waffles.data_classes.UniqueChannel
   :members:

ChannelMap
-----------------
.. inheritance-diagram::  waffles.data_classes.ChannelMap

.. autoclass:: waffles.data_classes.ChannelMap
   :members:

ChannelWs
-----------------
.. inheritance-diagram::  waffles.data_classes.ChannelWs

.. autoclass:: waffles.data_classes.ChannelWs
   :members:

ChannelWsGrid
-----------------
.. inheritance-diagram::  waffles.data_classes.ChannelWsGrid

.. autoclass:: waffles.data_classes.ChannelWsGrid
   :members:

IODict
-----------------
.. inheritance-diagram::  waffles.data_classes.IODict

.. autoclass:: waffles.data_classes.IODict
   :members:

IPDict
-----------------
.. inheritance-diagram::  waffles.data_classes.IPDict

.. autoclass:: waffles.data_classes.IPDict
   :members:

ORDict
-----------------
.. inheritance-diagram::  waffles.data_classes.ORDict

.. autoclass:: waffles.data_classes.ORDict
   :members:



