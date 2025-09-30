GENERAL DATA CLASSES
======================

.. admonition:: **Waveforms related classes**

   The following classes are related to the waveforms:

   * WaveformAdcs: implements the adcs array of a Waveform.
   * Waveform: implements a Waveform from WaveformAdcs.
   * WaveformSet: implements a set of Waveforms.

WaveformAdcs
-----------------

.. autoclass:: waffles.WaveformAdcs
   :members:

Waveform
-----------------
.. inheritance-diagram::  waffles.Waveform

.. autoclass:: waffles.Waveform
   :members:

WaveformSet
-----------------
.. autoclass:: waffles.WaveformSet
   :members:


.. admonition:: **Analysis related classes**
   
      The following classes are related to the analysis of the waveforms:

      * BasicWfAna: implements a basic analysis over a waveform. **Each analyser can create a particular class for their analysis (i.e. BeamWfAna.py) from BasicWfAna as template.**
      * WfAna: stands for Waveform Analysis and implements an analysis over an arbitrary WaveformAdcs.
      * WfAnaResult: stores the results of an analysis inside the waveform object.
      * WfPeak: implements a peak which has been spotted in the adcs attribute of a certain Waveform object.
      * CalibrationHistogram: implements a histogram for the calibration of the waveforms.
      * TrackedHistogram: implements a histogram for the tracking of the waveforms.
      * WafflesAnalysis: implements a Waffles Analysis, fixing a common interface and workflow for all Waffles analyses.
      * StoreWfAna: implements a dummy analysis which is performed over a certain WaveformAdcs object, which simply stores the given input parameters as if they were the result of a real analysis.
      * PeakFindingWfAna: implements a peak finding analysis over a waveform. 
      
BasicWfAna
-----------------
.. inheritance-diagram::  waffles.BasicWfAna

.. autoclass:: waffles.BasicWfAna
   :members:

WfAna
-----------------
.. inheritance-diagram::  waffles.WfAna

.. autoclass:: waffles.WfAna
   :members:

WfAnaResult
-----------------
.. inheritance-diagram::  waffles.WfAnaResult

.. autoclass:: waffles.WfAnaResult
   :members:

WfPeak
-----------------
.. inheritance-diagram::  waffles.WfPeak

.. autoclass:: waffles.WfPeak
   :members:

CalibrationHistogram
----------------------
.. inheritance-diagram::  waffles.CalibrationHistogram

.. autoclass:: waffles.CalibrationHistogram
   :members:

TrackedHistogram
------------------
.. inheritance-diagram::  waffles.TrackedHistogram
   
.. autoclass:: waffles.TrackedHistogram
   :members:

WafflesAnalysis
------------------
.. inheritance-diagram::  waffles.WafflesAnalysis

.. autoclass:: waffles.WafflesAnalysis
   :members:

StoreWfAna
------------------
.. inheritance-diagram::  waffles.StoreWfAna

.. autoclass:: waffles.StoreWfAna
   :members:

PeakFindingWfAna
------------------
.. inheritance-diagram::  waffles.PeakFindingWfAna

.. autoclass:: waffles.PeakFindingWfAna
   :members:


.. admonition:: **Beam Information related classes**
   
      The following classes are related to the Beam Information:

      * BeamInfo: implements a BeamInfo, containing information about the beam particle (time of flight, momentum, cherenkov bits).
      * BeamEvent: implements a BeamEvent, inheriting from Event and extends the base class with beam information (BeamInfo).
      * BeamWfAna: implements a basic analysis which is performed over a certain WaveformAdcs object.

BeamInfo
------------------
.. inheritance-diagram::  waffles.BeamInfo

.. autoclass:: waffles.BeamInfo
   :members:

BeamEvent
------------------
.. inheritance-diagram::  waffles.BeamEvent

.. autoclass:: waffles.BeamEvent
   :members:

BeamWfAna
------------------
.. inheritance-diagram::  waffles.BeamWfAna

.. autoclass:: waffles.BeamWfAna
   :members:


.. admonition:: **Structural classes**
   
      The following classes are related to the structure of the data:

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

.. autoclass:: waffles.Map
   :members:

UniqueChannel
------------------

.. autoclass:: waffles.UniqueChannel
   :members:

ChannelMap
-----------------
.. inheritance-diagram::  waffles.ChannelMap

.. autoclass:: waffles.ChannelMap
   :members:

ChannelWs
-----------------
.. inheritance-diagram::  waffles.ChannelWs

.. autoclass:: waffles.ChannelWs
   :members:

ChannelWsGrid
-----------------

.. autoclass:: waffles.ChannelWsGrid
   :members:

IODict
-----------------

.. autoclass:: waffles.IODict
   :members:

IPDict
-----------------
.. inheritance-diagram::  waffles.IPDict

.. autoclass:: waffles.IPDict
   :members:

ORDict
-----------------
.. inheritance-diagram::  waffles.ORDict

.. autoclass:: waffles.ORDict
   :members: