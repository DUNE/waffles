.. _notebook:

👾 **EXAMPLES**
================


With the utilities defined in ``WaveformSet`` you can load waveforms from a ROOT file, filter them and plot them.
You need to specify if the data is self-triggered or full streaming, and the fraction of the data you want to read.

For a first quick examination of the data you can follow the example in :doc:`00_CheckData <00_CheckData>` (download it locally if necessary). 
If everything is working as expected the output should look similar to:

.. plotly::
      
    import plotly.express
    plotly.io.read_json('00_CheckData_apa3.json')

.. plotly::
      
    import plotly.express
    plotly.io.read_json('00_CheckData_apa4.json')


Next step is analyzing the waveforms (which will allow us to make calibration histograms, heatmpas, etc).
For that ... 🚧


.. toctree::   
    :maxdepth: 2

    00_CheckData