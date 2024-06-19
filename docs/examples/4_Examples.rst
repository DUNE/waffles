.. _notebook:

👾 **EXAMPLES**
================


With the utilities defined in ``WaveformSet`` you can load waveforms from a ROOT file, filter them and plot them.
You need to specify if the data is self-triggered or full streaming, and the fraction of the data you want to read.
For the moment no mixing of data types is implemented in the same set.


.. code-block:: python

    filepath = '/path/to/run_26607_0_dataflow0_datawriter_0_decode.root'  

    mywvfset = WaveformSet.from_ROOT_file(  filepath,                              # path to the root file                       
                                            bulk_data_tree_name = 'raw_waveforms', # 'raw_waveforms' branch from the root file
                                            meta_data_tree_name = 'metadata',      # 'metadata' branch from the root file
                                            read_full_streaming_data = False,      # self-triggered (False) data
                                            start_fraction=0.0,                    # start reading from the beginning
                                            stop_fraction=1.0)                     # stop reading at the end


You can check the attributes of the object, for example the number of waveforms, the available channels, the number of samples, etc:


.. code-block:: python

    print(mywvfset.AvailableChannels.keys()) # available channels
    print(mywvfset.PointsPerWf)              # number of samples per waveform
    print(len(mywvfset.Waveform))            # how many waveforms match the fiter

You can also create a set from an already existing one, for example to filter the waveforms by endpoint and channel (``wf_filter = WaveformSet.match_endpoint_and_channel``).
For making plots we first generate an auxiliary grid to get the indices of the waveforms inside the set, so that we can plot the selected number of waveforms in the physical position:

.. code-block:: python

    nrows = 10
    ncols = 4

    aux = mywvfset.get_grid_of_wf_idcs( nrows,
                                        ncols,
                                        #wfs_per_axes = None,
                                        wf_filter = WaveformSet.match_endpoint_and_channel,
                                        filter_args = apa_3,
                                        max_wfs_per_axes = 2)

Then we plot:

.. code-block:: python

    figure = psu.make_subplots( rows = nrows,
                                cols = ncols)

    figure = mywvfset.plot( *analysis_args,
                            nrows = nrows,
                            ncols = ncols,
                            figure = figure,
                            wfs_per_axes = None,
                            grid_of_wf_idcs = aux,
                            share_x_scale = True,
                            share_y_scale = True,
                            mode = 'overlay',
                            analysis_label=label,
                            plot_analysis_markers=True,
                            show_baseline_limits = True,
                            show_baseline = True,
                            show_general_integration_limits = False,
                            show_spotted_peaks = True,
                            show_peaks_integration_limits = False,
                            time_bins = 512,
    #                        time_bins = math.floor(mywvfset.PointsPerWf/2),
                            adc_bins = 200,
                            adc_range_above_baseline = 100,
                            adc_range_below_baseline = 300,
                            # adc_range_above_baseline = 1500,
                            # adc_range_below_baseline = 1500,
                            detailed_label=True,
                            **analysis_kwargs)

    figure.update_layout(   width = 1100,
                            height=1200,
                            showlegend = True)
    figure.show()