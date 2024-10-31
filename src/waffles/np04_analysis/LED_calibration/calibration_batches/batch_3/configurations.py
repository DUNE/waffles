# batch_N is a 4-levels nested dictionary where:
# - the first key level is the APA number (1,2,3,4)
# - the second key level is the PDE value (0.4, 0.45, 0.5)
# - the third key level is the run number
# - the fourth key level is the LED configuration
#     - an LED configuration is a 3-tuple whose format is (channel_mask, ticks_width, pulse_bias_percent)
configs = {}

# In the third calibration batch (29-30/07/2024), APA 1 data was not acquired
for apa in range(2, 5):
    configs[apa] = {}

    for pde in [ 0.4, 0.45, 0.5 ]:
        configs[apa][pde] = {}

configs[2][0.40] = {
    28481: (50, 20, 1400),
    28483: (50, 20, 1800),
    # 2024/10/28: Cannot access data from this run:
    # Unable to open file /tmp/np04hd_raw_run028485_0000_dataflow0_datawriter_0_20240730T123701.hdf5 (File accessibility) File has been truncated
    # 28485: (50, 20, 2200),
    28486: (50, 20, 2800),
    # 2024/10/28: Cannot access data from this run:
    # The first chunk apparently contains no waveform. Inspection of further chunk is needed.
    # 28487: (50, 20, 3400),
    28488: (50, 20, 4000),
}

configs[2][0.45] = {
    28489: (50, 20, 1400),
    28491: (50, 20, 1800),
    28492: (50, 20, 2200),
    28493: (50, 20, 2800),
    28494: (50, 20, 3400),
    28495: (50, 20, 4000),
}

configs[2][0.50] = {
    28496: (50, 20, 1400),
    28497: (50, 20, 1800),
    28498: (50, 20, 2200),
    28499: (50, 20, 2800),
    28500: (50, 20, 3400),
    28501: (50, 20, 4000),
}

configs[3][0.40] = {
    28361: (1, 1, 1400),
    # 2024/10/29: Rucio is not able to find data for this run
    # Details: Data identifier 'hd-protodune:hd-protodune_28362' not found
    # 28362: (1, 1, 1600),
    28364: (1, 1, 1800),
    28365: (12, 1, 2000),
    28366: (12, 1, 2200),
}

configs[3][0.45] = {
    28368: (1, 1, 1400),
    28369: (1, 1, 1600),
    28370: (1, 1, 1800),
    28371: (12, 1, 2000),
    28372: (12, 1, 2200),
}

configs[3][0.50] = {
    28373: (1, 1, 1400),
    28374: (1, 1, 1600),
    # 2024/10/29: Cannot access data from this run:
    # Unable to open file /tmp/np04hd_raw_run028375_0000_dataflow0_datawriter_0_20240729T230744.hdf5 (File accessibility) File has been truncated
    # 28375: (1, 1, 1800),
    28376: (12, 1, 2000),
    28377: (12, 1, 2200),
}

configs[4] = configs[3]
