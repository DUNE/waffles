
from waffles.np04_analysis.LED_calibration.calibration_batches.batch_1.LED_configuration_to_channel \
    import config_to_channels as batch_1
from waffles.np04_analysis.LED_calibration.calibration_batches.batch_2.LED_configuration_to_channel \
    import config_to_channels as batch_2
from waffles.np04_analysis.LED_calibration.calibration_batches.batch_3.LED_configuration_to_channel \
    import config_to_channels as batch_3

# run_to_config is a 6-levels nested dictionary where:
# - the first key level is an integer which labels a certain measurements batch
# - the second key level is the APA number (1,2,3,4)
# - the third key level is the PDE value (0.4, 0.45, 0.5)
# - the fourth key level is the LED configuration
#     - an LED configuration is a 3-tuple whose format is (channel_mask, ticks_width, pulse_bias_percent)
# - the fifth key level is the endpoint number
# - the sixth key level is a list of channels for the given endpoint
config_to_channels = {}

# P.e. config_to_channels[1][3][0.45][(1 , 1, 1400)][111] gives the channels of endpoint 111 from APA 3 for which 
# the data taken with the LED configuration (1 , 1, 1400) during the first measurement batch should be used.

config_to_channels[1] = batch_1
config_to_channels[2] = batch_2
config_to_channels[3] = batch_3