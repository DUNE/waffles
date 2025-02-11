from waffles.np04_analysis.led_calibration.configs.calibration_batches.batch_1.configurations import configs as batch_1
from waffles.np04_analysis.led_calibration.configs.calibration_batches.batch_2.configurations import configs as batch_2
from waffles.np04_analysis.led_calibration.configs.calibration_batches.batch_3.configurations import configs as batch_3

# run_to_config is a 5-levels nested dictionary where:
# - the first key level is an integer which labels a certain measurements batch
# - the first key level is the APA number (1,2,3,4)
# - the second key level is the PDE value (0.4, 0.45, 0.5)
# - the third key level is the run number
# - the fourth key level is the LED configuration
#     - an LED configuration is a 3-tuple whose format is (channel_mask, ticks_width, pulse_bias_percent)
run_to_config = {}

# P.e. run_to_config[1][3][0.45] gives a dictionary where the keys are the runs of the first calibration
# batch where data for APA 3 at 0.45 PDE was taken. In turn, run_to_config[2][3][0.45][27914] gives the LED 
# configuration for run 27914

run_to_config[1] = batch_1
run_to_config[2] = batch_2
run_to_config[3] = batch_3