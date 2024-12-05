from waffles.np04_analysis.led_calibration.configs.calibration_batches.batch_1.metadata import metadata as batch_1
from waffles.np04_analysis.led_calibration.configs.calibration_batches.batch_2.metadata import metadata as batch_2
from waffles.np04_analysis.led_calibration.configs.calibration_batches.batch_3.metadata import metadata as batch_3

# metadata is a dictionary of dictionaries
# - the first key level is an integer which labels a certain measurements batch
# - the values are metadata dictionaries for the given measurements batch
metadata = {}

# P.e. metadata[2] gives the metadata dictionary for the second measurements batch

metadata[1] = batch_1
metadata[2] = batch_2
metadata[3] = batch_3