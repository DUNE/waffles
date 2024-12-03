from waffles.np04_analysis.LED_calibration.calibration_batches.batch_1.excluded_channels import excluded_channels as batch_1
from waffles.np04_analysis.LED_calibration.calibration_batches.batch_2.excluded_channels import excluded_channels as batch_2
from waffles.np04_analysis.LED_calibration.calibration_batches.batch_3.excluded_channels import excluded_channels as batch_3

# excluded_channels is a 5-levels nested dictionary where:
# - the first key level is an integer which labels a certain measurements batch
# - the second key level is the APA number (1,2,3,4)
# - the third key level is the PDE value (0.4, 0.45, 0.5)
# - the fourth key level is the endpoint
# - the fifth key level is a list of (excluded) channels
excluded_channels = {}

# P.e. excluded_channels[1][1][0.40][105] gives the list of channels for endpoint 105 that are excluded for 
# the first calibration batch, for APA 1 at 0.40 PDE

excluded_channels[1] = batch_1
excluded_channels[2] = batch_2
excluded_channels[3] = batch_3