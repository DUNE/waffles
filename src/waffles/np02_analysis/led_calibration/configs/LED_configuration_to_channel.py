# config_to_channels is a 5-levels nested dictionary where:
# - the first key level is the APA number (1,2,3,4)
# - the second key level is the PDE value (0.4, 0.45, 0.5)
# - the third key level is the LED configuration
#     - an LED configuration is a 3-tuple whose format is (channel_mask, ticks_width, pulse_bias_percent)
# - the fourth key level is the endpoint number
# - the fifth key level is a list of channels for the given endpoint
config_to_channels = {}

# P.e. config_to_channels[2][0.45][(50, 20, 2200)] gives the endpoints and channels from APA 2 at 0.45
# PDE which should be calibrated using the data collected using the LED configuration (50, 20, 2200)

for det_id in range(1, 3):
    config_to_channels[det_id] = {}

    for pde in [ 0.4 ]:
        config_to_channels[det_id][pde] = {}

config_to_channels[1][0.40][(8, 1, 3000)] = {
    107: [0, 7, 20, 27, 40, 42, 45,47]
}

config_to_channels[2][0.40][(8, 1, 3000)] = {
    107: [10, 17, 30, 37, 41, 43, 44, 46]
}

config_to_channels[1][0.40][(8, 1, 3100)] = {
    107: [0, 7, 20, 27, 40, 42, 45,47]
}

config_to_channels[2][0.40][(8, 1, 3100)] = {
    107: [10, 17, 30, 37, 41, 43, 44, 46]
}

config_to_channels[1][0.40][(8, 1, 3200)] = {
    107: [0, 7, 20, 27, 40, 42, 45,47]
}

config_to_channels[2][0.40][(8, 1, 3200)] = {
    107: [10, 17, 30, 37, 41, 43, 44, 46]
}


