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

# In the third calibration batch (29-30/07/2024), APA 1 data was not acquired
for apa in range(2, 5):
    config_to_channels[apa] = {}

    for pde in [ 0.4, 0.45, 0.5 ]:
        config_to_channels[apa][pde] = {}

config_to_channels[2][0.40][(50, 20, 1400)] = {
    109: [ 7, 1, 3, 15, 47, 45]
}

config_to_channels[2][0.40][(50, 20, 1800)] = {
109: [31, 5, 42, 40, 41, 43, 44, 46]+\
# Ideally one would calibrate these channels using data from (50, 20, 2200)
# LED-config., but that data is not available in this particular case.
    [21, 37, 33, 4, 12]
}

# config_to_channels[2][0.40][(50, 20, 2200)] = {
#     109: [21, 37, 35, 33, 34, 2, 4, 12]
# }
# This data (APA 2, PDE 0.40, using (50, 20, 2200) LED-config.) was acquired during run 28485
# 2024/10/28: Cannot access data from this run

config_to_channels[2][0.40][(50, 20, 2800)] = {
    109: [27, 25, 23, 24, 26, 32, 6]+\
# Ideally one would calibrate these channels using data from (50, 20, 2200)
# LED-config., but that data is not available in this particular case.
    [35, 34, 2]+\
# Ideally one would calibrate these channels using data from (50, 20, 3400)
# LED-config., but that data is not available in this particular case.
    [36, 0]
}

# config_to_channels[2][0.40][(50, 20, 3400)] = {
#     109: [22, 30, 36, 0]
# }
# This data (APA 2, PDE 0.40, using (50, 20, 3400) LED-config.) was acquired during run 28487
# 2024/10/28: Cannot access data from this run

config_to_channels[2][0.40][(50, 20, 4000)] = {
    109: [20]+\
# Ideally one would calibrate these channels using data from (50, 20, 3400)
# LED-config., but that data is not available in this particular case.
    [22, 30]
}


config_to_channels[2][0.45][(50, 20, 1400)] = {
    109: [45, 47, 15, 1, 3, 7]
}

config_to_channels[2][0.45][(50, 20, 1800)] = {
    109: [41, 43, 44, 46, 40, 42, 12, 4, 5, 31]
}

config_to_channels[2][0.45][(50, 20, 2200)] = {
    109: [33, 34, 35, 37, 21]
}

config_to_channels[2][0.45][(50, 20, 2800)] = {
    109: [6, 2, 0, 36, 30, 32, 23, 24, 27]
}

config_to_channels[2][0.45][(50, 20, 3400)] = {
    109: [25, 22, 26]
}

config_to_channels[2][0.45][(50, 20, 4000)] = {
    109: [20]
}


config_to_channels[2][0.50][(50, 20, 1400)] = {
    109: [47, 45, 15, 1, 3]
}

config_to_channels[2][0.50][(50, 20, 1800)] = {
    109: [41, 43, 44, 46, 42, 40, 12, 4, 7, 5, 31]
}

config_to_channels[2][0.50][(50, 20, 2200)] = {
    109: [2, 34, 33, 37, 35, 23, 21]
}

config_to_channels[2][0.50][(50, 20, 2800)] = {
    109: [6, 0, 36, 32, 30, 24, 27, 25]
}

config_to_channels[2][0.50][(50, 20, 3400)] = {
    109: [22, 26]
}

config_to_channels[2][0.50][(50, 20, 4000)] = {
    109: [20]
}


config_to_channels[3][0.40][(1 , 1, 1400)] = {
    111: [11, 13, 15, 17, 21, 23, 44, 45, 46, 47]+\
# Ideally one would calibrate these channels using data from (1, 1, 1600)
# LED-config., but that data is not available in this particular case.
    [14]
}

# config_to_channels[3][0.40][(1 , 1, 1600)] = {
#     111: [10, 12, 14, 16, 20, 24, 26, 30, 32, 40, 41, 42, 43]
# }
# This data (APA 3, PDE 0.40, using (1, 1, 1600) LED-config.) was acquired during run 28362
# 2024/10/29: Rucio is not able to find data for this run

config_to_channels[3][0.40][(1 , 1, 1800)] = {
    111: [0, 2, 5, 7, 31, 33, 34, 35, 37]+\
# Ideally one would calibrate these channels using data from (1, 1, 1600)
# LED-config., but that data is not available in this particular case.
    [10, 12, 16, 24, 26, 30, 32, 40, 41, 42, 43]
}

config_to_channels[3][0.40][(12, 1, 2000)] = {
    111: [6, 27, 25, 22, 20, 36]
}

config_to_channels[3][0.40][(12, 1, 2200)] = {
    111: [1, 3, 4]
}


config_to_channels[3][0.45][(1 , 1, 1400)] = {
    111: [11, 13, 15, 17, 21, 23, 44, 45, 46, 47]
}

config_to_channels[3][0.45][(1 , 1, 1600)] = {
    111: [32, 30, 41, 43, 16, 14, 12, 40, 42, 26, 24]
}

config_to_channels[3][0.45][(1 , 1, 1800)] = {
    111: [34, 33, 31, 0, 2, 5, 7, 37, 35, 10, 27, 25]
}

config_to_channels[3][0.45][(12, 1, 2000)] = {
    111: [6, 36, 22, 20]
}

config_to_channels[3][0.45][(12, 1, 2200)] = {
    111: [1, 3, 4]
}


config_to_channels[3][0.50][(1 , 1, 1400)] = {
    111: [11, 13, 15, 17, 21, 23, 44, 45, 46, 47]
}

config_to_channels[3][0.50][(1 , 1, 1600)] = {
    111: [32, 30, 41, 43, 16, 14, 12, 40, 42, 26, 24]+\
# Ideally one would calibrate these channels using data from (1, 1, 1800)
# LED-config., but that data is not available in this particular case.
    [34, 31, 2, 5, 7, 37, 35, 10]
}

# config_to_channels[3][0.50][(1 , 1, 1800)] = {
#     111: [34, 33, 31, 36, 0, 2, 5, 7, 37, 35, 10, 27, 25]
# }
# This data (APA 3, PDE 0.50, using (1, 1, 1800) LED-config.) was acquired during run 28375
# 2024/10/29: Cannot access data from this run

config_to_channels[3][0.50][(12, 1, 2000)] = {
    111: [6, 22, 20]+\
# Ideally one would calibrate these channels using data from (1, 1, 1800)
# LED-config., but that data is not available in this particular case.
    [36, 33, 0, 27, 25]
}

config_to_channels[3][0.50][(12, 1, 2200)] = {
    111: [1, 3, 4]
}


config_to_channels[4][0.40][(1, 1, 1400)] = {
    112: [27, 25, 21, 23, 37, 35, 31],
    113: [0],
}

# config_to_channels[4][0.40][(1, 1, 1600)] = {
#     112: [16, 22, 20, 24, 32, 33, 34, 47, 45],
#     113: [2, 5, 7],
# }
# This data (APA 4, PDE 0.40, using (1, 1, 1600) LED-config.) was acquired during run 28362
# 2024/10/29: Rucio is not able to find data for this run

config_to_channels[4][0.40][(1, 1, 1800)] = {
    112: [0, 2, 5, 7, 1, 3, 4, 6, 10, 12, 15, 17, 14, 13, 11, 26, 30, 36, 40, 42]+\
# Ideally one would calibrate these channels using data from (1, 1, 1600)
# LED-config., but that data is not available in this particular case.
        [16, 22, 20, 24, 32, 33, 34, 47, 45],
    113: []+\
# Ideally one would calibrate these channels using data from (1, 1, 1600)
# LED-config., but that data is not available in this particular case.
        [2, 5, 7]
}

config_to_channels[4][0.40][(12, 1, 2000)] = {}  # Not used

config_to_channels[4][0.40][(12, 1, 2200)] = {}  # Not used


config_to_channels[4][0.45][(1, 1, 1400)] = {
    112: [27, 25, 37, 35, 31, 21, 23],
    113: [0],
}

config_to_channels[4][0.45][(1, 1, 1600)] = {
    112: [16, 14, 24, 32, 30, 33, 34, 45, 47, 22, 20],
    113: [2, 5],
}

config_to_channels[4][0.45][(1 , 1, 1800)] = {
    112: [0, 2, 5, 7, 1, 3, 4, 6, 10, 12, 15, 17, 13, 11, 26, 36, 40, 42],
    113: [7]
}

config_to_channels[4][0.45][(12, 1, 2000)] = {} # Not used

config_to_channels[4][0.45][(12, 1, 2200)] = {} # Not used


config_to_channels[4][0.50][(1 , 1, 1400)] = {
    112: [27, 25, 21, 23, 37, 35, 31],
    113: [0]
}

config_to_channels[4][0.50][(1 , 1, 1600)] = {
    112: [16, 14, 22, 20, 24, 32, 30, 33, 34, 47, 45]+\
# Ideally one would calibrate these channels using data from (1, 1, 1800)
# LED-config., but that data is not available in this particular case.
    [0, 2, 5, 7, 1, 3, 4, 6, 10, 12, 15, 17, 13, 11, 26, 36, 40, 42],
    113: [2, 5]+\
# Ideally one would calibrate these channels using data from (1, 1, 1800)
# LED-config., but that data is not available in this particular case.
    [7],
}

# config_to_channels[4][0.50][(1 , 1, 1800)] = {
#     112: [0, 2, 5, 7, 1, 3, 4, 6, 10, 12, 15, 17, 13, 11, 26, 36, 40, 42],
#     113: [7]
# }
# This data (APA 4, PDE 0.50, using (1, 1, 1800) LED-config.) was acquired during run 28375
# 2024/10/29: Cannot access data from this run

config_to_channels[4][0.50][(12, 1, 2000)] = {} # Not used

config_to_channels[4][0.50][(12, 1, 2200)] = {} # Not used