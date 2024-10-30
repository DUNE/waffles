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

for apa in range(1, 5):
    config_to_channels[apa] = {}

    for pde in [ 0.4, 0.45, 0.5 ]:
        config_to_channels[apa][pde] = {}

config_to_channels[1][0.40][(50, 20, 1400)] = {
    105: [4, 6, 23, 21, 12, 15, 17]
}

config_to_channels[1][0.40][(50, 20, 1800)] = {
    104: [17, 15, 12, 10, 11, 13, 14, 16],
    105: [7, 5, 2, 0, 3, 24],
    107: [12, 10],
}

config_to_channels[1][0.40][(50, 20, 2200)] = {
    104: [5, 1, 3, 4, 6],
    105: [1, 26],
    107: [15, 7, 5],
}

config_to_channels[1][0.40][(50, 20, 2800)] = {
    104: [7, 2, 0],
    105: [10],
    107: [17, 0, 2],
}


config_to_channels[1][0.45][(50, 20, 1400)] = {
    105: [4, 6, 23, 21, 12, 15, 17]
}

config_to_channels[1][0.45][(50, 20, 1800)] = {
    104: [17, 15, 12, 10, 11, 13, 14, 16],
    105: [5, 2, 0, 3, 24],
    107: [12, 10],
}

config_to_channels[1][0.45][(50, 20, 2200)] = {
    104: [5, 2, 1, 3, 4, 6],
    105: [7, 1, 26],
    107: [17, 15, 5, 7],
}

config_to_channels[1][0.45][(50, 20, 2800)] = {
    104: [7, 0], 
    105: [10], 
    107: [0, 2]
}


config_to_channels[1][0.50][(50, 20, 1400)] = {
    105: [4, 6, 23, 21, 12, 17]
}

config_to_channels[1][0.50][(50, 20, 1800)] = {
    104: [17, 15, 12, 10, 11, 13, 14, 16],
    105: [5, 2, 0, 3, 24, 15],
    107: [12, 10],
}

config_to_channels[1][0.50][(50, 20, 2200)] = {
    104: [3, 4, 6],
    105: [7, 1, 26],
    107: [17, 15, 5, 7],
}

config_to_channels[1][0.50][(50, 20, 2800)] = {
    104: [7, 5, 2, 0, 1],
    105: [10],
    107: [0, 2],
}


config_to_channels[2][0.40][(50, 20, 1400)] = {
    109: [ 7, 1, 3, 15, 47, 45]+\
# Ideally one would calibrate these channels using data from (50, 20, 1800)
# LED-config., but that data is not available in this particular case.
    [41, 42]
}

# config_to_channels[2][0.40][(50, 20, 1800)] = {
# 109: [31, 5, 42, 40, 41, 43, 44, 46]
# }
# This data (APA 2, PDE 0.40, using (50, 20, 1800) LED-config.) was acquired during run 28149
# 2024/10/25: Rucio is not able to find data for run 28149

config_to_channels[2][0.40][(50, 20, 2200)] = {
    109: [21, 37, 35, 33, 34, 2, 4, 12]+\
# Ideally one would calibrate these channels using data from (50, 20, 1800)
# LED-config., but that data is not available in this particular case.
    [40, 31, 5, 43, 44, 46]
}

config_to_channels[2][0.40][(50, 20, 2800)] = {
    109: [27, 25, 23, 24, 26, 32, 6]
}

config_to_channels[2][0.40][(50, 20, 3400)] = {
    109: [22, 30, 36, 0]
}

config_to_channels[2][0.40][(50, 20, 4000)] = {
    109: [20]
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
    111: [11, 13, 15, 17, 21, 23, 44, 45, 46, 47]
}

config_to_channels[3][0.40][(1 , 1, 1600)] = {
    111: [10, 12, 14, 16, 24, 26, 30, 32, 40, 41, 42, 43]
}

config_to_channels[3][0.40][(1 , 1, 1800)] = {
    111: [0, 2, 5, 7, 31, 33, 34, 35, 37]
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
    111: [11, 13, 15, 17, 21, 23, 44, 45, 46, 47]+\
# Ideally one would calibrate these channels using data from (1, 1, 1600)
# LED-config., but that data is not available in this particular case.
    [32, 14, 12, 24]}

# config_to_channels[3][0.50][(1 , 1, 1600)] = {
# 111: [32, 30, 41, 43, 16, 14, 12, 40, 42, 26, 24]
# }
# This data (APA 3, PDE 0.50, using (1, 1, 1600) LED-config.) was acquired during run 28177
# 2024/10/25: Rucio is not able to find data for run 28149

config_to_channels[3][0.50][(1 , 1, 1800)] = {
    111: [34, 33, 31, 36, 0, 2, 5, 7, 37, 35, 10, 27, 25]+\
# Ideally one would calibrate these channels using data from (1, 1, 1600)
# LED-config., but that data is not available in this particular case.
    [30, 41, 43, 16, 40, 42, 26]}

config_to_channels[3][0.50][(12, 1, 2000)] = {
    111: [6, 22, 20]
}

config_to_channels[3][0.50][(12, 1, 2200)] = {
    111: [1, 3, 4]
}


config_to_channels[4][0.40][(1, 1, 1400)] = {
    112: [27, 25, 21, 23, 37, 35, 31],
    113: [0],
}

config_to_channels[4][0.40][(1, 1, 1600)] = {
    112: [16, 22, 20, 24, 32, 33, 34, 47, 45],
    113: [2, 5, 7],
}

config_to_channels[4][0.40][(1, 1, 1800)] = {
    112: [0, 2, 5, 7, 1, 3, 4, 6, 10, 12, 15, 17, 14, 13, 11, 26, 30, 36, 40, 42]
}

config_to_channels[4][0.40][(12, 1, 2000)] = {}  # Not used

config_to_channels[4][0.40][(12, 1, 2200)] = {}  # Not used


config_to_channels[4][0.45][(1, 1, 1400)] = {
    112: [27, 25, 37, 35, 31, 21, 23],
    113: [0],
}

config_to_channels[4][0.45][(1, 1, 1600)] = {
    112: [16, 14, 24, 32, 30, 33, 34, 45, 47, 22, 20],
    113: [2, 5, 7],
}

# Based on batch 1, one would calibrate 113-7 based on (1, 1, 1800).
# However, for some reason, there is not data from endpoint 113 for this data taking (run 28167)
config_to_channels[4][0.45][(1 , 1, 1800)] = {
    112: [0, 2, 5, 7, 1, 3, 4, 6, 10, 12, 15, 17, 13, 11, 26, 36, 40, 42]
}

config_to_channels[4][0.45][(12, 1, 2000)] = {} # Not used

config_to_channels[4][0.45][(12, 1, 2200)] = {} # Not used


config_to_channels[4][0.50][(1 , 1, 1400)] = {
    112: [27, 25, 21, 23, 37, 35, 31]+\
# Ideally one would calibrate these channels using data from (1, 1, 1600)
# LED-config., but that data is not available in this particular case.
    [22, 32, 33],
    113: [0]+\
# Ideally one would calibrate these channels using data from (1, 1, 1600)
# LED-config., but that data is not available in this particular case.
    [2]
}

# config_to_channels[4][0.50][(1 , 1, 1600)] = {
# 112: [16, 14, 22, 20, 24, 32, 30, 33, 34, 47, 45],
# 113: [2, 5]
# }
# This data (APA 3, PDE 0.50, using (1, 1, 1600) LED-config.) was acquired during run 28177
# 2024/10/25: Rucio is not able to find data for run 28149

config_to_channels[4][0.50][(1 , 1, 1800)] = {
    112: [0, 2, 5, 7, 1, 3, 4, 6, 10, 12, 15, 17, 13, 11, 26, 36, 40, 42]+\
# Ideally one would calibrate these channels using data from (1, 1, 1600)
# LED-config., but that data is not available in this particular case.
    [14, 16, 30, 34, 20, 47, 45, 24],
    113: [7]+\
# Ideally one would calibrate these channels using data from (1, 1, 1600)
# LED-config., but that data is not available in this particular case.
    [5]
}

config_to_channels[4][0.50][(12, 1, 2000)] = {} # Not used

config_to_channels[4][0.50][(12, 1, 2200)] = {} # Not used