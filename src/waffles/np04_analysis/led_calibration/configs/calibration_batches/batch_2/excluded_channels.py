# excluded_channels is a 4-levels nested dictionary where:
# - the first key level is the APA number (1,2,3,4)
# - the second key level is the PDE value (0.4, 0.45, 0.5)
# - the third key level is the endpoint
# - the fourth key level is a list of (excluded) channels
excluded_channels = {}  

for apa in range(1, 5):
    excluded_channels[apa] = {}

    for pde in [ 0.4, 0.45, 0.5 ]:
        excluded_channels[apa][pde] = {}

excluded_channels[1][0.40] = {
    107: []
}

excluded_channels[1][0.45] = {
    107: []
}

excluded_channels[1][0.50] = {
    107: []
}


excluded_channels[2][0.40] = {
    109: [17, 11, 13, 14, 16, 10, 7]
}

excluded_channels[2][0.45] = {
    109: [17, 11, 13, 14, 16, 10]
}

excluded_channels[2][0.50] = {
    109: [17, 11, 13, 14, 16, 10]
}


excluded_channels[3][0.40] = {
    111: [31]
}

excluded_channels[3][0.45] = {
    111: [31]
}

excluded_channels[3][0.50] = {
    111: [15, 31]
}


excluded_channels[4][0.40] = {
    112: [11, 33], 
    113: [0, 2, 7]
}

excluded_channels[4][0.45] = {
    112: [11], 
    113: [7]
}

excluded_channels[4][0.50] = {
    112: [11, 22], 
    113: [2]
}
