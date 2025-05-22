# excluded_channels is a 4-levels nested dictionary where:
# - the first key level is the APA number (1,2,3,4)
# - the second key level is the PDE value (0.4, 0.45, 0.5)
# - the third key level is the endpoint
# - the fourth key level is a list of (excluded) channels
excluded_channels = {}  


for det_id in range(1, 3):
    excluded_channels[det_id] = {}

    for pde in [ 0.4 ]:
        excluded_channels[det_id][pde] = {}

excluded_channels[1][0.40] = {
    107: [0]
}
