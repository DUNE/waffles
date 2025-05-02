# batch_N is a 4-levels nested dictionary where:
# - the first key level is the APA number (1,2,3,4)
# - the second key level is the PDE value (0.4, 0.45, 0.5)
# - the third key level is the run number
# - the fourth key level is the LED configuration
#     - an LED configuration is a 3-tuple whose format is (channel_mask, ticks_width, pulse_bias_percent)
configs = {}
  
for det_id in range(1, 3):
    
    configs[det_id] = {}
        
    for pde in [ 0.4 ]:
        configs[det_id][pde] = {}

configs[1][0.40] = {
        35867: (8, 1, 3200)
}

configs[2][0.40] = {
        35867: (8, 1, 3200)
}
    



