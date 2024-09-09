configs = {}    # batch_N is a 4-levels nested dictionary where:
                # - the first key level is the APA number (1,2,3,4)
                # - the second key level is the PDE value (0.4, 0.45, 0.5)
                # - the third key level is the run number
                # - the fourth key level is the LED configuration
                #     - an LED configuration is a 3-tuple whose format is (channel_mask, ticks_width, pulse_bias_percent)
    
for apa in range(1, 5):
    configs[apa] = {}

    for pde in [ 0.4, 0.45, 0.5 ]:
        configs[apa][pde] = {}

configs[1][0.40] = {27898 : (50, 20, 1400),
                    27899 : (50, 20, 1800),
                    27900 : (50, 20, 2200),
                    27921 : (50, 20, 2800)}

configs[1][0.45] = {27902 : (50, 20, 1400),
                    27903 : (50, 20, 1800),
                    27904 : (50, 20, 2200),
                    27901 : (50, 20, 2800)}

configs[1][0.50] = {27905 : (50, 20, 1400),
                    27906 : (50, 20, 1800),
                    27907 : (50, 20, 2200),
                    27908 : (50, 20, 2800)}

configs[2] = configs[1]

configs[3][0.40] = {27917 : (1 , 1 , 1400),
                    27918 : (1 , 1 , 1600),
                    27919 : (1 , 1 , 1800),
                    27920 : (12, 1 , 2000)}

configs[3][0.45] = {27913 : (1 , 1 , 1400),
                    27914 : (1 , 1 , 1600),
                    27915 : (1 , 1 , 1800),
                    27916 : (12, 1 , 2000)}

configs[3][0.50] = {27909 : (1 , 1 , 1400),
                    27910 : (1 , 1 , 1600),
                    27911 : (1 , 1 , 1800),
                    27912 : (12, 1 , 2000)}

configs[4] = configs[3]