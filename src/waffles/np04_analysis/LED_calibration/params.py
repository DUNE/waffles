
""" ------------- Initialize ------------- """

# Path where to look for the root files or the pickle files
#data_folderpath = '/Users/acervera/HEP/DUNE/ProtoDUNE-HD/PDS/data_taking/waffles/docs/examples'
#data_folderpath = 'data'
# Path where to save the plots
plots_saving_filepath = ''

# select the configuration
#batch  = 3  # 1, ...
#apa_no = 3  # 1, 2, 3, 4
#pde    = 0.45  # 0.40, 0.45, 0.50

# overvoltages for HPK and FBK for different PDEs
hpk_ov = {0.4 : 2.0, 0.45 : 3.5, 0.50 : 4.0}
fbk_ov = {0.4 : 3.5, 0.45 : 4.5, 0.50 : 7.0}
ov_no  = {0.4 : 1,   0.45 : 2,   0.50 : 3  }

""" ------------- Parameters for charge histrogram -------------"""

# parameters for the waveform analysis
analysis_label = 'standard'

# lower integration limits 
starting_tick_apa1 = {
            27818: 621,
            27820: 615,
            27822: 615,
            27823: 615,
            27824: 615,
            27825: 615,
            27826: 615,
            27827: 632,
            27828: 626,
            27898: 635,
            27899: 635,
            27900: 618,
            27921: 602,
            27901: 615,
            27902: 615,
            27903: 615,
            27904: 630,
            27905: 620,
            27906: 610,
            27907: 608,
            27908: 602
        }

starting_tick_apa234 = 125

# baseline limits
baseline_limits_apa1   = [100, 400]
baseline_limits_apa234 = [0, 100, 900, 1000]

# Integration window width
integ_window = 40

# get the number of bins for the charge histogram
nbins = 130 #led_utils.get_nbins_for_charge_histo(pde,apa_no) 

""" ------------- Parameters for peak fitting -------------"""

# Maximum number of peaks to fit
max_peaks = 2

# Minimal prominence, as a fraction of the y-range, for a peak to be detected
prominence = 0.15 # [0.10 - 0.2]

# The number of points to fit on either side of the peak maximum
# P.e. setting this to 2 will fit 5 points in total: the maximum and 2 points on either side
half_points_to_fit = 2 # [2 - 3] 

initial_percentage=0.15 

percentage_step=0.05