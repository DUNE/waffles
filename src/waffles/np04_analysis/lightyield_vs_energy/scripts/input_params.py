waveform_pkl_file = f"/afs/cern.ch/work/a/anbalbon/public/reading_beamrun_NEW/run027361_ALL_hdf5.pkl"
period = 'June'
energy = 3 # GeV
run = 27361

# Channel info
apa = 2
endpoint_selected = 109
channel_selected = 35

#Baseline info 
baseline_limits = {1: [0, 100, 900, 1000], 2: [0, 100, 900, 1000]}
baseliner_std_cut = 3.
baseliner_type = 'mean'
baseline_analysis_label = 'baseliner'
null_baseline_analysis_label = 'null_baseliner'

# Integral info
deviation_from_baseline = 0.3
lower_limit_correction = 0
upper_limit_correction = 0
integration_analysis_label = 'integrator'

bias_info = {
    'June': { 'FBK' : 4.5, 'HPK' : 3., 'batch' : 1 }, # FBK OV, HPK  OV, batch number to use for p.e. calibration
    'July': { 'FBK' : 4.5, 'HPK' : 2.5, 'batch' : 1 },
    'August': { 'FBK' : 4.5, 'HPK' : 2.5, 'batch' : 4 }, 
    'September': { 'FBK' : 4.5, 'HPK' : 2.5, 'batch' : 6 }}

june_calibration_csv = "/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/data/june_calibration_results.csv"
other_calibration_csv = "/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/data/other_calibration_results.csv"