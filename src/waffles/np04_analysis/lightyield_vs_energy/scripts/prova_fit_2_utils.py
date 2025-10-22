from waffles.np04_analysis.lightyield_vs_energy.imports import *
from waffles.np04_analysis.lightyield_vs_energy.utils import *


#############################

# Analysis type: bin or channel study

bin_study = False

channel_study = True

#############################

# Analysis data to used

integral_before = True
integral_deconv = False
integral_deconv_filtered = False

analysis_list = [name for name, flag in zip(['integral_before', 'integral_deconv', 'integral_deconv_filtered'], [integral_before, integral_deconv, integral_deconv_filtered]) if flag]

#############################

# Channels to analyze

ch_start = 0
ch_end = 47
ch_range = range(ch_start, ch_end + 1)  # Inclusive range for channels

ch_range_wholeAPA = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37, 40, 41, 42, 43, 44, 45, 46, 47]

#############################

# Parameters for the bin study 

bin_start = 44 
bin_stop = 44

bin_list = range(bin_start, bin_stop + 1)

###########################

# HISTOGRAM OPTIONS

histogram_bin_zero_extremes = True
histogram_bin_error = True

global_hist_dic = {
    'N_bin': 20, #tipical
    'label': lambda energy, counts, N_bin: f"E={energy}GeV - all data ({N_bin} bins, {counts} counts)",
    'color': 'green',
    'alpha': 0.6,
    'color errorbar' : 'green'}

centered_hist_dic = {
    'N_bin': 20,
    'label': lambda energy, counts, N_bin: f"E={energy}GeV - centered data ({N_bin} bins, {counts} counts)",
    'color': 'orange',
    'alpha': 0.6,
    'color errorbar' : 'red'}

#############################

# Other parameters

N_sigma = 0.8 # To select the central region

output_folder = "/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output/fit_analysis" # Where to save output

plotly_show = False

###########################

# FIT AVAILABLE
global_gaussian_fit = False    # Set to True if you want to fit a Gaussian distribution
global_langau_fit = True # Set to True if you want to fit a Landau-Gauss distribution
global_langau_peak_fit = True
global_bi_gaussian_fit = False # Set to True if you want to fit a bi-Gaussian distribution

centered_gaussian_fit = False  # Set to True if you want to fit a centered Gaussian distribution
centered_langau_fit = False # Set to True if you want to fit a Landau-Gauss distribution
centered_bi_gaussian_fit = False # Set to True if you want to fit a centered bi-Gaussian distribution

global_mean = True
centered_mean = False


###########################

# FIT PARAMETERS - GLOBAL HISTOGRAM

global_gaussian_fit_dic = {
    'fit function': gaussian,
    'area': 'global',
    'p0': lambda mu, sigma, counts: [mu, sigma*0.6, max(counts)], #mean, sigma, amplitude
    'bounds': lambda mu, sigma, counts: (
        [mu - 0.7*sigma, sigma*0.5, max(counts)*0.9],
        [mu + 0.7*sigma, sigma*0.8, max(counts)*1.15]),
    'color': 'darkgreen',
    'label' : lambda chi, popt, perr: f"Global gaussian fit (χ²={'{:.2e}'.format(chi) if chi>100 else '{:.2f}'.format(chi)}): \nμ = {to_scientific_notation(popt[0], float(perr[0]))} \nσ = {to_scientific_notation(popt[1], perr[1])} \nA = {to_scientific_notation(popt[2], perr[2])}",
    'linear plot': 0 #indice del parametro di popt da usare come y per fare il plot lineare
}


global_langau_fit_dic = {
    'fit function': langau,
    'area': 'global',
    'p0': lambda mu, sigma, counts: [mu, 0.3*sigma, sigma*0.4, max(counts)], #mpv (most probable value), eta (% of sigma), sigma, amplitude
    'bounds': lambda mu, sigma, counts: (
        [mu - 0.7*sigma, sigma*0.1, sigma*0.3, max(counts)*0.85], 
        [mu + 0.7*sigma, sigma*0.7, sigma*0.7, max(counts)*1.15]),
    'color': 'skyblue',
    'label' : lambda chi, popt, perr: f"Global langau fit (χ²={'{:.2e}'.format(chi) if chi>100 else '{:.2f}'.format(chi)}): \nmpv = {to_scientific_notation(popt[0], float(perr[0]))} \neta = {to_scientific_notation(popt[1], perr[1])} \nσ = {to_scientific_notation(popt[2], perr[2])} \nA = {to_scientific_notation(popt[3], perr[3])}",
    'linear plot': 0 #indice del parametro di popt da usare come y per fare il plot lineare
}  

global_langau_peak_fit_dic = {
    'fit function': langau,
    'area': 'global',
    'p0': lambda mu, sigma, counts: [mu, 0.3*sigma, sigma*0.6, max(counts)], #mpv (most probable value), eta (% of sigma), sigma, amplitude
    'bounds': lambda mu, sigma, counts: (
        [mu - 0.7*sigma, sigma*0.1, sigma*0.5, max(counts)*0.85], 
        [mu + 0.7*sigma, sigma*0.7, sigma*0.8, max(counts)*1.15]),
    'color': 'blue',
    'label' : lambda chi, popt, perr: f"Peak value of langau fit (χ²={'{:.2e}'.format(chi) if chi>100 else '{:.2f}'.format(chi)}): \nmpv = {to_scientific_notation(popt[0], float(perr[0]))} \neta = {to_scientific_notation(popt[1], perr[1])} \nσ = {to_scientific_notation(popt[2], perr[2])} \nA = {to_scientific_notation(popt[3], perr[3])}",
    'linear plot': 0 #indice del parametro di popt da usare come y per fare il plot lineare
}  


global_bi_gaussian_fit_dic = {
    'fit function': bi_gaussian,
    'area': 'global',
    'p0': lambda mu, sigma, counts: [mu - 0.1*sigma, sigma*0.6, max(counts), mu + 0.1*sigma, sigma*0.6, max(counts)*0.5], #mean1, sigma1, amplitude1, mean2, sigma2, amplitude2
    'bounds': lambda mu, sigma, counts: (
        [mu - 0.7*sigma, sigma*0.5, max(counts)*0.9, mu - 0.7*sigma, sigma*0.5, max(counts)*0.2], 
        [mu + 0.7*sigma, sigma*0.8,max(counts)*1.3, mu + 0.7*sigma, sigma*0.7, max(counts)*0.9]),
    'color': 'blueviolet',
    'label' : lambda chi, popt, perr: f"Global bi-gaussian fit (χ²={'{:.2e}'.format(chi) if chi>100 else '{:.2f}'.format(chi)}): \nμ1 = {to_scientific_notation(popt[0], float(perr[0]))} \nσ1 = {to_scientific_notation(popt[1], perr[1])} \nA1 = {to_scientific_notation(popt[2], perr[2])} \nμ2 = {to_scientific_notation(popt[3], float(perr[3]))} \nσ2 = {to_scientific_notation(popt[4], perr[4])} \nA2 = {to_scientific_notation(popt[5], perr[5])}",
    'linear plot': 0 #indice del parametro di popt da usare come y per fare il plot lineare
}  

##

variable_names_global = ["global_gaussian_fit", "global_langau_fit", "global_langau_peak_fit", "global_bi_gaussian_fit"]
fit_characteristics_global = {var: globals()[f"{var}_dic"] for var in variable_names_global if f"{var}_dic" in globals()}
active_fits_global = {var: fit_characteristics_global[var] for var in variable_names_global if globals()[var]}

###########################

# FIT PARAMETERS - CENTERED HISTOGRAM

centered_gaussian_fit_dic = {
    'fit function': gaussian,
    'area': 'centered',
    'p0': lambda mu, sigma, counts: [mu, sigma*0.6, max(counts)], #mean, sigma, amplitude
    'bounds': lambda mu, sigma, counts: (
        [mu - 0.4*sigma, sigma*0.5, max(counts)*0.9],
        [mu + 0.4*sigma, sigma*0.8, max(counts)*1.15]),
    'color': 'lime',
    'label' : lambda chi, popt, perr: f"Centered gaussian fit (χ²={'{:.2e}'.format(chi) if chi>100 else '{:.2f}'.format(chi)}): \nμ = {to_scientific_notation(popt[0], float(perr[0]))} \nσ = {to_scientific_notation(popt[1], perr[1])} \nA = {to_scientific_notation(popt[2], perr[2])}",
    'linear plot': 0 #indice del parametro di popt da usare come y per fare il plot lineare
}

centered_langau_fit_dic = {
    'fit function': langau,
    'area': 'centered',
    'p0': lambda mu, sigma, counts: [mu, 0.3*sigma, sigma*0.6, max(counts)], #mpv (most probable value), eta (% of sigma), sigma, amplitude
    'bounds': lambda mu, sigma, counts: (
        [mu - 0.7*sigma, sigma*0.1, sigma*0.5, max(counts)*0.85], 
        [mu + 0.7*sigma, sigma*0.7, sigma*0.8, max(counts)*1.15]),
    'color': 'darkblue',
    'label' : lambda chi, popt, perr: f"Centered langau fit (χ²={'{:.2e}'.format(chi) if chi>100 else '{:.2f}'.format(chi)}): \nmpv = {to_scientific_notation(popt[0], float(perr[0]))} \neta = {to_scientific_notation(popt[1], perr[1])} \nσ = {to_scientific_notation(popt[2], perr[2])} \nA = {to_scientific_notation(popt[3], perr[3])}",
    'linear plot': 0 #indice del parametro di popt da usare come y per fare il plot lineare
}  

centered_bi_gaussian_fit_dic = {
    'fit function': bi_gaussian,
    'area': 'centered',
    'p0': lambda mu, sigma, counts: [mu - 0.1*sigma, sigma*0.6, max(counts), mu + 0.1*sigma, sigma*0.6, max(counts)*0.5], #mean1, sigma1, amplitude1, mean2, sigma2, amplitude2
    'bounds': lambda mu, sigma, counts: (
        [mu - 0.4*sigma, sigma*0.5, max(counts)*0.9, mu - 0.4*sigma, sigma*0.5, max(counts)*0.2], 
        [mu + 0.4*sigma, sigma*0.8,max(counts)*1.3, mu + 0.4*sigma, sigma*0.7, max(counts)*0.9]),
    'color': 'fuchsia',
    'label' : lambda chi, popt, perr: f"Centered bi-gaussian fit (χ²={'{:.2e}'.format(chi) if chi>100 else '{:.2f}'.format(chi)}): \nμ1 = {to_scientific_notation(popt[0], float(perr[0]))} \nσ1 = {to_scientific_notation(popt[1], perr[1])} \nA1 = {to_scientific_notation(popt[2], perr[2])} \nμ2 = {to_scientific_notation(popt[3], float(perr[3]))} \nσ2 = {to_scientific_notation(popt[4], perr[4])} \nA2 = {to_scientific_notation(popt[5], perr[5])}",
    'linear plot': 0 #indice del parametro di popt da usare come y per fare il plot lineare
}  

##

variable_names_centered = ["centered_gaussian_fit", "centered_langau_fit", "centered_bi_gaussian_fit"]
fit_characteristics_centered = {var: globals()[f"{var}_dic"] for var in variable_names_centered if f"{var}_dic" in globals()}
active_fits_centered = {var: fit_characteristics_centered[var] for var in variable_names_centered if globals()[var]}

 
###########################