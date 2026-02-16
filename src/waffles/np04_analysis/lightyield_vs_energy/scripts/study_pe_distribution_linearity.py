'''
For the full APA 1
- study che <Npe> distribution at each energy, doing a fit with Langauss + Gaussian 
- study the gaussian mean vs energy + linear and parabolic fit
- study the langauss x peak vs energy (muon stability)
- study the gaussian sigma vs energy ** NEW


- Input data from merging_results_apa12.ipynb 
'''

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np
# from pylandau import langau
from landaupy import langauss as lg

from scipy.optimize import minimize_scalar
from scipy.optimize import curve_fit
import pickle
import pandas as pd

from scipy.odr import ODR, Model, RealData



plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    #'xtick.labelsize': 12,
    #'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,  # backup if you don't specify dpi in savefig
})


dict_info = {1 : {'filepath' : "/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output/apa1_vs_apa2/1GeV/apa12_study/all_apa1_mean_photoelectrons.csv",
                'run': 27343,
                'which binning' : 'width',
                'bin width': 2,
                'pe_separation':0,
                'use_given_parameters' : False,
                'use mean' : False},
            2 :{'filepath' : "/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output/apa1_vs_apa2/2GeV/apa12_study/all_apa1_mean_photoelectrons.csv",
                'run': 27355,
                'which binning' : 'bins',
                'bins' : 150,
                'bin width': 2,
                'pe_separation': 110,
                'initial_guess' : [ 100, 5, 10, 100, 120,  50, 100 ],
                'bounds_low' : [ 85, 0.1, 0.2, 0, 100, 0.1,  0],
                'bounds_high' : [ 150, 10, 20, np.inf, 180, 200, np.inf ],
                'use_given_parameters' : True
                },
             3:{'filepath' : "/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output/apa1_vs_apa2/3GeV/apa12_study/all_apa1_mean_photoelectrons.csv", #"/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/scripts/Renan_scripts/pe_histogram_data/3GeV_APA1_ok.csv",
                'run': 27361,
                'which binning' : 'width',
                'bin width': 4,
                'pe_separation': 150,
                'initial_guess' : [ 89, 0.7, 12.95, 500, 212.85,  54.58, 30 ],
                'bounds_low' : [ 85, 0.6, 9, 400, 200, 47,  20],
                'bounds_high' : [ 95, 1.2, 15, 700, 230, 60, 60 ],
                'use_given_parameters' : False},
             5:{'filepath' : "/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output/apa1_vs_apa2/5GeV/apa12_study/all_apa1_mean_photoelectrons.csv",
                'run': 27367,
                'which binning' : 'width',
                'bin width': 5,
                'pe_separation': 190,
                'use_given_parameters' : False},
             7: {'filepath' : "/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output/apa1_vs_apa2/7GeV/apa12_study/all_apa1_mean_photoelectrons.csv", #"/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/scripts/Renan_scripts/pe_histogram_data/7GeV_APA1_ok.csv",
                 'run' : 27374,
                 'bin width': 8,
                'which binning' : 'width',
                'pe_separation':200,
                'use_given_parameters' : False},
            }


def r2_score(y, y_fit):
    """
    Calcola il coefficiente di determinazione R².
    """
    ss_res = np.sum((y - y_fit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot
    return r2

def r2_score_weighted(y, y_fit, yerr):
    """
    Calcola R² pesato.
    """
    w = 1 / yerr**2
    y_mean_w = np.average(y, weights=w)
    ss_res_w = np.sum(w * (y - y_fit)**2)
    ss_tot_w = np.sum(w * (y - y_mean_w)**2)
    r2_w = 1 - ss_res_w / ss_tot_w
    return r2_w

def chi2_func(y, y_fit, yerr, n_params):
    """
    Calcola chi-quadro e chi-quadro ridotto.
    """
    residuals = (y - y_fit) / yerr
    chi2_val = np.sum(residuals**2)
    ndf = len(y) - n_params
    chi2_red = chi2_val / ndf
    return chi2_val, chi2_red, ndf


def linear(x, A, B):
    return A + B*x

def linear_array(params, x):
    a,b = params
    return linear(x,a,b)

def parabola(x,A,B,C):
    return A + B*x + C*x*x

def parabola_array(params, x):
    a, b, c = params
    return parabola(x,a,b,c)

def langauss(x, mpv, eta, sigma, A):
    return A * lg.pdf(x, mpv, eta, sigma)

def gaussian(x, mu, sigma, A):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Combined model: Langauss + Gaussian
def langauss_plus_gauss(x,
                        mpv, eta, sigma_lg, A,
                        mu, sigma_g, B):
    return (
        langauss(x, mpv, eta, sigma_lg, A)
        + gaussian(x, mu, sigma_g, B)
    )

def langauss_plus_gauss_array(params, x):
    mpv, eta, sigma_lg, A, mu, sigma_g, B = params
    return langauss_plus_gauss(x, mpv, eta, sigma_lg, A, mu, sigma_g, B)


def find_peak(params):
    mpv, eta, sigma, A = params

    # --- Prevent very small sigma from breaking the optimizer
    sigma = max(sigma, 1e-3)
    search_width = max(0.1, 5 * sigma)  # Ensure minimum range

    result = minimize_scalar(
        lambda x: -langauss(x, mpv, eta, sigma, A),
        bounds=(mpv - search_width, mpv + search_width),
        method='bounded'
    )
    return result.x


def propagate_error(find_peak, params, errors, epsilon=1e-5):
    peak = find_peak(params)
    partials = []

    for i in range(len(params)):
        params_eps_plus = params.copy()
        params_eps_minus = params.copy()

        params_eps_plus[i] += epsilon
        params_eps_minus[i] -= epsilon

        f_plus = find_peak(params_eps_plus)
        f_minus = find_peak(params_eps_minus)

        derivative = (f_plus - f_minus) / (2 * epsilon)
        partials.append(derivative)

    # --- Now propagate the errors
    squared_terms = [(partials[i] * errors[i])**2 for i in range(len(params))]
    total_error = np.sqrt(sum(squared_terms))

    return peak, total_error


#####################################################################################
#####################################################################################
#####################################################################################


dict_energy_linearity = {}

# ----------------------------------------------------------------------
# 1. 2-3-5-7 GeV analysis
# ----------------------------------------------------------------------

for energy in [2, 3, 5, 7]:
    print(f'\n ---------------------\nAnalysis {energy} GeV ...')

    # --- All data from merging_results_apa12.ipynb output
    df = pd.read_csv(dict_info[energy]['filepath'])

    # --- First selection 
    select = np.asarray([s for s in df.PhotonA_y_x if s > 50])
    
    # --- Histogram build on fixed bin width or number of bins  
    if dict_info[energy]['which binning'] == 'width':
        width = dict_info[energy]['bin width']
        bins = np.arange(0, np.max(select) + width, width)
    else: 
        bins = dict_info[energy]['bins']

    # --- Histogram
    count, b = np.histogram(select, bins=bins)
    bin_centers = (b[:-1] + b[1:]) / 2
    bin_width = b[1] - b[0]

    # --- Second selection 
    # x_fit_data and y_fit_data MUST BE USED FOR THE FIT!!!!!!!!!!!!!
    mask = count > 0
    x_fit_data = bin_centers[mask]
    y_fit_data = count[mask]
    y_err = np.sqrt(y_fit_data)
    x_err = np.full_like(x_fit_data, 0.5 * bin_width)


    # --- Initial params values and bounds
    # From input dict or automatically computed 
    if dict_info[energy]['use_given_parameters']:
        initial_guess = dict_info[energy]['initial_guess']
        bounds_low = dict_info[energy]['bounds_low']
        bounds_high = dict_info[energy]['bounds_high']
    else: 
        langauss_pe = select[select < dict_info[energy]['pe_separation']]
        gauss_pe    = select[select > dict_info[energy]['pe_separation']]

        mask_bin_lg = bin_centers < dict_info[energy]['pe_separation']
        mask_bin_g  = bin_centers > dict_info[energy]['pe_separation']

        initial_guess = [
            np.median(langauss_pe),      # mpv
            3,                         # eta
            np.std(langauss_pe)/2,         # sigma_lg
            np.max(count[mask_bin_lg]),  # A_lg
            np.mean(gauss_pe),           # mu
            np.std(gauss_pe),            # sigma_g
            np.max(count[mask_bin_g])    # A_g
        ]

        bounds_low = [
            50, 1, 5,  0,
            150,   np.std(gauss_pe)-20,  np.max(count[mask_bin_g])/2
        ]

        bounds_high = [
            150,   10.0, 25, np.inf,
            np.max(select),  np.std(gauss_pe)+20, np.inf
        ]


    # --- First fit, only y_err 
    params_initial, covariance_initial = curve_fit(
        langauss_plus_gauss,
        x_fit_data,
        y_fit_data,
        p0=initial_guess,
        sigma=y_err,
        absolute_sigma=True,
        bounds=(bounds_low, bounds_high),
        maxfev=30000,
        method='trf'
    )


    # --- x_err propagation 
    epsilon = 1e-5
    dfdx = (langauss_plus_gauss(x_fit_data + epsilon, *params_initial)
            - langauss_plus_gauss(x_fit_data - epsilon, *params_initial)) / (2 * epsilon)

    sigma_x = 0.5 * bin_width

    sigma_tot = np.sqrt(y_err**2 + (dfdx * sigma_x)**2)

    # --- Second fit, with sigma_tot = y_err + x_err
    params, covariance = curve_fit(
    langauss_plus_gauss,
    x_fit_data,
    y_fit_data,
    p0=params_initial,
    sigma=sigma_tot,
    absolute_sigma=True,
    bounds=(0.85*params_initial, 1.15*params_initial),
    maxfev=30000,
    method='trf'
    )

    ## If you want to use just the first fit results....
    # params = params_initial
    # covariance = covariance_initial


    ## Alternative way - it doesn't work for 3GeV (no bounds can be given)
    # data = RealData(x_fit_data, y_fit_data, sx=x_err, sy=y_err)
    # model = Model(langauss_plus_gauss_array)
    # odr = ODR(data, model, beta0=params_initial)
    # out = odr.run()
    # params = out.beta
    # errors = out.sd_beta

    
    # --- Params and error extraction 
    errors = np.sqrt(np.diag(covariance))
    mpv, eta, sigma_lg, A_lg, mu, sigma_g, A_g = params 
    empv, eeta, esigma_lg, eA_lg, emu, esigma_g, eA_g = errors

    # --- Fit function computation, to draw
    x_fit = np.linspace(min(b), max(b), 10000)
    y_fit = langauss_plus_gauss(x_fit, *params) # Guassian + Langauss
    y_lg = langauss(x_fit, mpv, eta, sigma_lg, A_lg) # Langauss only
    y_g  = gaussian(x_fit, mu, sigma_g, A_g) # Gaussian onluy


    # --- Intersection point computation (if required)
    # intersection_mask = (x_fit >= mpv) & (x_fit <= mu)
    # x_sub  = x_fit[intersection_mask]
    # y_g_sub  = y_g[intersection_mask]
    # y_lg_sub = y_lg[intersection_mask]
    # intersection_idx = np.argmin(np.abs(y_g_sub - y_lg_sub))
    # x_intersection = x_sub[intersection_idx]

    # --- Langauss peak position
    peak, error_peak = propagate_error(find_peak, params[:4], errors[:4], epsilon=1e-5)

    # --- Chi2 and R2
    y_fit_pred = langauss_plus_gauss(x_fit_data, *params)
    r_squared = r2_score(y_fit_data, y_fit_pred)
    chi2, chi2_rid, ndf = chi2_func(y_fit_data, y_fit_pred, y_err, len(params))


    # --- Plot
    plt.figure(figsize=(8, 5))
    plt.stairs(values=count, edges=b, color='orange', label='Data')
    plt.plot(x_fit, y_fit, 'k-', lw=2, label='Langauss + Gaussian')
    plt.plot(x_fit, y_lg, 'r--', lw=1, label='Langauss')
    plt.plot(x_fit, y_g,  'b--', lw=1, label='Gaussian')
    # plt.axvline(x_intersection, color='gold', linestyle='--', linewidth=1, label="Intersection")


    info_text = (
        f"Langauss:\n"
        f"$x_{{mpv}}$ = {mpv:.2f} ± {empv:.2f}\n"
        f"$\\chi$ = {eta:.2f} ± {eeta:.2f}\n"
        f"$\\sigma$ = {sigma_lg:.2f} ± {esigma_lg:.2f}\n"
        f"$A$ = {A_lg:.0f} ± {eA_lg:.0f}\n"
        f"$x_{{0}}$ = {peak:.2f} ± {error_peak:.2f}\n\n"

        f"Gaussian:\n"
        f"$\\mu$ = {mu:.2f} ± {emu:.2f}\n"
        f"$\\sigma$ = {sigma_g:.2f} ± {esigma_g:.2f}\n"
        f"$A$ = {A_g:.0f} ± {eA_g:.0f}\n\n"

        f"R$^2$ = {r_squared:.3f}\n"
        f"$\chi^2_{{rid}}$ = {chi2_rid:.2f}\n\n"

        #f"Intersection = {x_intersection:.0f}\n\n"

        f"bins width = {dict_info[energy]['bin width']:.0f}"
    )

    ax = plt.gca()
    box = AnchoredText(
        info_text,
        loc='upper right',
        frameon=True
    )
    ax.add_artist(box)

    plt.title(f"Energy {energy} GeV - Run 0{dict_info[energy]['run']} - APA 1")
    plt.xlabel(r'$\langle N_{\mathrm{PE}} \rangle$')
    plt.ylabel('Counts [AU]')
    plt.xlim(0, mu*2)
    plt.ylim(0, max(count) * 1.2)
    plt.legend(loc='upper left', ncol=1)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()

    plt.text(0.7, 0.95, r'$\bf{ProtoDUNE\!-\!HD}$ Preliminary',
         transform=plt.gca().transAxes,
         fontsize=11, ha='right', va='top')

    plt.savefig(f'/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output/apa1_vs_apa2/pedistribution_linearity/{energy}GeV.png')
    plt.close()

    # --- Saving data for next analysis 
    dict_energy_linearity[energy] = {'mpv' : mpv, 'eta' : eta, 'sigma_lg' : sigma_lg, 'A_lg' : A_lg, 'mu' : mu, 'sigma_g': sigma_g, 'A_g' : A_g, 'peak': peak,
                                     'empv' : empv, 'eeta' : eeta, 'esigma_lg' : esigma_lg, 'eA_lg' : eA_lg, 'emu' : emu, 'esigma_g': esigma_g, 'eA_g' : eA_g, 'epeak': error_peak}

    print(f'DONE !!\n\n ')

#####################################################################################
#####################################################################################

# ----------------------------------------------------------------------
# 2. 1 GeV analysis
# ----------------------------------------------------------------------

energy = 1

# --- All data from merging_results_apa12.ipynb output
df = pd.read_csv(dict_info[energy]['filepath'])

print(f'\n ---------------------\nAnalysis {energy} GeV ...')

# --- First selection 
select = np.asarray([s for s in df.PhotonA_y_x if (s > 10) and (s< 150)])

# --- Histogram build on fixed bin width or number of bins  
if dict_info[energy]['which binning'] == 'width':
    width = dict_info[energy]['bin width']
    bins = np.arange(0, np.max(select) + width, width)
else: 
    bins = dict_info[energy]['bins']

# --- Histogram
count, b = np.histogram(select, bins=bins)
bin_centers = (b[:-1] + b[1:]) / 2
bin_width = b[1] - b[0]

# --- Second selection 
# x_fit_data and y_fit_data MUST BE USED FOR THE FIT!!!!!!!!!!!!!
mask = count > 0
x_fit_data = bin_centers[mask]
y_fit_data = count[mask]
y_err = np.sqrt(y_fit_data)

# --- Initial params values and bounds
# From input dict or automatically computed (both for gaussian and langauss)
if dict_info[energy]['use_given_parameters']:
        langauss_initial_guess = dict_info[energy]['langauss_initial_guess']
        lagauss_bounds_low = dict_info[energy]['lagauss_bounds_low']
        langauss_bounds_high = dict_info[energy]['langauss_bounds_high']

        gaussian_initial_guess = dict_info[energy]['gaussian_initial_guess']
        gaussian_bounds_low = dict_info[energy]['gaussian_bounds_low']
        gaussian_bounds_high = dict_info[energy]['gaussian_bounds_high']
else: 
    langauss_initial_guess = [np.median(select), 3, np.std(select)/2, np.max(select)]   
    lagauss_bounds_low = [ 50, 1, 5,  0] 
    langauss_bounds_high = [ 100,   10.0, 25, np.inf]

    gaussian_initial_guess = [ np.mean(select), np.std(select), np.max(select) ]
    gaussian_bounds_low = [50, 5,  10]
    gaussian_bounds_high = [100,  25, np.inf]

# --- Langauss fit 

# First fit with only y_err
langauss_params_initial, langauss_covariance_initial = curve_fit(langauss, x_fit_data,  y_fit_data, p0=langauss_initial_guess, sigma=y_err, absolute_sigma=True, bounds=(lagauss_bounds_low, langauss_bounds_high), maxfev=30000, method='trf' )

# x_err propagation
epsilon = 1e-5
dfdx = (langauss(x_fit_data + epsilon, *langauss_params_initial)- langauss(x_fit_data - epsilon, *langauss_params_initial)) / (2 * epsilon)
sigma_x = 0.5 * bin_width
sigma_tot = np.sqrt(y_err**2 + (dfdx * sigma_x)**2)

# Second fit with y_tot
langauss_params, langauss_covariance = curve_fit(
langauss,
x_fit_data,
y_fit_data,
p0=langauss_params_initial,
sigma=sigma_tot,
absolute_sigma=True,
bounds=(0.85*langauss_params_initial, 1.15*langauss_params_initial),
maxfev=30000,
method='trf'
)

# Parameters and error extrapolation
langauss_errors = np.sqrt(np.diag(langauss_covariance))
mpv, eta, sigma_lg, A_lg = langauss_params
empv, eeta, esigma_lg, eA_lg = langauss_errors
langauss_peak, langauss_error_peak = propagate_error(find_peak, langauss_params[:4], langauss_errors[:4], epsilon=1e-5)

# Fit info for plot + Chi2 and R2
x_fit = np.linspace(min(b), max(b), 10000)
langauss_y_fit = langauss(x_fit, *langauss_params)
langauss_y_fit_pred = langauss(x_fit_data, *langauss_params)
langauss_r_squared = r2_score(y_fit_data, langauss_y_fit_pred)
langauss_chi2, langauss_chi2_rid, langauss_ndf = chi2_func(y_fit_data, langauss_y_fit_pred, y_err, len(langauss_params))


# --- Gaussian fit - easy (no x_err)
gaussian_params, gaussian_covariance = curve_fit(gaussian, x_fit_data, y_fit_data, p0=gaussian_initial_guess, sigma=y_err, absolute_sigma=True, bounds=(gaussian_bounds_low, gaussian_bounds_high), maxfev=30000, method='trf' )
gaussian_errors = np.sqrt(np.diag(gaussian_covariance))
mu, sigma_g, A_g = gaussian_params
emu, esigma_g, eA_g = gaussian_errors
x_fit = np.linspace(min(b), max(b), 10000)
gaussian_y_fit = gaussian(x_fit, *gaussian_params)
gaussian_y_fit_pred = gaussian(x_fit_data, *gaussian_params)
gaussian_r_squared = r2_score(y_fit_data, gaussian_y_fit_pred)
gaussian_chi2, gaussian_chi2_rid, gaussian_ndf = chi2_func(y_fit_data, gaussian_y_fit_pred, y_err, len(gaussian_params))

# --- Plot 
plt.figure(figsize=(8, 5))
plt.stairs(values=count, edges=b, color='orange', label='Data')
plt.plot(x_fit, langauss_y_fit, 'k-', lw=2, label='Langauss')
# plt.plot(x_fit, gaussian_y_fit,  'r--', lw=1, label='Gaussian')

info_text = (
    f"Langauss:\n"
    f"$x_{{mpv}}$ = {mpv:.2f} ± {empv:.2f}\n"
    f"$\\chi$ = {eta:.2f} ± {eeta:.2f}\n"
    f"$\\sigma$ = {sigma_lg:.2f} ± {esigma_lg:.2f}\n"
    f"$A$ = {A_lg:.0f} ± {eA_lg:.0f}\n"
    f"$x_{{0}}$ = {langauss_peak:.2f} ± {langauss_error_peak:.2f}\n\n"
    f"R$^2$ = {langauss_r_squared:.3f}\n"
    f"$\chi^{2}_{{rid}}$ = {langauss_chi2_rid:.2f} \n\n"

    # f"Gaussian:\n"
    # f"$\\mu$ = {mu:.2f} ± {emu:.2f}\n"
    # f"$\\sigma$ = {sigma_g:.2f} ± {esigma_g:.2f}\n"
    # f"$A$ = {A_g:.0f} ± {eA_g:.0f}\n\n"
    # f"R$^2$ = {gaussian_r_squared:.3f}\n"
    # f"$\chi^{2}_{{rid}}$ = {gaussian_chi2_rid:.2f} \n\n"
    f"bins width = {dict_info[energy]['bin width']:.0f}"
)

ax = plt.gca()
box = AnchoredText(
    info_text,
    loc='upper right',
    frameon=True
)
ax.add_artist(box)

plt.title(f"Energy {energy} GeV - Run 0{dict_info[energy]['run']} - APA 1")
plt.xlabel(r'$\langle N_{\mathrm{PE}} \rangle$ [AU]')
plt.ylabel('Counts [AU]')
plt.xlim(0, mu*2)
plt.ylim(0, max(count) * 1.2)
plt.legend(loc='upper left', ncol=1)
plt.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.5)
plt.tight_layout()

plt.text(0.7, 0.95, r'$\bf{ProtoDUNE\!-\!HD}$ Preliminary',
         transform=plt.gca().transAxes,
         fontsize=11, ha='right', va='top')

plt.savefig(f'/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output/apa1_vs_apa2/pedistribution_linearity/{energy}GeV.png')
plt.close()

# --- Saving data for next analysis 
dict_energy_linearity[energy] = {'mpv' : mpv, 'eta' : eta, 'sigma_lg' : sigma_lg, 'A_lg' : A_lg, 'mu' : mu, 'sigma_g': sigma_g, 'A_g' : A_g, 'peak' : langauss_peak,
                                    'empv' : empv, 'eeta' : eeta, 'esigma_lg' : esigma_lg, 'eA_lg' : eA_lg, 'emu' : emu, 'esigma_g': esigma_g, 'eA_g' : eA_g, 'epeak' : langauss_error_peak}

print(f'DONE !!\n\n ')

#####################################################################################
#####################################################################################

# ----------------------------------------------------------------------
# 3. Gaussian mean vs energy analysis
# ----------------------------------------------------------------------

print(f'Gaussian mean vs energy...')

energies = np.array([2, 3, 5, 7])
means = np.array([dict_energy_linearity[e]['mu'] for e in energies])
means_errors = np.array([dict_energy_linearity[e]['emu'] for e in energies])

if dict_info[1]['use mean']: # Use gaussian or langauss for 1 GeV 
    energies = np.append(energies, 1)
    means = np.append(means, dict_energy_linearity[1]['mu'])
    means_errors = np.append(means_errors, dict_energy_linearity[1]['emu'])
else:
    energies = np.append(energies, 1)
    means = np.append(means, dict_energy_linearity[1]['peak'])
    means_errors = np.append(means_errors, dict_energy_linearity[1]['epeak'])

means_errors = means_errors + 0.05 * means
energies_errors = 0.05 * energies

data_color = "#000000"
linear_color = "#25E000"
parab_color = "#001EFF"


plt.figure(figsize=(8, 5))
plt.errorbar(energies, means, xerr=energies_errors, yerr=means_errors,
             fmt='o', color=data_color, markersize=4, elinewidth=1.5, capsize=3, alpha=0.8, label = 'Data')

data = RealData(energies, means, sx=energies_errors, sy=means_errors)

# --- Linear fit 
model_1 = Model(linear_array)
odr_1 = ODR(data, model_1, beta0=curve_fit(linear, energies, means)[0])
out_1 = odr_1.run()
A_1, B_1 = out_1.beta
eA_1, eB_1 = out_1.sd_beta
ndf_1 = (len(energies) - len(out_1.beta))
chi2rid_1 = out_1.sum_square / (len(energies)-len(out_1.beta))
r2_1 = r2_score(means, linear(energies,A_1, B_1))
x_fit = np.linspace(min(energies)-0.5, max(energies)+0.5, 100)
y_fit_1 = linear(x_fit, A_1, B_1)
plt.plot(x_fit, y_fit_1, color=linear_color, linewidth=2, label=f'Fit: y = A + Bx \nA = {A_1:.2f} ± {eA_1:.2f} \nB = {B_1:.2f} ± {eB_1:.2f} \n$R^2$ = {r2_1:.3f}') #\n$\chi^{2}_{{rid}}$ = {chi2rid_1:.3f}


# --- Parabolic fit
model_2 = Model(parabola_array)
odr_2 = ODR(data, model_2, beta0=curve_fit(parabola, energies, means)[0])
out_2 = odr_2.run()
A_2, B_2, C_2 = out_2.beta
eA_2, eB_2, eC_2 = out_2.sd_beta
ndf_2 = (len(energies) - len(out_2.beta))
chi2rid_2 = out_2.sum_square / (len(energies)-len(out_2.beta))
r2_2 = r2_score(means, parabola(energies,A_2, B_2, C_2 ))
y_fit_2 = parabola(x_fit, A_2, B_2, C_2)
plt.plot(x_fit, y_fit_2, color=parab_color, linewidth=2, label=f'Fit: y = A + Bx + Cx$^{2}$\nA = {A_2:.2f} ± {eA_2:.2f} \nB = {B_2:.2f} ± {eB_2:.2f} \nC = {C_2:.2f} ± {eC_2:.2f}\n$R^2$ = {r2_2:.3f}') #$\chi^{2}_{{rid}}$ = {chi2rid_2:.3f}


plt.text(0.7, 0.95, r'$\bf{ProtoDUNE\!-\!HD}$ Preliminary', transform=plt.gca().transAxes, fontsize=11, ha='right', va='top')
plt.xlabel(r"$\langle E_{beam} \rangle$ [GeV]")
plt.ylabel(r"Gaussian mean $\langle N_{\mathrm{PE}} \rangle$")
plt.legend()
plt.title('Calorimetric linearity')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(f'/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output/apa1_vs_apa2/pedistribution_linearity/linearity.png')
plt.close()

#####################################################################################
#####################################################################################

# ----------------------------------------------------------------------
# 4. Langauss peak vs energy analysis
# ----------------------------------------------------------------------

print(f'Langauss peak vs energy...')

energies = np.array([2, 3, 5, 7])
peaks = np.array([dict_energy_linearity[e]['peak'] for e in energies])
peaks_errors = np.array([dict_energy_linearity[e]['epeak'] for e in energies])
peaks_errors = peaks_errors + 0.05 * peaks

popt, pcov = curve_fit(linear, energies, peaks, sigma=peaks_errors, absolute_sigma=True)
A, B = popt
eA, eB = np.sqrt(np.diag(pcov))

plt.figure(figsize=(8, 5))
plt.errorbar(energies, peaks, yerr=peaks_errors, fmt='o', color='green', label='Data', capsize=3, markersize=2)

# x_fit = np.linspace(min(energies)-0.5, max(energies)+0.5, 100)
# y_fit = linear(x_fit, A, B)
# plt.plot(x_fit, y_fit, 'r-', label=f'Fit: y = A + Bx\nA = {A:.2f} ± {eA:.2f} \nB = {B:.2f} ±{eB:.2f}')

plt.xlabel(r"$\langle E_{beam} \rangle$ [GeV]")
plt.ylabel(r"Langauss peak $\langle N_{\mathrm{PE}} \rangle$")
plt.legend()
plt.title('Muon peak vs energy')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f'/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output/apa1_vs_apa2/pedistribution_linearity/muon_validation.png')
plt.close()


#####################################################################################
#####################################################################################

# ----------------------------------------------------------------------
# 5. Gaussian sigma vs energy analysis
# ----------------------------------------------------------------------

print(f'Gaussian sigma vs energy...')

energies = np.array([1, 2, 3, 5, 7]) # Using gaussian result also for 1 GeV 
energies_errors = 0.05 * energies

# GAUSSIAN DATA
means = np.array([dict_energy_linearity[e]['mu'] for e in energies])
means_errors = np.array([dict_energy_linearity[e]['emu'] for e in energies])
means_errors = means_errors + 0.05 * means_errors 

sigma = np.array([dict_energy_linearity[e]['sigma_g'] for e in energies])
sigma_errors = np.array([dict_energy_linearity[e]['esigma_g'] for e in energies])
sigma_errors = sigma_errors + 0.05 * sigma_errors # keep it??

## LANGAUSS DATA
# means = np.array([dict_energy_linearity[e]['peak'] for e in energies])
# means_errors = np.array([dict_energy_linearity[e]['peak'] for e in energies])
# means_errors = means_errors + 0.05 * means_errors 

# sigma = np.array([dict_energy_linearity[e]['sigma_lg'] for e in energies])
# sigma_errors = np.array([dict_energy_linearity[e]['esigma_lg'] for e in energies])
#sigma_errors = sigma_errors + 0.05 * sigma_errors # keep it??

x = energies
ex = energies_errors
y = sigma / means
ey = y* np.sqrt((sigma_errors/sigma)**2 + (means_errors/means)**2)

data_color = "#000000"
linear_color = "#25E000"
parab_color = "#001EFF"



plt.figure(figsize=(8, 5))
plt.errorbar(x, y, xerr=ex, yerr=ey,
             fmt='o', color=data_color, markersize=4, elinewidth=1.5, capsize=3, alpha=0.8, label = 'Data')

data = RealData(x, y, sx=ex, sy=ey)

# --- Fit 
# model_1 = Model(linear_array)
# odr_1 = ODR(data, model_1, beta0=curve_fit(linear, energies, means)[0])
# out_1 = odr_1.run()
# A_1, B_1 = out_1.beta
# eA_1, eB_1 = out_1.sd_beta
# ndf_1 = (len(energies) - len(out_1.beta))
# chi2rid_1 = out_1.sum_square / (len(energies)-len(out_1.beta))
# r2_1 = r2_score(means, linear(energies,A_1, B_1))
# x_fit = np.linspace(min(energies)-0.5, max(energies)+0.5, 100)
# y_fit_1 = linear(x_fit, A_1, B_1)
# plt.plot(x_fit, y_fit_1, color=linear_color, linewidth=2, label=f'Fit: y = A + Bx \nA = {A_1:.2f} ± {eA_1:.2f} \nB = {B_1:.2f} ± {eB_1:.2f} \n$R^2$ = {r2_1:.3f}') #\n$\chi^{2}_{{rid}}$ = {chi2rid_1:.3f}


plt.text(0.7, 0.95, r'$\bf{ProtoDUNE\!-\!HD}$ Preliminary', transform=plt.gca().transAxes, fontsize=11, ha='right', va='top')
plt.xlabel(r"$\langle E_{beam} \rangle$ [GeV]")
plt.ylabel(r"Gaussian $\dfrac{\sigma}{\mu}$ [AU]")
plt.legend()
plt.title('Energy resolution')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(f'/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output/apa1_vs_apa2/pedistribution_linearity/energy_resolution.png')
plt.close()


print('\n\nDONE!!\n')