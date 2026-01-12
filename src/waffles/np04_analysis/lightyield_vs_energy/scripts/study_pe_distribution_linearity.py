import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np
# from pylandau import langau
from landaupy import langauss as lg

from scipy.optimize import minimize_scalar
from scipy.optimize import curve_fit
import pickle
import pandas as pd


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
                'use mean' : True},
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
                'bin width': 2,
                'pe_separation': 150,
                'use_given_parameters' : False},
             5:{'filepath' : "/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output/apa1_vs_apa2/5GeV/apa12_study/all_apa1_mean_photoelectrons.csv",
                'run': 27367,
                'which binning' : 'width',
                'bin width': 4,
                'pe_separation': 190,
                'use_given_parameters' : False},
             7: {'filepath' : "/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output/apa1_vs_apa2/7GeV/apa12_study/all_apa1_mean_photoelectrons.csv", #"/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/scripts/Renan_scripts/pe_histogram_data/7GeV_APA1_ok.csv",
                 'run' : 27374,
                 'bin width': 6,
                'which binning' : 'width',
                'pe_separation':200,
                'use_given_parameters' : False},
            }

def linear(x, A, B):
    return A + B*x


def langauss(x, mpv, eta, sigma, A):
    return A * lg.pdf(x, mpv, eta, sigma)

# Fit function wrapper
def fit_function(x, mpv, eta, sigma, A):
    return langauss(x, mpv, eta, sigma, A)

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


def find_peak(params):
    mpv, eta, sigma, A = params

    # Prevent very small sigma from breaking the optimizer
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

    # Now propagate the errors
    squared_terms = [(partials[i] * errors[i])**2 for i in range(len(params))]
    total_error = np.sqrt(sum(squared_terms))

    return peak, total_error


#####################################################################################
#####################################################################################
#####################################################################################


dict_energy_linearity = {}

for energy in [2, 3, 5, 7]:
    print(f'\n ---------------------\nAnalysis {energy} GeV ...')

    # ----------------------------------------------------------------------
    # 0. Preparation
    # ----------------------------------------------------------------------

    df = pd.read_csv(dict_info[energy]['filepath'])

    # ----------------------------------------------------------------------
    # 1. Histogram
    # ----------------------------------------------------------------------

    select = np.asarray([s for s in df.PhotonA_y_x if s > 50])
    
    if dict_info[energy]['which binning'] == 'width':
        width = dict_info[energy]['bin width']
        bins = np.arange(0, np.max(select) + width, width)
    else: 
        bins = dict_info[energy]['bins']

    count, b = np.histogram(select, bins=bins)
    bin_centers = (b[:-1] + b[1:]) / 2
    bin_width = b[1] - b[0]

    # ----------------------------------------------------------------------
    # 2. Restrict fit region
    # ----------------------------------------------------------------------

    mask = count > 0
    x_fit_data = bin_centers[mask]
    y_fit_data = count[mask]
    y_err = np.sqrt(y_fit_data)

    # ----------------------------------------------------------------------
    # 3. and 4. Initial guesses (data-driven) and bounds
    # ----------------------------------------------------------------------

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

    # ----------------------------------------------------------------------
    # 5. Fit
    # ----------------------------------------------------------------------

    params, covariance = curve_fit(
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

    errors = np.sqrt(np.diag(covariance))

    # ----------------------------------------------------------------------
    # 6. Extract parameters
    # ----------------------------------------------------------------------

    mpv, eta, sigma_lg, A_lg, mu, sigma_g, A_g = params
    empv, eeta, esigma_lg, eA_lg, emu, esigma_g, eA_g = errors

    # ----------------------------------------------------------------------
    # 7. Evaluate fit
    # ----------------------------------------------------------------------

    x_fit = np.linspace(min(b), max(b), 10000)
    y_fit = langauss_plus_gauss(x_fit, *params)

    # Components
    y_lg = langauss(x_fit, mpv, eta, sigma_lg, A_lg)
    y_g  = gaussian(x_fit, mu, sigma_g, A_g)


    # Intersection point
    intersection_mask = (x_fit >= mpv) & (x_fit <= mu)
    x_sub  = x_fit[intersection_mask]
    y_g_sub  = y_g[intersection_mask]
    y_lg_sub = y_lg[intersection_mask]
    intersection_idx = np.argmin(np.abs(y_g_sub - y_lg_sub))
    x_intersection = x_sub[intersection_idx]

    # ----------------------------------------------------------------------
    # 8. Langauss peak position
    # ----------------------------------------------------------------------

    peak, error_peak = propagate_error(find_peak, params[:4], errors[:4], epsilon=1e-5)

    # ----------------------------------------------------------------------
    # 9. R²
    # ----------------------------------------------------------------------

    y_fit_pred = langauss_plus_gauss(bin_centers, *params)

    ss_res = np.sum((count - y_fit_pred) ** 2)
    ss_tot = np.sum((count - np.mean(count)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    # ----------------------------------------------------------------------
    # 10. Plot
    # ----------------------------------------------------------------------

    plt.figure(figsize=(8, 5))
    plt.stairs(values=count, edges=b, color='orange', label='Data')
    plt.plot(x_fit, y_fit, 'k-', lw=2, label='Langauss + Gaussian')
    plt.plot(x_fit, y_lg, 'r--', lw=1, label='Langauss')
    plt.plot(x_fit, y_g,  'b--', lw=1, label='Gaussian')
    plt.axvline(x_intersection, color='gold', linestyle='--', linewidth=1, label="Intersection")


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

        f"R$^2$ = {r_squared:.3f}\n\n"

        f"Intersection = {x_intersection:.0f}\n\n"

        f"bins width = {dict_info[energy]['bin width']:.0f}"
    )

    ax = plt.gca()
    box = AnchoredText(
        info_text,
        loc='upper right',
        frameon=True
    )
    ax.add_artist(box)

    plt.title(f"{energy} GeV/c - Run 0{dict_info[energy]['run']} - APA 1")
    plt.xlabel(r'$\langle N_{\mathrm{PE}} \rangle$')
    plt.ylabel('Counts [AU]')
    plt.xlim(0, mu*2)
    plt.ylim(0, max(count) * 1.2)
    plt.legend(loc='upper left', ncol=1)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()

    plt.savefig(f'/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output/apa1_vs_apa2/pedistribution_linearity/{energy}GeV.png')
    plt.close()

    # ----------------------------------------------------------------------
    # 11. Saving data
    # ----------------------------------------------------------------------

    dict_energy_linearity[energy] = {'mpv' : mpv, 'eta' : eta, 'sigma_lg' : sigma_lg, 'A_lg' : A_lg, 'mu' : mu, 'sigma_g': sigma_g, 'A_g' : A_g, 'peak': peak,
                                     'empv' : empv, 'eeta' : eeta, 'esigma_lg' : esigma_lg, 'eA_lg' : eA_lg, 'emu' : emu, 'esigma_g': esigma_g, 'eA_g' : eA_g, 'epeak': error_peak}

    print(f'DONE !!\n\n ')


# ----------------------------------------------------------------------
# 12. 1 GeV analysis
# ----------------------------------------------------------------------
energy = 1
df = pd.read_csv(dict_info[energy]['filepath'])

print(f'\n ---------------------\nAnalysis {energy} GeV ...')


select = np.asarray([s for s in df.PhotonA_y_x if (s > 10) and (s< 150)])

if dict_info[energy]['which binning'] == 'width':
    width = dict_info[energy]['bin width']
    bins = np.arange(0, np.max(select) + width, width)
else: 
    bins = dict_info[energy]['bins']

count, b = np.histogram(select, bins=bins)
bin_centers = (b[:-1] + b[1:]) / 2
bin_width = b[1] - b[0]

mask = count > 0
x_fit_data = bin_centers[mask]
y_fit_data = count[mask]
y_err = np.sqrt(y_fit_data)

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

# Langauss
langauss_params, langauss_covariance = curve_fit(langauss, x_fit_data,  y_fit_data, p0=langauss_initial_guess, sigma=y_err, absolute_sigma=True, bounds=(lagauss_bounds_low, langauss_bounds_high), maxfev=30000, method='trf' )
langauss_errors = np.sqrt(np.diag(langauss_covariance))
mpv, eta, sigma_lg, A_lg = langauss_params
empv, eeta, esigma_lg, eA_lg = langauss_errors

langauss_peak, langauss_error_peak = propagate_error(find_peak, langauss_params[:4], langauss_errors[:4], epsilon=1e-5)

x_fit = np.linspace(min(b), max(b), 10000)
langauss_y_fit = langauss(x_fit, *langauss_params)
langauss_y_fit_pred = langauss(bin_centers, *langauss_params)
langauss_ss_res = np.sum((count - langauss_y_fit_pred) ** 2)
langauss_ss_tot = np.sum((count - np.mean(count)) ** 2)
langauss_r_squared = 1 - langauss_ss_res / langauss_ss_tot

# Gaussian
gaussian_params, gaussian_covariance = curve_fit(gaussian, x_fit_data, y_fit_data, p0=gaussian_initial_guess, sigma=y_err, absolute_sigma=True, bounds=(gaussian_bounds_low, gaussian_bounds_high), maxfev=30000, method='trf' )
gaussian_errors = np.sqrt(np.diag(gaussian_covariance))
mu, sigma_g, A_g = gaussian_params
emu, esigma_g, eA_g = gaussian_errors

x_fit = np.linspace(min(b), max(b), 10000)
gaussian_y_fit = gaussian(x_fit, *gaussian_params)
gaussian_y_fit_pred = gaussian(bin_centers, *gaussian_params)
gaussian_ss_res = np.sum((count - gaussian_y_fit_pred) ** 2)
gaussian_ss_tot = np.sum((count - np.mean(count)) ** 2)
gaussian_r_squared = 1 - gaussian_ss_res / gaussian_ss_tot

# Plot
plt.figure(figsize=(8, 5))
plt.stairs(values=count, edges=b, color='orange', label='Data')
plt.plot(x_fit, langauss_y_fit, 'r--', lw=1, label='Langauss')
plt.plot(x_fit, gaussian_y_fit,  'b--', lw=1, label='Gaussian')


info_text = (
    f"Langauss:\n"
    f"$x_{{mpv}}$ = {mpv:.2f} ± {empv:.2f}\n"
    f"$\\chi$ = {eta:.2f} ± {eeta:.2f}\n"
    f"$\\sigma$ = {sigma_lg:.2f} ± {esigma_lg:.2f}\n"
    f"$A$ = {A_lg:.0f} ± {eA_lg:.0f}\n"
    f"$x_{{0}}$ = {langauss_peak:.2f} ± {langauss_error_peak:.2f}\n"
    f"R$^2$ = {langauss_r_squared:.3f}\n\n"

    f"Gaussian:\n"
    f"$\\mu$ = {mu:.2f} ± {emu:.2f}\n"
    f"$\\sigma$ = {sigma_g:.2f} ± {esigma_g:.2f}\n"
    f"$A$ = {A_g:.0f} ± {eA_g:.0f}\n\n"
    f"R$^2$ = {gaussian_r_squared:.3f}\n"
    f"bins width = {dict_info[energy]['bin width']:.0f}"
)

ax = plt.gca()
box = AnchoredText(
    info_text,
    loc='upper right',
    frameon=True
)
ax.add_artist(box)

plt.title(f"{energy} GeV/c - Run 0{dict_info[energy]['run']} - APA 1")
plt.xlabel(r'$\langle N_{\mathrm{PE}} \rangle$')
plt.ylabel('Counts [AU]')
plt.xlim(0, mu*2)
plt.ylim(0, max(count) * 1.2)
plt.legend(loc='upper left', ncol=1)
plt.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.5)
plt.tight_layout()

plt.savefig(f'/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output/apa1_vs_apa2/pedistribution_linearity/{energy}GeV.png')
plt.close()

dict_energy_linearity[energy] = {'mpv' : mpv, 'eta' : eta, 'sigma_lg' : sigma_lg, 'A_lg' : A_lg, 'mu' : mu, 'sigma_g': sigma_g, 'A_g' : A_g, 'peak' : langauss_peak,
                                    'empv' : empv, 'eeta' : eeta, 'esigma_lg' : esigma_lg, 'eA_lg' : eA_lg, 'emu' : emu, 'esigma_g': esigma_g, 'eA_g' : eA_g, 'epeak' : langauss_error_peak}


print(f'DONE !!\n\n ')



# ----------------------------------------------------------------------
# 12. Linear regression of Mu vs energy
# ----------------------------------------------------------------------

print(f'Linear fit...')

energies = np.array([2, 3, 5, 7])
means = np.array([dict_energy_linearity[e]['mu'] for e in energies])
means_errors = np.array([dict_energy_linearity[e]['emu'] for e in energies])


if 'use mean':
    energies = np.append(energies, 1)
    means = np.append(means, dict_energy_linearity[1]['mu'])
    means_errors = np.append(means_errors, dict_energy_linearity[1]['emu'])
else:
    energies = np.append(energies, 1)
    means = np.append(means, dict_energy_linearity[1]['peak'])
    means_errors = np.append(means_errors, dict_energy_linearity[1]['epeak'])

means_errors = means_errors + 0.05 * means

popt, pcov = curve_fit(linear, energies, means, sigma=means_errors, absolute_sigma=True)
A, B = popt
eA, eB = np.sqrt(np.diag(pcov))

plt.figure(figsize=(8, 5))
plt.errorbar(energies, means, yerr=means_errors, fmt='o', color='blue', label='Data', capsize=3, markersize=2)

x_fit = np.linspace(min(energies)-0.5, max(energies)+0.5, 100)
y_fit = linear(x_fit, A, B)
plt.plot(x_fit, y_fit, 'r-', label=f'Fit: y = A + Bx\nA = {A:.2f} ± {eA:.2f} \nB = {B:.2f} ±{eB:.2f}')

plt.xlabel("Momentum (GeV/c)")
plt.ylabel(r"Gaussian mean $\langle N_{\mathrm{PE}} \rangle$")
plt.legend()
plt.title('Calorimetric linearity')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f'/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output/apa1_vs_apa2/pedistribution_linearity/linearity.png')
plt.close()

print(f'DONE !!\n\n')



# ----------------------------------------------------------------------
# 13. Linear regression of Peak vs energy
# ----------------------------------------------------------------------

print(f'Linear fit...')

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

plt.xlabel("Momentum (GeV/c)")
plt.ylabel(r"Langauss peak $\langle N_{\mathrm{PE}} \rangle$")
plt.legend()
plt.title('Muon peak vs energy')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f'/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output/apa1_vs_apa2/pedistribution_linearity/muon_validation.png')
plt.close()

print(f'DONE !!\n\n')