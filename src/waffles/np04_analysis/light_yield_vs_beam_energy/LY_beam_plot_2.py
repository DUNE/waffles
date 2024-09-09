import plotly.graph_objects as go
#from useful_data_SELF import *
from useful_data_FULL import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import os

    

def gaussian(x, mu, sigma, A):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def linear_fit(x, A, B):
    return A * x + B


def chi_squared(observed, expected):
        return np.sum(((observed - expected)** 2) / expected)


def round_to_significant(value, error):
    error_order = -int(np.floor(np.log10(error)))
    rounded_error = round(error, error_order)
    rounded_value = round(value, error_order)
    return rounded_value, rounded_error


def to_scientific_notation(value, error):
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa_value = value / 10**exponent
    mantissa_error = error / 10**exponent
    mantissa_value, mantissa_error = round_to_significant(mantissa_value, mantissa_error)
    return mantissa_value, mantissa_error, exponent


def LY_vs_energy_plot(x,y,y_err,Apa,Endpoint,Ch):
    fig, ax = plt.subplots(figsize=(7,4))
    ax.errorbar(x, y, yerr=y_err, fmt='o', capsize=5, color='darkblue', ecolor='blue', elinewidth=2)
    
    popt, pcov = curve_fit(linear_fit, x, y, sigma=y_err, absolute_sigma=True)
    A_fit, B_fit = popt
    perr = np.sqrt(np.diag(pcov))
    A_err, B_err = perr
    
    A_mantissa, A_err_mantissa, A_exp = to_scientific_notation(A_fit, A_err)
    B_mantissa, B_err_mantissa, B_exp = to_scientific_notation(B_fit, B_err)
    
    x_fit = np.linspace(min(x)-1, max(x)+1, 100)
    y_fit = linear_fit(x_fit, A_fit, B_fit)
    ax.plot(x_fit, y_fit, linestyle ='--', color = 'deepskyblue', label=f'Linear fit: \n$A=({A_mantissa} \pm {A_err_mantissa}) \\times 10^{{{A_exp}}}$ \n$B=({B_mantissa} \pm {B_err_mantissa}) \\times 10^{{{B_exp}}}$')
    ax.legend()
    ax.set_xlabel('Energy (GeV)')
    ax.set_ylabel('Mean integrated charge')
    fig.suptitle(f'Integrated charge vs Energy beam \nApa {Apa} Endpoint {Endpoint} Channel {Ch}')
    
    output_folder = f'{working_directory}/Apa{Apa}_End{Endpoint}_Ch{Ch}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    plt.tight_layout()    
    plt.savefig(f"{output_folder}/LY_vs_Energy_Apa{Apa}_End{Endpoint}_Ch{Ch}.png", dpi=300)
    plt.close()
    
    
def filter_data(data, min_limit):
    new_data = []
    for d in data:
        if (d > min_limit):
            new_data.append(d)
    data = new_data
    
    bins = int(np.sqrt(len(data))) + 10
    counts, edges = np.histogram(data, bins=bins, range=(min(data), max(data)))
    
    mean_value = np.mean(data)
    bin_index_mean = np.digitize(mean_value, edges) - 1
    
    empty_bins_count = 0
    one_bins_count = 0
    remove_start_index = None
    for i in range(bin_index_mean + 1, len(counts)):
        if counts[i] == 0:
            empty_bins_count += 1
        elif counts[i] == 1:
            one_bins_count +=1
        else:
            empty_bins_count = 0
        
        if (empty_bins_count > 4) or ((one_bins_count> 2) and (empty_bins_count > 2)):
            remove_start_index = i + 1 
            break

    new_data = data
    if remove_start_index is not None and remove_start_index < len(edges):
        remove_start_edge = edges[remove_start_index]
        new_data = []
        for k in data:
            if k < remove_start_edge:
                new_data.append(k)

    return new_data

#############################################################################################

global working_directory 
working_directory = '/afs/cern.ch/user/a/anbalbon/waffles/src/waffles/np04_analysis/light_yield_vs_beam_energy'

                
for Ch in CH_data.keys():
    
    output_folder = f'{working_directory}/Apa{Apa}_End{Endpoint}_Ch{Ch}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    datasets_full = [subdict['int_list'] for subdict in CH_data[Ch].values() if 'int_list' in subdict]
    energies_full = [int(s) for s in list(CH_data[Ch].keys())]
    
    energies_list = [x_val for x_val, y_val in zip(energies_full, datasets_full) if y_val != 0]
    datasets = [y_val for y_val in datasets_full if y_val != 0]

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    mu_list = []
    mu_err_list = []
    sigma_list = []
    sigma_err_list = []
    A_list = []
    A_err_list = []
    energies = []

    for i, ax in enumerate(axes.flatten()[:len(datasets)]): 
        if len(datasets[i]) == 0:
            continue
        
        data = filter_data(datasets[i], min_charge_filter)
        bins = int(np.sqrt(len(data))) + 10
        start = min(data) - 100
        stop = max(data) + 100
        
        ax.hist(data, bins=bins, range=(start, stop), alpha=0.6, color='g', edgecolor='black')
        #ax.axvline(x=67000, color='r', linestyle='--', linewidth=2, label='x = 25000')
        ax.set_xlabel('Integrated charge')
        ax.set_ylabel('Counts')
        ax.set_title(f'Energy: {energies_list[i]} GeV')
        
        if len(data) < 85:
            print(f'Few data for Ch {Ch} at {energies_list[i]} GeV: --> Saving mean and std')
            mu_list.append(np.mean(data))
            mu_err_list.append(np.mean(data) / np.sqrt(len(data)))
            sigma_list.append(np.std(data))
            sigma_err_list.append(0)
            A_list.append(0)
            A_err_list.append(0)
            energies.append(energies_list[i])
            continue

        
        try: 
            counts, edges = np.histogram(data, bins=bins, range=(start, stop))
        
            bin_centers = 0.5 * (edges[1:] + edges[:-1])
            initial_guess = [np.mean(data), np.std(data), np.max(counts)]
            popt, pcov = curve_fit(gaussian, bin_centers, counts, p0=initial_guess)

            mu_fit, sigma_fit, A_fit = popt
            
            perr = np.sqrt(np.diag(pcov))
            mu_err, sigma_err, A_err = perr
            
            mean_mantissa, mean_err_mantissa, mean_exp = to_scientific_notation(mu_fit, mu_err)
            std_mantissa, std_err_mantissa, std_exp = to_scientific_notation(sigma_fit, sigma_err)
            A_mantissa, A_err_mantissa, A_exp = to_scientific_notation(A_fit, A_err)
            
            '''expected_counts = gaussian(bin_centers, *popt)
            chi2 = chi_squared(counts, expected_counts)
            dof = len(counts) - len(popt) 
            chi2_rid = chi2 / dof if dof > 0 else np.nan '''
            
            fit_string = f'Gaussian fit: \n$\\mu=({mean_mantissa} \pm {mean_err_mantissa}) \\times 10^{{{mean_exp}}}$ \n$\\sigma=({std_mantissa} \pm {std_err_mantissa}) \\times 10^{{{std_exp}}}$ \n$A=({A_mantissa} \pm {A_err_mantissa}) \\times 10^{{{A_exp}}}$' #\n $\\chi^{{2}}_{{rid}} = {chi2_rid:.3f}$'

            x_fit = np.linspace(min(data)-100, max(data)+100, 1000)  
            y_fit = gaussian(x_fit, mu_fit, sigma_fit, A_fit)

            ax.plot(x_fit, y_fit, 'r--', linewidth=2, label=fit_string)
            ax.legend(fontsize='small')

            mu_list.append(mu_fit)
            mu_err_list.append(mu_err)
            sigma_list.append(sigma_fit)
            sigma_err_list.append(sigma_err)
            A_list.append(A_fit)
            A_err_list.append(A_err)
            energies.append(energies_list[i])
        
        except RuntimeError as e:
            print(f'Fit failed for Ch {Ch} at {energies_list[i]} GeV: {e} --> Saving mean and std')
            mu_list.append(np.mean(data))
            mu_err_list.append(np.mean(data) / np.sqrt(len(data)))
            sigma_list.append(np.std(data))
            sigma_err_list.append(0)
            A_list.append(0)
            A_err_list.append(0)
            energies.append(energies_list[i])

        
    plt.suptitle(f"Charge histogram: Apa {Apa} Endpoint {Endpoint} Channel {Ch}")    
    plt.tight_layout()
    plt.savefig(f"{output_folder}/Charge_hist_Apa{Apa}_End{Endpoint}_Ch{Ch}.png")
    plt.close()
    print(f'Ch {Ch}: hist')
    
    LY_vs_energy_plot(energies,mu_list,mu_err_list,Apa,Endpoint,Ch)
    print(f'Ch {Ch}: plot\n')