from waffles.np04_analysis.lightyield_vs_energy.imports import *
from waffles.np04_analysis.lightyield_vs_energy.utils import *



##############


N_sigma = 0.8

output_folder = "/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output/fit_analysis"

#############################


# CHANNELS TO ANALYZE

ch_start = 35
ch_end = 35
ch_range = range(ch_start, ch_end + 1)  # Inclusive range for channels

###########################

# ANALYSIS TO USE

analysis_list = ['integral_before'] #'integral_deconv', 'integral_deconv_filtered'
# analysis_list = ['integral_before', 'integral_deconv', 'integral_deconv_filtered']

###########################

# PLOT OPTIONS

plotly_show = True

###########################

# HISTOGRAM OPTIONS

global_hist_dic = {
    'N_bin': 40,
    'label': lambda energy, data_array: f"E={energy}GeV - all data ({global_hist_dic['N_bin']} bins, {len(data_array)} counts)",
    'color': 'green',
    'alpha': 0.6 }

centered_hist_dic = {
    'N_bin': 20,
    'label': lambda energy, data_array: f"E={energy}GeV - centered data ({centered_hist_dic['N_bin']} bins, {len(data_array)} counts)",
    'color': 'orange',
    'alpha': 0.6 }


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
    'p0': lambda mu, sigma, counts: [mu, 0.3*sigma, sigma*0.6, max(counts)], #mpv (most probable value), eta (% of sigma), sigma, amplitude
    'bounds': lambda mu, sigma, counts: (
        [mu - 0.7*sigma, sigma*0.1, sigma*0.5, max(counts)*0.85], 
        [mu + 0.7*sigma, sigma*0.7, sigma*0.8, max(counts)*1.15]),
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
        [mu - 0.4*sigma, sigma*0.1, sigma*0.5, max(counts)*0.9], 
        [mu + 0.4*sigma, sigma*0.7, sigma*0.8, max(counts)*1.2]),
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

print('\nReading beam run info...')
with open(f"/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/data/beam_run_info.json", "r") as file:
    run_set = json.load(file)['A']
    print('done\n')
    

print('\nReading analysis results...')
with open(f"/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output/set_A_self_beam_end109_data250514_all_100725_lightyield.json", "r") as file:
    result_info = json.load(file)
    print('done\n')
    

NEW_result = {}
    
    
analysis_results = {}

for key in find_analysis_keys(result_info):
    analysis_results[key] = {}

for apa, apa_dic in result_info.items():
    
    for key in find_analysis_keys(result_info):
        analysis_results[key][apa] = {}
    
    print(f' ----------------------------\n\n\t APA {apa}\n')

    for end, end_dic in apa_dic.items():
        
        for key in find_analysis_keys(result_info):
            analysis_results[key][apa][end] = {}
        
        for ch, ch_dic in end_dic.items():
            if int(ch) in ch_range:
                print(f"Endpoint {end} - Channel {ch}")
                
                for integral_label, integral_info in ch_dic['Analysis'].items():
                    if integral_label in analysis_list:
                        
                        # Matplotlib plot
                        fig, ax = plt.subplots(3, 2, figsize=(15, 10))
                        plt.suptitle(f'Endpoint {end} - Channel {ch}')
                        ax = ax.flatten()
                        i = 0 #axis index
                        
                        # Plotly plot
                        rows, cols = 3, 4
                        fig_plotly = make_subplots(rows=rows, cols=cols,subplot_titles=['Energy 1 GeV (Global)','Energy 1 GeV (Central)', 'Energy 2 GeV (Global)','Energy 2 GeV (Central)', 'Energy 3 GeV (Global)','Energy 3 GeV (Central)', 'Energy 5 GeV (Global)','Energy 5 GeV (Central)', 'Energy 7 GeV (Global)','Energy 7 GeV (Central)', 'Linear fit'])
                        fig_plotly.update_layout( title_text=f'Endpoint {end} - Channel {ch}: {integral_label} analysis', title_x=0.5, barmode='overlay')
                        
                        color_dict = {fit_name: fit_props['color'] for fit_name, fit_props in {**active_fits_centered, **active_fits_global}.items()}
                        for fit_name in {**active_fits_global, **active_fits_centered}.keys():
                            NEW_result[fit_name] = {'energy': np.array([]), 'peak value': np.array([]), 'peak error': np.array([])}

                        for energy, histo_info in integral_info['LY data'].items():
                            row = (i // 2) + 1  # From 1 to 3
                            col_0 = ((i % 2) * 2) + 1  # 1 or 3 (global hist)
                            col_1 = col_0 + 1          # 2 or 4 (centered hist)
                            
                            if histo_info:
                                # All data
                                data_array_original = -np.array(histo_info['histogram data']) # Working with positive values
                                mu = np.mean(data_array_original) #global mean
                                sigma = np.std(data_array_original) #global std
                                
                                # excluding outliers - Global
                                mask_global = (data_array_original >= mu - 3*sigma) & (data_array_original <= mu + 3*sigma)
                                data_array_global = data_array_original[mask_global]
                                
                                # Centered data
                                mask_centered = (data_array_global >= mu - N_sigma*sigma) & (data_array_global <= mu + N_sigma*sigma)
                                data_array_centered = data_array_global.copy()[mask_centered]
                                mu_centered = np.mean(data_array_centered) #centered mean
                                sigma_centered = np.std(data_array_centered) #centered std
                                                            
                                for hist_type, hist_dic, active_analysis, data_array, col in zip(['global', 'centered'], [global_hist_dic, centered_hist_dic], [active_fits_global, active_fits_centered], [data_array_global, data_array_centered], [col_0, col_1]):
                                    if hist_type == 'centered':
                                        continue
                                    
                                    # if hist_type == 'centered': 
                                    #     counts, bins = np.histogram(data_array, bins=hist_dic['N_bin'], density=False)
                                    # else: 
                                    counts, bins, _ = ax[i].hist(data_array, bins=hist_dic['N_bin'], range=(np.min(data_array), np.max(data_array)), density=False, alpha=hist_dic['alpha'], color=hist_dic['color'], label=hist_dic['label'](energy, data_array))
                                    
                                    bin_centers = 0.5 * (bins[:-1] + bins[1:])
                                    ax[i].set_xlabel("Integrated Charge")
                                    ax[i].set_ylabel("Counts")
                                    ax[i].set_title(f"Energy: {energy} GeV")
                                    
                                    fig_plotly.add_trace(go.Histogram(x=data_array, xbins=dict(start=np.min(data_array), end=np.max(data_array), size=(np.max(data_array)-np.min(data_array))/hist_dic['N_bin']), nbinsx=hist_dic['N_bin'], histnorm='', marker_color=hist_dic['color'], opacity=hist_dic['alpha'], name=hist_dic['label'](energy, data_array).replace('\n','<br>')), row=row, col=col)
                                    fig_plotly.update_xaxes(title_text="Integrated Charge", range=[mu - 3*sigma, mu + 3*sigma], row=row, col=col)
                                    fig_plotly.update_yaxes(title_text="Counts", row=row, col=col)

                                    #Selected region for central histogram
                                    if hist_type == 'global':
                                        ax[i].axvline(mu - N_sigma*sigma, color='black', linestyle='--', linewidth=1, label=f"μ-{N_sigma}σ")
                                        ax[i].axvline(mu + N_sigma*sigma, color='black', linestyle='--', linewidth=1, label=f"μ+{N_sigma}σ")
                                        
                                        fig_plotly.add_vline( x= mu - N_sigma*sigma,line=dict(color='black', dash='dash'), annotation_text=f"μ-{N_sigma}σ", annotation_position="top left",  row=row, col=col)
                                        fig_plotly.add_vline( x= mu + N_sigma*sigma,line=dict(color='black', dash='dash'), annotation_text=f"μ+{N_sigma}σ", annotation_position="top right",  row=row, col=col)

                                    #Global mean value
                                    if global_mean:
                                        label = f"Global mean: {to_scientific_notation(mu,sigma)}"
                                        color = 'blue'
                                        ax[i].axvline(mu, color=color, linestyle='--', linewidth=1, label=label)
                                        fig_plotly.add_trace(go.Scatter(x=[mu, mu], y=[0, max(counts)*1.2], mode='lines', name=label, line=dict(color=color, dash='dash'), showlegend = (hist_type == 'global')), row=row, col=col)

                                    
                                    # Central mean value
                                    if centered_mean:
                                        label = f"Centered mean: {to_scientific_notation(mu_centered,sigma_centered)}"
                                        color = 'red'
                                        ax[i].axvline(mu_centered, color=color, linestyle='--', linewidth=1, label=label)
                                        fig_plotly.add_trace(go.Scatter(x=[mu_centered, mu_centered], y=[0, max(counts)*1.2], mode='lines', name=label, line=dict(color=color, dash='dash'), showlegend = (hist_type == 'global')), row=row, col=col)
                                    
                                    
                                    # CHARGE DISTRIBUTION FITTING
                                    for fit_name, fit_dic in active_analysis.items():
                                        try: 
                                            bin_centers, counts,hist_sigma_par, hist_absolute_sigma = prepare_histogram_curve_fit_inputs(counts, bin_centers, histogram_bin_zero_extremes=False, histogram_bin_error=False)
                                            popt, pcov = curve_fit(fit_dic['fit function'], bin_centers, counts, sigma = hist_sigma_par, absolute_sigma=hist_absolute_sigma, p0=fit_dic['p0'](mu, sigma, counts),bounds = fit_dic['bounds'](mu, sigma, counts))
                                            # popt, pcov = curve_fit(fit_dic['fit function'], bin_centers, counts, p0=fit_dic['p0'](mu, sigma, counts),bounds = fit_dic['bounds'](mu, sigma, counts)) # original 
                                            
                                            perr = np.sqrt(np.diag(pcov))
                                            chi = chi2_ridotto_distribution(counts, bin_centers, fit_dic['fit function'], popt)
                                            x_fit = np.linspace(min(data_array), max(data_array), 1000)
                                            
                                            if fit_name != "global_langau_peak_fit":
                                                ax[i].plot(x_fit, fit_dic['fit function'](x_fit, *popt), color=fit_dic['color'], lw=1, label=fit_dic['label'](chi, popt, perr))
                                                fig_plotly.add_trace(go.Scatter(x=x_fit, y=fit_dic['fit function'](x_fit, *popt), mode='lines', name=fit_dic['label'](chi, popt, perr).replace('\n','<br>'), line=dict(color=fit_dic['color']), showlegend=True ), row=row, col=col)

                                        except Exception as e:
                                            popt = np.zeros(len(fit_dic['p0'](mu, sigma, counts)))
                                            perr = np.zeros(len(fit_dic['p0'](mu, sigma, counts)))
                                            print(f"Error fitting {fit_name}: {e}")
                                        
                                        if (popt[0] != 0) and (fit_name != "global_langau_peak_fit"):
                                            peak, error_peak = propagate_error(find_peak, popt, perr)
                                            label = f'Langau peak = {to_scientific_notation(peak, error_peak)}'
                                            color = 'gold'
                                            ax[i].axvline(peak, color=color, linestyle='--', linewidth=1, label=label)
                                            fig_plotly.add_trace(go.Scatter(x=[peak, peak], y=[0, max(counts)*1.2], mode='lines', name=label, line=dict(color=color, dash='dash')), row=row, col=col)
                                            #fig_plotly.add_trace(go.Scatter(x=[popt[0], popt[0]], y=[0, max(counts)*1.2], mode='lines', name='mpv', line=dict(color='orange', dash='dash')), row=row, col=col)
                                                
                                            NEW_result[fit_name]['energy'] = np.append(NEW_result[fit_name]['energy'],int(energy))
                                            NEW_result[fit_name]['peak value'] = np.append(NEW_result[fit_name]['peak value'],peak)
                                            NEW_result[fit_name]['peak error'] = np.append(NEW_result[fit_name]['peak error'],error_peak) 
                                        
                                        else:   
                                            NEW_result[fit_name]['energy'] = np.append(NEW_result[fit_name]['energy'],int(energy))
                                            NEW_result[fit_name]['peak value'] = np.append(NEW_result[fit_name]['peak value'],popt[fit_dic['linear plot']])
                                            NEW_result[fit_name]['peak error'] = np.append(NEW_result[fit_name]['peak error'],perr[fit_dic['linear plot']]) 
                                             
                        
                            ax[i].legend(fontsize=5, ncol=2)
                            i+=1
                        
                        # LINEAR FIT
                        for fit_name, fit_result in NEW_result.items():
                            if len(fit_result['energy']) > 2:
                                popt, pcov = curve_fit(linear_fit, fit_result['energy'], fit_result['peak value'],sigma=fit_result['peak error'])
                                perr = np.sqrt(np.diag(pcov))
                                chi = chi2_ridotto_xy(fit_result['energy'], fit_result['peak value'], fit_result['peak error'],linear_fit, popt)
                                
                                label = f"y=ax+b (χ²={'{:.2e}'.format(chi) if chi>100 else '{:.2f}'.format(chi)})\na = {to_scientific_notation(popt[0], perr[0])} \nb = {to_scientific_notation(popt[1], perr[1])}"
                                
                                ax[i].errorbar(fit_result['energy'], fit_result['peak value'], yerr=fit_result['peak error'], color = color_dict[fit_name], fmt='o', label=fit_name.replace('_', ' ').title())
                                ax[i].plot(fit_result['energy'], linear_fit(fit_result['energy'], popt[0], popt[1]),linestyle='-', color= color_dict[fit_name], label=label) 
                                ax[i].legend(fontsize=4, ncol=3, loc='upper left')
                                
                                fig_plotly.add_trace( go.Scatter(x=fit_result['energy'], y=fit_result['peak value'], error_y=dict(type='data', array=fit_result['peak error'], visible=True), mode='markers', name=fit_name.replace('_', ' ').title(), marker=dict(color=color_dict[fit_name]), showlegend=True), row=3, col=3)
                                fig_plotly.add_trace( go.Scatter(x=fit_result['energy'], y=linear_fit(fit_result['energy'], popt[0], popt[1]), mode='lines', name=label.replace('\n','<br>'), line=dict(color=color_dict[fit_name]), showlegend=True ), row=3, col=3)
                        
                        fig_plotly.update_layout(legend=dict(orientation="v", y=1, x=1, xanchor="left", yanchor="top", font=dict(size=8)), margin=dict(r=200))
                        fig_plotly.write_image(f"{output_folder}/{integral_label}/{integral_label}_apa{apa}_end{end}_ch{ch}.png", format='png', width=2000, height=1300)
                        if plotly_show:
                            fig_plotly.show()
                        
                        plt.tight_layout()
                        analysis_results[integral_label][apa][end][ch] = fig

                            
for key in find_analysis_keys(result_info):
    for apa, apa_dic in analysis_results[key].items():
        APA_pdf_file = PdfPages(f"{output_folder}/{key}/{key}_apa{apa}_prova_fit_NEW.pdf")
        for end, end_dic in apa_dic.items():
            for ch, fig in end_dic.items():
                APA_pdf_file.savefig(fig)
                plt.close(fig)
        APA_pdf_file.close()