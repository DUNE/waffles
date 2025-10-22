from waffles.np04_analysis.lightyield_vs_energy.scripts.prova_fit_2_utils import *

# original data
# with open("/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output/merged_data_ALL_ALL_140925_lightyield.json", "r") as file: 
#     original_data = json.load(file)
#     print('done\n')

# analysis info
pickle_name = "integral_before_apa2_channels0-47" #"integral_before_apa2_ch35_bins10-50"
pickle_file = f'/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output/fit_analysis/integral_before/{pickle_name}.pkl'

energy_list = ['1', '2', '3', '5', '7'] 
apa = '2'
end = '109'
integral_label = 'integral_before'
fit_name = 'global_langau_fit'



# Reading input file
with open(pickle_file, 'rb') as f:
    analysis_results = pickle.load(f)

# Info contained 
ch_list = np.array(list(analysis_results[integral_label][apa][end].keys()))
bin_list = np.array(list(analysis_results[integral_label][apa][end][ch_list[0]].keys()))
analysis_list = np.array(list(analysis_results.keys()))
fit_list = np.array(list(analysis_results[integral_label][apa][end][ch_list[0]][bin_list[0]]['linear fit'].keys()))


# result = {}
# for ch in ch_list:
#     result[ch] = {}
#     for energy in energy_list:
#         result[ch][energy] = {'Nbin': bin_list, 'mpv': np.array([]), 'mpv error': np.array([]), 'chi2rid': np.array([])}
#         for N_bin in bin_list:
#             result[ch][energy]['mpv'] = np.append(result[ch][energy]['mpv'],analysis_results[integral_label][apa][end][ch][N_bin][energy][fit_name]['parameters'][0])
#             result[ch][energy]['mpv error'] = np.append(result[ch][energy]['mpv error'],analysis_results[integral_label][apa][end][ch][N_bin][energy][fit_name]['error'][0])
#             result[ch][energy]['chi2rid'] = np.append(result[ch][energy]['chi2rid'],analysis_results[integral_label][apa][end][ch][N_bin][energy][fit_name]['chi2rid'])



# # Searching for bin with best chi2
# if len(bin_list)>1:
#     for ch, ch_dic in result.items():
#         for energy, energy_dic in ch_dic.items():
#             idx_nearest = np.argmin(np.abs(energy_dic['chi2rid'] - 1))
#             n_bin_value = energy_dic['Nbin'][idx_nearest]
#             result[ch][energy]['best chi'] = {'chi2rid': energy_dic['chi2rid'][idx_nearest], 'Nbin': n_bin_value, 'mpv': energy_dic['mpv'][idx_nearest], 'mpv error': energy_dic['mpv error'][idx_nearest]}

# # Plotting chi2 as function of bin number
# if len(bin_list) > 0:
#     for ch, ch_dic in result.items():
#         fig, ax = plt.subplots(3, 2, figsize=(15, 10))
#         plt.suptitle(f"CH {ch}")
#         ax = ax.flatten()
#         i = 0
#         energy_list = list(ch_dic.keys())
#         for energy, energy_dic in ch_dic.items():
#             ax[i].scatter(energy_dic['Nbin'], energy_dic['chi2rid'])
#             ax[i].scatter(result[ch][energy]['best chi']['Nbin'], result[ch][energy]['best chi']['chi2rid'], color='red', label='best chi2rid: {:.2f} at Nbin={}'.format(result[ch][energy]['best chi']['chi2rid'], result[ch][energy]['best chi']['Nbin']))
#             ax[i].set_xlabel("# bins")
#             ax[i].set_ylabel("chi2_rid")
#             ax[i].grid(True)
#             ax[i].legend(fontsize=10)
#             if i < len(energy_list) - 1:
#                 ax[i].set_title(f"Energy: {energy} GeV")
#             i += 1
#         fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#         fig.savefig( f'/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output/fit_analysis/{integral_label}/bin_chi_study_ch{ch}.png')


# # Plotting mpv as function of bin number (senza barra di errore) + istogramma a fianco 
# from matplotlib import gridspec
# import matplotlib.cm as cm

# if len(bin_list) > 0:
#     for ch, ch_dic in result.items():
#         fig = plt.figure(figsize=(18, 10), constrained_layout=True)
#         plt.suptitle(f"CH {ch} - MPV value vs #bins (1)", fontsize=16)
#         gs = gridspec.GridSpec(3, 4, figure=fig, wspace=0.05, hspace=0.2)
#         colors = cm.Set2.colors
#         i = 0
#         energy_list = list(ch_dic.keys())

#         for energy, energy_dic in ch_dic.items():
#             row, col = divmod(i, 2)
#             ax_scatter = fig.add_subplot(gs[row, col*2:col*2+1])
#             mean_val, std_val = np.mean(energy_dic['mpv']), np.std(energy_dic['mpv'])
#             ax_scatter.fill_between(energy_dic['Nbin'], mean_val - std_val, mean_val + std_val, color='yellow', alpha=0.3, label=f"±1σ (std={std_val:.2f})")
#             ax_scatter.scatter(energy_dic['Nbin'], energy_dic['mpv'], color=colors[i % len(colors)], label=f"mpv")
#             ax_scatter.axhline(mean_val, color="red", linestyle="--", label=f"mean={mean_val:.2f}")
#             ax_scatter.scatter(result[ch][energy]['best chi']['Nbin'], result[ch][energy]['best chi']['mpv'], color='red', label='best chi2rid: {:.2f} at Nbin={}'.format(result[ch][energy]['best chi']['chi2rid'], result[ch][energy]['best chi']['Nbin']))
#             ax_scatter.set_xlabel("# bins")
#             ax_scatter.set_ylabel("mpv")
#             ax_scatter.grid(True, linestyle=":", alpha=0.7)
#             if i < len(energy_list): 
#                 ax_scatter.set_title(f"Energy: {energy} GeV")
#             ax_scatter.legend(fontsize=8)

#             ax_hist = fig.add_subplot(gs[row, col*2+1], sharey=ax_scatter)
#             counts, bins = np.histogram(energy_dic['mpv'], bins='auto'); bin_centers = 0.5 * (bins[:-1] + bins[1:])
#             ax_hist.barh(bin_centers, counts, height=(bins[1] - bins[0]), color=colors[(i+1) % len(colors)], alpha=0.7, edgecolor="black")
#             ax_hist.set_xlabel("counts")
#             ax_hist.grid(True, linestyle=":", alpha=0.7)
#             plt.setp(ax_hist.get_yticklabels(), visible=False)

#             i += 1

#         fig.savefig(f'/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output/fit_analysis/{integral_label}/bin_mpv1_study_ch{ch}.png')

# # Plotting mpv as function of bin number (with error bar) 
# if len(bin_list) > 0:
#     for ch, ch_dic in result.items():
#         fig, ax = plt.subplots(3, 2, figsize=(15, 10))
#         plt.suptitle(f"CH {ch} - MPV value vs #bins (2)", fontsize=16)
#         ax = ax.flatten()
#         i = 0
#         energy_list = list(ch_dic.keys())
#         for energy, energy_dic in ch_dic.items():
#             ax[i].errorbar(energy_dic['Nbin'], energy_dic['mpv'], yerr=energy_dic['mpv error'], fmt='o', capsize=3, label="mpv" )
#             ax[i].errorbar(result[ch][energy]['best chi']['Nbin'], result[ch][energy]['best chi']['mpv'], yerr=result[ch][energy]['best chi']['mpv error'], fmt='o', color='red', ecolor='red', capsize=3, label='best chi2rid: {:.2f} at Nbin={}'.format(result[ch][energy]['best chi']['chi2rid'], result[ch][energy]['best chi']['Nbin']))
#             mean_val, std_val = np.mean(energy_dic['mpv']), np.std(energy_dic['mpv'])
#             ax[i].fill_between(energy_dic['Nbin'], mean_val - std_val, mean_val + std_val, color='yellow', alpha=0.3, label=f"±1σ (std={std_val:.2f}")
#             ax[i].axhline(mean_val, color="black", linestyle="--", label=f"mean={mean_val:.2f}")
#             ax[i].set_xlabel("# bins")
#             ax[i].set_ylabel("mpv")
#             ax[i].grid(True)
#             if i < len(energy_list):
#                 ax[i].set_title(f"Energy: {energy} GeV")
#             ax[i].legend(fontsize=10)
#             i += 1
#         fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#         fig.savefig(f'/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output/fit_analysis/{integral_label}/bin_mpv2_study_ch{ch}.png')



# # Select which number of bins to use for each energy 
# # bin_dic_selection = {"1": 44, "2": 44, "3": 44, "4": 44, "5": 44, "7": 44} # decided by the USER
# bin_dic_selection = {chiave: result[ch][chiave]['best chi']['Nbin'] for chiave in energy_list} # Best chi

# # Linear fit with best chi2 bins
# if len(bin_list) > 0:
#     for ch, ch_dic in result.items():
#         fig, ax = plt.subplots(3, 2, figsize=(15, 10))
#         plt.suptitle(f"CH {ch} - Linear fit with best chi2 bins", fontsize=16)
#         ax = ax.flatten()
#         i = 0
#         energy_list = list(ch_dic.keys())
#         peak_list = np.array([])
#         peak_error_list = np.array([])
#         energy_list_fit = np.array([])
#         for energy, energy_dic in ch_dic.items():
#             if i < len(energy_list):
#                 N_bin = bin_dic_selection[energy]
#                 counts = analysis_results[integral_label][apa][end][ch][N_bin][energy][fit_name]['hist data']['counts']
#                 bin_edges = analysis_results[integral_label][apa][end][ch][N_bin][energy][fit_name]['hist data']['bins']
#                 bin_centers = bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
#                 ax[i].stairs(counts, bin_edges, fill=True, color=global_hist_dic['color'], alpha=global_hist_dic['alpha'], label=global_hist_dic['label'](energy, sum(analysis_results[integral_label][apa][end][ch][N_bin][energy][fit_name]['hist data']['counts']), N_bin))
                
#                 errors = np.sqrt(counts)
#                 errors[errors == 0] = 1.0
#                 ax[i].errorbar(bin_centers, counts, yerr=errors, fmt='o', markersize=4, ecolor=global_hist_dic['color errorbar'], elinewidth=1.5, capsize=0, markerfacecolor=global_hist_dic['color errorbar'], markeredgewidth=0)

                
#                 if fit_name != "global_langau_peak_fit":
#                     x_fit = np.linspace(min(bin_edges), max(bin_edges), 1000)
#                     popt = analysis_results[integral_label][apa][end][ch][N_bin][energy][fit_name]['parameters']
#                     perr = analysis_results[integral_label][apa][end][ch][N_bin][energy][fit_name]['error']
#                     chi = analysis_results[integral_label][apa][end][ch][N_bin][energy][fit_name]['chi2rid']
#                     ax[i].plot(x_fit, active_fits_global[fit_name]['fit function'](x_fit, *popt), color=active_fits_global[fit_name]['color'], lw=3, label=active_fits_global[fit_name]['label'](chi, popt, perr))
                    
#                     peak = analysis_results[integral_label][apa][end][ch][N_bin][energy]['global_langau_peak_fit']['parameters']
#                     error_peak = analysis_results[integral_label][apa][end][ch][N_bin][energy]['global_langau_peak_fit']['error']
#                     ax[i].axvline(peak, color='gold', linestyle='--', linewidth=2, label=f'Langau peak = {to_scientific_notation(peak, error_peak)}')

                    
#                     peak_list= np.append(peak_list,peak)
#                     peak_error_list= np.append(peak_error_list,error_peak)
#                     energy_list_fit= np.append(energy_list_fit,int(energy))

#                     ax[i].text(min(x_fit)*0.9, max(counts)+5, r"$\bf{ProtoDUNE\text{-}HD}$"+"\nPreliminary", fontsize=10, color="black", va="top", ha="left", zorder=10)

#             ax[i].set_xlabel("Integrated signal")
#             ax[i].set_ylabel("Counts")
#             ax[i].grid(True)
#             ax[i].legend(fontsize=9)
#             i += 1

#         #linear
#         if len(peak_list)>2:
#             popt, pcov = curve_fit(linear_fit, energy_list_fit, peak_list,sigma=peak_error_list)
#             perr = np.sqrt(np.diag(pcov))
#             chi = chi2_ridotto_xy(energy_list_fit, peak_list, peak_error_list, linear_fit, popt)
            
#             ax[i].errorbar(energy_list_fit, peak_list, yerr=peak_error_list, color='blue', fmt='o', label=fit_name.replace('_', ' ').title(), markersize=6, elinewidth=2, capsize=2)  
#             ax[i].plot(energy_list_fit, linear_fit(energy_list_fit, popt[0], popt[1]), linestyle='-', color='blue', label=f"y=ax+b (χ²={'{:.2e}'.format(chi) if chi>100 else '{:.2f}'.format(chi)})\na = {to_scientific_notation(popt[0], perr[0])} \nb = {to_scientific_notation(popt[1], perr[1])}", linewidth=3)  
#             ax[i].legend(fontsize=10, ncol=3, loc='upper left')  
#             ax[i].set_xlabel("Energy (GeV)", fontsize=15)  
#             ax[i].set_ylabel("Integrated signal", fontsize=15)  
#             ax[i].tick_params(axis='both', which='major', labelsize=12)
#             ax[i].text(energy_list_fit[-2], peak_list[2], r"$\bf{ProtoDUNE\text{-}HD}$ Preliminary", fontsize=10, color="black", va="top", ha="left", zorder=10)
 
#         fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#         fig.savefig(f'/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output/fit_analysis/{integral_label}/linear_fit_best_chi_study_ch{ch}.png')


# più fit lineari sullo stesso grafico 
my_ch_list = ['1', '7', '15', '20', '21', '22', '23', '24', '25', '30', '31', '35', '45']
#my_ch_list = ch_list  # all channels
if len(bin_list) == 1 and len(ch_list) > 1:
    N_bin = bin_list[0]
    plt.figure(figsize=(8,6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(my_ch_list)))  # palette automatica
    
    for i, ch in enumerate(my_ch_list):
        if fit_name in analysis_results[integral_label][apa][end][ch][N_bin]['linear fit']:
            dic = analysis_results[integral_label][apa][end][ch][N_bin]['linear fit'][fit_name]
            
            x = np.array(dic['x'][1:])
            y = np.array(dic['y'][1:])
            ey = np.array(dic['ey'][1:])

            popt = dic['parameters']  # es. [m, q]
            perr = dic['error']
            chi2 = dic['chi2rid']

            # scatter con errorbar
            plt.errorbar(x, y, yerr=ey, fmt='o', color=colors[i], label=f"ch {ch}, chi2rid {chi2:.2f}")

            # retta del fit
            x_fit = np.linspace(min(x), max(x), 200)
            y_fit = popt[0]*x_fit + popt[1]
            plt.plot(x_fit, y_fit, '-', color=colors[i])

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Fit multipli - bin {N_bin}")
    plt.legend(ncol=4, fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(f"/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output/fit_analysis/integral_before/ALL_channel_fit_bin{N_bin}.png", dpi=300, bbox_inches="tight")
    plt.close()


# FUNZIONA: distribuzione di MPV e MPV ERROR per ogni canale, sommando tutti i bin
# result = {}

# if bin_study:
#     for ch, ch_dic in analysis_results[integral_label][apa][end].items():
#         result[ch] = {'mpv': np.array([]), 'mpv error': np.array([])}
        
#         for N_bin, N_bin_dic in ch_dic.items():
#             mpv = N_bin_dic['linear fit']["global_langau_fit"]['parameters'][0]
#             mpv_error = N_bin_dic['linear fit']["global_langau_fit"]['error'][0]
#             result[ch]['mpv'] = np.append(result[ch]['mpv'], mpv)
#             result[ch]['mpv error'] = np.append(result[ch]['mpv error'], mpv_error)
        
#         result[ch]['mpv mean'] = np.mean(result[ch]['mpv'])
#         result[ch]['mpv std'] = np.std(result[ch]['mpv'])

#         result[ch]['mpv error mean'] = np.mean(result[ch]['mpv error'])
#         result[ch]['mpv error std'] = np.std(result[ch]['mpv error'])

#         # Istogramma MPV
#         plt.figure(figsize=(6,4))
#         plt.hist(result[ch]['mpv'], bins='auto', alpha=0.7, color='blue', edgecolor='black')
#         plt.title(f'CH {ch} | MPV Mean={result[ch]["mpv mean"]:.2f}, MPV Std={result[ch]["mpv std"]:.2f}')
#         plt.xlabel('MPV value')
#         plt.ylabel('Counts')
#         plt.grid(True, linestyle='--', alpha=0.5)
#         plt.tight_layout()
#         plt.savefig(f"/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output/fit_analysis/integral_before/hist_mpv_binning_ch{ch}.png")
#         plt.close()

#         # Istogramma MPV ERROR
#         plt.figure(figsize=(6,4))
#         plt.hist(result[ch]['mpv error'], bins='auto', alpha=0.7, color='green', edgecolor='black')
#         plt.title(f'CH {ch} | MPV ERROR Mean={result[ch]["mpv error mean"]:.2f}, MPV ERROR Std={result[ch]["mpv error std"]:.2f}')
#         plt.xlabel('MPV ERROR value')
#         plt.ylabel('Counts')
#         plt.grid(True, linestyle='--', alpha=0.5)
#         plt.tight_layout()
#         plt.savefig(f"/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output/fit_analysis/integral_before/hist_mpv_error_binning_ch{ch}.png")
#         plt.close()