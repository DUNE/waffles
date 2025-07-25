from waffles.np04_analysis.lightyield_vs_energy.scripts.prova_fit_2_utils import *

pickle_file = '/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output/fit_analysis/output.pkl'

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


result = {}
for ch in ch_list:
    result[ch] = {}
    for energy in energy_list:
        result[ch][energy] = {'Nbin': bin_list, 'mpv': np.array([]), 'mpv error': np.array([]), 'chi2rid': np.array([])}
        for N_bin in bin_list:
            result[ch][energy]['mpv'] = np.append(result[ch][energy]['mpv'],analysis_results[integral_label][apa][end][ch][N_bin][energy][fit_name]['parameters'][0])
            result[ch][energy]['mpv error'] = np.append(result[ch][energy]['mpv error'],analysis_results[integral_label][apa][end][ch][N_bin][energy][fit_name]['error'][0])
            result[ch][energy]['chi2rid'] = np.append(result[ch][energy]['chi2rid'],analysis_results[integral_label][apa][end][ch][N_bin][energy][fit_name]['chi2rid'])



# Searching for bin with best chi2
if len(bin_list)>1:
    for ch, ch_dic in result.items():
        for energy, energy_dic in ch_dic.items():
            idx_nearest = np.argmin(np.abs(energy_dic['chi2rid'] - 1))
            n_bin_value = energy_dic['Nbin'][idx_nearest]
            result[ch][energy]['best chi'] = {'chi2rid': energy_dic['chi2rid'][idx_nearest], 'Nbin': n_bin_value}

# Plotting chi2 as function of bin number
if len(bin_list)>0:
    for ch, ch_dic in result.items():
        fig, ax = plt.subplots(3, 2, figsize=(15, 10))
        plt.suptitle(f"CH {ch}")
        ax = ax.flatten()
        i = 0 #axis index
        for energy, energy_dic in ch_dic.items():
            ax[i].scatter(energy_dic['Nbin'], energy_dic['chi2rid'])
            i+=1
        fig.savefig(f'/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output/fit_analysis/{integral_label}/bin_chi_study_ch{ch}.png')



    # best_Nbin_per_channel_energy = {}
    # for ch, ch_dic in analysis_results[integral_label][apa][end].items():
    #     best_Nbin_per_channel_energy[ch] = {}
    #     for i, energy in enumerate(energy_list):
    #         min_distance = float('inf')
    #         best_Nbin = None
    #         best_chi = None
    #         for N_bin in bin_list:
    #             if N_bin not in ch_dic:
    #                 continue
    #             bin_dic = ch_dic[N_bin]
    #             if energy in bin_dic and fit_name in bin_dic[energy]:
    #                 chi_value = bin_dic[energy][fit_name]['chi2rid']
    #                 mpv_value = bin_dic[energy][fit_name]['parameters'][0]
    #                 mpv_error_value = bin_dic[energy][fit_name]['error'][0]
                    
    #                 if chi_value is None or np.isnan(chi_value):
    #                     continue

    #                 distance = abs(chi_value - 1)
    #                 if distance < min_distance:
    #                     min_distance = distance
    #                     best_Nbin = N_bin
    #                     best_chi = chi_value

                    

#             best_Nbin_per_channel_energy[ch][energy]={'bin best': best_Nbin, 'chi best':best_chi}

#     print(best_Nbin_per_channel_energy)


# mpv = bin_dic['linear fit'][fit_name]['y'][0]
# mpv_error = bin_dic['linear fit'][fit_name]['ey'][0]

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