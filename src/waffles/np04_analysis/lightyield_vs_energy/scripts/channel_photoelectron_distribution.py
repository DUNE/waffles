# Using the output of channel_photoelectron_distribution_json.ipynb --> events connected to a good trigger
# study the phtotoelectron distributon channel by channel 
# then linearity 


from waffles.np04_analysis.lightyield_vs_energy.scripts.utils import *

plt.rcParams.update({
    'font.size': 7,
    'axes.titlesize': 13,
    'axes.labelsize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 7,
    'figure.dpi': 300,  
})


def main(all_channel_data, input_folder, output_folder):
    bin_enrgy_dic = {1: 5, 2: 5, 3: 8, 5: 10, 7: 15}  # Photoelectron width of the histogram at different energies
    # bin_enrgy_dic = {1: 10, 2: 10, 3: 10, 5: 10, 7: 10}
    analysis_result_dict = {}
    cols = ['APA', 'Endpoint', 'Channel', 'Energies', 'Gaussian mu', 'Gaussian emu', 'Gaussian params', 'Gaussian eparams', 'Gaussian chi2rid', 'Langauss peak', 'Langauss epeak', 'Langauss params', 'Langauss eparams', 'Langauss chi2rid']
    analysis_result_df = pd.DataFrame(columns=cols)
    df_rows = []

    conteggi_l = 0
    conteggi_g = 0

    for apa in [1]: 
        analysis_result_dict[apa] = {}
        APA_pdf_file = PdfPages(f"{output_folder}/APA{apa}_PE_study.pdf")
        for endpoint in which_endpoints_in_the_APA(apa): #["104"]: #
            analysis_result_dict[apa][endpoint] = {}
            for channel in sorted(list(channel_vendor_map[endpoint].keys())): # channel_dict in channel_vendor_map[endpoint].items(): ["1","2","3","4","5","6"]: #
                print(f"Processing APA {apa} - Endpoint {endpoint} - Channel {channel}")
                analysis_result_dict[apa][endpoint][channel] = {}

                fig, ax = plt.subplots(4, 2, figsize=(12, 15))
                plt.suptitle(f'Endpoint {endpoint} - Channel {channel}', fontsize =16)
                ax = ax.flatten()
                i = 0


                # ----------------------------------------------------------------------
                # 2. 1, 2, 3, 5, 7 GeV analysis with Languss (ey+ex) and Gaussian (ey)
                # ----------------------------------------------------------------------

                for energy in all_channel_data.keys():
                    # --- All data from output of channel_photoelectron_distribution_json.ipynb
                    distribution_data = np.array(all_channel_data[energy][str(endpoint)][str(channel)]['n_pe'])

                    if len(distribution_data) > 150:

                        # --- Raw distribution info
                        data_mu = np.mean(distribution_data)
                        data_sigma = np.std(distribution_data)

                        # --- First selection
                        n_data_sigma = 6
                        mask = (distribution_data <= data_mu + n_data_sigma * data_sigma) # mask = distribution_data >= 0 # to include all data
                        filtered_hist_data = distribution_data[mask]
                        filtered_mu = np.mean(filtered_hist_data)
                        filtered_sigma = np.std(filtered_hist_data)

                        # --- Fixed bin width 
                        width = bin_enrgy_dic[energy]
                        bins = np.arange(np.min(filtered_hist_data) - width, np.max(filtered_hist_data) + width, width)

                        # --- Histogram 
                        count, b = np.histogram(filtered_hist_data, bins=bins)
                        bin_centers = (b[:-1] + b[1:]) / 2
                        bin_width = b[1] - b[0]
                        count_error = np.sqrt(count)
                        bin_center_max = bin_centers[np.argmax(count)]

                        # --- Plot 
                        ax[i].stairs(values=count, edges=b, color='orange', label=f'Data ({len(filtered_hist_data)})')
                        #ax[i].axvline(data_mu, color='green', linestyle='--', linewidth=1.5, label=f"Data mu: {data_mu:.0f}")

                        # --- Second selection
                        n_fit_sigma = 2
                        fit_range_min = min(distribution_data) #bin_center_max - n_fit_sigma * filtered_sigma
                        fit_range_max = bin_center_max + n_fit_sigma * filtered_sigma
                        mask = (bin_centers >= fit_range_min) & (bin_centers <= fit_range_max) & (count > 0) # mask = count > 0
                        filtered_x_fit = bin_centers[mask]
                        filtered_y_fit = count[mask]
                        filtered_y_error = count_error[mask]
                        filtered_fit_mu = np.mean(filtered_x_fit)
                        filtered_fit_sigma = np.std(filtered_x_fit)
                        filtered_fit_max = max(filtered_y_fit)
                        #ax[i].axvline((filtered_mu), color='blue', linestyle='--', linewidth=1.5, label=f"Fit mean ({n_fit_sigma} std): {filtered_fit_mu:.0f}")

                        # --- Plot fit range
                        ax[i].axvline((fit_range_min), color='gold', linestyle='--', linewidth=1.5, label=f"Fit limits ")
                        ax[i].axvline((fit_range_max), color='gold', linestyle='--', linewidth=1.5)


                        #################
                        # --- Gaussian fit 
                        
                        # Initial values and bounds
                        n_filtered_sigma = 0.5
                        g_initial_guess = [
                            bin_center_max, 
                            filtered_fit_sigma, 
                            filtered_fit_max
                        ]

                        g_bounds_low = [
                            bin_center_max-n_filtered_sigma*filtered_fit_sigma,
                            filtered_fit_sigma*0.7,
                            filtered_fit_max*0.7
                        ]

                        g_bounds_high = [
                            bin_center_max+n_filtered_sigma*filtered_fit_sigma,
                            filtered_fit_sigma*1.2,
                            filtered_fit_max*1.2
                        ]

                        # First fit with only y_err
                        g_params_initial, g_covariance_initial = curve_fit(gaussian, filtered_x_fit, filtered_y_fit, p0=g_initial_guess, sigma=filtered_y_error, absolute_sigma=True, bounds=(g_bounds_low, g_bounds_high), maxfev=30000, method='trf')

                        # x_err propagation
                        epsilon = 1e-5
                        dfdx = (gaussian(filtered_x_fit + epsilon, *g_params_initial)- gaussian(filtered_x_fit - epsilon, *g_params_initial)) / (2 * epsilon)
                        sigma_x = 0.5 * bin_width
                        sigma_tot = np.sqrt(filtered_y_error**2 + (dfdx * sigma_x)**2)

                        # Second fit with y_tot
                        g_params, g_covariance = curve_fit(
                        gaussian,
                        filtered_x_fit,
                        filtered_y_fit,
                        p0=g_params_initial,
                        sigma=sigma_tot,
                        absolute_sigma=True,
                        bounds=(0.85*g_params_initial, 1.15*g_params_initial),
                        maxfev=30000,
                        method='trf'
                        )

                        # g_params = g_params_initial
                        # g_covariance = g_covariance_initial

                        # Parameters and error extrapolation
                        g_mu, g_sigma, g_A = g_params
                        g_emu, g_esigma, g_eA  = np.sqrt(np.diag(g_covariance))

                        # Fit info for plot + Chi2 and R2
                        x_fit = np.linspace(min(bin_centers), max(bin_centers), 10000)
                        g_y_fit = gaussian(x_fit, *g_params)
                        ax[i].plot(x_fit, g_y_fit, 'b-', lw=2, label='Gaussian')
                        ax[i].axvline(g_mu, color='deepskyblue', linestyle='--', linewidth=1.5, label=f"Gaussian $\mu$")

                        g_chi2, g_chi2_rid, g_ndf = chi2_func(filtered_y_fit, gaussian(filtered_x_fit, *g_params), filtered_y_error, len(g_params))
                        g_r_squared = r2_score(filtered_y_fit, gaussian(filtered_x_fit, *g_params))
                        # g_r_squared = 1 - np.sum((count - gaussian(bin_centers, *g_params)) ** 2) / np.sum((count - np.mean(count)) ** 2)


                        #################
                        # --- Langauss fit

                        # Initial values and bounds
                        l_initial_guess = [
                            bin_centers[np.argmax(count)], # mpv
                            g_sigma/3, # eta
                            g_sigma,   # sigma
                            np.sum(count) * bin_width *1.5 # A
                        ]
                        
                        l_bounds_low = [
                            b[0] + bin_width,     # mpv
                            g_sigma/3-10,       # eta
                            g_sigma-10,       # sigma
                            np.sum(count) * bin_width * 1   # A
                        ]
                        l_bounds_high = [
                            b[-1] - bin_width,# mpv
                            g_sigma/3+10, # eta
                            g_sigma+5,   # sigma
                            np.sum(count) * bin_width * 2 # A
                        ]
                        
  
                        # First fit with only y_err
                        l_params_initial, l_covariance_initial = curve_fit(langauss, filtered_x_fit, filtered_y_fit, p0=l_initial_guess, sigma=filtered_y_error, absolute_sigma=True, bounds=(l_bounds_low, l_bounds_high), maxfev=30000, method='trf')

                        # x_err propagation
                        epsilon = 1e-5
                        dfdx = (langauss(filtered_x_fit + epsilon, *l_params_initial)- langauss(filtered_x_fit - epsilon, *l_params_initial)) / (2 * epsilon)
                        sigma_x = 0.5 * bin_width
                        sigma_tot = np.sqrt(filtered_y_error**2 + (dfdx * sigma_x)**2)

                        # Second fit with y_tot
                        l_params, l_covariance = curve_fit(
                        langauss,
                        filtered_x_fit,
                        filtered_y_fit,
                        p0=l_params_initial,
                        sigma=sigma_tot,
                        absolute_sigma=True,
                        bounds=(0.85*l_params_initial, 1.15*l_params_initial),
                        maxfev=30000,
                        method='trf'
                        )

                        # l_params = l_params_initial
                        # l_covariance = l_covariance_initial

                        # Parameters and error extrapolation
                        l_mpv, l_eta, l_sigma, l_A = l_params
                        l_empv, l_eeta, l_esigma, l_eA  = np.sqrt(np.diag(l_covariance))

                        # Fit info for plot + Chi2 and R2
                        x_fit = np.linspace(min(bin_centers), max(bin_centers), 10000)
                        l_y_fit = langauss(x_fit, *l_params)
                        ax[i].plot(x_fit, l_y_fit, 'm-', lw=2, label='Langauss')
                        
                        l_chi2, l_chi2_rid, l_ndf = chi2_func(filtered_y_fit, langauss(filtered_x_fit, *l_params), sigma_tot, len(l_params))
                        l_r_squared = r2_score(filtered_y_fit, langauss(filtered_x_fit, *l_params))
                        #l_r_squared = 1 - np.sum((count - langauss(bin_centers, *l_params)) ** 2) / np.sum((count - np.mean(count)) ** 2)

                        # Langauss peak
                        l_peak, l_epeak = propagate_error(find_peak, l_params, np.sqrt(np.diag(l_covariance)), epsilon=1e-5)
                        ax[i].axvline(l_peak, color='fuchsia', linestyle='--', linewidth=1.5, label=f"Langauss peak $x_0$")


                        # Plot info
                        info_text = (
                            f"Gaussian:\n"
                            f"$\\mu$ = {g_mu:.2f} ± {g_emu:.2f}\n"
                            f"$\\sigma$ = {g_sigma:.2f} ± {g_esigma:.2f}\n"
                            f"$A$ = {g_A:.0f} ± {g_eA:.0f}\n"
                            f"$\chi^2$ = {g_chi2_rid:.3f}\n"
                            f"R$^2$ = {g_r_squared:.3f}\n\n"

                            f"Langauss:\n"
                            f"$x_{{mpv}}$ = {l_mpv:.2f} ± {l_empv:.2f}\n"
                            f"$\\chi$ = {l_eta:.2f} ± {l_eeta:.2f}\n"
                            f"$\\sigma$ = {l_sigma:.2f} ± {l_esigma:.2f}\n"
                            f"$A$ = {l_A:.0f} ± {l_eA:.0f}\n"
                            f"$x_{{0}}$ = {l_peak:.2f} ± {l_epeak:.2f}\n"
                            f"$\chi^2$ = {l_chi2_rid:.3f}\n"
                            f"R$^2$ = {l_r_squared:.3f}\n\n"

                            f"bins width = {bin_width:.0f}"
                        )

                        box = AnchoredText(
                            info_text,
                            loc='upper right',
                            frameon=True
                        )
                        ax[i].add_artist(box)

                        ax[i].legend(loc='upper center') 

                        # Saving results
                        analysis_result_dict[apa][endpoint][channel][energy] = {} 
                        analysis_result_dict[apa][endpoint][channel][energy]['gaussian'] = {}   
                        analysis_result_dict[apa][endpoint][channel][energy]['gaussian']['params'] = g_params.tolist()
                        analysis_result_dict[apa][endpoint][channel][energy]['gaussian']['eparams'] = np.sqrt(np.diag(g_covariance)).tolist()
                        analysis_result_dict[apa][endpoint][channel][energy]['gaussian']['r_squared'] = g_r_squared
                        analysis_result_dict[apa][endpoint][channel][energy]['gaussian']['bin_width'] = bin_width
                        analysis_result_dict[apa][endpoint][channel][energy]['gaussian']['mu'] = g_mu
                        analysis_result_dict[apa][endpoint][channel][energy]['gaussian']['emu'] = g_emu
                        analysis_result_dict[apa][endpoint][channel][energy]['gaussian']['sigma'] = g_sigma
                        analysis_result_dict[apa][endpoint][channel][energy]['gaussian']['esigma'] = g_esigma
                        analysis_result_dict[apa][endpoint][channel][energy]['gaussian']['A'] = g_A
                        analysis_result_dict[apa][endpoint][channel][energy]['gaussian']['eA'] = g_eA


                        analysis_result_dict[apa][endpoint][channel][energy]['langauss'] = {}   
                        analysis_result_dict[apa][endpoint][channel][energy]['langauss']['params'] = l_params.tolist()
                        analysis_result_dict[apa][endpoint][channel][energy]['langauss']['eparams'] = np.sqrt(np.diag(l_covariance)).tolist()
                        analysis_result_dict[apa][endpoint][channel][energy]['langauss']['r_squared'] = l_r_squared
                        analysis_result_dict[apa][endpoint][channel][energy]['langauss']['bin_width'] = bin_width
                        analysis_result_dict[apa][endpoint][channel][energy]['langauss']['mpv'] = l_mpv
                        analysis_result_dict[apa][endpoint][channel][energy]['langauss']['empv'] = l_empv
                        analysis_result_dict[apa][endpoint][channel][energy]['langauss']['eta'] = l_eta
                        analysis_result_dict[apa][endpoint][channel][energy]['langauss']['eeta'] = l_eeta
                        analysis_result_dict[apa][endpoint][channel][energy]['langauss']['sigma'] = l_sigma
                        analysis_result_dict[apa][endpoint][channel][energy]['langauss']['esigma'] = l_esigma
                        analysis_result_dict[apa][endpoint][channel][energy]['langauss']['A'] = l_A
                        analysis_result_dict[apa][endpoint][channel][energy]['langauss']['eA'] = l_eA
                        analysis_result_dict[apa][endpoint][channel][energy]['langauss']['peak'] = l_peak
                        analysis_result_dict[apa][endpoint][channel][energy]['langauss']['epeak'] = l_epeak

                        if l_r_squared > g_r_squared:
                            conteggi_l +=1
                        else:
                            conteggi_g +=1

                        
                    ax[i].set_title(f"{energy} GeV/c")
                    ax[i].set_xlabel(r'$N_{\mathrm{PE}}$')
                    ax[i].set_ylabel('Counts [AU]')
                    # ax[i].set_.xlim(0, mu*2)
                    # ax[i].set_.ylim(0, max(count) * 1.2)
                    ax[i].grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.5)
                    
                    i+=1
    

    
                energies = np.asarray(list(analysis_result_dict[apa][endpoint][channel].keys()))
                energies_errors = 0.05 * energies

                if len(energies) > 2:

                    # ----------------------------------------------------------------------
                    # 2. Gaussian mean or Lnaguss peak vs energy analysis
                    # ----------------------------------------------------------------------


                    # --- GAUSSIAN ANALYSIS 

                    g_means = np.array([analysis_result_dict[apa][endpoint][channel][e]['gaussian']['mu'] for e in energies ])
                    g_means_errors = np.array([analysis_result_dict[apa][endpoint][channel][e]['gaussian']['emu'] for e in energies])
                    g_means_errors = g_means_errors + 0.05 * g_means

                    ax[i].errorbar(energies, g_means, xerr = energies_errors, yerr=g_means_errors, fmt='o', color='deepskyblue', label='Gaussian $\mu$', capsize=3, markersize=2)

                    g_data = RealData(energies, g_means, sx=energies_errors, sy=g_means_errors)
                    g_model = Model(linear_array)
                    g_odr = ODR(g_data, g_model, beta0=curve_fit(linear, energies, g_means, sigma=g_means_errors, absolute_sigma=True)[0])
                    g_out = g_odr.run()
                    g_popt =  g_out.beta
                    g_popt_error = g_out.sd_beta

                    # g_popt, g_pcov = curve_fit(linear, energies, g_means, sigma=g_means_errors, absolute_sigma=True)
                    # g_popt_error = np.sqrt(np.diag(g_pcov))

                    g_A, g_B = g_popt
                    g_eA, g_eB = g_popt_error

                    g_r_squared = r2_score(g_means, linear(energies, *g_popt))
                    g_chi2rid = g_out.sum_square / (len(energies)-len(g_popt))
                    #g_chi2rid = (np.sum(((g_means - linear(energies, *g_popt)) / g_means_errors) ** 2)) / (len(energies) - 2)

                    x_fit = np.linspace(min(energies)-0.5, max(energies)+0.5, 100)
                    y_fit = linear(x_fit, g_A, g_B)
                    ax[i].plot(x_fit, y_fit, 'b-', label=f'Gaussian fit: y = A + Bx\nA = {g_A:.2f} ± {g_eA:.2f} \nB = {g_B:.2f} ± {g_eB:.2f} \n$R^2$ = {g_r_squared:.2f}') #\n$\\chi^2_{{rid}}$ = {g_chi2rid:.2f} 

                    # --- LANGAUSS ANALYSIS 
                    l_peaks = np.array([analysis_result_dict[apa][endpoint][channel][e]['langauss']['peak'] for e in energies])
                    l_peak_errors = np.array([analysis_result_dict[apa][endpoint][channel][e]['langauss']['epeak'] for e in energies])
                    l_peak_errors = l_peak_errors + 0.05 * l_peaks

                    ax[i].errorbar(energies, l_peaks, xerr = energies_errors, yerr=l_peak_errors, fmt='o', color='fuchsia', label='Langauss peak $x_0$', capsize=3, markersize=2)

                    l_data = RealData(energies, l_peaks, sx=energies_errors, sy=l_peak_errors)
                    l_model = Model(linear_array)
                    l_odr = ODR(l_data, l_model, beta0=curve_fit(linear, energies, l_peaks, sigma=l_peak_errors, absolute_sigma=True)[0])
                    l_out = l_odr.run()
                    l_popt =  l_out.beta
                    l_popt_error = l_out.sd_beta

                    #l_popt, l_pcov = curve_fit(linear, energies, l_peaks, sigma=l_peak_errors, absolute_sigma=True)
                    # l_popt_error = np.sqrt(np.diag(l_pcov))

                    l_A, l_B = l_popt
                    l_eA, l_eB = l_popt_error

                    l_r_squared = r2_score(l_peaks, linear(energies, *l_popt))
                    l_chi2rid = l_out.sum_square / (len(energies)-len(l_popt))
                    # l_chi2rid = (np.sum(((l_peaks - linear(energies, *l_popt)) / l_peak_errors) ** 2)) / (len(energies) - 2)

                    x_fit = np.linspace(min(energies)-0.5, max(energies)+0.5, 100)
                    y_fit = linear(x_fit, l_A, l_B)
                    ax[i].plot(x_fit, y_fit, 'm-', label=f'Langauss fit: y = A + Bx\nA = {l_A:.2f} ± {l_eA:.2f} \nB = {l_B:.2f} ± {l_eB:.2f} \n$R^2$ = {l_r_squared:.2f}') #\n$\\chi^2_{{rid}}$ = {l_chi2rid:.2f}
                    

                    ax[i].set_xlabel(r"$\langle E_{beam} \rangle$ [GeV]")
                    ax[i].set_ylabel(r'$\langle N_{\mathrm{PE}} \rangle$ [AU]')
                    ax[i].legend()
                    ax[i].set_title('Calorimetric linearity')
                    ax[i].grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.5)

                    # Saving results
                    analysis_result_dict[apa][endpoint][channel]['linearity'] = {}
                    analysis_result_dict[apa][endpoint][channel]['linearity']['gaussian'] = {}
                    analysis_result_dict[apa][endpoint][channel]['linearity']['gaussian']['params'] = g_popt.tolist()
                    analysis_result_dict[apa][endpoint][channel]['linearity']['gaussian']['eparams'] = g_popt_error.tolist()
                    analysis_result_dict[apa][endpoint][channel]['linearity']['gaussian']['chi2rid'] = g_chi2rid
                    analysis_result_dict[apa][endpoint][channel]['linearity']['gaussian']['A'] = g_A
                    analysis_result_dict[apa][endpoint][channel]['linearity']['gaussian']['eA'] = g_eA
                    analysis_result_dict[apa][endpoint][channel]['linearity']['gaussian']['B'] = g_B
                    analysis_result_dict[apa][endpoint][channel]['linearity']['gaussian']['eB'] = g_eB


                    analysis_result_dict[apa][endpoint][channel]['linearity']['langauss'] = {}
                    analysis_result_dict[apa][endpoint][channel]['linearity']['langauss']['params'] = l_popt.tolist()
                    analysis_result_dict[apa][endpoint][channel]['linearity']['langauss']['eparams'] = l_popt_error.tolist()
                    analysis_result_dict[apa][endpoint][channel]['linearity']['langauss']['chi2rid'] = l_chi2rid
                    analysis_result_dict[apa][endpoint][channel]['linearity']['langauss']['A'] = l_A
                    analysis_result_dict[apa][endpoint][channel]['linearity']['langauss']['eA'] = l_eA
                    analysis_result_dict[apa][endpoint][channel]['linearity']['langauss']['B'] = l_B
                    analysis_result_dict[apa][endpoint][channel]['linearity']['langauss']['eB'] = l_eB


                    df_rows.append({'APA':apa, 'Endpoint':endpoint, 'Channel':channel, 'Energies':energies, 'Gaussian mu':g_means, 'Gaussian emu':g_means_errors, 'Gaussian params':g_popt, 'Gaussian eparams': g_popt_error.tolist(), 'Gaussian chi2rid':g_chi2rid, 'Langauss peak': l_peaks, 'Langauss epeak':l_peak_errors, 'Langauss params':l_popt, 'Langauss eparams':l_popt_error.tolist(), 'Langauss chi2rid':l_chi2rid})

                    i+=1

                    # ----------------------------------------------------------------------
                    # 3. sigma/mean vs energy analysis
                    # ----------------------------------------------------------------------


                    # --- GAUSSIAN ANALYSIS 

                    g_sigma = np.array([analysis_result_dict[apa][endpoint][channel][e]['gaussian']['sigma'] for e in energies ])
                    g_sigma_errors = np.array([analysis_result_dict[apa][endpoint][channel][e]['gaussian']['esigma'] for e in energies])
                    #g_sigma_errors = g_sigma_errors + 0.05 * g_means

                    x = energies
                    ex = energies_errors
                    g_y = g_sigma / g_means
                    g_ey = g_y* np.sqrt((g_sigma_errors/g_sigma)**2 + (g_means_errors/g_means)**2)

                    ax[i].errorbar(energies, g_y, xerr = ex, yerr=g_ey, fmt='o', color='deepskyblue', label='Gaussian $\dfrac{\sigma}{\mu}$', capsize=3, markersize=2)

                    # --- LANGAUSS ANALYSIS 
                    l_sigma = np.array([analysis_result_dict[apa][endpoint][channel][e]['langauss']['sigma'] for e in energies ])
                    l_sigma_errors = np.array([analysis_result_dict[apa][endpoint][channel][e]['langauss']['esigma'] for e in energies])
                    #l_sigma_errors = g_sigma_errors + 0.05 * g_means

                    x = energies
                    ex = energies_errors
                    l_y = l_sigma / l_peak
                    l_ey = l_y* np.sqrt((l_sigma_errors/l_sigma)**2 + (l_peak_errors/l_peak)**2)


                    ax[i].errorbar(x, l_y, xerr = ex, yerr=l_ey, fmt='o', color='fuchsia', label='Langauss $\dfrac{\sigma}{x_0}$', capsize=3, markersize=2)

                    ax[i].set_xlabel(r"$\langle E_{beam} \rangle$ [GeV]")
                    ax[i].set_ylabel(r'$\dfrac{\sigma_N}{N_{PE}}$')
                    ax[i].legend()
                    ax[i].set_title('Energy resolution')
                    ax[i].grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.5)        


                    i+=1

                    # ----------------------------------------------------------------------
                    # 4. sigma vs energy analysis
                    # ----------------------------------------------------------------------


                    # --- GAUSSIAN ANALYSIS 

                    g_sigma = np.array([analysis_result_dict[apa][endpoint][channel][e]['gaussian']['sigma'] for e in energies ])
                    g_sigma_errors = np.array([analysis_result_dict[apa][endpoint][channel][e]['gaussian']['esigma'] for e in energies])
                    #g_sigma_errors = g_sigma_errors + 0.05 * g_means

                    ax[i].errorbar(energies, g_sigma, xerr = energies_errors, yerr=g_sigma_errors, fmt='o', color='deepskyblue', label='Gaussian $\sigma$', capsize=3, markersize=2)

                    # --- LANGAUSS ANALYSIS 
                    l_sigma = np.array([analysis_result_dict[apa][endpoint][channel][e]['langauss']['sigma'] for e in energies ])
                    l_sigma_errors = np.array([analysis_result_dict[apa][endpoint][channel][e]['langauss']['esigma'] for e in energies])
                    #l_sigma_errors = g_sigma_errors + 0.05 * g_means

                    ax[i].errorbar(energies, l_sigma, xerr = energies_errors, yerr=l_sigma_errors, fmt='o', color='fuchsia', label='Langauss $\sigma$', capsize=3, markersize=2)

                    ax[i].set_xlabel(r"$\langle E_{beam} \rangle$ [GeV]")
                    ax[i].set_ylabel(r'$\sigma$')
                    ax[i].legend()
                    ax[i].set_title('Energy resolution')
                    ax[i].grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.5)        

                
                plt.tight_layout()
                APA_pdf_file.savefig(fig)
                plt.close(fig)

        APA_pdf_file.close()

        print(f"conteggi landau migliore: {conteggi_l}")
        print(f"conteggi gaussian migliore: {conteggi_g}")
        print()

    analysis_result_df = pd.concat([analysis_result_df, pd.DataFrame(df_rows)], ignore_index=True)
    analysis_result_df.to_csv(f"{output_folder}/PE_study_results.csv", index=False)


    


    with open(f"{output_folder}/PE_study_results.json", "w") as f:
        json.dump(analysis_result_dict, f, indent=4)


if __name__ == "__main__":
    input_folder = f"/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output/apa1_vs_apa2"
    output_folder = f"/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output/single_channels_study"

    all_channel_data = {1:{}, 2:{}, 3:{}, 5:{}, 7:{}}
    for energy in all_channel_data.keys():
        with open(f"{input_folder}/{energy}GeV/channel_study/channels_data.json", "r") as f:
            all_channel_data[energy] = json.load(f)

    main(all_channel_data, input_folder, output_folder)