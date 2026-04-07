from utils import *

def main(energies, apa_studied, input_folder, output_folder, adiacent_channels_map, strategies, pairs, p2_evaluation = False):
    energy_dict =  {1: 0, 2: 118, 3: 119, 5: 175, 7: 253} # Crossing point between langauss and gaussian distribution - to use for beam event selection trigger 

    all_data = {}

    for energy in energies: 
        apa1_mean_pe_threshold = energy_dict[energy]

        apa12_energy_folder = f"{input_folder}/apa1_vs_apa2/{energy}GeV"

        merged_dict = load_energy_json_dict(energy, apa12_energy_folder, output_folder)


        # SELECTING TRIGGERS INDEXES WHERE APA1 MEAN PE > THRESHOLD
        dic_trigger_index = {}
        for _to_, to_dict in merged_dict.items(): # looking at each "0_to_10", "10_to_20", ...
            index_list = []
            for i, trigger_mean_pe in enumerate(to_dict['1']['mean']): # looking at each trigger
                try:
                    if trigger_mean_pe > apa1_mean_pe_threshold: # checking if the mean photoelectrons in APA1 is above threshold
                        index_list.append(i)
                except:
                    continue
            dic_trigger_index[_to_] = index_list # saving the list of trigger indexes that passed the threshold for this "_to_"


    

        adiacent_channels_dict = {}

        for apa, adiacent_channels_apa in adiacent_channels_map.items():
            adiacent_channels_apa_list = []

            for ch_pairs in adiacent_channels_apa:

                ch_pair_info = copy.deepcopy(ch_pairs)

                for i in range(2):
                    ch_pair_info[i]['NPE'] = []
                    ch_pair_info[i]['NPE_err'] = []

                apa1 = str(ch_pairs[0]['apa'])
                end1 = str(ch_pairs[0]['end'])
                ch1  = str(ch_pairs[0]['ch'])

                apa2 = str(ch_pairs[1]['apa'])
                end2 = str(ch_pairs[1]['end'])
                ch2  = str(ch_pairs[1]['ch'])

                for _to_, index_list in dic_trigger_index.items():
                    for index in index_list:

                        cd1 = merged_dict.get(_to_, {}).get(apa1, {}).get("channel_dic")
                        cd2 = merged_dict.get(_to_, {}).get(apa2, {}).get("channel_dic")

                        ch_data1 = None
                        ch_data2 = None

                        if isinstance(cd1, list) and index < len(cd1):
                            ch_data1 = cd1[index].get(end1, {}).get(ch1)

                        if isinstance(cd2, list) and index < len(cd2):
                            ch_data2 = cd2[index].get(end2, {}).get(ch2)

                        if ch_data1 and ch_data2:
                            
                            n_pe1 = ch_data1['n_pe']
                            e_n_pe1 = ch_data1['e_n_pe']
                            n_pe2 = ch_data2['n_pe']
                            e_n_pe2 = ch_data2['e_n_pe']

                            ch_pair_info[0]['NPE'].append(n_pe1)
                            ch_pair_info[0]['NPE_err'].append(e_n_pe1)
                            ch_pair_info[1]['NPE'].append(n_pe2)
                            ch_pair_info[1]['NPE_err'].append(e_n_pe2)
                
                
                for i in range(2):
                    ch_pair_info[i]['NPE_mean'] = np.mean(ch_pair_info[i]['NPE']) if ch_pair_info[i]['NPE'] else 0
                    ch_pair_info[i]['NPE_std'] = np.std(ch_pair_info[i]['NPE']) if ch_pair_info[i]['NPE'] else 0
                    ch_pair_info[i]['NPE_err_mean'] = ch_pair_info[i]['NPE_mean'] / np.sqrt(len(ch_pair_info[i]['NPE'])) if ch_pair_info[i]['NPE'] else 0
                

                adiacent_channels_apa_list.append(ch_pair_info)
            
            adiacent_channels_dict[apa] = adiacent_channels_apa_list
            
        all_data[energy] = adiacent_channels_dict
        
    colors = {1: "red", 2: "green", 3: "blue", 5: "orange", 7: "purple"}

    energies = np.array(energies)

    APA_pdf_file = PdfPages(f"{output_folder}/multi_pair_analysis_{pairs}_pairs_energyproblem.pdf")

    df_calibration = pd.read_csv(f"/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/data/calibration/calibration_final.csv")
    df_calibration = df_calibration[
        (df_calibration['batch'] == 1) &
        (df_calibration['APA'] == int(apa)) &
        (
            ((df_calibration['vendor'] == 'FBK') & (np.isclose(df_calibration['OV_V'], 4.5))) |
            ((df_calibration['vendor'] == 'HPK') & (np.isclose(df_calibration['OV_V'], 3)))
        )
    ]    
    df_calibration = df_calibration[['endpoint', 'channel', 'snr', 'snr_error', 'SPE_mean_amplitude', 'gain', 'gain_error', 'std_0', 'std_0_error']]
  
    all_results = []
    for apa, adiacent_channels_apa in adiacent_channels_map.items():
        for ch_pairs in adiacent_channels_apa:
            n_rows = len(strategies)+1
            if p2_evaluation:
                n_cols = len(energies) + 2
            else:
                n_cols = len(energies) + 1
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
            fig.subplots_adjust(
                hspace=0.3,
                wspace=0.4,   
                top=0.92,
                bottom=0.05
            )            
            plt.suptitle(f'end{ch_pairs[0]["end"]}ch{ch_pairs[0]["ch"]} - end{ch_pairs[1]["end"]}ch{ch_pairs[1]["ch"]}', fontsize =22)
            
            
            all_mu       = {s: [] for s in strategies}
            all_mu_err   = {s: [] for s in strategies}
            all_sigma        = {s: [] for s in strategies}
            all_sigma_err    = {s: [] for s in strategies}
            all_mean = {s: [] for s in strategies}
            all_std = {s: [] for s in strategies}
            all_mean_pe_1 = []
            all_mean_pe_2 = []  
            all_mean_pe_1_err = []
            all_mean_pe_2_err = []

            scatter_plot_results = {}
            valid_energies = []
            for j, energy in enumerate(energies):
                valid_energy = False

                ax = axes[0, j]

                for data_ch_pair in all_data[energy][apa]:
                    if (data_ch_pair[0]['end'] == ch_pairs[0]['end'] and data_ch_pair[0]['ch'] == ch_pairs[0]['ch'])  and (data_ch_pair[1]['end'] == ch_pairs[1]['end'] and data_ch_pair[1]['ch'] == ch_pairs[1]['ch']):
                        x = np.array(data_ch_pair[0]['NPE'])
                        ex = np.array(data_ch_pair[0]['NPE_err'])
                        y = np.array(data_ch_pair[1]['NPE'])
                        ey = np.array(data_ch_pair[1]['NPE_err'])

                        if len(x) < 100:
                            print(f"Warning: Only {len(x)} events for energy {energy} GeV, APA {apa}, channels {ch_pairs[0]['end']}ch{ch_pairs[0]['ch']} and {ch_pairs[1]['end']}ch{ch_pairs[1]['ch']}. Results might not be reliable.")
                            continue
                        else:
                            valid_energies.append(energy)
                            valid_energy = True

                        all_mean_pe_1.append(data_ch_pair[0]['NPE_mean'])
                        all_mean_pe_1_err.append(data_ch_pair[0]['NPE_err_mean'])
                        all_mean_pe_2.append(data_ch_pair[1]['NPE_mean'])
                        all_mean_pe_2_err.append(data_ch_pair[1]['NPE_err_mean'])

                        ax.errorbar(
                            x, y,
                            xerr=ex, yerr=ey,
                            fmt='o',
                            markersize=5,
                            capsize=3,
                            elinewidth=1,
                            markeredgecolor='black',
                            markerfacecolor=colors[energy],
                            label="Data"
                        )

                        xmin = np.minimum(np.min(x), np.min(y)) - 50
                        xmax = np.maximum(np.max(x), np.max(y)) + 50

                        # Reference line y = x
                        ax.plot([xmin, xmax], [xmin, xmax], 'k--', label='Expected y = x')

                        # ----- ODR linear fit -----
                        data = RealData(x, y, sx=ex, sy=ey)
                        model = Model(linear_array)
                        odr = ODR(data, model, beta0=curve_fit(linear, x, y, sigma=ey, absolute_sigma=True)[0])
                        out = odr.run()

                        A, B = out.beta
                        eA, eB = out.sd_beta

                        y_fit = linear(x, A, B)

                        r_squared = r2_score(y, y_fit)
                        # chi2rid = out.sum_square / (len(x) - len(out.beta))
                        _, chi2rid, _= chi2_func(y, y_fit, ey, len(out.beta))

                        label_fit = (
                            f'y = A + Bx\n'
                            f'A = ({fmt(A, eA)})\n'
                            f'B = ({fmt(B, eB)})\n'
                            f'$R^2$ = {r_squared:.3f}\n'
                            # rf'$\chi^2_{{rid}}$ = {chi2rid:.2f}'
                        )

                        ax.plot([xmin, xmax],
                                linear_array([A,B], np.array([xmin, xmax])),
                                'r-', label=label_fit)
                        
                        scatter_plot_results[energy] = {'A': (A, eA), 'B': (B, eB), 'r_squared': r_squared, 'chi2rid': chi2rid}

                if valid_energy:
                    ax.set_xlim(xmin, xmax)
                    ax.set_ylim(xmin, xmax)
                    ax.legend(loc='upper left', fontsize=8)
                else:
                    ax.text(0.5, 0.5,
                            "No valid data",
                            transform=ax.transAxes,
                            ha='center', va='center')
                    ax.set_axis_off()

                ax.set_xlabel(
                    f"$N_{{PE}}$ - end{ch_pairs[0]['end']}ch{ch_pairs[0]['ch']}",
                    fontsize=11
                )
                ax.set_ylabel(
                    f"$N_{{PE}}$ - end{ch_pairs[1]['end']}ch{ch_pairs[1]['ch']}",
                    fontsize=11
                )
                ax.set_title(f"{energy} GeV", fontsize=13)
                ax.grid(True)
                
                
                if not valid_energy:
                    continue

                for i, residual_strategy in enumerate(strategies):

                    # HISTOGRAM OF RESIDUALS
                    ax = axes[i+1, j]
                    if residual_strategy =='N1-N2':
                        residuals =  x-y 
                        label_x = r"$N_1 - N_2$"
                    
                    elif residual_strategy == '(N1-N2)/((N1+N2)/2)':
                        residuals = (1/np.sqrt(2))* (x-y)/((x+y)/2)
                        label_x = r"$ \sqrt{2} \cdot \frac{N_A - N_B}{N_A + N_B}$"

                    elif residual_strategy == '(N1/<N1>) - (N2/<N2>)':
                        residuals = (1/np.sqrt(2))* ((x/data_ch_pair[0]['NPE_mean']) - (y/data_ch_pair[1]['NPE_mean']))
                        label_x = r"$\frac{1}{\sqrt{2}} \cdot  \!\left( \frac{N_1}{\langle N_1 \rangle} - \frac{N_2}{\langle N_2 \rangle} \right)$"
                       
                    else:
                        continue

                    N = len(residuals)
                    mean_0 = np.mean(residuals)
                    std_0  = np.std(residuals, ddof=1)

                    Range = [np.percentile(residuals, 0.3), np.percentile(residuals, 99.8)]
                    bins = 50

                    ax.hist(
                        residuals,
                        bins=bins,
                        range=Range,
                        color=colors[energy],
                        alpha=0.8,
                        label=f"N = {N}\n"+rf"$\mu$ = {mean_0:.2}"+"\n"+rf"$\sigma$ = {std_0:.2f}"
                    )

                    # ===== Gaussian fit of histogram =====
                    counts, bin_edges = np.histogram(residuals, bins=bins, range=Range)
                    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                    bin_width = bin_edges[1] - bin_edges[0]

                    mask = counts > 0
                    x_fit = bin_centers[mask]
                    y_fit = counts[mask]

                    sx = np.full_like(x_fit, bin_width / 2)
                    sy = np.sqrt(y_fit)

                    data_g = RealData(x_fit, y_fit, sx=sx, sy=sy)
                    model_g = Model(gaussian_array)

                    odr_g = ODR(data_g, model_g, beta0=[mean_0, std_0, np.max(y_fit)])
                    out_g = odr_g.run()

                    mu, sigma, A_g = out_g.beta
                    emu, esigma, eA_g = out_g.sd_beta

                    y_model_g = gaussian(x_fit, mu, sigma, A_g)
                    r_squared = r2_score(y_fit, y_model_g)
                    # chi2rid = out_g.sum_square / (len(x_fit)-len(out_g.beta))
                    _, chi2rid, _ = chi2_func(y_fit, y_model_g, sy, len(out_g.beta))

                    label_fit = (
                        rf"$\mu$ = {fmt(mu, emu)}"
                        "\n"
                        rf"$\sigma$ = {fmt(sigma, esigma)}"
                        "\n"
                        f"A = {fmt(A_g,eA_g)}"
                        "\n"
                        f"R$^2$ = {r_squared:.3f}"
                        "\n"
                        # rf"$\chi^2_{{rid}}$ = {chi2rid:.3f}"
                    )

                    x_plot = np.linspace(Range[0], Range[1], 500)
                    y_plot = gaussian(x_plot, mu, sigma, A_g)

                    ax.plot(x_plot, y_plot, 'k-', linewidth=2, label=label_fit)
                    ax.axvline(mu, linestyle='--')

                    ax.set_title(f"{energy} GeV", fontsize=13)
                    ax.set_ylabel("Counts", fontsize=11)
                    ax.set_xlabel(label_x, fontsize=11)

                    ax.legend(loc="upper left", fontsize=8)
                    ax.grid(True)

                    all_mu[residual_strategy].append(mu)
                    all_mu_err[residual_strategy].append(emu)
                    all_sigma[residual_strategy].append(sigma)
                    all_sigma_err[residual_strategy].append(esigma)
                    all_mean[residual_strategy].append(mean_0)
                    all_std[residual_strategy].append(std_0)
                

            # # ENERGY RESOLUTION
            energy_resolution_result = {}
            
            n_valid = len(all_mean_pe_1)
            if n_valid <= 3:
                print(f"Skipping energy resolution: only {n_valid} valid energies")

                for i in range(len(strategies)):
                    ax = axes[i+1, -1]
                    ax.text(0.5, 0.5,
                            f"Not enough energies\n({n_valid} < 3)",
                            transform=ax.transAxes,
                            ha='center', va='center')
                    ax.set_axis_off()

            else:
                if p2_evaluation:
                    ax = axes[0, -1]

                    ### ELECTONS
                    x= np.array(valid_energies)
                    ex = 0.05*x

                    row0 = df_calibration[(df_calibration['endpoint'] == ch_pairs[0]["end"]) & (df_calibration['channel'] == ch_pairs[0]["ch"])]
                    row1 = df_calibration[(df_calibration['endpoint'] == ch_pairs[1]["end"]) & (df_calibration['channel'] == ch_pairs[1]["ch"])]

                    snr1 = row0['snr'].values[0]
                    snr2 = row1['snr'].values[0]
                    snr1_err = row0['snr_error'].values[0]
                    snr2_err = row1['snr_error'].values[0]

                    sigma0_1 = row0['std_0'].values[0]
                    sigma0_2 = row1['std_0'].values[0]
                    sigma0_1_err = row0['std_0_error'].values[0]
                    sigma0_2_err = row1['std_0_error'].values[0]

                    gain1 = row0['gain'].values[0]
                    gain2 = row1['gain'].values[0]
                    gain1_err = row0['gain_error'].values[0]
                    gain2_err = row1['gain_error'].values[0]

                    all_mean_pe_1 = np.array(all_mean_pe_1, dtype=float)
                    all_mean_pe_2 = np.array(all_mean_pe_2, dtype=float)
                    all_mean_pe_1_err = np.array(all_mean_pe_1_err, dtype=float)
                    all_mean_pe_2_err = np.array(all_mean_pe_2_err, dtype=float)

                    A = ((sigma0_1 / gain1) / all_mean_pe_1)**2
                    B = ((sigma0_2 / gain2) / all_mean_pe_2)**2
                    y= np.sqrt(A + B)
                    eA = A**2 *((sigma0_1_err/sigma0_1)**2+(gain1_err/gain1)**2+(all_mean_pe_1_err/all_mean_pe_1)**2)
                    eB = B**2 *((sigma0_2_err/sigma0_2)**2+(gain2_err/gain2)**2+(all_mean_pe_2_err/all_mean_pe_2)**2)
                    ey = np.sqrt(eA + eB) / y
                    label_p2 = r'$\sqrt{\left(\frac{\sigma_{0,A}/gain_A}{\langle N_{PE,A} \rangle}\right)^2 + \left(\frac{\sigma_{0,B}/gain_B}{\langle N_{PE,B} \rangle}\right)^2}$'

                    ax.errorbar(x,
                    y,
                    xerr = ex,
                    yerr = ey,
                    marker='o',
                    linestyle='',
                    markersize=4,
                    linewidth=1,
                    capsize=3,
                    color = 'red',
                    label=f"Only electrons"
                    )               
                    
                    data = RealData(x, y, sx=ex, sy=ey)
                    model = Model(noise_term_fit_array)

                    odr = ODR(data, model, beta0=np.array([0]))
                    odr.set_job(fit_type=0)
                    out = odr.run()

                    p2_fixed = out.beta[0]
                    ep2_fixed = out.sd_beta[0]

                    y_fit = noise_term_fit_array(out.beta, x)                        
                    # chi2_red = out.sum_square / (len(x) - len(out.beta)) 
                    _, chi2_red, _= chi2_func(y, y_fit, ey, len(out.beta))
                    r_squared = r2_score(y,y_fit)
            
                    y_fit_label = (f"Fit "
                    r'$y=\frac{c}{x}$'
                    f'\n'
                    f"c = {fmt(p2_fixed,ep2_fixed)}\n"
                    f"$R^2$ = {r_squared:.3f}\n"
                    # r"$\chi^2_{rid}$" +f" = {chi2_red:.3f}"
                    )

                    x_plot = np.linspace(np.min(x), np.max(x), 500)
                    y_plot = noise_term_fit_array(out.beta, x_plot)
                    ax.plot(x_plot, y_plot, label=y_fit_label, color = 'orange')


                    ### PROTONS
                    x_protons = np.array([0.43,1.27,2.20,4.15,6.12]) 
                    ex_protons = 0.05*x_protons
                    y_protons = y
                    ey_protons = ey 
                    ax.errorbar(x_protons,
                    y_protons,
                    xerr = ex_protons,
                    yerr = ey_protons,
                    marker='o',
                    linestyle='',
                    markersize=4,
                    linewidth=1,
                    capsize=3,
                    color = 'blue',
                    label=f"Only protons"
                    )

                    data_protons = RealData(x_protons, y_protons, sx=ex_protons, sy=ey_protons)
                    model_protons = Model(noise_term_fit_array)
                    odr_protons = ODR(data_protons, model_protons, beta0=np.array([0]))
                    odr_protons.set_job(fit_type=0)
                    out_protons = odr_protons.run()
                    p2_fixed_protons = out_protons.beta[0]
                    ep2_fixed_protons = out_protons.sd_beta[0]
                    y_fit_protons = noise_term_fit_array(out_protons.beta, x_protons)
                    # chi2_red_protons = out_protons.sum_square / (len(x_protons) - len(out_protons.beta))
                    _, chi2_red_protons, _= chi2_func(y_protons, y_fit_protons, ey_protons, len(out_protons.beta))
                    r_squared_protons = r2_score(y_protons,y_fit_protons)
                    y_fit_label_protons = (f"Fit protons "
                    r'$y=\frac{c}{x}$'
                    f'\n'
                    f"c = {fmt(p2_fixed_protons,ep2_fixed_protons)}\n"
                    f"$R^2$ = {r_squared_protons:.3f}\n"
                    # r"$\chi^2_{rid}$" +f" = {chi2_red_protons:.3f}"
                    )
                    x_plot_protons = np.linspace(np.min(x_protons), np.max(x_protons), 500)
                    y_plot_protons = noise_term_fit_array(out_protons.beta, x_plot_protons)
                    ax.plot(x_plot_protons, y_plot_protons, label=y_fit_label_protons, color = 'skyblue')

                    ax.set_xlabel(r"$\langle K_{beam} \rangle$ [GeV]")
                    ax.set_ylabel(label_p2)
                    ax.set_title(f"Noise term", fontsize=13)
                    ax.grid(True)
                    ax.set_xlim(0, 8)
                    ax.legend(fontsize=9)   


                for i, residual_strategy in enumerate(strategies):

                    ax = axes[i+1, -2]

                    x = np.array(valid_energies)
                    ex = 0.05*x

                    all_sigma[residual_strategy] = np.array(all_sigma[residual_strategy])
                    all_sigma_err[residual_strategy] = np.array(all_sigma_err[residual_strategy])
                    all_mean_pe_1 = np.array(all_mean_pe_1)
                    all_mean_pe_1_err = np.array(all_mean_pe_1_err)

                    if residual_strategy == 'N1-N2':
                        y = (1/np.sqrt(2))* (all_sigma[residual_strategy]/all_mean_pe_1)
                        ey = y * np.sqrt((all_sigma_err[residual_strategy]/all_sigma[residual_strategy])**2+(all_mean_pe_1_err/all_mean_pe_1)**2)
                        y_label = r'$\frac{1}{\sqrt{2}} \cdot \sigma(N_1 - N_2) \cdot \frac{1}{\langle N_1 \rangle}$'
                    elif residual_strategy == '(N1-N2)/((N1+N2)/2)':
                        y = all_sigma[residual_strategy]
                        ey = all_sigma_err[residual_strategy]
                        y_label = r'$\sigma\!\left(\sqrt{2} \cdot \frac{N_A - N_B}{N_A + N_B}\right)$'
                    elif residual_strategy == '(N1/<N1>) - (N2/<N2>)':
                        y = all_sigma[residual_strategy]
                        ey =  all_sigma_err[residual_strategy]
                        y_label = r'$\sigma\!\left(\frac{1}{\sqrt{2}} \cdot  \!\left( \frac{N_1}{\langle N_1 \rangle} - \frac{N_2}{\langle N_2 \rangle} \right)\right)$'
                    else:
                        continue

                    ### ONLY ELECTRONS
                    ax.errorbar(x,
                        y,
                        xerr = ex,
                        yerr = ey,
                        marker='o',
                        linestyle='',
                        markersize=4,
                        linewidth=1,
                        capsize=3,
                        color = 'red',
                        label=f"Only electrons "
                        )

                    p0_init = np.min(y)                  
                    p1_init = np.sqrt(np.maximum(y[0]**2 - p0_init**2, 1e-6)) * np.sqrt(x[0])            
                    p2_init = 0.1                       
                    
                    data = RealData(x, y, sx=ex, sy=ey)
                    model = Model(energy_resolution_fit_array)

                    odr = ODR(data, model, beta0=np.array([p0_init, p1_init, p2_init]))
                    odr.set_job(fit_type=0)
                    out = odr.run()

                    p0, p1, p2 = out.beta
                    ep0, ep1, ep2 = out.sd_beta

                    y_fit = energy_resolution_fit(x,p0,p1,p2)
                    # chi2_red = out.sum_square / (len(x) - len(out.beta)) 
                    _, chi2_red, _= chi2_func(y, y_fit, ey, len(out.beta))
                    r_squared = r2_score(y,y_fit)
            
                    y_fit_label = (f"Fit only ELECTRONS: "
                    r'$y=\sqrt{a^2 + \left(\frac{b}{\sqrt{x}}\right)^2 + \left(\frac{c}{x}\right)^2}$'
                    f'\n'
                    f"a = {fmt(p0,ep0)}\n"
                    f"b = {fmt(p1,ep1)}\n"
                    f"c = {fmt(p2,ep2)}\n"
                    f"$R^2$ = {r_squared:.3f}\n"
                    # r"$\chi^2_{rid}$" +f" = {chi2_red:.3f}"
                    )

                    x_plot = np.linspace(np.min(x), np.max(x), 500)
                    y_plot = energy_resolution_fit(x_plot, p0, p1, p2)
                    ax.plot(x_plot, y_plot, color = "orange", label=y_fit_label)

                    ##### PROTONS

                    x_protons = np.array([0.43,1.27,2.20,4.15,6.12])
                    ex_protons = 0.05*x_protons
                    y_protons = y 
                    ey_protons = ey 

                    ax.errorbar(x_protons,
                    y_protons,
                    xerr = ex_protons,
                    yerr = ey_protons,
                    marker='o',
                    linestyle='',
                    markersize=4,
                    linewidth=1,
                    capsize=3,
                    color = 'blue',
                    label=f"Only protons"
                    )

                    p0_init_protons = np.min(y)                  
                    p1_init_protons = np.sqrt(np.maximum(y[0]**2 - p0_init_protons**2, 1e-6)) * np.sqrt(x_protons[0])            
                    p2_init_protons = 0.1                       
                    
                    data_protons = RealData(x_protons, y_protons, sx=ex_protons, sy=ey_protons)
                    model = Model(energy_resolution_fit_array)

                    odr_protons = ODR(data_protons, model, beta0=np.array([p0_init_protons, p1_init_protons, p2_init_protons]))
                    odr_protons.set_job(fit_type=0)
                    out_protons = odr_protons.run()

                    p0_protons, p1_protons, p2_protons = out_protons.beta
                    ep0_protons, ep1_protons, ep2_protons = out_protons.sd_beta

                    y_fit_protons = energy_resolution_fit(x_protons,p0_protons,p1_protons,p2_protons)
                    # chi2_red = out.sum_square / (len(x) - len(out.beta)) 
                    _, chi2_red_protons, _= chi2_func(y_protons, y_fit_protons, ey_protons, len(out_protons.beta))
                    r_squared_protons = r2_score(y_protons,y_fit_protons)
            
                    y_fit_label = (f"Protons fit: "
                    r'$y=\sqrt{a^2 + \left(\frac{b}{\sqrt{x}}\right)^2 + \left(\frac{c}{x}\right)^2}$'
                    f'\n'
                    f"a = {fmt(p0_protons,ep0_protons)}\n"
                    f"b = {fmt(p1_protons,ep1_protons)}\n"
                    f"c = {fmt(p2_protons,ep2_protons)}\n"
                    f"$R^2$ = {r_squared_protons:.3f}\n"
                    # r"$\chi^2_{rid}$" +f" = {chi2_red_protons:.3f}"
                    )

                    x_plot_protons = np.linspace(np.min(x_protons), np.max(x_protons), 500)
                    y_plot_protons = energy_resolution_fit(x_plot_protons, p0_protons, p1_protons, p2_protons)
                    ax.plot(x_plot_protons, y_plot_protons, color = "skyblue", label=y_fit_label)

                    ax.set_xlabel(r"$\langle K_{beam} \rangle$ [GeV]")
                    ax.set_ylabel(y_label)
                    ax.set_title(f"Energy Resolution", fontsize=13)
                    ax.grid(True)
                    ax.set_xlim(0, 8)
                    ax.legend(fontsize=7)


                    if p2_evaluation:
                        ax = axes[i+1, -1]

                        ### ELECTRONS

                        ax.errorbar(x,
                        y,
                        xerr = ex,
                        yerr = ey,
                        marker='o',
                        linestyle='',
                        markersize=4,
                        linewidth=1,
                        capsize=3,
                        color = 'red',
                        label=f"Only electrons"
                        )

                        fit_func = make_energy_resolution_fit_p2_fixed(p2_fixed)
                        data = RealData(x, y, sx=ex, sy=ey)
                        model = Model(fit_func)

                        odr_n = ODR(data, model, beta0=np.array([p0_init, p1_init]))
                        odr_n.set_job(fit_type=0)
                        out_n = odr_n.run()

                        p0_n, p1_n = out_n.beta
                        ep0_n, ep1_n = out_n.sd_beta

                        y_fit_n = fit_func(out_n.beta, x)
                        # chi2_red = out.sum_square / (len(x) - len(out.beta)) 
                        _, chi2_red_n, _= chi2_func(y, y_fit_n, ey, len(out_n.beta))
                        r_squared_n = r2_score(y,y_fit_n)
                
                        y_fit_label_n = (f"Electrons fit: "
                        r'$y=\sqrt{a^2 + \left(\frac{b}{\sqrt{x}}\right)^2 + \left(\frac{c_{FIX}}{x}\right)^2}$'
                        f'\n'
                        f"a = {fmt(p0_n,ep0_n)}\n"
                        f"b = {fmt(p1_n,ep1_n)}\n"
                        r"$c_{FIX}$" + f" = {fmt(p2_fixed,ep2_fixed)}\n"
                        f"$R^2$ = {r_squared_n:.3f}\n"
                        # r"$\chi^2_{rid}$" +f" = {chi2_red_n:.3f}"
                        )

                        x_plot_n = np.linspace(np.min(x), np.max(x), 500)
                        y_plot_n = fit_func(out_n.beta,x_plot_n)
                        ax.plot(x_plot_n, y_plot_n, color = 'orange', label=y_fit_label_n)


                        ### PROTONS
                        ax.errorbar(x_protons,
                        y_protons,
                        xerr = ex_protons,
                        yerr = ey_protons,
                        marker='o',
                        linestyle='',
                        markersize=4,
                        linewidth=1,
                        capsize=3,
                        color = 'blue',
                        label=f"Only protons"
                        )

                        fit_func_protons = make_energy_resolution_fit_p2_fixed(p2_fixed_protons)
                        data_protons = RealData(x_protons, y_protons, sx=ex_protons, sy=ey_protons)
                        model_protons = Model(fit_func_protons)
                        odr_protons_n = ODR(data_protons, model_protons, beta0=np.array([p0_init_protons, p1_init_protons]))
                        odr_protons_n.set_job(fit_type=0)
                        out_protons_n = odr_protons_n.run()
                        p0_protons_n, p1_protons_n = out_protons_n.beta
                        ep0_protons_n, ep1_protons_n = out_protons_n.sd_beta
                        y_fit_protons_n = fit_func_protons(out_protons_n.beta, x_protons)
                        # chi2_red_protons_n = out_protons_n.sum_square / (len(x_protons) - len(out_protons_n.beta))
                        _, chi2_red_protons_n, _= chi2_func(y_protons, y_fit_protons_n, ey_protons, len(out_protons_n.beta))
                        r_squared_protons_n = r2_score(y_protons,y_fit_protons_n)

                        y_fit_label_protons_n = (f"Protons fit: "
                        r'$y=\sqrt{a^2 + \left(\frac{b}{\sqrt{x}}\right)^2 + \left(\frac{c_{FIX}}{x}\right)^2}$'
                        f'\n'
                        f"a = {fmt(p0_protons_n,ep0_protons_n)}\n"
                        f"b = {fmt(p1_protons_n,ep1_protons_n)}\n"
                        r"$c_{FIX}$" + f" = {fmt(p2_fixed_protons,ep2_fixed_protons)}\n"
                        f"$R^2$ = {r_squared_protons_n:.3f}\n"
                        # r"$\chi^2_{rid}$" +f" = {chi2_red_protons_n:.3f}" 
                        )

                        x_plot_protons_n = np.linspace(np.min(x_protons), np.max(x_protons), 500)
                        y_plot_protons_n = fit_func_protons(out_protons_n.beta, x_plot_protons_n)
                        ax.plot(x_plot_protons_n, y_plot_protons_n, color = "skyblue", label=y_fit_label_protons_n)


                        ax.set_xlabel(r"$\langle K_{beam} \rangle$ [GeV]")
                        ax.set_ylabel(y_label)
                        ax.set_title(f"Energy Resolution - c correction", fontsize=13)
                        ax.grid(True)
                        ax.set_xlim(0, 8)
                        ax.legend(fontsize=7)

                    energy_resolution_result[residual_strategy] = {'x': x, 'ex': ex,
                        'y': y,
                        'ey': ey,
                        'fit_params': {
                            'p0': (p0, ep0),
                            'p1': (p1, ep1),
                            'p2': (p2, ep2)},
                        'chi2_red': chi2_red,
                        'r_squared': r_squared
                        }


        
                APA_pdf_file.savefig(fig)
            # plt.show()
            plt.close(fig)


            ch_pair_result = {
                'apa': apa,
                'channel_pair': ch_pairs,
                'energies': energies,
                'strategies': strategies,
                'all_mu': all_mu,
                'all_mu_err': all_mu_err,
                'all_sigma': all_sigma,
                'all_sigma_err': all_sigma_err,
                'all_mean': all_mean,
                'all_std': all_std,
                'all_mean_pe_1': all_mean_pe_1,
                'all_mean_pe_1_err': all_mean_pe_1_err,
                'all_mean_pe_2': all_mean_pe_2,
                'all_mean_pe_err_2': all_mean_pe_2_err,
                'scatter_plot' : scatter_plot_results,
                'energy_resolution': energy_resolution_result
            }

            all_results.append(ch_pair_result)

    APA_pdf_file.close()

    with open(f"{output_folder}/data_{pairs}_pairs_energyproblem.pkl", "wb") as f:
        pickle.dump(all_results, f)





if __name__ == "__main__":

    energies = [1,2,3,5,7] #[1,2,3,5,7]

    apa_studied = ['1'] #['1', '2']

    pairs = 'adjacent' # 'adjacent' or 'all'

    strategies = [
        # 'N1-N2',
        '(N1-N2)/((N1+N2)/2)',
        # '(N1/<N1>) - (N2/<N2>)'
    ]

    input_folder = f"/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output"
    output_folder= f"{input_folder}/adjacent_chanels_study"

    if pairs == 'adjacent':
        adjacent_channels_apa1_map , adjacent_channels_apa2_map, _ = adjacent_channel_info() #ADJACENT CHANNELS
    elif pairs == 'all':
        adjacent_channels_apa1_map , adjacent_channels_apa2_map, _  = all_channels_info_possible_pairs() # ALL POSSIBLE COMBINATION
    else:
        raise ValueError("Invalid value for 'pairs'. Use 'adjacent' or 'all'.")

    adiacent_channels_map = {}
    if '1' in apa_studied:
        adiacent_channels_map['1'] = adjacent_channels_apa1_map
    if '2' in apa_studied:
        adiacent_channels_map['2'] = adjacent_channels_apa2_map

    # adiacent_channels_map = {'1': [[{'apa' : 1, 'end': 105, 'ch': 15},{'apa' : 1, 'end': 105, 'ch': 17}]]}
    # adiacent_channels_map = {'1': [[{'apa' : 1, 'end': 104, 'ch': 15},{'apa' : 1, 'end': 104, 'ch': 12}]]}
    # adiacent_channels_map = {'1': [[{'apa' : 1, 'end': 104, 'ch': 15},{'apa' : 1, 'end': 104, 'ch': 12}], [{'apa' : 1, 'end': 104, 'ch': 2},{'apa' : 1, 'end': 104, 'ch': 0}],[{'apa' : 1, 'end': 104, 'ch': 3},{'apa' : 1, 'end': 104, 'ch': 4}]], '2' : [[{'apa' : 2, 'end': 109, 'ch': 37},{'apa' : 2, 'end': 109, 'ch': 35}], [{'apa' : 2, 'end': 109, 'ch': 31},{'apa' : 2, 'end': 109, 'ch': 33}], [{'apa' : 2, 'end': 109, 'ch': 34},{'apa' : 2, 'end': 109, 'ch': 36}], [{'apa' : 2, 'end': 109, 'ch': 2},{'apa' : 2, 'end': 109, 'ch': 0}] ]}    



    main(energies, apa_studied, input_folder, output_folder, adiacent_channels_map, strategies, pairs, p2_evaluation= True)
