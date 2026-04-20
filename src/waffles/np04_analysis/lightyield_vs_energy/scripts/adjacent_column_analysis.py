from utils import *

# def main(energies, apa_studied, input_folder, output_folder, strategies, p2_evaluation= False):
#     energy_dict =  {1: 0, 2: 118, 3: 119, 5: 175, 7: 253} # Crossing point between langauss and gaussian distribution - to use for beam event selection trigger 

#     all_data = {}

#     for energy in energies: 
#         apa1_mean_pe_threshold = energy_dict[energy]

#         apa12_energy_folder = f"{input_folder}/apa1_vs_apa2/{energy}GeV"

#         merged_dict = load_energy_json_dict(energy, apa12_energy_folder, output_folder)


#         # SELECTING TRIGGERS INDEXES WHERE APA1 MEAN PE > THRESHOLD
#         dic_trigger_index = {}
#         for _to_, to_dict in merged_dict.items(): # looking at each "0_to_10", "10_to_20", ...
#             index_list = []
#             for i, trigger_mean_pe in enumerate(to_dict['1']['mean']): # looking at each trigger
#                 try:
#                     if trigger_mean_pe > apa1_mean_pe_threshold: # checking if the mean photoelectrons in APA1 is above threshold
#                         index_list.append(i)
#                 except:
#                     continue
#             dic_trigger_index[_to_] = index_list # saving the list of trigger indexes that passed the threshold for this "_to_"

        

#         for apa in apa_studied:
#             apa_trigger_data = []
#             if apa == '1':
#                 columns_map = apa1_columns_channels()
#             else: 
#                 continue 
            
#             for _to_, index_list in dic_trigger_index.items():
#                 for index in index_list:
#                     column_pe_dict = {col: {'all_data': [], 'all_data_err': [],'mean': None, 'std': None, 'mean_err': None} for col in range(1, 5)}                    
#                     event = merged_dict[_to_][str(apa)]['channel_dic'][index]
#                     for end, channels in event.items():
#                         for channel, data in channels.items():
#                             for column in range(1, 5):
#                                 for column_channel_dict in columns_map[column]:
#                                     if (int(channel) == column_channel_dict['ch'] and int(end) == column_channel_dict['end']):
#                                         value = data.get('n_pe')
#                                         value_err = data.get('n_pe_err')
#                                         if value is not None:
#                                             column_pe_dict[column]['all_data'].append(value)
#                                             column_pe_dict[column]['all_data_err'].append(value_err)

#                     for column in range(1, 5):
#                         values = column_pe_dict[column]['all_data']
#                         errors = column_pe_dict[column]['all_data_err']

#                         if len(values) > 0:
#                             mean = np.mean(values)
#                             std = np.std(values, ddof=1) if len(values) > 1 else 0
#                             mean_err = std / np.sqrt(len(values)) if len(values) > 1 else 0

#                             column_pe_dict[column]['mean'] = mean
#                             column_pe_dict[column]['std'] = std
#                             column_pe_dict[column]['mean_err'] = mean_err

#                     apa_trigger_data.append(column_pe_dict)
#         all_data[energy] = {apa: apa_trigger_data}      
# 

def main(energies, apa_studied, input_folder, output_folder, strategies, p2_evaluation= False, weighted_kinetic_energy: bool = False):

    energy_dict = {1: 0, 2: 118, 3: 119, 5: 175, 7: 253}

    all_data = {}
    all_data["energies"] = energies
    all_data["channel_pe"] = {}

    for energy in energies:

        apa1_mean_pe_threshold = energy_dict[energy]

        apa12_energy_folder = f"{input_folder}/apa1_vs_apa2/{energy}GeV"

        merged_dict = load_energy_json_dict(energy, apa12_energy_folder, output_folder)

        # SELECTING TRIGGERS INDEXES WHERE APA1 MEAN PE > THRESHOLD
        dic_trigger_index = {}
        for _to_, to_dict in merged_dict.items():
            index_list = []
            for i, trigger_mean_pe in enumerate(to_dict['1']['mean']):
                try:
                    if trigger_mean_pe > apa1_mean_pe_threshold:
                        index_list.append(i)
                except:
                    continue
            dic_trigger_index[_to_] = index_list


        for apa in apa_studied:

            apa_trigger_data = []

            if apa not in all_data["channel_pe"]:
                all_data["channel_pe"][apa] = {}

            channel_pe_all = {}

            if apa == '1':
                columns_map = apa1_columns_channels()
            else:
                continue


            for _to_, index_list in dic_trigger_index.items():

                for index in index_list:

                    column_pe_dict = {
                        col: {'all_data': [], 'all_data_err': [], 'mean': None, 'std': None, 'mean_err': None}
                        for col in range(1, 5)
                    }

                    event = merged_dict[_to_][str(apa)]['channel_dic'][index]

                    for end, channels in event.items():

                        end = int(end)

                        if end not in channel_pe_all:
                            channel_pe_all[end] = {}

                        for channel, data in channels.items():

                            channel = int(channel)

                            value = data.get('n_pe')
                            value_err = data.get('n_pe_err')

                            if value is not None:

                                if channel not in channel_pe_all[end]:
                                    channel_pe_all[end][channel] = {
                                        'values': [],
                                        'errors': []
                                    }

                                channel_pe_all[end][channel]['values'].append(value)
                                channel_pe_all[end][channel]['errors'].append(value_err)

                            for column in range(1, 5):
                                for column_channel_dict in columns_map[column]:

                                    if (channel == column_channel_dict['ch'] and end == column_channel_dict['end']):

                                        if value is not None:
                                            column_pe_dict[column]['all_data'].append(value)
                                            column_pe_dict[column]['all_data_err'].append(value_err)


                    for column in range(1, 5):

                        values = column_pe_dict[column]['all_data']
                        errors = column_pe_dict[column]['all_data_err']

                        if len(values) > 0:

                            mean = np.mean(values)
                            std = np.std(values, ddof=1) if len(values) > 1 else 0
                            mean_err = std / np.sqrt(len(values)) if len(values) > 1 else 0

                            column_pe_dict[column]['mean'] = mean
                            column_pe_dict[column]['std'] = std
                            column_pe_dict[column]['mean_err'] = mean_err


                    apa_trigger_data.append(column_pe_dict)


            # COMPUTE MEAN PER CHANNEL FOR THIS ENERGY
            for end, channels in channel_pe_all.items():

                if end not in all_data["channel_pe"][apa]:
                    all_data["channel_pe"][apa][end] = {}

                for channel, data in channels.items():

                    values = data['values']

                    if len(values) == 0:
                        continue

                    mean = np.mean(values)
                    std = np.std(values, ddof=1) if len(values) > 1 else 0
                    mean_err = std / np.sqrt(len(values)) if len(values) > 1 else 0

                    if channel not in all_data["channel_pe"][apa][end]:
                        all_data["channel_pe"][apa][end][channel] = {
                            "mean": [],
                            "std": [],
                            "mean_err": []
                        }

                    all_data["channel_pe"][apa][end][channel]["mean"].append(mean)
                    all_data["channel_pe"][apa][end][channel]["std"].append(std)
                    all_data["channel_pe"][apa][end][channel]["mean_err"].append(mean_err)


            all_data[energy] = {
                apa: apa_trigger_data
            }


    colors = {1: "red", 2: "green", 3: "blue", 5: "orange", 7: "purple"}

    energies = np.array(energies)

    

    if weighted_kinetic_energy:
        APA_pdf_file = PdfPages(f"{output_folder}/adjacent_column_analysis_weighted_kinetic_energy.pdf")  
    else: 
        APA_pdf_file = PdfPages(f"{output_folder}/adjacent_column_analysis.pdf")

    

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
    for apa in apa_studied:
        for column_pair in [(1,2), (2,3), (3,4)]:
            fig, axes = plt.subplots(len(strategies)+1, len(energies)+1, figsize=(6*len(energies), 12))
            fig.subplots_adjust(
                hspace=0.3,
                wspace=0.4,   
                top=0.92,
                bottom=0.05
            )       

            plt.suptitle(f'APA {apa} - columns {column_pair[0]} and {column_pair[1]}', fontsize =22)
            
            
            all_mu       = {s: [] for s in strategies}
            all_mu_err   = {s: [] for s in strategies}
            all_sigma        = {s: [] for s in strategies}
            all_sigma_err    = {s: [] for s in strategies}
            all_mean = {s: [] for s in strategies}
            all_std = {s: [] for s in strategies}
            all_mean_pe = []
            all_mean_pe_err = []

            scatter_plot_results = {}
            valid_energies = []

            for j, energy in enumerate(energies):
                valid_energy = False
                ax = axes[0, j]

                triggers = all_data[energy][apa]
                column_0, column_1 = column_pair

                x, ex = [], []
                y, ey = [], []

                for trigger in triggers:
                    sum0 = np.sum(trigger[column_0]['all_data'])
                    sum1 = np.sum(trigger[column_1]['all_data'])

                    errs0 = [v for v in trigger[column_0]['all_data_err'] if v is not None]
                    sum0_err = np.sqrt(np.sum(np.array(errs0)**2)) if errs0 else 0

                    errs1 = [v for v in trigger[column_1]['all_data_err'] if v is not None]
                    sum1_err = np.sqrt(np.sum(np.array(errs1)**2)) if errs1 else 0


                    if sum0 is not None and sum1 is not None:
                        x.append(sum0)
                        ex.append(sum0_err)
                        y.append(sum1)
                        ey.append(sum1_err)

                x = np.array([v for v in x if np.isfinite(v)])
                y = np.array([v for v in y if np.isfinite(v)])
                ex = np.array([v for v in ex if np.isfinite(v)])
                ey = np.array([v for v in ey if np.isfinite(v)])

                ex = np.where(ex == 0, 1e-6, ex)
                ey = np.where(ey == 0, 1e-6, ey)

                if len(x) < 100:
                    print(f"Warning: Only {len(x)} events for energy {energy} GeV, APA {apa}, channels {ch_pairs[0]['end']}ch{ch_pairs[0]['ch']} and {ch_pairs[1]['end']}ch{ch_pairs[1]['ch']}. Results might not be reliable.")
                    continue
                else:
                    valid_energies.append(energy)
                    valid_energy = True

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

                all_mean_pe.append(np.mean(x))
                all_mean_pe_err.append(np.std(x) / np.sqrt(len(x)))
                
            
                xmin = np.minimum(np.min(x), np.min(y)) - 50
                xmax = np.maximum(np.max(x), np.max(y)) + 50

                # Reference line y = x
                ax.plot([xmin, xmax], [xmin, xmax], 'k--', label='Expected y = x')

                # ----- ODR linear fit -----
                data = RealData(x, y, sx=ex, sy=ey)
                model = Model(linear_array)
                odr = ODR(data, model, beta0=[0,1])
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
                    f"$N_{{PE}}$ - column {column_0}",
                    fontsize=11
                )
                ax.set_ylabel(
                    f"$N_{{PE}}$ - column {column_1}",
                    fontsize=11
                )
                ax.set_title(f"{energy} GeV", fontsize=13)
                ax.grid(True)
                
                
                if not valid_energy:
                    continue

                for i, residual_strategy in enumerate(strategies):

                    # HISTOGRAM OF RESIDUALS
                    ax = axes[i+1, j]
                    # if residual_strategy =='N1-N2':
                    #     residuals =  x-y 
                    #     label_x = r"$N_1 - N_2$"
                    
                    if residual_strategy == '(N1-N2)/((N1+N2)/2)':
                        residuals = (1/np.sqrt(2))* (x-y)/((x+y)/2)
                        label_x = r"$ \frac{1}{\sqrt{2}} \cdot \frac{N_1 - N_2}{(N_1 + N_2) /2}$"

                    # elif residual_strategy == '(N1/<N1>) - (N2/<N2>)':
                    #     residuals = (1/np.sqrt(2))* ((x/data_ch_pair[0]['NPE_mean']) - (y/data_ch_pair[1]['NPE_mean']))
                    #     label_x = r"$\frac{1}{\sqrt{2}} \cdot  \!\left( \frac{N_1}{\langle N_1 \rangle} - \frac{N_2}{\langle N_2 \rangle} \right)$"
                       
                    else:
                        continue

                    residuals = residuals[np.isfinite(residuals)]
                    if len(residuals) == 0:
                        print(f"Warning: No valid residuals for energy {energy}")
                        continue

                    N = len(residuals)
                    mean_0 = np.mean(residuals)
                    std_0  = np.std(residuals, ddof=1)

                    Range = [np.percentile(residuals, 0.3), np.percentile(residuals, 99.8)]
                    bins = 30

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

                    ax.set_title(f"{energy} GeV — {residual_strategy}", fontsize=13)
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

            
            n_valid = len(all_mean_pe)
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
                    if weighted_kinetic_energy:
                        x, ex = weighted_mean_kinetic_energy_function(np.array(valid_energies))
                    else: 
                        #classic
                        x= np.array(valid_energies)
                        ex = 0.05*x

                    y = np.zeros_like(x, dtype=float)
                    z_err = np.zeros_like(x, dtype=float)
                    ey = np.zeros_like(x, dtype=float)

                    for column_number in column_pair:
                        for ch in apa1_columns_channels()[column_number]:

                            # recupero SNR e errore
                            row = df_calibration[
                                (df_calibration['endpoint'] == ch["end"]) &
                                (df_calibration['channel'] == ch["ch"])
                            ]

                            if row.empty:
                                continue

                            snr = row['snr'].values[0]
                            snr_err = row['snr_error'].values[0]
                            spe = row['SPE_mean_amplitude'].values[0]
                            std_0 = row['std_0'].values[0]
                            std_0_err = row['std_0_error'].values[0]
                            gain = row['gain'].values[0]
                            gain_err = row['gain_error'].values[0]

                            # mean PE array per energia
                            mean_pe = np.array(all_data['channel_pe'][apa][ch["end"]][ch["ch"]]['mean'])
                            mean_pe_err = np.array(all_data['channel_pe'][apa][ch["end"]][ch["ch"]]['mean_err'])

                            mean_pe_safe = np.where(mean_pe > 0, mean_pe, 1e-10)

                            # Method 1: sum of 1/(SNR * mean_PE) in quadrature --> ANNA proceedings version
                            # f = 1 / (snr * mean_pe_safe)
                            # y += f**2
                            # sigma_f = np.sqrt( (snr_err / (snr**2 * mean_pe_safe))**2 + (mean_pe_err / (snr * mean_pe_safe**2))**2 )
                            # ey += (f / np.sqrt(y) * sigma_f)**2
                            # y_label_p2fixed = r"$\sqrt{\sum_{i} \left(\frac{1}{\mathrm{SNR}_i \cdot \langle N_{PE,i} \rangle}\right)^2}$"

                            #Method 2: sum of (SPE_ampl/SNR)/(mean_PE) in quadrature --> ANSELMO
                            # f2 = (spe / snr) / mean_pe_safe
                            # y += f2**2
                            # sigma_f2 = np.sqrt( (spe * snr_err / (snr**2 * mean_pe_safe))**2 + (mean_pe_err / (snr * mean_pe_safe**2))**2 )
                            # ey += (f2 / np.sqrt(y) * sigma_f2)**2
                            # y_label_p2fixed = r"$\sqrt{\sum_{i} \left(\frac{SPE_{i} / \mathrm{SNR}_i}{\langle N_{PE,i} \rangle}\right)^2}$"

                            #Method 3: sum of (std_0 / SPE_ampl) / N_PE in quadrature --> 


                            #Method 4: sum of (std_0 / gain) / N_PE in quadrature --> NEW
                            z = ((std_0 / gain) / mean_pe_safe)**2
                            y += z
                            # sigma_z = np.sqrt( (std_0 * gain_err / (gain**2 * mean_pe_safe))**2 + (mean_pe_err / (gain * mean_pe_safe**2))**2 )
                            # ey += (z / np.sqrt(y) * sigma_z)**2
                            y_label_p2fixed = r"$\sqrt{\sum_{i} \left(\frac{\sigma_{0,i} / \mathrm{gain}_i}{\langle N_{PE,i} \rangle}\right)^2}$"
                            z_err+= (z**2) * ((std_0_err/std_0)**2+(gain_err/gain)**2+(mean_pe_err/mean_pe_safe)**2)

                    # radice finale
                    y = np.sqrt(y)

                    ey = np.sqrt(z_err) / y

                    ax.errorbar(x,
                    y,
                    xerr = ex,
                    yerr = ey,
                    marker='o',
                    linestyle='',
                    markersize=4,
                    linewidth=1,
                    capsize=3,
                    label=f"Data"
                    )               
                    
                    data = RealData(x, y, sx=ex, sy=ey)
                    model = Model(noise_term_fit_array)

                    odr = ODR(data, model, beta0=np.array([0.01]))
                    odr.set_job(fit_type=0)
                    out = odr.run()

                    p2_fixed = out.beta[0]
                    ep2_fixed = out.sd_beta[0]

                    y_fit = noise_term_fit_array(out.beta, x)                        
                    # chi2_red = out.sum_square / (len(x) - len(out.beta)) 
                    _, chi2_red, _= chi2_func(y, y_fit, ey, len(out.beta))
                    r_squared = r2_score(y,y_fit)
            
                    y_fit_label = (f"Fit "
                    r'$y=\frac{p_2}{x}$'
                    f'\n'
                    f"p2 = {fmt(p2_fixed,ep2_fixed)}\n"
                    f"$R^2$ = {r_squared:.3f}\n"
                    # r"$\chi^2_{rid}$" +f" = {chi2_red:.3f}"
                    )

                    x_plot = np.linspace(np.min(x), np.max(x), 500)
                    y_plot = noise_term_fit_array(out.beta, x_plot)
                    ax.plot(x_plot, y_plot, label=y_fit_label, color = 'fuchsia')

                    ax.set_xlabel(r"$\langle E_{beam} \rangle$ [GeV]")
                    ax.set_ylabel(y_label_p2fixed)
                    ax.set_title(f"Noise term", fontsize=13)
                    ax.grid(True)
                    ax.set_xlim(0, 8)
                    ax.legend(fontsize=9)   


                for i, residual_strategy in enumerate(strategies):
                    

                    ax = axes[i+1, -1]

                    if weighted_kinetic_energy:
                        x, ex = weighted_mean_kinetic_energy_function(np.array(valid_energies))
                    else: 
                        #classic
                        x= np.array(valid_energies)
                        ex = 0.05*x

                    all_sigma[residual_strategy] = np.array(all_sigma[residual_strategy])
                    all_sigma_err[residual_strategy] = np.array(all_sigma_err[residual_strategy])
                    all_mean_pe = np.array(all_mean_pe)
                    all_mean_pe_err = np.array(all_mean_pe_err)

                    # if residual_strategy == 'N1-N2':
                    #     y = (1/np.sqrt(2))* (all_sigma[residual_strategy]/all_mean_pe)
                    #     ey = y * np.sqrt((all_sigma_err[residual_strategy]/all_sigma[residual_strategy])**2+(all_mean_pe_err/all_mean_pe)**2)
                    #     y_label = r'$\frac{1}{\sqrt{2}} \cdot \sigma(N_1 - N_2) \cdot \frac{1}{\langle N_1 \rangle}$'
                    if residual_strategy == '(N1-N2)/((N1+N2)/2)':
                        y = all_sigma[residual_strategy]
                        ey = all_sigma_err[residual_strategy]
                        y_label = r'$\sigma\!\left(\frac{1}{\sqrt{2}} \cdot \frac{N_1 - N_2}{(N_1 + N_2) /2}\right)$'
                    # elif residual_strategy == '(N1/<N1>) - (N2/<N2>)':
                    #     y = all_sigma[residual_strategy]
                    #     ey =  all_sigma_err[residual_strategy]
                    #     y_label = r'$\sigma\!\left(\frac{1}{\sqrt{2}} \cdot  \!\left( \frac{N_1}{\langle N_1 \rangle} - \frac{N_2}{\langle N_2 \rangle} \right)\right)$'
                    else:
                        continue
                        

                    ax.errorbar(x,
                        y,
                        xerr = ex,
                        yerr = ey,
                        marker='o',
                        linestyle='',
                        markersize=4,
                        linewidth=1,
                        capsize=3,
                        label=f"Data"
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
            
                    y_fit_label = (f"Fit "
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
                    ax.plot(x_plot, y_plot, label=y_fit_label)


                    if p2_evaluation:
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
                
                        y_fit_label_n = (f"Fit "
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
                        ax.plot(x_plot_n, y_plot_n, label=y_fit_label_n)

                    ax.set_xlabel(r"$\langle E_{beam} \rangle$ [GeV]")
                    ax.set_ylabel(y_label)
                    ax.set_title(f"Energy Resolution", fontsize=13)
                    ax.grid(True)
                    ax.set_xlim(0, 8)
                    ax.legend(fontsize=9)

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


    

            ch_pair_result = {
                'apa': apa,
                'channel_pair': column_pair,
                'energies': energies,
                'strategies': strategies,
                'all_mu': all_mu,
                'all_mu_err': all_mu_err,
                'all_sigma': all_sigma,
                'all_sigma_err': all_sigma_err,
                'all_mean': all_mean,
                'all_std': all_std,
                'all_mean_pe': all_mean_pe,
                'all_mean_pe_err': all_mean_pe_err,
                'scatter_plot' : scatter_plot_results,
                'energy_resolution': energy_resolution_result
            }

            all_results.append(ch_pair_result)


            APA_pdf_file.savefig(fig)
            plt.close(fig)





           
    APA_pdf_file.close()

    # with open(f"{output_folder}/data_{pairs}_pairs.pkl", "wb") as f:
    #     pickle.dump(all_results, f)
        

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

    main(energies, apa_studied, input_folder, output_folder, strategies, p2_evaluation= True, weighted_kinetic_energy=True)