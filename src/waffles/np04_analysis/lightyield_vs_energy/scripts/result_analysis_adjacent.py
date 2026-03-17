
from utils import *

def pair_to_key(pair):
    return (
        pair[0]['end'], pair[0]['ch'],
        pair[1]['end'], pair[1]['ch']
    )

# --- INPUT ---
apa_studied = [1]  # solo APA 1, solo APA 2, o entrambe [1,2]
strategy = '(N1-N2)/((N1+N2)/2)' #'(N1-N2)/((N1+N2)/2)'   'N1-N2'  '(N1/<N1>) - (N2/<N2>)'

pairs = 'adjacent' # 'adjacent' or 'all'
energies = [1, 2, 3, 5, 7]

input_folder = f"/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output/adjacent_chanels_study"
output_folder = os.path.join(input_folder, f"strategy_{strategy.replace('/', '_')}_{pairs}_pairs")
os.makedirs(output_folder, exist_ok=True)

with open(f"{input_folder}/data_{pairs}_pairs.pkl", "rb") as f:
    all_results = pickle.load(f)

adiacent_channels_1, adiacent_channels_2, _ = adjacent_channel_info()

############################################################################################################
""" ENERGY RESOLUTION: grid 10x3 with eenrgy resolution fit """

n_rows = 10
n_cols = 3*len(apa_studied)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), constrained_layout=True)
axes = axes.flatten()



apa_columns = {1: 0, 2: 0}  
current_col = 0
all_channel_pairs = []

if 1 in apa_studied:
    apa_columns[1] = current_col
    for row in range(n_rows):
        for col in range(3):
            idx = row*3 + col
            if idx < len(adiacent_channels_1):
                all_channel_pairs.append({
                    'apa': 1,
                    'ch_pair': adiacent_channels_1[idx],
                    'subplot_idx': row*n_cols + current_col + col
                })
    current_col += 3

if 2 in apa_studied:
    apa_columns[2] = current_col
    for row in range(n_rows):
        for col in range(3):
            idx = row*3 + col
            if idx < len(adiacent_channels_2):
                all_channel_pairs.append({
                    'apa': 2,
                    'ch_pair': adiacent_channels_2[idx],
                    'subplot_idx': row*n_cols + current_col + col
                })
    current_col += 3

for entry in all_channel_pairs:
    apa = entry['apa']
    ch_pairs = entry['ch_pair']
    subplot_idx = entry['subplot_idx']
    ax = axes[subplot_idx]

    ax.set_title(f"APA{apa} - {ch_pairs[0]['end']}ch{ch_pairs[0]['ch']}/"
                 f"{ch_pairs[1]['end']}ch{ch_pairs[1]['ch']}", fontweight='bold', fontsize=9)

    ch_result = None
    for r in all_results:
        ch0, ch1 = r['channel_pair']
        if (ch0['end'] == ch_pairs[0]['end'] and ch0['ch'] == ch_pairs[0]['ch'] and
            ch1['end'] == ch_pairs[1]['end'] and ch1['ch'] == ch_pairs[1]['ch']):
            ch_result = r
            break

    if ch_result is None or strategy not in ch_result['energy_resolution']:
        ax.text(0.5, 0.5, "No valid data", ha='center', va='center', transform=ax.transAxes)
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 0.5)
        continue

    energy_res = ch_result['energy_resolution'][strategy]
    x, ex = energy_res['x'], energy_res['ex']
    y, ey = energy_res['y'], energy_res['ey']
    fit = energy_res['fit_params']
    p0, ep0 = fit['p0']
    p1, ep1 = fit['p1']
    p2, ep2 = fit['p2']
    r_squared = energy_res['r_squared']
    chi2_red = energy_res['chi2_red']

    y_fit_vals = energy_resolution_fit(x, p0, p1, p2)

    ax.errorbar(x, y, xerr=ex, yerr=ey, fmt='o', markersize=4, capsize=3)
    x_fit = np.linspace(0, 8, 200)
    y_fit = energy_resolution_fit(x_fit, p0, p1, p2)
    ax.plot(x_fit, y_fit, 'r-')

    legend_text = (f"p0 = {fmt(p0,ep0)}\n"
                   f"p1 = {fmt(p1,ep1)}\n"
                   f"p2 = {fmt(p2,ep2)}\n"
                   f"R² = {r_squared:.3f}\n"
                   r"χ²_red = "+f"{chi2_red:.3f}")
    ax.legend([legend_text], loc='upper right', fontsize=7)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 0.5)
    ax.set_xlabel(r"$\langle E_{beam} \rangle$ [GeV]")
    ax.set_ylabel(r"$\sigma_{E}/E$")
    ax.grid(True)

plt.suptitle(f"Energy Resolution - Strategy: {strategy}", fontsize=16)
plt.savefig(f"{output_folder}/energy_resolution_grid_APA{apa_studied}.png", dpi=200)
plt.close()


############################################################################################################
""" SCATTER PLOT of Npe chA-chB: intercept A, and angular coefficient B and ΔB = B-1  """


fig2, axes2 = plt.subplots(len(energies), 2,
                           figsize=(14, 12),
                           constrained_layout=True)



apa_map = {}
for p in adiacent_channels_1:
    apa_map[pair_to_key(p)] = 1
for p in adiacent_channels_2:
    apa_map[pair_to_key(p)] = 2

for row, energy in enumerate(energies):

    x_labels = []
    A_vals, A_errs = [], []
    B_vals, B_errs = [], []

    for r in all_results:

        key = pair_to_key(r['channel_pair'])
        apa = apa_map.get(key, None)

        if apa not in apa_studied:
            continue

        if energy not in r['energies']:
            continue

        ch_name = (
            f"{r['channel_pair'][0]['end']}ch{r['channel_pair'][0]['ch']}/"
            f"{r['channel_pair'][1]['end']}ch{r['channel_pair'][1]['ch']}"
        )

        x_labels.append(ch_name)

        scatter_info = r['scatter_plot'].get(
            energy,
            {'A': (np.nan, np.nan), 'B': (np.nan, np.nan)}
        )

        A_vals.append(scatter_info['A'][0])
        A_errs.append(scatter_info['A'][1])

        B_vals.append(scatter_info['B'][0])
        B_errs.append(scatter_info['B'][1])

    x_pos = np.arange(len(x_labels))

    # =======================
    # Column 1 — Intercept A
    # =======================
    axA = axes2[row, 0]
    axA.errorbar(x_pos, A_vals, yerr=A_errs,
                 fmt='o', markersize=4, capsize=3)

    axA.axhline(0, linestyle='--')
    axA.set_ylabel("Intercept A")
    axA.set_title(f"E = {energy} GeV")

    if row == len(energies) - 1:
        axA.set_xticks(x_pos)
        axA.set_xticklabels(x_labels, rotation=90, fontsize=7)
    else:
        axA.set_xticks(x_pos)
        axA.set_xticklabels([])

    axA.grid(True)

    # =======================
    # Columns 2 — Angular coefficient B
    # =======================
    axB = axes2[row, 1]
    axB.errorbar(x_pos, B_vals, yerr=B_errs,
                 fmt='o', markersize=4, capsize=3)

    axB.axhline(1, linestyle='--')
    axB.set_ylabel("Slope B")
    axB.set_title(f"E = {energy} GeV")
    axB.set_ylim(0.5, 1.6)

    if row == len(energies) - 1:
        axB.set_xticks(x_pos)
        axB.set_xticklabels(x_labels, rotation=90, fontsize=7)
    else:
        axB.set_xticks(x_pos)
        axB.set_xticklabels([])

    axB.grid(True)

plt.suptitle(
    f"Scatter Plot Linear Fit Parameters — APA {apa_studied}",
    fontsize=16
)

plt.savefig(
    f"{output_folder}/scatter_fit_params_APA{apa_studied}.png",
    dpi=200
)

plt.close()

############################################################################################################
""" ENERGY RESOLUTION: scatter plot of fit parameters p0, p1, p2 for all pairs"""

fig4, axes4 = plt.subplots(3, 1,
                           figsize=(10, 12),
                           constrained_layout=True)

# --- Mappa coppia-canali → APA ---



apa_map = {}
for p in adiacent_channels_1:
    apa_map[pair_to_key(p)] = 1
for p in adiacent_channels_2:
    apa_map[pair_to_key(p)] = 2

x_labels = []
p0_vals, p0_errs = [], []
p1_vals, p1_errs = [], []
p2_vals, p2_errs = [], []

for r in all_results:

    key = pair_to_key(r['channel_pair'])
    apa = apa_map.get(key, None)

    if apa not in apa_studied:
        continue

    energy_res = r.get('energy_resolution', {}).get(strategy)
    if energy_res is None:
        continue

    fit = energy_res['fit_params']

    ch_name = (
        f"{r['channel_pair'][0]['end']}ch{r['channel_pair'][0]['ch']}/"
        f"{r['channel_pair'][1]['end']}ch{r['channel_pair'][1]['ch']}"
    )

    x_labels.append(ch_name)

    p0_vals.append(fit['p0'][0])
    p0_errs.append(fit['p0'][1])

    p1_vals.append(fit['p1'][0])
    p1_errs.append(fit['p1'][1])

    p2_vals.append(fit['p2'][0])
    p2_errs.append(fit['p2'][1])


x_pos = np.arange(len(x_labels))


# ======================
# Row 1 — p0
# ======================
ax0 = axes4[0]
ax0.errorbar(x_pos, p0_vals, yerr=p0_errs,
             fmt='o', markersize=4, capsize=3)

mean_p0 = np.nanmean(p0_vals)
std_p0  = np.nanstd(p0_vals)

ax0.axhline(mean_p0, linestyle='--',
            label=f"Mean = {mean_p0:.4f}\nStd = {std_p0:.4f}")

ax0.set_ylabel("p0")
ax0.set_xticks(x_pos)
ax0.set_xticklabels([])
ax0.legend()
ax0.grid(True)


# ======================
# Row 2 — p1
# ======================
ax1 = axes4[1]
ax1.errorbar(x_pos, p1_vals, yerr=p1_errs,
             fmt='o', markersize=4, capsize=3)

mean_p1 = np.nanmean(p1_vals)
std_p1  = np.nanstd(p1_vals)

ax1.axhline(mean_p1, linestyle='--',
            label=f"Mean = {mean_p1:.4f}\nStd = {std_p1:.4f}")

ax1.set_ylabel("p1")
ax1.set_xticks(x_pos)
ax1.set_xticklabels([])
ax1.legend()
ax1.grid(True)


# ======================
# Row 3 — p2
# ======================
ax2 = axes4[2]
ax2.errorbar(x_pos, p2_vals, yerr=p2_errs,
             fmt='o', markersize=4, capsize=3)

mean_p2 = np.nanmean(p2_vals)
std_p2  = np.nanstd(p2_vals)

ax2.axhline(mean_p2, linestyle='--',
            label=f"Mean = {mean_p2:.4f}\nStd = {std_p2:.4f}")

ax2.set_ylabel("p2")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(x_labels, rotation=90, fontsize=7)

ax2.set_ylim(0, 1)  # range richiesto

ax2.legend()
ax2.grid(True)


plt.suptitle(
    f"Energy Resolution Fit Parameters — APA {apa_studied}",
    fontsize=16
)

plt.savefig(
    f"{output_folder}/energy_resolution_params_summary_APA{apa_studied}.png",
    dpi=200
)


############################################################################################################
""" ENERGY RESOLUTION - STOCHASTIC TERM p1: heatmap """

n_rows = 10
n_cols = 3 * len(apa_studied) 
fig5, ax5 = plt.subplots(figsize=(10, 12), constrained_layout=True)

p1_grid = np.full((n_rows, n_cols), np.nan) 
labels_grid = np.full((n_rows, n_cols), '', dtype=object)  



all_channel_pairs = []
current_col = 0

if 1 in apa_studied:
    for row in range(n_rows):
        for col in range(3):
            idx = row*3 + col
            if idx < len(adiacent_channels_1):
                all_channel_pairs.append({
                    'apa': 1,
                    'ch_pair': adiacent_channels_1[idx],
                    'row': row,
                    'col': current_col + col
                })
    current_col += 3

if 2 in apa_studied:
    for row in range(n_rows):
        for col in range(3):
            idx = row*3 + col
            if idx < len(adiacent_channels_2):
                all_channel_pairs.append({
                    'apa': 2,
                    'ch_pair': adiacent_channels_2[idx],
                    'row': row,
                    'col': current_col + col
                })
    current_col += 3

for entry in all_channel_pairs:
    row = entry['row']
    col = entry['col']
    ch_pair = entry['ch_pair']

    ch_result = None
    for r in all_results:
        ch0 = r['channel_pair'][0]
        ch1 = r['channel_pair'][1]
        if (ch0['end'] == ch_pair[0]['end'] and ch0['ch'] == ch_pair[0]['ch'] and
            ch1['end'] == ch_pair[1]['end'] and ch1['ch'] == ch_pair[1]['ch']):
            ch_result = r
            break

    if ch_result is not None and strategy in ch_result['energy_resolution']:
        fit_info = ch_result['energy_resolution'][strategy]['fit_params']['p1']
        p1_val = fit_info[0]
        p1_err = fit_info[1]

        p1_grid[row, col] = p1_val
        labels_grid[row, col] = f"{ch_pair[0]['end']}ch{ch_pair[0]['ch']}/" \
                                f"{ch_pair[1]['end']}ch{ch_pair[1]['ch']}\n" \
                                f"{p1_val:.3f}±{p1_err:.3f}"

cmap = plt.cm.viridis
norm = mcolors.Normalize(vmin=np.nanmin(p1_grid), vmax=np.nanmax(p1_grid))

im = ax5.imshow(p1_grid, cmap=cmap, norm=norm)

for i in range(n_rows):
    for j in range(n_cols):
        if not np.isnan(p1_grid[i, j]):
            ax5.text(j, i, labels_grid[i, j],
                     ha='center', va='center', color='white', fontsize=8)

ax5.set_xticks([])
ax5.set_yticks([])

ax5.set_title(f"Heatmap p1 — APA {apa_studied}", fontsize=16)
plt.colorbar(im, ax=ax5, label="p1 value")

plt.savefig(f"{output_folder}/p1_heatmap_APA{apa_studied}.png", dpi=200)
plt.close()



############################################################################################################
""" SCATTER PLOT: p1 vs mean LY for all available paris """



ly_map = {}

for ch_list in [adiacent_channels_1, adiacent_channels_2]:
    for pair in ch_list:
        ch0, ch1 = pair
        if ('ly' in ch0 and not np.isnan(ch0['ly'])) and ('ly' in ch1 and not np.isnan(ch1['ly'])):
            ly_mean = (ch0['ly'] + ch1['ly'])/2
            ly_err = np.sqrt(ch0['ly_err']**2 + ch1['ly_err']**2)
            key = (ch0['end'], ch0['ch'], ch1['end'], ch1['ch'])
            ly_map[key] = (ly_mean, ly_err)

pair_info = []

for r in all_results:
    ch0, ch1 = r['channel_pair']
    key = (ch0['end'], ch0['ch'], ch1['end'], ch1['ch'])
    
    apa = None
    if key in [(pair[0]['end'], pair[0]['ch'], pair[1]['end'], pair[1]['ch']) for pair in adiacent_channels_1]:
        apa = 1
    elif key in [(pair[0]['end'], pair[0]['ch'], pair[1]['end'], pair[1]['ch']) for pair in adiacent_channels_2]:
        apa = 2
    if apa not in apa_studied:
        continue

    if key not in ly_map:
        continue
    ly_mean, ly_err = ly_map[key]

    if strategy not in r['energy_resolution']:
        continue
    
    p1_val, p1_err = r['energy_resolution'][strategy]['fit_params']['p1']

    if (
        p1_val is None or
        np.isnan(p1_val) or
        p1_val <= 0 or
        p1_err is None or
        np.isnan(p1_err) or
        p1_err <= 0
    ):
        continue

    pair_name = f"{ch0['end']}ch{ch0['ch']}-{ch1['end']}ch{ch1['ch']}"
    pair_info.append({
        'x': ly_mean,
        'ex': ly_err,
        'y': p1_val,
        'ey': p1_err,
        'label': pair_name
    })

if len(pair_info) == 0:
    print("Nessuna coppia da plottare!")
else:
    fig, ax = plt.subplots(figsize=(10,6))
    x_vals = [p['x'] for p in pair_info]
    y_vals = [p['y'] for p in pair_info]
    x_errs = [p['ex'] for p in pair_info]
    y_errs = [p['ey'] for p in pair_info]
    labels = [p['label'] for p in pair_info]

    colors = plt.cm.tab20(np.linspace(0,1,len(pair_info)))

    for i in range(len(pair_info)):
        ax.errorbar(x_vals[i], y_vals[i], xerr=x_errs[i], yerr=y_errs[i],
                    fmt='o', color=colors[i], capsize=3, label=labels[i])
        
    x_arr = np.array(x_vals)
    y_arr = np.array(y_vals)
    xerr_arr = np.array(x_errs)
    yerr_arr = np.array(y_errs)

    mask = (~np.isnan(x_arr)) & (~np.isnan(y_arr)) & \
        (~np.isnan(xerr_arr)) & (~np.isnan(yerr_arr))

    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    xerr_arr = xerr_arr[mask]
    yerr_arr = yerr_arr[mask]

    xerr_arr[xerr_arr <= 0] = np.median(xerr_arr[xerr_arr > 0])
    yerr_arr[yerr_arr <= 0] = np.median(yerr_arr[yerr_arr > 0])

    model = Model(inv_sqrt_model)
    data = RealData(x_arr, y_arr, sx=xerr_arr, sy=yerr_arr)
    ODR_instance = ODR(data, model, beta0=[np.mean(y_arr * np.sqrt(x_arr))])
    output = ODR_instance.run()

    A_fit = output.beta[0]
    A_err = output.sd_beta[0]
    
    chi2rid = output.res_var / (len(x_arr) - len(output.beta))
    r2_squared = r2_score(y_arr, inv_sqrt_model(output.beta, x_arr))

    x_fit = np.linspace(min(x_arr)*0.9, max(x_arr)*1.1, 300)
    y_fit = inv_sqrt_model([A_fit], x_fit)

    ax.plot(x_fit, y_fit,
            color='black',
            linewidth=2,
            label=rf"Fit $y=\frac{{A}}{{\sqrt{{x}}}}$"+"\n"+rf"$A = {fmt(A_fit,A_err)}$" +"\n" + rf"$R^2 = {r2_squared:.3f}$" +"\n" + rf"$\chi^2_{{red}} = {chi2rid:.3f}$")


    ax.set_xlabel(r"Mean LY [$N_{PE}$/GeV]")
    ax.set_ylabel("p1 parameter")
    ax.grid(True)
    ax.legend(fontsize=8, ncols=2)
    ax.set_title(f"Scatter plot p1 vs illumination — APA {apa_studied}")

    plt.tight_layout()
    plt.savefig(f"{output_folder}/p1_ly_scatter_plot_APA{apa_studied}.png", dpi=200)

    plt.close()
