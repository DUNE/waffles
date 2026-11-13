"""
Script version of calibration.ipynb.

Reads the runs from a runslist.txt (tab-separated text pasted from the shift
spreadsheet), and for each detector type (cathode/membrane):
  - loads the processed waveforms of the runs
  - removes saturated/bad waveforms
  - fits the charge histograms (SNR, gain, ...)
  - plots the persistence heatmaps
  - plots the average SPE waveforms
Plots are saved as .png (heatmaps, optional with --save-heatmaps, only as .html).
Everything is saved in <outdir>/YYYYMMDD_N/, together with a copy of the runslist.
If a folder with an identical runslist.txt already exists, it is reused.

Usage:
    python run_calibration.py --runslist runslist.txt --dettype cathode
"""
import argparse
import copy
import shutil
import sys
from datetime import datetime
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

from waffles.input_output.hdf5_structured import load_structured_waveformset
from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.UniqueChannel import UniqueChannel
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
from waffles.np02_utils.AutoMap import dict_module_to_uniqch, dict_uniqch_to_module, expand_modules
from waffles.np02_utils.PlotUtils import plot_detectors, fithist, select_spe, matplotlib_plot_WaveformSetGrid

# --- Fixed settings (same values as in calibration.ipynb) ----------------------
DEFAULT_DATADIR = "/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/"
ENDPOINT_BY_DETTYPE = {"cathode": 106, "membrane": 107}

# Bad waveform cut: any ADC in the first BAD_WF_NPOINTS points outside [min, max] is rejected
BAD_WF_NPOINTS = 500
BAD_WF_ADC_MAX = 16200
BAD_WF_ADC_MIN = 100

# Average SPE selection and amplitude search window
SPE_PRETRIGGER = 200
SPE_POSTTRIGGER = 400
SPE_THRESHOLD = 80
SPE_MAXIMUM_BEGIN = 250
SPE_MAXIMUM_END = 300

PLOT_WIDTH = 1300
PLOT_HEIGHT = 650
MATPLOTLIB_FIGSIZE = (16, 8)  # same as waffles matplotlib_plot_WaveformSetGrid default
CHARGE_HIST_XLABEL = "Charge [ADC*ns]"
CHARGE_HIST_YLABEL = "Counts"

# Results summary: modules outside these limits are flagged
SUMMARY_SNR_MIN = 2.5
SUMMARY_SNR_MAX = 10
SUMMARY_AVG_PE_MIN = 0.5

# Modules not used for calibration: not expected in the runslist
MODULES_NOT_EXPECTED = ["M1(1)", "M1(2)", "M2(1)", "M2(2)"]

# runslist.txt columns
COL_RUN = 0
COL_DATE = 1
COL_MODULES = -1

CSV_COLUMNS = ["date", "run", "endpoint", "channel", "module", "snr", "gain",
               "baseline_stddev", "spe_stddev", "avg_pe", "spe_amplitude"]


# --- runslist handling ---------------------------------------------------------
def parse_runslist(runslist: Path) -> tuple[str, dict[str, dict[int, list[str]]], dict[int, str]]:
    """
    Returns the date of the first run (YYYYMMDD), per detector type
    a dictionary run -> list of modules (e.g. {45420: ["C1(1)", "C1(2)"]})
    and a dictionary run -> date (dd/mm/yyyy).
    """
    available_modules = list(dict_module_to_uniqch.keys())
    run_to_module = {"cathode": {}, "membrane": {}}
    run_dates = {}
    date = None
    for line in runslist.read_text().splitlines():
        if not line.strip():
            continue
        cols = line.split("\t")
        run = int(cols[COL_RUN])
        run_dates[run] = cols[COL_DATE].strip()
        if date is None:
            date = datetime.strptime(cols[COL_DATE].strip(), "%d/%m/%Y").strftime("%Y%m%d")
        tokens = [t.strip() for t in cols[COL_MODULES].split(",") if t.strip()]
        for module in expand_modules(tokens, available_modules):
            if module not in dict_module_to_uniqch:
                print(f"WARNING: run {run}: unknown module '{module}', ignoring it.")
                continue
            dettype = "cathode" if module.startswith("C") else "membrane"
            run_to_module[dettype].setdefault(run, []).append(module)
    if date is None:
        raise ValueError(f"No runs found in {runslist}")
    return date, run_to_module, run_dates


def get_output_folder(outdir: Path, date: str, runslist: Path) -> Path:
    """Reuse a folder <date>_N with the same runslist.txt, otherwise create the next one."""
    content = runslist.read_text()
    counters = []
    for folder in outdir.glob(f"{date}_*"):
        suffix = folder.name.split("_")[-1]
        if not suffix.isdigit():
            continue
        counters.append(int(suffix))
        saved = folder / "runslist.txt"
        if saved.is_file() and saved.read_text() == content:
            print(f"Same runslist found, using {folder}")
            return folder
    folder = outdir / f"{date}_{max(counters, default=0) + 1}"
    folder.mkdir(parents=True)
    shutil.copy(runslist, folder / "runslist.txt")
    print(f"Created {folder}")
    return folder


# --- Loading -------------------------------------------------------------------
def merged_file(run, dettype, datadir) -> str:
    return f"{datadir}/processed/run{run:0d}_{dettype}/processed_merged_run{run:06d}_structured_{dettype}.hdf5"


def split_files(run, dettype, datadir) -> list[str]:
    return glob(f"{datadir}/processed/run{run:06d}_{dettype}/processed_*_run{run:06d}_*_{dettype}.hdf5")


def open_processed(run, dettype, datadir, channels=None, endpoints=None, nwaveforms=None, mergefiles=False, verbose=True):
    """
    Open the processed waveform set for a given run and detector type.
    """
    try:
        wfset = load_structured_waveformset(
            merged_file(run, dettype, datadir),
            max_to_load=nwaveforms,
            channels_filter=channels,
            endpoint_filter=endpoints
        )
    except:
        files = split_files(run, dettype, datadir)
        if verbose:
            print("List of files found:")
            print(files)
        if not mergefiles or len(files) == 1:
            wfset = load_structured_waveformset(files[0], max_to_load=nwaveforms, channels_filter=channels, endpoint_filter=endpoints, verbose=verbose)
        else:
            print("Merging files..")
            wfset = load_structured_waveformset(files[0], max_to_load=nwaveforms, channels_filter=channels, endpoint_filter=endpoints, verbose=verbose)
            for f in files[1:]:
                tmpwf = load_structured_waveformset(f, max_to_load=nwaveforms, channels_filter=channels, endpoint_filter=endpoints, verbose=verbose)
                wfset.merge(copy.deepcopy(tmpwf))
    return wfset


def create_wfset(run_to_unich, dettype, endpoint, datadir, nwaveforms, mergefiles) -> WaveformSet:
    wfset_full = None
    for run, channels in run_to_unich.items():
        wfset = open_processed(run, dettype, datadir, channels, [endpoint], nwaveforms=nwaveforms, verbose=True, mergefiles=mergefiles)
        if wfset_full is None:
            wfset_full = wfset
        else:
            wfset_full.merge(wfset)
        print(f"Loaded run {run}")
    return wfset_full


def filter_bad_waveforms(waveform: Waveform) -> bool:
    adcs = waveform.adcs[0:BAD_WF_NPOINTS]
    if np.any(adcs > BAD_WF_ADC_MAX):
        return False
    if np.any(adcs < BAD_WF_ADC_MIN):
        return False
    return True


# --- Average SPE ---------------------------------------------------------------
def plot_average_spe_and_store(wfset: WaveformSet, spe_amplitudes: dict, pretrigger, posttrigger, threshold, maximum_begin, maximum_end):
    """Same as waffles plot_average_spe, but stores the amplitude in spe_amplitudes[module]."""
    import matplotlib.pyplot as plt
    module = dict_uniqch_to_module[str(UniqueChannel(wfset.waveforms[0].endpoint, wfset.waveforms[0].channel))]
    try:
        wfset_spe = WaveformSet.from_filtered_WaveformSet(wfset, select_spe, pretrigger=pretrigger, posttrigger=posttrigger, threshold=threshold, show_progress=False)
    except:
        spe_amplitudes[module] = np.nan
        return

    spewf = np.array([wf.adcs - wf.analyses['std'].result['baseline'] for wf in wfset_spe.waveforms])
    spewf = np.mean(spewf, axis=0)
    plt.plot(spewf, label=module)
    plt.xlabel('Time [ticks]')
    plt.ylabel('Amplitude [ADCs]')
    plt.legend()
    maximum = np.max(spewf[maximum_begin:maximum_end])
    spe_amplitudes[module] = maximum


# --- Results -------------------------------------------------------------------
def average_pe(wfset: WaveformSet, gain: float, variable="integral") -> float:
    """Same computation as in waffles fithist (average charge within 2-98% quantiles / gain)."""
    charges = np.array([wf.analyses["std"].result[variable] for wf in wfset.waveforms if wf.analyses["std"].result[variable] is not np.nan])
    charges = charges[(charges >= np.quantile(charges, 0.02)) & (charges <= np.quantile(charges, 0.98))]
    return np.mean(charges) / gain if len(charges) > 0 else 0


def collect_results(wfset: WaveformSet, modules: list[str], spe_amplitudes: dict) -> list[dict]:
    grid = ChannelWsGrid.clusterize_waveform_set(wfset)
    rows = []
    for module in modules:
        uch = dict_module_to_uniqch[module]
        wfs = grid.get(uch.endpoint, {}).get(uch.channel)
        if wfs is None:
            continue
        row = dict.fromkeys(CSV_COLUMNS, np.nan)
        row.update(run=list(wfs.runs)[0], endpoint=uch.endpoint, channel=uch.channel, module=module,
                   spe_amplitude=spe_amplitudes.get(module, np.nan))
        result = wfs.waveforms[0].analyses["std"].result
        if "snr" in result:  # only set by fithist when the fit succeeded
            row.update(snr=result["snr"], gain=result["gain"],
                       baseline_stddev=result["gain"] / result["snr"],
                       spe_stddev=result["spe_stddev"],
                       avg_pe=average_pe(wfs, result["gain"]))
        rows.append(row)
    return rows


def summarize_results(rows: list[dict], dettype: str) -> list[str]:
    """One line per flagged module (fit failed, SNR or avg p.e. out of limits, no average SPE)."""
    lines = []
    for row in rows:
        problems = []
        if np.isnan(row["snr"]) and row["gain"] == 0:
            problems.append(f"SNR out of [{SUMMARY_SNR_MIN}, {SUMMARY_SNR_MAX}]")
        elif np.isnan(row["snr"]):
            problems.append("fit failed")
        elif row["snr"] < SUMMARY_SNR_MIN:
            problems.append(f"SNR = {row['snr']:.2f} (< {SUMMARY_SNR_MIN})")
        elif row["snr"] > SUMMARY_SNR_MAX:
            problems.append(f"SNR = {row['snr']:.2f} (> {SUMMARY_SNR_MAX})")
        if not np.isnan(row["snr"]) and row["avg_pe"] < SUMMARY_AVG_PE_MIN:
            problems.append(f"avg p.e. = {row['avg_pe']:.2f} (< {SUMMARY_AVG_PE_MIN})")
        if np.isnan(row["spe_amplitude"]) and not row["gain"] == 0:
            problems.append("no average SPE")
        if problems:
            lines.append(f"  {row['module']:<6} run {row['run']}  {', '.join(problems)}")
    header = f"===== Summary {dettype}: {len(lines)} of {len(rows)} modules flagged ====="
    return [header] + lines


def mask_snr_out_of_range(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with SNR out of [SUMMARY_SNR_MIN, SUMMARY_SNR_MAX]: snr and spe_amplitude -> NaN, other values -> 0."""
    out = (df["snr"] < SUMMARY_SNR_MIN) | (df["snr"] > SUMMARY_SNR_MAX)
    df.loc[out, ["gain", "baseline_stddev", "spe_stddev", "avg_pe"]] = 0
    df.loc[out, ["snr", "spe_amplitude"]] = np.nan
    return df


def update_csv(rows: list[dict], filename: Path, modules: list[str], run_dates: dict[int, str]) -> pd.DataFrame:
    """
    Create the csv, or update it: rows of modules already in the file are replaced,
    new modules are added and the others are kept. Rows are ordered as in modules.
    The SNR out of range masking is applied only to the csv: the returned df keeps
    the values (used in the summary).
    """
    new = pd.DataFrame(rows, columns=CSV_COLUMNS)
    new["date"] = new["run"].map(run_dates)
    if filename.is_file():
        old = pd.read_csv(filename)
        old = old[~old["module"].isin(new["module"])]
        df = pd.concat([old, new], ignore_index=True)
        print(f"Updated {filename} ({len(new)} modules replaced/added, {len(old)} kept)")
    else:
        df = new
        print(f"Saved {filename}")
    order = {m: i for i, m in enumerate(modules)}
    df = df.sort_values("module", key=lambda col: col.map(order)).reset_index(drop=True)
    mask_snr_out_of_range(df.copy()).to_csv(filename, index=False, float_format="%.2f", na_rep="NaN")
    return df


# --- Safeguards ----------------------------------------------------------------
def check_missing_files(run_to_module: dict[str, dict[int, list[str]]], dettypes: list[str], datadir) -> dict[str, list[int]]:
    """Returns, per detector type, the runs whose processed files are not found."""
    missing = {}
    for dettype in dettypes:
        runs = [run for run in run_to_module[dettype]
                if not Path(merged_file(run, dettype, datadir)).is_file() and not split_files(run, dettype, datadir)]
        if runs:
            missing[dettype] = runs
    return missing


def check_missing_modules(run_to_module: dict[str, dict[int, list[str]]], dettypes: list[str]) -> list[str]:
    """Returns the expected modules (from detector_groups) that are not in any run."""
    missing = []
    for dettype in dettypes:
        listed = {m for modules in run_to_module[dettype].values() for m in modules}
        group_low, group_high = detector_groups(dettype)
        missing += [m for m in group_low + group_high if m not in listed and m not in MODULES_NOT_EXPECTED]
    return missing


def check_duplicated_modules(run_to_module: dict[str, dict[int, list[str]]], dettypes: list[str]) -> list[str]:
    """Returns a message for each module listed more than once (fithist needs one run per module)."""
    duplicated = []
    for dettype in dettypes:
        runs_per_module = {}
        for run, modules in run_to_module[dettype].items():
            for m in modules:
                runs_per_module.setdefault(m, []).append(run)
        duplicated += [f"{m} is listed in runs {', '.join(map(str, runs))}"
                       for m, runs in runs_per_module.items() if len(runs) > 1]
    return duplicated


def ask_to_continue(question: str):
    if input(f"{question} [y/N] ").strip().lower() != "y":
        print("Stopped.")
        sys.exit(1)


# --- Main analysis -------------------------------------------------------------
def detector_groups(dettype: str) -> tuple[list[str], list[str]]:
    """Modules 1-4 (low) and 5-8 (high) of the cathode or membrane."""
    dletter = dettype.upper()[0]  # C or M
    group_low = [f"{dletter}{detnum}({chnum})" for detnum in range(1, 5) for chnum in range(1, 3)]
    group_high = [f"{dletter}{detnum}({chnum})" for detnum in range(5, 9) for chnum in range(1, 3)]
    return group_low, group_high


def process_dettype(dettype: str, run_to_module: dict[int, list[str]], run_dates: dict[int, str], folder: Path, args):
    endpoint = ENDPOINT_BY_DETTYPE[dettype]
    group_low, group_high = detector_groups(dettype)
    group_all = group_low + group_high

    run_to_unich = {r: [dict_module_to_uniqch[m].channel for m in modules] for r, modules in run_to_module.items()}

    wfset_full = create_wfset(run_to_unich, dettype, endpoint, args.datadir, args.nwaveforms, args.mergefiles)
    print("Loaded all waveforms:", len(wfset_full.waveforms))
    wfset = WaveformSet.from_filtered_WaveformSet(wfset_full, filter_bad_waveforms, show_progress=True)
    del wfset_full
    print(f"After removing saturated/bad waveforms: {len(wfset.waveforms)}")

    # Charge histograms + fit
    print(f"Creating and fitting the charge histograms of {dettype}")
    histargs = dict(dofit=True, fitmultigauss=True, configyaml=args.configyaml, print_results=False)
    figs = plot_detectors(wfset, group_all, plot_function=fithist, wf_func=histargs,
                          width=PLOT_WIDTH, height=PLOT_HEIGHT, return_fig=True)
    hist_info = {row["module"]: row for row in collect_results(wfset, group_all, {})}
    for fig, rows, cols, *_ in figs:
        plotly_lines_to_png(fig, rows, cols, folder / f"charge_hist_{dettype}.png", hist_info)
        if args.show:
            fig.show()

    if args.save_heatmaps:
        save_heatmaps(wfset, dettype, group_low, group_high, folder, args.show)

    # Average SPE
    spe_amplitudes = {}
    func_params = dict(
        spe_amplitudes=spe_amplitudes,
        pretrigger=SPE_PRETRIGGER,
        posttrigger=SPE_POSTTRIGGER,
        threshold=SPE_THRESHOLD,
        maximum_begin=SPE_MAXIMUM_BEGIN,
        maximum_end=SPE_MAXIMUM_END,
    )
    fig, _ = matplotlib_plot_WaveformSetGrid(wfset, plot_function=plot_average_spe_and_store, detector=group_all, func_params=func_params)
    fig.tight_layout()
    fig.savefig(folder / f"average_spe_{dettype}.png", bbox_inches="tight")
    print(f"Saved {folder / f'average_spe_{dettype}.png'}")
    if args.show:
        import matplotlib.pyplot as plt
        plt.show()

    save_results(wfset, group_all, spe_amplitudes, run_dates, dettype, folder)


def save_results(wfset: WaveformSet, modules: list[str], spe_amplitudes: dict, run_dates: dict[int, str], dettype: str, folder: Path):
    """Create/update and print the results csv, then print and write the summary (of the whole csv)."""
    csvfile = folder / f"calibration_results_{dettype}.csv"
    df = update_csv(collect_results(wfset, modules, spe_amplitudes), csvfile, modules, run_dates)
    summary = "\n".join(summarize_results(df.to_dict("records"), dettype))
    (folder / f"summary_{dettype}.txt").write_text(summary + "\n")
    print_results(dettype, folder)


def print_results(dettype: str, folder: Path):
    """Print the saved results csv and summary of dettype."""
    print(f"===== Results {dettype} =====")
    print((folder / f"calibration_results_{dettype}.csv").read_text(), end="")
    print((folder / f"summary_{dettype}.txt").read_text(), end="")


def save_heatmaps(wfset: WaveformSet, dettype: str, group_low: list[str], group_high: list[str], folder: Path, show: bool):
    """Persistence heatmaps of modules 1-4 and 5-8, saved as html."""
    print(f"Creating the heatmaps of {dettype}...")
    argsheat = dict(
        mode="heatmap",
        analysis_label="std",
        adc_range_above_baseline=500,
        adc_range_below_baseline=-150,
        adc_bins=125,
        time_bins=wfset.points_per_wf // 2,
        filtering=4,
        share_y_scale=False,
        share_x_scale=True,
        wfs_per_axes=5000,
        zlog=True,
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        return_fig=True,
    )
    for label, group in (("low", group_low), ("high", group_high)):
        figs = plot_detectors(wfset, group, **argsheat)
        for fig, *_ in figs:
            save_plotly(fig, folder / f"heatmap_{dettype}_{label}", show)


def save_plotly(fig, basename: Path, show: bool):
    fig.write_html(f"{basename}.html")
    print(f"Saved {basename}.html")
    if show:
        fig.show()


def plotly_lines_to_png(fig, rows: int, cols: int, filename: Path, info: dict):
    """
    Redraw a plotly subplot grid made only of line traces (e.g. charge histograms + fits)
    with matplotlib, so it can be saved as png without kaleido/Chrome.
    info[module] gives the run, endpoint, channel and snr used in the legends.
    """
    import matplotlib.pyplot as plt
    mfig, axs = plt.subplots(rows, cols, figsize=MATPLOTLIB_FIGSIZE)
    axs = np.array(axs).reshape(-1)
    module_of_ax = {}
    for trace in fig.data:
        ax = axs[int(trace.xaxis[1:] or 1) - 1]  # plotly axes 'x', 'x2', ... are row-major
        if trace.name is None:  # empty placeholder added by waffles for modules without data
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        if trace.line.shape == "hv":  # histogram: x are the bin edges (len(y) + 1)
            module = trace.name.split(";")[0]  # trace name is "module; snr=..."
            module_of_ax[ax] = module
            ax.stairs(trace.y, trace.x, color="black", linewidth=1, label=f"Data (run: {info[module]['run']})")
        elif trace.name.endswith("(MultiFit)"):
            snr = info[module_of_ax[ax]]["snr"]
            ax.plot(trace.x, trace.y, color="red", linewidth=1, label=f"Fit (SNR={snr:.2f})")
        else:  # single gaussian "(Fit i)"
            ax.plot(trace.x, trace.y, color="grey", linestyle="--", linewidth=1, label="_nolegend_")
    for ax, module in module_of_ax.items():
        row = info[module]
        ax.legend(title=f"{row['endpoint']}-{row['channel']}: {module}", fontsize=7, title_fontsize=7)
        ax.set_xlabel(CHARGE_HIST_XLABEL, fontsize=8)
        ax.set_ylabel(CHARGE_HIST_YLABEL, fontsize=8)
    mfig.tight_layout()
    mfig.savefig(filename)
    plt.close(mfig)
    print(f"Saved {filename}")


def main():
    parser = argparse.ArgumentParser(description="NP02 PDS calibration (SNR, gain, SPE amplitude) from a runslist.txt")
    parser.add_argument("--runslist", type=Path, default=Path("./runslist.txt"), help="Tab-separated runs list pasted from the spreadsheet")
    parser.add_argument("--outdir", type=Path, default=Path("./results"), help="Parent folder of the YYYYMMDD_N output folders")
    parser.add_argument("--dettype", choices=["cathode", "membrane", "both"], default="both")
    parser.add_argument("--datadir", default=DEFAULT_DATADIR, help="Folder containing processed/")
    parser.add_argument("--configyaml", default="ch_snr_parameters.yaml", help="Fit parameters per channel")
    parser.add_argument("--nwaveforms", type=int, default=None, help="Maximum number of waveforms to load per run (for quick tests)")
    parser.add_argument("--mergefiles", action="store_true", help="Merge all files of a run (default: only the first file)")
    parser.add_argument("--save-heatmaps", action="store_true", help="Also save the persistence heatmaps as html (slow, large files)")
    parser.add_argument("--show", action="store_true", help="Also show the plots interactively")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    date, run_to_module, run_dates = parse_runslist(args.runslist)

    dettypes = ["cathode", "membrane"] if args.dettype == "both" else [args.dettype]
    for dettype in dettypes:
        if not run_to_module[dettype]:
            print(f"No {dettype} modules in {args.runslist}, skipping.")
    dettypes = [d for d in dettypes if run_to_module[d]]

    missing_files = check_missing_files(run_to_module, dettypes, args.datadir)
    if missing_files:
        print(f"ERROR: processed files not found in {args.datadir}/processed/")
        print("The following runs are missing to process:")
        for dettype, runs in missing_files.items():
            print(f"{dettype.capitalize()}:")
            print(",".join(map(str, sorted(runs))))
        all_runs = set().union(*missing_files.values())
        print("All missing runs for rucio:")
        print(",".join(map(str, sorted(all_runs))))
        sys.exit(1)

    duplicated_modules = check_duplicated_modules(run_to_module, dettypes)
    if duplicated_modules:
        print("ERROR: each module must be in only one run:\n  " + "\n  ".join(duplicated_modules))
        sys.exit(1)

    missing_modules = check_missing_modules(run_to_module, dettypes)
    if missing_modules:
        print(f"WARNING: these modules are not in {args.runslist}: {', '.join(missing_modules)}")
        ask_to_continue("Continue anyway?")

    folder = get_output_folder(args.outdir, date, args.runslist)
    existing = [f.name for d in dettypes if (f := folder / f"calibration_results_{d}.csv").is_file()]
    if existing:
        print(f"WARNING: {folder} already has results ({', '.join(existing)}), the rows of the processed modules will be replaced.")
        print("         To keep them, run again with a different --outdir.")
        ask_to_continue("Overwrite?")
    for dettype in dettypes:
        print(f"\n===== Processing {dettype} =====")
        process_dettype(dettype, run_to_module[dettype], run_dates, folder, args)

    if len(dettypes) > 1:
        print("\n\n\n===== Final results (all detector types) =====")
        for dettype in dettypes:
            print_results(dettype, folder)


if __name__ == "__main__":
    main()
