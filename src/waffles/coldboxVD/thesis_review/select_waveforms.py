"""Diagnostic LED selection using noise measured in the same spybuffer file."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from waffles.coldboxVD.utils.spybuffer_reader import create_waveform_set_from_spybuffer
from waffles.data_classes.IPDict import IPDict
from waffles.data_classes.StoreWfAna import StoreWfAna
from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.utils.integral.WindowIntegrator import WindowIntegrator
from waffles.coldboxVD.thesis_review.calibration_helpers import (
    locate_led, optimize_window, fit_calibration, fitted_curve)
from waffles.coldboxVD.thesis_review.spe_metrics import estimate_spe_metrics
from waffles.coldboxVD.thesis_review.scan_report import create_report, create_debug_report

HERE = Path(__file__).resolve().parent


def keep_record(waveform: Waveform, mask: np.ndarray) -> bool:
    return bool(mask[waveform.record_number])


def save_figure(fig, destination):
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def mark_windows(ax, args):
    ax.axvspan(args.baseline_start, args.baseline_stop, color='tab:green', alpha=.12)
    if args.int_low is not None:
        ax.axvspan(args.int_low, args.int_up, color='tab:red', alpha=.15)
    ax.set_xlabel('Sample index (16 ns/sample)')
    ax.set_ylabel('ADC relative to baseline')


def stage_plot(centered, mask, name, args, out):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return
    rng = np.random.default_rng(2025)
    sample = rng.choice(indices, min(len(indices), args.plot_waveforms), replace=False)
    subset = centered[sample]
    # Quantiles affect only the display; all waveforms contribute to the CSV.
    lo, hi = np.quantile(subset, [.001, .999])
    pad = max((hi-lo)*.1, 2.)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    times = np.arange(args.length)
    axes[0].hist2d(np.tile(times, len(subset)), subset.ravel(),
                   bins=[args.length, 180], range=[[0, args.length], [lo-pad, hi+pad]],
                   norm=LogNorm(), cmap='viridis')
    axes[0].set_title(f'{name}: {len(indices)} waveforms; {len(sample)} shown')
    for row in subset[:30]:
        axes[1].plot(times, row, alpha=.2, linewidth=.6, color='tab:blue')
    axes[1].plot(times, centered[indices].mean(axis=0), color='black', label='Mean of all in stage')
    axes[1].legend()
    for ax in axes:
        mark_windows(ax, args)
    save_figure(fig, out / f'{name}.png')


def analyse_file(path, args, relative_parent=Path()):
    args = argparse.Namespace(**vars(args))  # Effective windows are local to each file.
    channel = int(path.stem.removeprefix('channel_'))
    size = path.stat().st_size
    if not size or size % (2*args.length):
        raise ValueError(f'{path}: empty file or incomplete uint16 waveform')
    available = size//(2*args.length)
    if args.expected_waveforms is not None and available != args.expected_waveforms:
        raise ValueError(f'{path}: {available} waveforms, expected {args.expected_waveforms}; check the transfer')
    out = args.output / relative_parent / path.stem
    out.mkdir(parents=True, exist_ok=True)
    wfset = create_waveform_set_from_spybuffer(
        filename=str(path), WFs=args.max_waveforms, length=args.length, config_channel=channel)
    raw = np.stack([wf.adcs for wf in wfset.waveforms]).astype(float)
    discovery, discovery_profile = locate_led(raw, args.invert)
    args.invert = discovery['invert']
    if args.baseline_stop is None:
        args.baseline_stop = discovery['baseline_stop']
    if not 0 <= args.baseline_start < args.baseline_stop < discovery['peak']:
        raise ValueError('Baseline window must precede the detected LED peak')
    pre = raw[:, args.baseline_start:args.baseline_stop]
    baseline = (np.mean(pre, axis=1) if args.baseline_estimator == 'mean'
                else np.median(pre, axis=1))
    centered = raw - baseline[:, None]
    # Unclipped standard deviation: clipping would underestimate the actual noise.
    rms = np.std(pre, axis=1)
    noise = float(np.median(rms))
    spread = float(1.4826*np.median(np.abs(rms-noise)))
    if not np.isfinite(noise) or noise <= 0:
        raise ValueError(f'{path}: cannot estimate a positive noise scale')
    rms_limit = max(noise + args.rms_mad*spread, args.rms_floor*noise)
    excursion_limit = args.excursion_sigma*noise
    pre_peak = np.max(np.abs(centered[:, args.baseline_start:args.baseline_stop]), axis=1)
    pass_rms = rms <= rms_limit
    pass_excursion = pre_peak <= excursion_limit
    # Keep a full-record flag for diagnostics; reject rails only near the LED.
    any_rail = (raw.min(axis=1) <= 0) | (raw.max(axis=1) >= args.adc_max)
    led_region = raw[:, max(0, discovery['edge']-12):min(args.length, discovery['peak']+161)]
    pass_rails = (led_region.min(axis=1) > 0) & (led_region.max(axis=1) < args.adc_max)
    # Exactly constant preimpulses are excluded from the noise-quality stage.
    mask0 = np.ones(len(raw), dtype=bool)
    mask1 = pass_rms & (rms > 0)
    mask2 = mask1 & pass_excursion
    mask3 = mask2 & pass_rails
    stages = [('00_original', mask0), ('01_rms', mask1),
              ('02_preimpulse', mask2), ('03_selected', mask3)]
    # Store the per-waveform baseline in waffles without modifying the raw ADCs.
    for wf, value in zip(wfset.waveforms, baseline):
        wf.analyse('baseline', StoreWfAna, IPDict({'baseline': float(value)}), overwrite=True)
    scans, validation_fit = [], None
    if args.int_low is None:
        try:
            args.int_low, args.int_up, scans, validation_fit, _ = optimize_window(
                wfset, centered, mask3, discovery, args.invert, args.max_peaks, out)
        except ValueError:
            for name, mask in stages:
                stage_plot(centered, mask, name, args, out)
            stage_plot(centered, ~mask3, '04_rejected', args, out)
            (out/'selection_only.json').write_text(json.dumps({
                'noise_adc': noise, 'rms_limit_adc': rms_limit,
                'excursion_limit_adc': excursion_limit, 'led_discovery': discovery,
                'counts': {name: int(mask.sum()) for name, mask in stages}}, indent=2)+'\n')
            raise
    if not args.baseline_stop <= args.int_low < args.int_up < args.length:
        raise ValueError('Inconsistent effective baseline/integration windows')
    training_fit = next((row for row in scans
                         if row['int_low'] == args.int_low
                         and row['int_up'] == args.int_up
                         and row['status'] == 'ok'), None)
    wfset.analyse('charge', WindowIntegrator,
                 IPDict({'baseline_analysis': 'baseline', 'inversion': args.invert,
                         'int_ll': args.int_low, 'int_ul': args.int_up,
                         'amp_ll': args.int_low, 'amp_ul': args.int_up}),
                 checks_kwargs={'points_no': args.length}, overwrite=True)
    charges = np.array([wf.analyses['charge'].result['integral'] for wf in wfset.waveforms])
    unselected_fit, _ = fit_calibration(wfset, args.max_peaks)
    if mask3.any():
        selected = WaveformSet.from_filtered_WaveformSet(wfset, keep_record, mask=mask3)
        fit_result, histogram = fit_calibration(selected, args.max_peaks)
        if validation_fit is not None:
            validation_ok = validation_fit['status'] == 'ok' and training_fit is not None
            if validation_ok and 'gain_adc_ns' in fit_result:
                gain_pull = abs(training_fit['gain_adc_ns']-validation_fit['gain_adc_ns'])/np.hypot(
                    training_fit['gain_error_adc_ns'], validation_fit['gain_error_adc_ns'])
                validation_ok = bool(gain_pull < 5)
                fit_result['training_validation_gain_pull'] = float(gain_pull)
            fit_result['window_validation_ok'] = bool(validation_ok)
            if not validation_ok and fit_result['status'] == 'ok':
                fit_result['status'] = 'review_required'
        print(f'{path.name}: selected {len(selected.waveforms)}/{len(raw)}')
    else:
        fit_result, histogram = {'status': 'no_selected_waveforms'}, None
        print(f'{path.name}: no waveforms selected; inspect diagnostics')
    spe_metrics, spe_mean_waveform = estimate_spe_metrics(
        centered[mask3], charges[mask3], fit_result, args.int_low, args.int_up,
        invert=args.invert) if mask3.any() else ({'status': 'no_selected_waveforms'}, None)
    (out / 'spe_metrics.json').write_text(json.dumps(spe_metrics, indent=2)+'\n')
    if spe_mean_waveform is not None:
        fig, ax = plt.subplots(figsize=(10, 4))
        signed_mean = (-1 if args.invert else 1)*spe_mean_waveform
        ax.plot(np.arange(args.length), signed_mean, label='Mean of 1-PE candidates')
        ax.axvspan(args.int_low, args.int_up, color='tab:red', alpha=.12,
                   label='Charge integration window')
        if 'peak_sample' in spe_metrics:
            ax.plot(spe_metrics['peak_sample'],
                    signed_mean[spe_metrics['peak_sample']], 'o', label='SPE peak')
        ax.set(xlim=(max(0, args.int_low-40), min(args.length-1, args.int_up+80)),
               xlabel='Sample index (16 ns/sample)', ylabel='ADC relative to baseline',
               title=f"SPE amplitude: {spe_metrics.get('status', 'unknown')}")
        ax.legend()
        save_figure(fig, out / '13_spe_waveform.png')
    with (out / 'waveform_metrics.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['record', 'baseline_adc', 'pre_rms_adc', 'pre_max_abs_adc',
                         'pass_rms', 'pass_preimpulse', 'pass_led_adc_rails', 'any_record_adc_rail',
                         'selected', 'charge_adc_ns'])
        for i in range(len(raw)):
            writer.writerow([i, baseline[i], rms[i], pre_peak[i], bool(mask1[i]),
                             bool(pass_excursion[i]), bool(pass_rails[i]), bool(any_rail[i]),
                             bool(mask3[i]), charges[i]])
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    module = next((part for part in path.parts if part in {'M1', 'M2', 'M3', 'M4', 'M5', 'M6'}), None)
    bias_folder = next((part for part in path.parts if 'DVbias_' in part), None)
    vgain = int(path.parent.name.removeprefix('vgain_')) if path.parent.name.startswith('vgain_') else None
    summary = {'input': str(path.resolve()), 'output_folder': str(out.resolve()),
               'channel': channel, 'module': module,
               'bias_folder': bias_folder, 'vgain': vgain, 'settings': settings,
               'available_waveforms': available, 'loaded_waveforms': len(raw),
               'noise_adc': noise, 'rms_mad_adc': spread, 'rms_limit_adc': rms_limit,
               'pre_excursion_limit_adc': excursion_limit,
               'counts': {name: int(mask.sum()) for name, mask in stages},
               'selected_fraction': float(mask3.mean()), 'charge_unit': 'ADC ns',
               'led_discovery': discovery,
               'window_optimization_training_fit': training_fit,
               'independent_validation_fit': validation_fit,
               'full_selected_sample_fit': fit_result,
               'spe_metrics': spe_metrics,
               'unselected_sample_fit': unselected_fit,
               'saturation_check_window': [max(0, discovery['edge']-12),
                                           min(args.length, discovery['peak']+161)],
               'integration': 'waffles WindowIntegrator; trapezoidal; endpoints inclusive'}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (out / 'fit_result.json').write_text(json.dumps(fit_result, indent=2)+'\n')
    (out / 'unselected_fit.json').write_text(json.dumps(unselected_fit, indent=2)+'\n')
    (out / 'integration_scan.json').write_text(json.dumps(scans, indent=2)+'\n')
    for name, mask in stages:
        stage_plot(centered, mask, name, args, out)
    stage_plot(centered, ~mask3, '04_rejected', args, out)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, values, limit, label in zip(axes, [rms, pre_peak], [rms_limit, excursion_limit],
                                      ['Preimpulse RMS [ADC]', 'Preimpulse max absolute excursion [ADC]']):
        upper = max(np.quantile(values, .995), limit*1.2)
        ax.hist(values, bins=100, range=(0, upper), histtype='step', label='Original')
        ax.hist(values[mask3], bins=100, range=(0, upper), histtype='step', label='Selected')
        ax.axvline(limit, color='red', label='Cut')
        ax.set(xlabel=label, ylabel='Waveforms', yscale='log')
        ax.legend()
    save_figure(fig, out / '05_noise_diagnostics.png')
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([name.split('_', 1)[1] for name, _ in stages], [mask.sum() for _, mask in stages])
    ax.set_ylabel('Waveforms retained (cumulative)')
    save_figure(fig, out / '06_selection_counts.png')
    # Common edges allow before/after comparisons; record all charges, including tails, in CSV.
    lo, hi = np.quantile(charges, [.001, .999])
    if lo == hi:
        lo, hi = lo-1, hi+1
    edges = np.linspace(lo, hi, args.bins+1)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax in axes:
        for name, mask in stages:
            ax.hist(charges[mask], bins=edges, histtype='step', label=name)
        ax.set_ylabel('Waveforms / bin')
        ax.legend()
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Charge [ADC ns], trapezoidal integral')
    save_figure(fig, out / '07_charge_histogram.png')
    if mask3.any():
        # A separate zoom keeps rare large pulses from hiding the PE structure.
        zlo, zhi = np.quantile(charges[mask3], [.01, .98])
        if zlo == zhi:
            zlo, zhi = zlo-1, zhi+1
        fig, ax = plt.subplots(figsize=(10, 4))
        zoom_edges = np.linspace(zlo, zhi, args.bins+1)
        ax.hist(charges, bins=zoom_edges, histtype='step', label='Original')
        ax.hist(charges[mask3], bins=zoom_edges, histtype='step', label='Selected')
        ax.set(xlabel='Charge [ADC ns]', ylabel='Waveforms / bin',
               title='Charge zoom: selected 1%–98% quantiles (display only)')
        ax.legend()
        save_figure(fig, out / '08_charge_zoom.png')
        fig, ax = plt.subplots(figsize=(10, 4))
        t = np.arange(args.length)
        ax.plot(t, centered.mean(axis=0), label='Original mean')
        ax.plot(t, centered[mask3].mean(axis=0), label='Selected mean')
        ax.set_xlim(max(0, args.int_low-40), min(args.length-1, args.int_up+80))
        mark_windows(ax, args)
        ax.legend()
        save_figure(fig, out / '09_pulse_zoom.png')
    if histogram is not None:
        x, model = fitted_curve(histogram)
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        errors = np.sqrt(np.maximum(histogram.counts, 1))
        for ax in axes[:2]:
            ax.errorbar(x, histogram.counts, yerr=errors, fmt='.', markersize=3, label='Selected')
            if model is not None:
                ax.plot(x, model, label='waffles multigauss_iminuit')
                mm = histogram.iminuit
                for i in range(len(mm.parameters)-4):
                    ax.axvline(mm.values['mean_baseline']+i*mm.values['gain'], alpha=.2, color='black')
            ax.legend()
            ax.set_ylabel('Waveforms / bin')
        axes[1].set_yscale('log')
        if model is not None:
            axes[2].plot(x, (histogram.counts-model)/errors, '.')
        axes[2].axhline(0, color='black')
        axes[2].set(xlabel='Charge [ADC ns]', ylabel='Pull')
        title = f"Fit: {fit_result['status']}"
        if 'snr' in fit_result:
            title += f"; SNR={fit_result['snr']:.3g} ± {fit_result['snr_error']:.2g}; chi2/ndof={fit_result['chi2_ndof']:.2f}"
        axes[0].set_title(title)
        save_figure(fig, out / '10_multigaussian_fit.png')
    if scans:
        fig, ax = plt.subplots(figsize=(9, 4))
        valid = [s for s in scans if 'snr' in s]
        ax.errorbar([s['int_up'] for s in valid], [s['snr'] for s in valid],
                    yerr=[s['snr_error'] for s in valid], fmt='o-', label='Training sample')
        ax.axvline(args.int_up, color='red', label='Chosen upper limit')
        if validation_fit and 'snr' in validation_fit:
            ax.errorbar([args.int_up], [validation_fit['snr']],
                        yerr=[validation_fit['snr_error']], fmt='s', label='Independent validation')
        ax.set(xlabel='Integration upper sample (inclusive)', ylabel='Fitted charge SNR')
        ax.legend()
        save_figure(fig, out / '11_integration_scan.png')
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(discovery_profile, label='Robust discovery profile')
    ax.axvline(discovery['edge'], label='Detected leading edge', color='black')
    mark_windows(ax, args)
    ax.legend()
    save_figure(fig, out / '12_led_discovery.png')
    print(f'Noise={noise:.3f} ADC; RMS limit={rms_limit:.3f}; preimpulse limit={excursion_limit:.3f}')
    print(f"Window={args.int_low}–{args.int_up}; fit={fit_result['status']}; SNR={fit_result.get('snr')}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=HERE/'data', help='Directory of channel_*.dat or one file')
    parser.add_argument('--recursive', action='store_true', help='Discover channel files inside vgain_* folders')
    parser.add_argument('--vgains', type=int, nargs='+', help='Only these VGAIN values when scanning folders')
    parser.add_argument('--channels', type=int, nargs='+', help='Only these channel numbers')
    parser.add_argument('--report-pdf', type=Path, help='One page per VGAIN for the single selected channel')
    parser.add_argument('--debug-pdf', action='store_true',
                        help='Also collect per-channel/VGAIN diagnostic PNGs into a local PDF')
    parser.add_argument('--output', type=Path, default=HERE/'output'/'selection')
    parser.add_argument('--length', type=int, default=1024)
    parser.add_argument('--max-waveforms', type=int, default=40000)
    parser.add_argument('--expected-waveforms', type=int,
                        help='Require this many complete records per file (checks downloaded scan data)')
    parser.add_argument('--baseline-start', type=int, default=0)
    parser.add_argument('--baseline-stop', type=int, default=None, help='Exclusive; default detected from LED edge')
    parser.add_argument('--baseline-estimator', choices=['median', 'mean'], default='median',
                        help='Per-waveform baseline; mean permits comparison with the C++ reference')
    parser.add_argument('--int-low', type=int, default=None, help='Manual override; provide int-up too')
    parser.add_argument('--int-up', type=int, default=None, help='Inclusive; default optimized with training fits')
    parser.add_argument('--max-peaks', type=int, default=6, help='waffles peak-search target, not component cap')
    parser.add_argument('--rms-mad', type=float, default=5.)
    parser.add_argument('--rms-floor', type=float, default=1.25)
    parser.add_argument('--excursion-sigma', type=float, default=5.)
    parser.add_argument('--adc-max', type=int, default=16383)
    parser.add_argument('--invert', action='store_true', default=None,
                        help='Force negative polarity; default detects polarity from data')
    parser.add_argument('--plot-waveforms', type=int, default=1000)
    parser.add_argument('--bins', type=int, default=300)
    args = parser.parse_args()
    if args.baseline_start < 0 or args.length < 128:
        parser.error('Require baseline-start >= 0 and length >= 128')
    if (args.int_low is None) != (args.int_up is None):
        parser.error('Provide both int-low and int-up, or neither')
    if args.int_low is not None and not 0 <= args.int_low < args.int_up < args.length:
        parser.error('Require 0 <= int-low < int-up < length')
    if args.max_peaks < 2:
        parser.error('At least two peaks required for calibration')
    if args.report_pdf is not None and (not args.channels or len(args.channels) != 1):
        parser.error('--report-pdf requires exactly one channel in --channels')
    if min(args.max_waveforms, args.plot_waveforms, args.bins, args.adc_max) <= 0:
        parser.error('Waveform counts, bins and ADC maximum must be positive')
    if args.expected_waveforms is not None and args.expected_waveforms <= 0:
        parser.error('Expected waveform count must be positive')
    if args.rms_mad <= 0 or args.rms_floor <= 1 or args.excursion_sigma <= 0:
        parser.error('Require rms-mad > 0, rms-floor > 1, excursion-sigma > 0')
    files = ([args.input] if args.input.is_file() else
             sorted(args.input.rglob('channel_*.dat') if args.recursive else args.input.glob('channel_*.dat')))
    if args.vgains:
        files = [p for p in files if p.parent.name.startswith('vgain_')
                 and int(p.parent.name.removeprefix('vgain_')) in args.vgains]
    if args.channels:
        files = [p for p in files if p.stem.startswith('channel_')
                 and int(p.stem.removeprefix('channel_')) in args.channels]
    if not files:
        parser.error(f'No channel_*.dat files found in {args.input}')
    summaries = []
    for path in files:
        relative_parent = path.parent.relative_to(args.input) if args.input.is_dir() else Path()
        try:
            result = analyse_file(path, args, relative_parent)
            summaries.append(result)
            if args.debug_pdf:
                create_debug_report(result, Path(result['output_folder'])/'debug_report.pdf')
        except ValueError as exc:
            failure = {'input': str(path), 'status': 'analysis_failed', 'message': str(exc),
                       'channel': int(path.stem.removeprefix('channel_')),
                       'vgain': int(path.parent.name.removeprefix('vgain_')) if path.parent.name.startswith('vgain_') else None,
                       'settings': {'input': str(args.input), 'output': str(args.output)}}
            out = args.output/relative_parent/path.stem
            out.mkdir(parents=True, exist_ok=True)
            failure['output_folder'] = str(out.resolve())
            (out/'failure.json').write_text(json.dumps(failure, indent=2)+'\n')
            if args.debug_pdf:
                create_debug_report(failure, out/'debug_report.pdf')
            summaries.append(failure)
            print(f'{path.name}: FAILED: {exc}')
    (args.output/'summary_all_channels.json').write_text(json.dumps(summaries, indent=2)+'\n')
    with (args.output/'scan_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['input', 'module', 'bias_folder', 'vgain',
            'channel', 'loaded', 'selected', 'selected_fraction', 'noise_adc', 'int_low',
            'int_up', 'gain_adc_ns', 'gain_error_adc_ns', 'snr', 'snr_error',
            'chi2_ndof', 'fit_status', 'candidate_gain_adc_ns', 'candidate_snr',
            'validation_status', 'unselected_snr', 'spe_status',
            'spe_amplitude_adc', 'spe_amplitude_stat_error_adc',
            'dynamic_range_pe', 'dynamic_range_error_pe',
            'unselected_pedestal_sigma_adc_ns', 'message'])
        writer.writeheader()
        for item in summaries:
            fit = item.get('full_selected_sample_fit', {})
            fit_ok = fit.get('status') == 'ok'
            validation = item.get('independent_validation_fit') or {}
            unselected = item.get('unselected_sample_fit') or {}
            spe = item.get('spe_metrics') or {}
            spe_ok = fit_ok and spe.get('status') == 'ok'
            settings = item.get('settings', {})
            counts = item.get('counts', {})
            writer.writerow({'input': item['input'], 'module': item.get('module'),
                'bias_folder': item.get('bias_folder'), 'vgain': item.get('vgain'),
                'channel': item.get('channel'), 'loaded': item.get('loaded_waveforms'),
                'selected': counts.get('03_selected'), 'selected_fraction': item.get('selected_fraction'),
                'noise_adc': item.get('noise_adc'), 'int_low': settings.get('int_low'),
                'int_up': settings.get('int_up'),
                'gain_adc_ns': fit.get('gain_adc_ns') if fit_ok else None,
                'gain_error_adc_ns': fit.get('gain_error_adc_ns') if fit_ok else None,
                'snr': fit.get('snr') if fit_ok else None,
                'snr_error': fit.get('snr_error') if fit_ok else None,
                'chi2_ndof': fit.get('chi2_ndof') if fit_ok else None,
                'fit_status': fit.get('status', item.get('status')),
                'candidate_gain_adc_ns': fit.get('gain_adc_ns'),
                'candidate_snr': fit.get('snr'),
                'validation_status': validation.get('status'),
                'unselected_snr': unselected.get('snr'),
                'spe_status': spe.get('status'),
                'spe_amplitude_adc': spe.get('spe_amplitude_corrected_adc') if spe_ok else None,
                'spe_amplitude_stat_error_adc': spe.get('spe_amplitude_stat_error_adc') if spe_ok else None,
                'dynamic_range_pe': spe.get('dynamic_range_pe') if spe_ok else None,
                'dynamic_range_error_pe': spe.get('dynamic_range_stat_error_pe') if spe_ok else None,
                'unselected_pedestal_sigma_adc_ns': unselected.get('pedestal_sigma_adc_ns'),
                'message': item.get('message')})
    if args.report_pdf is not None:
        create_report(summaries, args.report_pdf, args.channels[0])
    if any(s.get('status') == 'analysis_failed' for s in summaries):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
