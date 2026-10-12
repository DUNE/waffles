"""Charge-calibration scan and per-point diagnostic reports."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


def model_curve(x, fit):
    params = {k: v['value'] for k, v in fit['parameters'].items()}
    mu0, sigma0 = params['mean_baseline'], params['std_baseline']
    gain, propstd = params['gain'], params['propstd']
    model = params['scale_baseline']*np.exp(-.5*((x-mu0)/sigma0)**2)
    for key, scale in params.items():
        if key.startswith('scale_') and key.endswith('pe'):
            i = int(key.split('_')[1][:-2])
            sigma = np.sqrt(sigma0**2+i*propstd**2)
            model += scale*np.exp(-.5*((x-mu0-i*gain)/sigma)**2)
    return model


def measurement(fit, key, error_key, unit=''):
    if key not in fit:
        return 'not available'
    v = fit[key]
    error = fit.get(error_key)
    return f'{v:.3g} +/- {error:.2g} {unit}' if error is not None else f'{v:.3g} {unit}'


def point_label(summary, channel):
    bias = summary.get('bias_folder') or ''
    if 'DVbias_' in bias:
        bias = 'DVbias ' + bias.rsplit('DVbias_', 1)[-1]
    vgain = summary.get('vgain')
    return (f"{summary.get('module') or '?'}  {bias}  |  "
            f"VGAIN {vgain if vgain is not None else '?'}  |  channel {channel}")


def create_report(summaries, output_path, channel):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    channel_rows = [s for s in summaries if s.get('channel') == channel]
    if not channel_rows:
        raise ValueError(f'No analysis results for channel {channel}')
    channel_rows.sort(key=lambda s: (s.get('vgain') is None, s.get('vgain') or 0))
    with PdfPages(output_path) as pdf:
        for page_no, summary in enumerate(channel_rows, 1):
            fit = summary.get('full_selected_sample_fit') or {}
            input_path = Path(summary['input'])
            analysis_folder = summary.get('output_folder')
            metric_path = Path(analysis_folder)/'waveform_metrics.csv' if analysis_folder else None
            fig = plt.figure(figsize=(11.69, 8.27), constrained_layout=True)
            grid = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], height_ratios=[3, 1])
            has_model = 'parameters' in fit
            ax = fig.add_subplot(grid[0, 0] if has_model else grid[:, 0])
            residual = fig.add_subplot(grid[1, 0], sharex=ax) if has_model else None
            panel = fig.add_subplot(grid[:, 1]); panel.axis('off')
            fig.suptitle(point_label(summary, channel),
                         fontsize=15, fontweight='bold')
            if metric_path is not None and metric_path.exists():
                with metric_path.open() as f:
                    rows = list(csv.DictReader(f))
                charge = np.array([float(r['charge_adc_ns']) for r in rows])
                selected = np.array([r['selected'] == 'True' for r in rows])
                values = charge[selected]
                if len(values):
                    domain = fit.get('histogram_domain')
                    if domain is None:
                        domain = np.quantile(values, [.005, .99]).tolist()
                    if domain[0] == domain[1]:
                        domain = [domain[0]-1, domain[1]+1]
                    bins = int(fit.get('histogram_bins', 80))
                    counts, edges = np.histogram(values, bins=bins, range=domain)
                    x = (edges[:-1]+edges[1:])/2
                    errors = np.sqrt(np.maximum(counts, 1))
                    ax.errorbar(x, counts, yerr=errors, fmt='.', markersize=3,
                                color='#075b9a', label='Selected waveforms')
                    if has_model:
                        model = model_curve(x, fit)
                        ax.plot(x, model, color='#d44c22', linewidth=2,
                                label='Waffles multigaussian fit')
                        residual.plot(x, (counts-model)/errors, '.', markersize=3, color='#075b9a')
                        for i in range(fit['gaussian_components']):
                            peak = fit['parameters']['mean_baseline']['value'] + i*fit['gain_adc_ns']
                            if domain[0] <= peak <= domain[1]:
                                ax.axvline(peak, color='gray', alpha=.18)
                    ax.legend(loc='upper right', fontsize=9)
            else:
                ax.text(.5, .5, 'Charge histogram unavailable', ha='center', va='center', transform=ax.transAxes)
            ax.set_ylabel('Waveforms per bin')
            ax.set_title('Charge spectrum and fit' if has_model else 'Charge spectrum', loc='left')
            if residual is not None:
                residual.axhline(0, color='black', linewidth=.8)
                residual.set(xlabel='Integrated charge [ADC ns]', ylabel='Pull')
            else:
                ax.set_xlabel('Integrated charge [ADC ns]')
                ax.text(.03, .97, 'No reliable fit', ha='left', va='top',
                        color='#aa2424', transform=ax.transAxes)
            settings = summary.get('settings', {})
            counts = summary.get('counts', {})
            validation = summary.get('independent_validation_fit') or {}
            unselected = summary.get('unselected_sample_fit') or {}
            spe = summary.get('spe_metrics') or {}
            info = [
                ('Fit status', fit.get('status', summary.get('status', 'unknown'))),
                ('Waveforms', f"{counts.get('03_selected', '?')} / {summary.get('loaded_waveforms', '?')}"),
                ('Noise', measurement(summary, 'noise_adc', None, 'ADC')),
                ('Baseline samples', f"[{settings.get('baseline_start', '?')}, {settings.get('baseline_stop', '?')})"),
                ('Integration samples', f"[{settings.get('int_low', '?')}, {settings.get('int_up', '?')}]"),
                ('Pedestal mean', measurement(fit, 'pedestal_adc_ns', None, 'ADC ns')),
                ('Pedestal width', measurement(fit, 'pedestal_sigma_adc_ns', 'pedestal_sigma_error_adc_ns', 'ADC ns')),
                ('Gain', measurement(fit, 'gain_adc_ns', 'gain_error_adc_ns', 'ADC ns')),
                ('SNR after cuts', measurement(fit, 'snr', 'snr_error')),
                ('SNR before cuts', measurement(unselected, 'snr', 'snr_error')),
                ('SPE status', spe.get('status', 'not run')),
                ('SPE amplitude', measurement(spe, 'spe_amplitude_corrected_adc',
                                              'spe_amplitude_stat_error_adc', 'ADC')),
                ('Dynamic range', measurement(spe, 'dynamic_range_pe',
                                               'dynamic_range_stat_error_pe', 'PE')),
                ('Chi2 / ndof', measurement(fit, 'chi2_ndof', None)),
                ('Fit entries', str(fit.get('histogram_entries', '?'))),
                ('Validation status', validation.get('status', 'not run')),
                ('Validation SNR', measurement(validation, 'snr', 'snr_error')),
            ]
            y = .96
            for label, value in info:
                panel.text(.04, y, label, transform=panel.transAxes, fontsize=10, va='top')
                panel.text(.54, y, str(value), transform=panel.transAxes, fontsize=10, va='top')
                y -= .052
            panel.text(.04, .02,
                'SNR is gain / pedestal width; compare before and after cuts.\n'
                'Shown SPE/DR errors are statistical only; fit errors depend on the model and window.',
                transform=panel.transAxes, fontsize=8, color='#555555', va='bottom')
            fig.text(.98, .015, f'{page_no} / {len(channel_rows)}', ha='right', fontsize=8)
            pdf.savefig(fig)
            plt.close(fig)
    return output_path


def create_debug_report(summary, output_path):
    """Collect the existing PNG diagnostics for one channel and one VGAIN."""
    analysis_folder = Path(summary['output_folder'])
    groups = [
        ('LED position and integration', [
            ('12_led_discovery.png', 'LED discovery'),
            ('09_pulse_zoom.png', 'Pulse and integration window'),
            ('11_integration_scan.png', 'Integration-window scan'),
            ('05_noise_diagnostics.png', 'Baseline-noise diagnostics')]),
        ('Waveform selection', [
            ('00_original.png', 'Original waveforms'),
            ('01_rms.png', 'After RMS cut'),
            ('02_preimpulse.png', 'After pre-LED excursion cut'),
            ('03_selected.png', 'Selected waveforms')]),
        ('Rejected waveforms and charge', [
            ('04_rejected.png', 'Rejected waveforms'),
            ('06_selection_counts.png', 'Cumulative selection'),
            ('07_charge_histogram.png', 'Charge, full range'),
            ('08_charge_zoom.png', 'Charge, peak region')]),
        ('Calibration fit', [('10_multigaussian_fit.png', 'Multigaussian fit and pulls')]),
        ('SPE amplitude', [('13_spe_waveform.png', 'Mean waveform of 1-PE candidates')]),
    ]
    available = [(heading, [(analysis_folder/name, caption) for name, caption in plots
                            if (analysis_folder/name).is_file()])
                 for heading, plots in groups]
    available = [(heading, plots) for heading, plots in available if plots]
    if not available:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    label = point_label(summary, summary.get('channel', '?'))
    with PdfPages(output_path) as pdf:
        for number, (heading, plots) in enumerate(available, 1):
            fig = plt.figure(figsize=(16.54, 11.69))
            fig.suptitle(f'{label}\n{heading}', y=.985, fontsize=17, fontweight='bold')
            fig.subplots_adjust(left=.03, right=.98, bottom=.04, top=.87,
                                wspace=.05, hspace=.14)
            if heading == 'Calibration fit':
                ax = fig.add_subplot(1, 2, 1)
                ax.imshow(plt.imread(plots[0][0]))
                ax.set_axis_off()
                panel = fig.add_subplot(1, 2, 2)
                panel.set_axis_off()
                fit = summary.get('full_selected_sample_fit') or {}
                unselected = summary.get('unselected_sample_fit') or {}
                settings = summary.get('settings') or {}
                validation = summary.get('independent_validation_fit') or {}
                spe = summary.get('spe_metrics') or {}
                lines = [
                    ('Fit status', fit.get('status', summary.get('status', '?'))),
                    ('Selected / loaded', f"{(summary.get('counts') or {}).get('03_selected', '?')} / {summary.get('loaded_waveforms', '?')}"),
                    ('Baseline samples', f"[{settings.get('baseline_start', '?')}, {settings.get('baseline_stop', '?')})"),
                    ('Integration samples', f"[{settings.get('int_low', '?')}, {settings.get('int_up', '?')}]"),
                    ('Pedestal width', measurement(fit, 'pedestal_sigma_adc_ns', 'pedestal_sigma_error_adc_ns', 'ADC ns')),
                    ('Gain', measurement(fit, 'gain_adc_ns', 'gain_error_adc_ns', 'ADC ns')),
                    ('SNR after cuts', measurement(fit, 'snr', 'snr_error')),
                    ('SNR before cuts', measurement(unselected, 'snr', 'snr_error')),
                    ('SPE amplitude', measurement(spe, 'spe_amplitude_corrected_adc',
                                                   'spe_amplitude_stat_error_adc', 'ADC')),
                    ('Dynamic range', measurement(spe, 'dynamic_range_pe',
                                                    'dynamic_range_stat_error_pe', 'PE')),
                    ('Chi2 / ndof', measurement(fit, 'chi2_ndof', None)),
                    ('Validation', validation.get('status', 'not run')),
                ]
                y = .94
                for name, value in lines:
                    panel.text(.02, y, name, fontsize=12, va='top', transform=panel.transAxes)
                    panel.text(.43, y, str(value), fontsize=12, va='top', transform=panel.transAxes)
                    y -= .070
            else:
                axes = fig.subplots(1, 1) if len(plots) == 1 else fig.subplots(2, 2)
                axes = np.atleast_1d(axes)
                for ax, (path, caption) in zip(axes.flat, plots):
                    ax.imshow(plt.imread(path))
                    ax.set_title(caption, fontsize=12)
                    ax.set_axis_off()
                for ax in list(axes.flat)[len(plots):]:
                    ax.set_axis_off()
            fig.text(.98, .01, f'{number} / {len(available)}', ha='right', fontsize=9)
            pdf.savefig(fig)
            plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summaries', type=Path, required=True, help='summary_all_channels.json')
    parser.add_argument('--channel', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    create_report(json.loads(args.summaries.read_text()), args.output, args.channel)


if __name__ == '__main__':
    main()
