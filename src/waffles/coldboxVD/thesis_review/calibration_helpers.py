"""Automatic window discovery and wrappers around the existing waffles fitter."""
import warnings
import numpy as np
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter1d
from scipy.stats import trim_mean
from scipy.signal import find_peaks
from waffles.data_classes.CalibrationHistogram import CalibrationHistogram
from waffles.data_classes.IPDict import IPDict
from waffles.utils.integral.WindowIntegrator import WindowIntegrator
from waffles.utils.fit_peaks.fit_peaks import fit_peaks_of_CalibrationHistogram
from waffles.utils.numerical_utils import multigaussfit


def robust_profile(adcs):
    # Trimming only constructs a discovery template; it does not reject charge records.
    return gaussian_filter1d(trim_mean(adcs, .02, axis=0), 1.5)


def locate_led(raw, invert=False):
    offset = np.median(raw, axis=1)
    centered = raw-offset[:, None]
    if invert is None:
        profile_positive = robust_profile(centered)
        rise_positive = profile_positive[4:]-profile_positive[:-4]
        rise_negative = -rise_positive
        left, right = max(64, len(profile_positive)//16), len(profile_positive)-32
        positive_score = np.max(rise_positive[left:right-4])
        negative_score = np.max(rise_negative[left:right-4])
        invert = bool(negative_score > positive_score)
        profile = -profile_positive if invert else profile_positive
    else:
        profile = robust_profile((-1 if invert else 1)*centered)
    # A sharp, time-coherent leading edge is less sensitive to long accidental tails.
    rise = profile[4:] - profile[:-4]
    middle = np.median(rise)
    spread = 1.4826*np.median(np.abs(rise-middle))
    left = max(64, len(profile)//16)
    right = len(profile)-32
    edge = left + int(np.argmax(rise[left:right-4])) + 2
    significance = float((rise[edge-2]-middle)/max(spread, 1e-9))
    if significance < 8:
        raise ValueError('No clear coherent LED leading edge (discovery significance < 8)')
    # Search only near this edge, not at later accidental pulses.
    peak = edge + int(np.argmax(profile[edge:min(len(profile), edge+32)]))
    baseline_stop = edge-12
    if baseline_stop < 64:
        raise ValueError('Insufficient preimpulse for automatic baseline estimation')
    return {'edge': edge, 'peak': peak, 'baseline_start': 0,
            'baseline_stop': baseline_stop, 'edge_significance': significance,
            'invert': bool(invert)}, profile


def integrate(wfset, low, up, invert):
    wfset.analyse('charge', WindowIntegrator,
                 IPDict({'baseline_analysis': 'baseline', 'inversion': invert,
                         'int_ll': int(low), 'int_ul': int(up),
                         'amp_ll': int(low), 'amp_ul': int(up)}),
                 checks_kwargs={'points_no': wfset.points_per_wf}, overwrite=True)


def fit_calibration(wfset, max_peaks=6):
    charges = np.array([wf.analyses['charge'].result['integral'] for wf in wfset.waveforms])
    if len(charges) < 200:
        return {'status': 'insufficient_statistics'}, None
    lo, hi = np.quantile(charges, [.005, .99])
    iqr = np.subtract(*np.percentile(charges, [75, 25]))
    width = 2*iqr/len(charges)**(1/3)
    if not hi > lo or width <= 0:
        return {'status': 'degenerate_charge_distribution'}, None
    bins = int(np.clip(np.ceil((hi-lo)/width), 64, 256))
    wf0 = wfset.waveforms[0]
    low_samples = wf0.analyses['charge'].int_ll
    high_samples = wf0.analyses['charge'].int_ul
    quiet_samples = max(16, min(64, low_samples//2))
    quiet_rms = np.median([np.std(wf.adcs[:quiet_samples]) for wf in wfset.waveforms])
    pedestal_scale = quiet_rms*np.sqrt(high_samples-low_samples+1)*wf0.time_step_ns
    # The samples may be correlated, so use an equal-length pre-LED integral
    # rather than sqrt(N) scaling to reject artificially narrow fit pedestals.
    quiet_start = max(16, min(96, low_samples//3))
    quiet_stop = quiet_start + high_samples-low_samples+1
    quiet_integral_sigma = None
    if quiet_stop < low_samples:
        quiet_charges = np.array([
            trapezoid(
                wf.adcs[quiet_start:quiet_stop]
                - wf.analyses['baseline'].result['baseline'],
                dx=wf.time_step_ns)
            for wf in wfset.waveforms])
        quiet_median = np.median(quiet_charges)
        quiet_integral_sigma = float(1.4826*np.median(np.abs(quiet_charges-quiet_median)))
    attempts = []
    last_histogram = None
    # Fixed fallback order, not a search for the largest fitted SNR.
    for prominence in [.15, .08, .03]:
        hist = CalibrationHistogram.from_WaveformSet(
            wfset, bins, np.array([lo, hi]), 'integral', 'charge')
        last_histogram = hist
        centers = (hist.edges[:-1]+hist.edges[1:])/2
        smooth = gaussian_filter1d(hist.counts.astype(float), .8)
        peaks, _ = find_peaks(smooth, prominence=.03*smooth.max(), distance=4)
        if not len(peaks):
            return {'status': 'unresolved_peaks'}, hist
        # Baseline-subtracted pedestal should be near zero. LED intensity can
        # make the 1-PE peak taller, so the histogram maximum is not a label.
        pedestal_index = int(peaks[np.argmin(np.abs(centers[peaks]))])
        if abs(centers[pedestal_index]) > max(3*pedestal_scale, 2*width):
            return {'status': 'pedestal_not_identified', 'nearest_peak_adc_ns': float(centers[pedestal_index])}, hist
        candidates = peaks[peaks > pedestal_index+4]
        if not len(candidates):
            return {'status': 'unresolved_peaks', 'histogram_bins': bins,
                    'histogram_domain': [float(lo), float(hi)]}, hist
        gain_seed = float(centers[candidates[0]]-centers[pedestal_index])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                converged = fit_peaks_of_CalibrationHistogram(
                    hist, max_peaks=max_peaks, prominence=prominence,
                    initial_percentage=.2, percentage_step=.1,
                    return_last_addition_if_fail=True, fit_type='multigauss_iminuit',
                    weigh_fit_by_poisson_sigmas=True, half_points_to_fit=3)
            mm = getattr(hist, 'iminuit', None)
            if mm is None:
                attempts.append({'prominence': prominence, 'status': 'no_valid_minimum'})
                continue
            # Keep the existing waffles model/minimizer. Anchor PE spacing to two
            # observed smoothed maxima, avoiding a half-spacing solution with empty peaks.
            mm.values['gain'] = gain_seed
            mm.limits['gain'] = (.7*gain_seed, 1.3*gain_seed)
            mm.values['mean_baseline'] = float(centers[pedestal_index])
            left = pedestal_index
            while left > 0 and smooth[left] > .6065*smooth[pedestal_index]:
                left -= 1
            mm.values['std_baseline'] = max(centers[pedestal_index]-centers[left], width)
            mm.values['propstd'] = .25*gain_seed
            fixed_out_of_range_components = []
            for key in mm.parameters:
                if key.startswith('scale_'):
                    position = 0 if key == 'scale_baseline' else int(key.split('_')[1][:-2])
                    mm.values[key] = max(float(np.interp(
                        centers[pedestal_index]+position*gain_seed, centers, hist.counts)), 1e-3)
                    if position:
                        # Waffles may add more PE components than the recorded
                        # charge range can constrain.  Their freely varying
                        # scales make HESSE singular even when the visible
                        # pedestal and 1-PE peaks fit well.  Suppress only a
                        # component whose seed center lies more than two seed
                        # widths beyond the last fitted histogram bin.
                        component_sigma = np.hypot(
                            mm.values['std_baseline'],
                            mm.values['propstd']*np.sqrt(position))
                        if (centers[pedestal_index]+position*gain_seed
                                > hi+2*component_sigma):
                            mm.values[key] = 1e-6
                            mm.fixed[key] = True
                            fixed_out_of_range_components.append(key)
            mm.strategy = 2
            mm.migrad(ncall=30000)
            mm.hesse()
            if not mm.valid or mm.covariance is None:
                attempts.append({'prominence': prominence, 'status': 'no_valid_seeded_minimum'})
                continue
            gain = float(mm.values['gain'])
            sigma = float(mm.values['std_baseline'])
            if gain <= 0 or sigma <= 0 or mm.ndof <= 0:
                attempts.append({'prominence': prominence, 'status': 'unphysical_parameters'})
                continue
            snr = gain/sigma
            cov = mm.covariance
            variance = (cov['gain', 'gain']/sigma**2
                        + gain**2*cov['std_baseline', 'std_baseline']/sigma**4
                        - 2*gain*cov['gain', 'std_baseline']/sigma**3)
            if variance < 0 or not np.isfinite(variance):
                attempts.append({'prominence': prominence, 'status': 'invalid_snr_variance'})
                continue
            reduced = float(mm.fval/mm.ndof)
            # A numerical minimum alone does not establish identifiable PE peaks.
            scale = float(mm.values['scale_1pe'])
            scale_error = float(mm.errors['scale_1pe'])
            resolved = gain/sigma >= 1.5 and scale > 3*scale_error
            at_gain_boundary = gain < .72*gain_seed or gain > 1.28*gain_seed
            # Only a gross discrepancy is vetoed: the fit pedestal may be wider
            # than the quiet integral because the LED response has extra jitter.
            noise_width_ok = (quiet_integral_sigma is None or quiet_integral_sigma <= 0
                              or sigma >= .5*quiet_integral_sigma)
            quality_ok = bool(mm.fmin.has_accurate_covar and resolved and reduced < 5
                              and not at_gain_boundary and noise_width_ok)
            result = {'status': 'ok' if quality_ok else 'review_required',
                      'converged': True, 'accurate_covariance': bool(mm.fmin.has_accurate_covar),
                      'pe_identification_heuristic': bool(resolved),
                      'gain_adc_ns': gain, 'gain_error_adc_ns': float(mm.errors['gain']),
                      'pedestal_adc_ns': float(mm.values['mean_baseline']),
                      'pedestal_sigma_adc_ns': sigma,
                      'quiet_integral_sigma_adc_ns': quiet_integral_sigma,
                      'pedestal_noise_width_ok': bool(noise_width_ok),
                      'pedestal_sigma_error_adc_ns': float(mm.errors['std_baseline']),
                      'snr': snr, 'snr_error': float(np.sqrt(variance)),
                      'gain_sigma_covariance': float(cov['gain', 'std_baseline']),
                      'gain_seed_adc_ns': gain_seed, 'gain_seed_limits': [.7*gain_seed, 1.3*gain_seed],
                      'gain_at_seed_boundary': bool(at_gain_boundary),
                      'chi2': float(mm.fval), 'ndof': int(mm.ndof), 'chi2_ndof': reduced,
                      'histogram_domain': [float(lo), float(hi)], 'histogram_bins': bins,
                      'histogram_entries': int(hist.counts.sum()),
                      'loaded_entries': len(charges), 'prominence': prominence,
                      'fixed_out_of_range_components': fixed_out_of_range_components,
                      'gaussian_components': len(hist.gaussian_fits_parameters['mean']),
                      'parameters': {k: {'value': float(mm.values[k]), 'error': float(mm.errors[k])}
                                     for k in mm.parameters},
                      'covariance': np.asarray(mm.covariance).tolist(),
                      'parameter_order': list(mm.parameters), 'previous_attempts': attempts}
            return result, hist
        except (ValueError, RuntimeError, IndexError, KeyError, ZeroDivisionError) as exc:
            attempts.append({'prominence': prominence, 'status': 'exception', 'message': str(exc)})
    return {'status': 'fit_failed', 'attempts': attempts}, last_histogram


def optimize_window(wfset, centered, mask, discovery, invert, max_peaks, out=None):
    """Split retained records: one half to optimize and the other to validate."""
    from waffles.data_classes.WaveformSet import WaveformSet
    indices = np.flatnonzero(mask)
    train_indices = indices[::2]
    validation_indices = indices[1::2]
    if len(train_indices) < 200 or len(validation_indices) < 200:
        raise ValueError('At least 400 selected waveforms required for window optimization')
    train = WaveformSet(*[wfset.waveforms[i] for i in train_indices])
    profile = robust_profile((-1 if invert else 1)*centered[train_indices])
    peak = discovery['peak']
    edge = discovery['edge']
    reference = float(np.median(profile[:discovery['baseline_stop']]))
    amplitude = profile[peak]-reference
    if amplitude <= 0:
        raise ValueError('Non-positive LED template amplitude')
    low = peak
    while low > edge-12 and profile[low] > reference+.1*amplitude:
        low -= 1
    low = max(discovery['baseline_stop']+1, low-1)
    last = min(len(profile)-1, peak+160)
    # A narrow prompt window can resolve PE peaks which a long integration
    # smears out. Always probe it; the wide coarse grid alone misses such cases.
    short = np.arange(peak+3, min(last, peak+30)+1, 3)
    candidates = np.unique(np.r_[short, np.linspace(peak+3, last, 10).astype(int)])
    scans = []
    for phase in ['coarse', 'refinement']:
        for up in candidates:
            if any(row['int_up'] == up for row in scans):
                continue
            integrate(train, low, int(up), invert)
            fit, _ = fit_calibration(train, max_peaks)
            scans.append({'int_low': int(low), 'int_up': int(up), 'phase': phase, **fit})
        valid = [row for row in scans if row['status'] == 'ok']
        if not valid:
            break
        coarse_best = max(valid, key=lambda row: row['snr'])['int_up']
        candidates = np.arange(max(peak+3, coarse_best-17), min(last, coarse_best+17)+1, 3)
    scans.sort(key=lambda row: row['int_up'])
    if out is not None:
        import json
        (out/'integration_scan.json').write_text(json.dumps(scans, indent=2)+'\n')
    valid = [row for row in scans if row['status'] == 'ok']
    if not valid:
        # Preserve a charge histogram for a weak/unresolved scan point. Never
        # extend the fallback far into the tail merely because the fit failed.
        threshold = reference + .2*amplitude
        fallback_up = min(last, peak+12)
        for up in range(peak+3, min(last-4, peak+24)):
            if np.all(profile[up:up+5] < threshold):
                fallback_up = up
                break
        return int(low), int(fallback_up), scans, {
            'status': 'no_reliable_training_fit', 'window_method': 'pulse_template_fallback'}, profile
    highest_snr = max(valid, key=lambda row: row['snr'])
    # Integrating further into the tail can give a tiny, noisy SNR increase
    # while blurring the PE structure. Prefer the shortest window whose
    # training SNR is compatible with the maximum within one fit uncertainty.
    compatible = [row for row in valid
                  if row['snr'] >= highest_snr['snr']-highest_snr['snr_error']]
    best = min(compatible, key=lambda row: row['int_up'])
    validation = WaveformSet(*[wfset.waveforms[i] for i in validation_indices])
    integrate(validation, low, best['int_up'], invert)
    validation_fit, _ = fit_calibration(validation, max_peaks)
    return int(low), int(best['int_up']), scans, validation_fit, profile


def fitted_curve(hist):
    centers = (hist.edges[:-1]+hist.edges[1:])/2
    mm = getattr(hist, 'iminuit', None)
    model = None if mm is None else multigaussfit(centers, *list(mm.values))
    return centers, model
