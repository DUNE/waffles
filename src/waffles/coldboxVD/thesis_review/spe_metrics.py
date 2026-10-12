"""Estimate SPE pulse amplitude and dynamic range from an accepted LED fit.

The charge selection and 0/2-PE correction follow fgalizzi/pds-ana,
Class/_c/SPE.cpp and Class/private_methods.hpp (commit 2fcabc1).  The charge
values passed here must be the *same Waffles integrals* used for the fit.
The input waveforms must already have their per-record baselines subtracted.
No external noise run or module-specific amplitude threshold is required.

The reference C++ computes a preliminary peak-to-trough dynamic range, but
the scan CSV uses 2**14 / (corrected mean-waveform maximum).  ``dynamic_range_pe``
below is the latter quantity.  We subtract the small residual mean pre-LED
offset from the waveform maximum, as the C++ mean baseline is zero by design.
"""

from typing import Optional, Tuple
import numpy as np
from scipy.special import ndtr


def _fit_components(fit: dict) -> Tuple[float, float, float, list]:
    """Read the Waffles multigaussian parameterization as unnormalized peaks."""
    pars = fit['parameters']
    pedestal = float(pars['mean_baseline']['value'])
    gain = float(pars['gain']['value'])
    sigma0 = float(pars['std_baseline']['value'])
    propstd = float(pars['propstd']['value'])
    if not all(np.isfinite(v) for v in (pedestal, gain, sigma0, propstd)):
        raise ValueError('Non-finite Gaussian fit parameter')
    if gain <= 0 or sigma0 <= 0 or propstd < 0:
        raise ValueError('Non-physical Gaussian fit parameter')

    components = []
    k = 0
    while True:
        name = 'scale_baseline' if k == 0 else f'scale_{k}pe'
        if name not in pars:
            break
        scale = float(pars[name]['value'])
        if not np.isfinite(scale) or scale < 0:
            raise ValueError(f'Invalid Gaussian amplitude {name}')
        components.append({'pe': k, 'height': scale,
                           'mean': pedestal + k * gain,
                           'sigma': float(np.hypot(sigma0, np.sqrt(k) * propstd))})
        k += 1
    if len(components) < 3:
        raise ValueError('Need fitted 0-, 1-, and 2-PE Gaussian components')
    return pedestal, gain, sigma0, components


def _component_area(component: dict, lower: float, upper: float) -> float:
    sigma = component['sigma']
    return float(component['height'] * sigma * np.sqrt(2 * np.pi)
                 * (ndtr((upper-component['mean']) / sigma)
                    - ndtr((lower-component['mean']) / sigma)))


def estimate_spe_metrics(
    centered_waveforms_adc: np.ndarray,
    charges_adc_ns: np.ndarray,
    fit: dict,
    int_low: int,
    int_up: int,
    *,
    invert: bool = False,
    adc_levels: int = 2**14,
    minimum_candidates: int = 100,
) -> Tuple[dict, Optional[np.ndarray]]:
    """Return ``(diagnostic, mean_SPE_waveform_or_None)``.

    ``centered_waveforms_adc`` and ``charges_adc_ns`` contain only the
    calibration-selected waveforms, in the same order.  The latter are the
    trapezoidal charge integrals returned by Waffles ``WindowIntegrator``.
    The candidate selection is strict ``mu0 + gain +/- sigma0``.  All PE
    Gaussian components are integrated over ``mu1 +/- sigma1`` to compute
    the reference 0/2-PE contamination correction.  Higher-PE contamination
    and candidate statistics are retained as quality diagnostics.

    ``spe_amplitude_stat_error_adc`` describes only waveform sampling at the
    measured peak index; fit-parameter/correction uncertainty and systematic
    differences between baseline or integration conventions are not included.
    """
    if fit.get('status') != 'ok':
        return {'status': 'fit_not_accepted', 'fit_status': fit.get('status')}, None
    waveforms = np.asarray(centered_waveforms_adc)
    charges = np.asarray(charges_adc_ns, dtype=float)
    if waveforms.ndim != 2 or charges.ndim != 1 or len(waveforms) != len(charges):
        raise ValueError('Expected N by samples waveforms and N charge values')
    if not len(waveforms) or not 0 < int_low < int_up < waveforms.shape[1]:
        raise ValueError('Invalid waveform count or integration window')
    if adc_levels <= 0 or minimum_candidates < 2:
        raise ValueError('adc_levels must be positive and minimum_candidates at least two')

    try:
        pedestal, gain, sigma0, components = _fit_components(fit)
    except (KeyError, TypeError, ValueError) as exc:
        return {'status': 'invalid_fit_components', 'reason': str(exc)}, None

    candidate_lower = pedestal + gain - sigma0
    candidate_upper = pedestal + gain + sigma0
    candidates = (charges > candidate_lower) & (charges < candidate_upper)
    n = int(candidates.sum())
    diagnostic = {
        'status': 'insufficient_spe_candidates' if n < minimum_candidates else 'ok',
        'candidate_charge_window_adc_ns': [float(candidate_lower), float(candidate_upper)],
        'candidate_count': n,
        'selected_waveform_count': int(len(charges)),
        'candidate_fraction': float(n / len(charges)),
        'fit_1pe_mean_adc_ns': float(pedestal + gain),
        'amplitude_peak_window_samples_inclusive': [int(int_low), int(int_up)],
        'adc_levels': int(adc_levels),
    }
    if n < minimum_candidates:
        return diagnostic, None
    diagnostic['candidate_mean_charge_adc_ns'] = float(charges[candidates].mean())

    # This is the mean of the raw baseline-subtracted waveforms, not of a
    # filtered/integrated representation.  Inversion matches WindowIntegrator.
    candidate_waveforms = waveforms[candidates].astype(float, copy=False)
    if invert:
        candidate_waveforms = -candidate_waveforms
    mean_waveform = candidate_waveforms.mean(axis=0)
    mean_pre = float(mean_waveform[:int_low].mean())
    peak_index = int(int_low + np.argmax(mean_waveform[int_low:int_up+1]))
    peak = float(mean_waveform[peak_index])
    trough = float(mean_waveform.min())
    raw_amplitude = peak - mean_pre
    if not np.isfinite(raw_amplitude) or raw_amplitude <= 0:
        return {**diagnostic, 'status': 'nonpositive_spe_amplitude'}, mean_waveform

    # C++: selected candidate mean = (N0*0 + N1*1 + N2*2) / Ntotal SPE.
    # The reference correction is 1 + (N2-N0)/Ntotal.  Its integrals use
    # mu1 +/- sigma1, while candidate waveform selection uses +/- sigma0.
    mu1, sigma1 = components[1]['mean'], components[1]['sigma']
    correction_lower, correction_upper = mu1-sigma1, mu1+sigma1
    areas = np.array([_component_area(c, correction_lower, correction_upper)
                      for c in components])
    area_total = float(areas.sum())
    if not np.isfinite(area_total) or area_total <= 0:
        return {**diagnostic, 'status': 'invalid_spe_correction'}, mean_waveform
    fractions = areas/area_total
    correction = float(1 - fractions[0] + fractions[2])
    general_correction = float(np.dot(np.arange(len(fractions)), fractions))
    if not np.isfinite(correction) or correction <= 0:
        return {**diagnostic, 'status': 'invalid_spe_correction'}, mean_waveform

    corrected_amplitude = raw_amplitude / correction
    dynamic_range = float(adc_levels / corrected_amplitude)
    # Standard error of the *selected mean*, including covariance between
    # the peak sample and its own waveform's pre-LED mean.  Peak position and
    # fit correction are held fixed, so this is not a total uncertainty.
    event_amplitudes = (candidate_waveforms[:, peak_index]
                        - candidate_waveforms[:, :int_low].mean(axis=1))
    raw_stat_error = float(event_amplitudes.std(ddof=1) / np.sqrt(n))
    corrected_stat_error = raw_stat_error / correction
    dynamic_range_stat_error = dynamic_range * raw_stat_error / raw_amplitude

    diagnostic.update({
        'peak_sample': peak_index,
        'mean_prepulse_offset_adc': mean_pre,
        'mean_peak_adc': peak,
        'mean_trough_adc': trough,
        'spe_amplitude_raw_adc': float(raw_amplitude),
        'spe_amplitude_raw_stat_error_adc': float(raw_stat_error),
        'spe_amplitude_stat_error_adc': float(corrected_stat_error),
        'spe_amplitude_corrected_adc': float(corrected_amplitude),
        'spe_amplitude_peak_to_peak_adc': float(peak-trough),
        'dynamic_range_pe': dynamic_range,
        'dynamic_range_stat_error_pe': float(dynamic_range_stat_error),
        'preliminary_peak_to_peak_dynamic_range_pe': float(adc_levels/(peak-trough))
        if peak > trough else None,
        'spe_correction_factor': correction,
        'spe_correction_generalized_all_pe': general_correction,
        'correction_charge_window_adc_ns': [float(correction_lower),
                                            float(correction_upper)],
        'fitted_fraction_0pe_in_correction_window': float(fractions[0]),
        'fitted_fraction_1pe_in_correction_window': float(fractions[1]),
        'fitted_fraction_2pe_in_correction_window': float(fractions[2]),
        'fitted_fraction_3pe_and_above_in_correction_window': float(fractions[3:].sum()),
        'fit_correction_uncertainty_included': False,
        'method': 'fgalizzi SPE candidate window and 0/2-PE correction; Waffles charge',
    })
    # The colleague's 0/2-PE formula assumes negligible 3+ PE overlap.  A
    # low 1-PE fraction or large correction needs visual review before a DR
    # point can be used as a performance measurement.
    if (fractions[1] < .70 or fractions[3:].sum() > .05
            or not .75 <= correction <= 1.25):
        diagnostic['status'] = 'review_required'
        diagnostic['reason'] = 'Large fitted contamination in the correction charge window'
    return diagnostic, mean_waveform
