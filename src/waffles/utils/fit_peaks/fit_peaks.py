from waffles.data_classes.CalibrationHistogram import CalibrationHistogram
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid

import math
import waffles.utils.fit_peaks.fit_peaks_utils as wuff

from waffles.Exceptions import GenerateExceptionMessage

from iminuit import Minuit

import numpy as np
import waffles.utils.numerical_utils as wun
from typing import Optional

class MultiGaussBinnedModel:
    """
    Picklable binned multi-gauss model.
    Parameter meaning (consistent with gaussian_bin_counts):
      - scale_* : CONTINUOUS amplitude (counts / x-unit). Model returns counts/bin after integration.
      - bkg     : constant background in counts/bin
      - bkgG    : gaussian background HEIGHT in counts/bin (converted internally to counts/x)
    """
    def __init__(self, pnames):
        self.pnames = list(pnames)

    def __call__(self, edges, *p):
        edges = np.asarray(edges)
        vals = {n: float(v) for n, v in zip(self.pnames, p)}

        # Representative bin width (needed to convert counts/bin <-> counts/x)
        bw = float(np.median(np.diff(edges)))
        bw = max(bw, 1e-12)

        amp0  = float(vals['scale_baseline'])   # counts/x
        mu0   = float(vals['mean_baseline'])
        s0    = float(vals['std_baseline'])
        gain  = float(vals['gain'])
        prop  = float(vals.get('propstd', 0.0))
        # We keep 'bkg' for backward compatibility, but we'll allow disabling it.
        bkg   = float(vals.get('bkg', 0.0))  # counts/bin (flat)

        # Optional: gaussian background (height in counts/bin)
        bkgG_h   = float(vals.get('bkgG', 0.0))          # counts/bin (height)
        bkgG_mu  = float(vals.get('bkgG_mean', mu0))
        bkgG_std = float(vals.get('bkgG_std', 10.0*s0))

        mu = wun.gaussian_bin_counts(edges, amp0, mu0, s0)  # counts/bin

        for name in self.pnames:
            if name.startswith("scale_") and name.endswith("pe"):
                k = int(name.split('_')[1].replace('pe', ''))
                amp_k  = float(vals[name])   # counts/x (density)
                mu_k   = mu0 + k * gain
                s_k    = math.sqrt(max(1e-24, s0*s0 + (prop*prop)*k))
                mu += wun.gaussian_bin_counts(edges, amp_k,  mu_k, s_k)

        # always include bkg (limits already enforce >=0)
        # Optional flat background (counts/bin)
        # If gaussian background is enabled, you usually want bkg=0 to avoid the global offset.
        if bkg > 0.0 and bkgG_h <= 0.0:
            mu = mu + bkg

        # Gaussian background (counts/bin height -> counts/x density -> integrate)
        if bkgG_h > 0.0 and bkgG_std > 0.0:
            bkgG_scale = bkgG_h / bw  # counts/x
            mu = mu + wun.gaussian_bin_counts(edges, bkgG_scale, bkgG_mu, bkgG_std)
        return np.clip(mu, 1e-12, None)


class PoissonDeviance:
    """Pickle-friendly Poisson deviance for binned counts."""
    def __init__(self, x, y, model):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.model = model

    def __call__(self, *p):
        mu = self.model(self.x, *p)
        y = self.y

        # Hard guards: invalid model output => huge penalty (keep Minuit alive)
        if (mu is None) or (not np.all(np.isfinite(mu))) or np.any(mu <= 0):
            return 1e30
        if (y is None) or (not np.all(np.isfinite(y))) or np.any(y < 0):
            return 1e30

        # Avoid inf/0 ratios without changing the physical region
        mu = np.clip(mu, 1e-12, 1e12)
        ratio = np.where(y > 0, y / mu, 1.0)
        ratio = np.clip(ratio, 1e-300, 1e300)

        term = np.where(y > 0, y * np.log(ratio), 0.0)
        dev = 2.0 * np.sum(mu - y + term)
        return dev if np.isfinite(dev) else 1e30
    
class PoissonDevianceBinned:
    """Pickle-friendly Poisson deviance for histogram bins."""
    def __init__(self, edges, y, model_binned):
        self.edges = np.asarray(edges)
        self.y = np.asarray(y)
        self.model = model_binned

    def __call__(self, *p):
        mu = self.model(self.edges, *p)
        y = self.y

        if (mu is None) or (not np.all(np.isfinite(mu))) or np.any(mu <= 0):
            return 1e30
        if (y is None) or (not np.all(np.isfinite(y))) or np.any(y < 0):
            return 1e30

        mu = np.clip(mu, 1e-12, 1e12)
        ratio = np.where(y > 0, y / mu, 1.0)
        ratio = np.clip(ratio, 1e-300, 1e300)

        term = np.where(y > 0, y * np.log(ratio), 0.0)
        dev = 2.0 * np.sum(mu - y + term)
        return dev if np.isfinite(dev) else 1e30

def fit_peaks_of_CalibrationHistogram(
    calibration_histogram: CalibrationHistogram,
    max_peaks: int,
    prominence: float,
    initial_percentage: float = 0.1,
    percentage_step: float = 0.1,
    return_last_addition_if_fail: bool = False,
    fit_type: str = 'independent_gaussians',
    half_points_to_fit: int = 2,
    std_increment_seed_fallback: float = 1e+2,
    ch_span_fraction_around_peaks: float = 0.05,
    verbosity: int = 2
) -> bool:
    # ---- Robustness knobs (kept internal on purpose) ----
    # Prevent sigma collapse / blow-up (main source of under/overestimation when peaks overlap)
    _STD_MIN_BINW_FRAC = 0.60   # sigma >= 0.6*bin_width
    _STD_MIN_SEED_FRAC = 0.20   # sigma >= 0.2*sigma_seed
    _STD_MAX_SEED_MULT = 4.00   # sigma <= 4*sigma_seed (soft physical prior)
    _STD_MAX_GAIN_FRAC = 0.90   # sigma <= 0.9*gain_seed (avoid merging peaks)
    _GAIN_MIN_FRAC     = 0.25   # gain >= 0.25*gain_seed (avoid gain->0 degeneracy)
    _GAIN_MAX_MULT     = 4.00   # gain <= 4*gain_seed
    _SEED_WIN_SIGMA_K  = 1.50   # window half-width in sigmas for scale seeding
    """For the given CalibrationHistogram object, 
    calibration_histogram, this function
        
        -   tries to find the first max_peaks whose 
            prominence is greater than the given prominence 
            parameter, using the scipy.signal.find_peaks() 
            function iteratively. This function delegates 
            this task to the
            wuff.__spot_first_peaks_in_CalibrationHistogram()
            function.

        -   Then, it fits a gaussian function to each one
            of the found peaks using the output of 
            the last call to scipy.signal.find_peaks()
            (which is returned by 
            wuff.__spot_first_peaks_in_CalibrationHistogram())
            as a seed for the fit.

        -   Finally, it stores the fit parameters in the
            gaussian_fits_parameters attribute of the given
            CalibrationHistogram object, according to its 
            structure, which can be found in the CalibrationHistogram
            class documentation.

    This function returns True if the number of found peaks
    matches the given max_peaks parameter, and False
    if it is smaller than max_peaks.
    
    Parameters
    ----------
    calibration_histogram: CalibrationHistogram
        The CalibrationHistogram object to fit peaks on
    max_peaks: int
        It must be a positive integer. It gives the
        maximum number of peaks that could be possibly
        fit. This parameter is passed to the 'max_peaks'
        parameter of the 
        wuff.__spot_first_peaks_in_CalibrationHistogram()
        function.
    prominence: float
        It must be greater than 0.0 and smaller than 1.0.
        It gives the minimal prominence of the peaks to 
        spot. This parameter is passed to the 'prominence' 
        parameter of the 
        wuff.__spot_first_peaks_in_CalibrationHistogram()
        function, where it is interpreted as the fraction 
        of the total amplitude of the histogram which is 
        required for a peak to be spotted as such. P.e. 
        setting prominence to 0.5, will prevent 
        scipy.signal.find_peaks() from spotting peaks 
        whose amplitude is less than half of the total 
        amplitude of the histogram.
    initial_percentage: float
        It must be greater than 0.0 and smaller than 1.0.
        This parameter is passed to the 'initial_percentage' 
        parameter of the 
        wuff.__spot_first_peaks_in_CalibrationHistogram()
        function. For more information, check the 
        documentation of such function.
    percentage_step: float
        It must be greater than 0.0 and smaller than 1.0.
        This parameter is passed to the 'percentage_step' 
        parameter of the 
        wuff.__spot_first_peaks_in_CalibrationHistogram() 
        function. For more information, check the 
        documentation of such function.
    return_last_addition_if_fail: bool
        This parameter is given to the
        return_last_addition_if_fail parameter of the
        wuff.__spot_first_peaks_in_CalibrationHistogram()
        function. It makes a difference only if the
        specified number of peaks (max_peaks) is not
        found. For more information, check the
        documentation of the 
        wuff.__spot_first_peaks_in_CalibrationHistogram()
        function.
    fit_type: str
        The supported values are 'independent_gaussians',
        'correlated_gaussians' and 'multigauss_iminuit'. If
        any other value is given, the
        'independent_gaussians' value will be used instead.
        If the 'independent_gaussians' value is used, the
        function will fit each peak independently, i.e. it
        will fit a gaussian function to each peak
        independently of the others. For more information on
        this type of fit, check the documentation of the
        wuff.__fit_independent_gaussians_to_calibration_histogram()
        function. If the 'correlated_gaussians' value is
        given, the function will fit all of the peaks at
        once using a fitting function which is a sum of
        gaussians whose means and standard deviations are
        correlated. For more information on this type of
        fit, check the documentation of the
        wuff.__fit_correlated_gaussians_to_calibration_histogram()
        function.
    half_points_to_fit: int
        This parameter is only used if the fit_type
        parameter is set to 'independent_gaussians'.
        It must be a positive integer. For each peak, it 
        gives the number of points to consider on either 
        side of the peak maximum, to fit each gaussian 
        function. I.e. if i is the iterator value for
        calibration_histogram.counts of the i-th peak, 
        then the histogram bins which will be considered 
        for the fit are given by the slice 
        calibration_histogram.counts[i - half_points_to_fit : i + half_points_to_fit + 1].
    std_increment_seed_fallback: float
        This parameter is only used if the fit_type
        parameter is set to 'correlated_gaussians'.
        In that case, it is given to the
        std_increment_seed_fallback parameter of the
        wuff.__fit_correlated_gaussians_to_calibration_histogram()
        function. For more information, check the
        documentation of such function.
    ch_span_fraction_around_peaks: float
        This parameter is only used if the fit_type
        parameter is set to 'correlated_gaussians'.
        In that case, it is given to the
        ch_span_fraction_around_peaks parameter of the
        wuff.__fit_correlated_gaussians_to_calibration_histogram()
        function. For more information, check the
        documentation of such function.

    Returns
    -------
    bool
        True if the number of found-and-fitted peaks matches
        the given max_peaks parameter, and False if it is
        smaller than max_peaks.
    """

    if max_peaks < 1:
        raise Exception(GenerateExceptionMessage(
            1,
            'fit_peaks_of_CalibrationHistogram()',
            f"The given max_peaks ({max_peaks}) "
            "must be greater than 0."))
    
    if prominence <= 0.0 or prominence >= 1.0:
        raise Exception(GenerateExceptionMessage( 
            2,
            'fit_peaks_of_CalibrationHistogram()',
            f"The given prominence ({prominence}) "
            "must be greater than 0.0 and smaller than 1.0."))
    
    if initial_percentage <= 0.0 or initial_percentage >= 1.0:
        raise Exception(GenerateExceptionMessage( 
            3,
            'fit_peaks_of_CalibrationHistogram()',
            f"The given initial_percentage ({initial_percentage})"
            " must be greater than 0.0 and smaller than 1.0."))

    if percentage_step <= 0.0 or percentage_step >= 1.0:
        raise Exception(GenerateExceptionMessage(
            4,
            'fit_peaks_of_CalibrationHistogram()',
            f"The given percentage_step ({percentage_step})"
            " must be greater than 0.0 and smaller than 1.0."))

    # -------- verbosity helper ----------
    def _v(level: int, msg: str):
        if verbosity >= level:
            print(msg)

    _v(1, f"[fit_peaks] Start: max_peaks={max_peaks}, prom={prominence:.3g}, "
           f"fit_type={fit_type}, half_pts={half_points_to_fit}, verb={verbosity}")
    
    calibration_histogram._CalibrationHistogram__reset_gaussian_fit_parameters()
    _v(3, "[fit_peaks] Reset previous gaussian_fits_parameters.")

    fFoundMax, spsi_output = wuff.__spot_first_peaks_in_CalibrationHistogram(   
        calibration_histogram,
        max_peaks,
        prominence,
        initial_percentage=initial_percentage,
        percentage_step=percentage_step,
        return_last_addition_if_fail=return_last_addition_if_fail
    )
    _v(1, f"[fit_peaks] spot_first_peaks: found={fFoundMax}")
    
    if fit_type == 'correlated_gaussians':
        fFitAll = wuff.__fit_correlated_gaussians_to_calibration_histogram(
            spsi_output,
            calibration_histogram,
            std_increment_seed_fallback=std_increment_seed_fallback,
            ch_span_fraction_around_peaks=ch_span_fraction_around_peaks
        )
    else:
        fFitAll = wuff.__fit_independent_gaussians_to_calibration_histogram(
            spsi_output,
            calibration_histogram,
            half_points_to_fit
        )
    _v(1, f"[fit_peaks] initial seed fits done: ok={fFitAll}")

    # --- NEW: if we found some peaks but fewer than requested, try to recover missing peaks ---
    # This is important for faint right-tail peaks: the initial prominence can miss them.
    if fit_type == "multigauss_iminuit":
        try:
            g = calibration_histogram.gaussian_fits_parameters
            n_seed = len(g.get("mean", [])) if isinstance(g, dict) else 0
        except Exception:
            n_seed = 0

        if 0 < n_seed < max_peaks:
            _v(1, f"[fit_peaks] Only {n_seed}/{max_peaks} seed peaks found — trying relaxed spotting.")
            # progressively relax prominence to catch small peaks on the right
            prom_grid = [
                max(1e-6, prominence * 0.80),
                max(1e-6, prominence * 0.60),
                max(1e-6, prominence * 0.40),
                max(1e-6, prominence * 0.25),
            ]
            for prom_ in prom_grid:
                calibration_histogram._CalibrationHistogram__reset_gaussian_fit_parameters()
                fFoundMax, spsi_output = wuff.__spot_first_peaks_in_CalibrationHistogram(
                    calibration_histogram,
                    max_peaks,
                    prom_,
                    initial_percentage=min(0.05, initial_percentage),
                    percentage_step=min(0.05, percentage_step),
                    return_last_addition_if_fail=True
                )
                # regenerate seeds with independent fits (stable)
                _ = wuff.__fit_independent_gaussians_to_calibration_histogram(
                    spsi_output, calibration_histogram, half_points_to_fit
                )
                try:
                    g2 = calibration_histogram.gaussian_fits_parameters
                    n2 = len(g2.get("mean", [])) if isinstance(g2, dict) else 0
                except Exception:
                    n2 = 0
                _v(2, f"[fit_peaks]  relaxed prom={prom_:.3g} -> seeds={n2}")
                if n2 >= max_peaks:
                    _v(1, "[fit_peaks]  recovered missing peak seeds.")
                    break

    # -------- NEW: Retry strategy if peak spotting / seeding failed --------
    # Defensive: if nothing was fitted yet (no peaks), attempt a few relaxed
    # peak-finding settings before giving up.
    def _have_fitted_any_peak(ch: CalibrationHistogram) -> bool:
        g = ch.gaussian_fits_parameters
        return (
            isinstance(g, dict)
            and 'mean' in g and 'scale' in g and 'std' in g
            and len(g['mean']) > 0
        )

    if not _have_fitted_any_peak(calibration_histogram):
        _v(1, "[fit_peaks] No peaks fitted after first pass — starting relaxed retries.")
        retry_grid = [
            # (prominence, initial_percentage, percentage_step)
            (max(1e-6, prominence * 0.5), 0.05, 0.05),
            (max(1e-6, prominence * 0.25), 0.05, 0.05),
            (max(1e-6, prominence * 0.25), 0.02, 0.05),
        ]
        for prom_, init_perc_, step_ in retry_grid:
            _v(2, f"[fit_peaks] Retry with prom={prom_:.3g}, init%={init_perc_:.2f}, step%={step_:.2f}")
            calibration_histogram._CalibrationHistogram__reset_gaussian_fit_parameters()
            fFoundMax, spsi_output = wuff.__spot_first_peaks_in_CalibrationHistogram(
                calibration_histogram,
                max_peaks,
                prom_,
                initial_percentage=init_perc_,
                percentage_step=step_,
                return_last_addition_if_fail=True
            )
            # Use independent fits to regenerate seeds
            fFitAll = wuff.__fit_independent_gaussians_to_calibration_histogram(
                spsi_output,
                calibration_histogram,
                half_points_to_fit
            )
            if _have_fitted_any_peak(calibration_histogram):
                _v(2, "[fit_peaks] Recovery fit succeeded.")
                break  # we recovered
    # --- Bin width (assume uniform bins; use robust estimator) ---
    # We'll still use a representative bin width for conversions between:
    #  - area (total counts) <-> height (counts/bin) for plotting/storage
    binw = float(np.median(np.diff(calibration_histogram.edges)))
    binw = max(binw, 1e-12)
    SQRT_2PI = math.sqrt(2.0 * math.pi)

    def _area_to_height_density(area: float, sigma: float) -> float:
        """Convert Gaussian area (total counts) -> peak height in counts / x-unit (density)."""
        sigma = max(float(sigma), 1e-12)
        return float(area) / (SQRT_2PI * sigma)

    def _bin_width_at(edges: np.ndarray, x: float) -> float:
        edges = np.asarray(edges)
        j = int(np.searchsorted(edges, x, side="right") - 1)
        j = max(0, min(j, len(edges) - 2))
        return max(float(edges[j + 1] - edges[j]), 1e-12)

    if fit_type != 'multigauss_iminuit':
        _v(1, f"[fit_peaks] Non-iminuit mode → returning {fFoundMax & fFitAll}")
        return fFoundMax & fFitAll

    n_peaks_found = len(calibration_histogram.gaussian_fits_parameters['scale'])
    _v(1, f"[fit_peaks] Proceeding with multigauss_iminuit, n_peaks_found={n_peaks_found}")


    # initialize parameters for iminuit
    # -------- multigauss_iminuit branch: robustness + GoF --------
    # Guard: if no seeds available, bail out safely.
    gfp = calibration_histogram.gaussian_fits_parameters
    if not (isinstance(gfp, dict)
            and 'scale' in gfp and 'mean' in gfp and 'std' in gfp
            and len(gfp['scale']) > 0 and len(gfp['mean']) > 0 and len(gfp['std']) > 0):
        _v(1, "[fit_peaks] Abort: no valid seeds available for iminuit.")
        return False

    # --- Modeling policy ---
    # (We will SWEEP this minimum later, starting at 4)
    # Keep a baseline minimum here; the sweep will override it.
    min_modeled_peaks = 2  # total peaks incl. baseline
    # Enforce equal sigma across peaks (propstd == 0)
    force_equal_std = False
    _v(3, f"[fit_peaks] Policy: force_equal_std={force_equal_std}")

    # --- Build immutable BASE seed vector (5 params) ---
    # IMPORTANT: In this branch we enforce a consistent physical meaning:
    #   scale_* : area (total expected counts) of each Gaussian component
    #   bkg     : constant background in counts/bin
    base_paramnames = [
        'scale_baseline', 'mean_baseline', 'std_baseline',
        'gain', 'propstd',
        'bkg',               # counts/bin
        'bkgG', 'bkgG_mean', 'bkgG_std'  # gaussian background (height counts/bin + shape)
    ]
    base_params = [
        1.0,                        # scale_baseline (area) – will be reseeded robustly below
        float(gfp['mean'][0][0]),   # mean_baseline
        float(gfp['std'][0][0]),    # std_baseline
    ]

    # Estimar gain como separación entre picos (no como posición absoluta)
    if n_peaks_found > 1:
        gain_seed = float(gfp['mean'][1][0] - gfp['mean'][0][0])
    else:
        # fallback si no hay 1pe: algo proporcional al ancho
        gain_seed = 2.0 * float(gfp['std'][0][0])

    # std_prop seed (si lo estás dejando libre)
    if force_equal_std:
        stdprop_seed = 0.0
    else:
        # una semilla suave, no crítica
        stdprop_seed = max(0.0, 0.5 * float(gfp['std'][0][0]))

    base_params.append(gain_seed)     # gain
    base_params.append(stdprop_seed)  # propstd

    # --- background seeds ---
    # If you prefer a "non-flat" background, seed the gaussian component and start bkg flat at ~0
    bkg_seed_bin = float(np.percentile(calibration_histogram.counts, 5))
    bkg_seed_bin = max(0.0, bkg_seed_bin)
    base_params.append(0.0)  # bkg (flat) starts at 0 to avoid visible offset by default

    # Gaussian background: height ~ low percentile, mean ~ center of domain, std ~ wide fraction of span
    x0 = float(0.5 * (calibration_histogram.edges[0] + calibration_histogram.edges[-1]))
    span = float(calibration_histogram.edges[-1] - calibration_histogram.edges[0])
    base_params.append(bkg_seed_bin)          # bkgG (height counts/bin)
    base_params.append(x0)                   # bkgG_mean
    base_params.append(max(1e-6, 0.35*span)) # bkgG_std (wide but finite) 

    # Estimates the number of peaks that should be fitted based on the histogram
    # Starts with a huge number of peaks, and then reduces it
    # Decide an upper bound for candidate model size; do not mutate base lists
    # Too many free scales makes the likelihood extremely degenerate and Minuit often becomes invalid.
    # Keep it close to what you actually see; the sweep can still increase model order if needed.
    n_peaks_to_fit_iminuit = min(n_peaks_found + 6, 20)

    data_x = (calibration_histogram.edges[:-1] + calibration_histogram.edges[1:]) * 0.5
    data_y = calibration_histogram.counts

    # ---------------------------
    # Local binned multi-gauss model (UNITS CONSISTENT)
    # scale_* : area (total counts)
    # bkg     : counts/bin (constant)
    # ---------------------------
    # NOTE: model is now MultiGaussBinnedModel (top-level, picklable)

    def _seed_scale_from_window(edges_all, y_all, mean, std, k=_SEED_WIN_SIGMA_K, eps=1e-12):
        """
        Robust AREA seed using a window around the peak (not a single bin).
        This is critical when peaks overlap: a single bin near 'mean' is easily contaminated,
        and Minuit compensates by shrinking/expanding sigma.
        """
        edges_all = np.asarray(edges_all)
        y_all = np.asarray(y_all)
        if not (np.isfinite(mean) and np.isfinite(std)) or std <= 0:
            return None

        lo = mean - k * std
        hi = mean + k * std
        i0 = int(np.searchsorted(edges_all, lo, side="right") - 1)
        i1 = int(np.searchsorted(edges_all, hi, side="left") - 1)
        i0 = max(0, i0)
        i1 = min(len(y_all) - 1, i1)
        if i1 < i0:
            return None

        edges_win = edges_all[i0:i1+2]
        y_win = y_all[i0:i1+1]

        # bkg is counts/bin (NOT density). Subtract per-bin directly.
        bkg_bin = float(base_params[5]) if len(base_params) > 5 else 0.0
        y_eff = y_win - max(0.0, bkg_bin)
        y_eff = np.clip(y_eff, 0.0, None)

        # expected counts/bin in each bin for unit AREA (scale=1)
        mu_unit = wun.gaussian_bin_counts(edges_win, 1.0, mean, std)
        mu_unit = np.clip(mu_unit, eps, None)

        # solve AREA from total counts in window (robust)
        num = float(np.sum(y_eff))
        den = float(np.sum(mu_unit))
        if not (np.isfinite(num) and np.isfinite(den)) or den <= eps:
            return None
        return num / max(den, eps)

    data_err = np.sqrt(data_y)
    data_err[data_err == 0] = 1  # avoid division by zero

    mu0_seed  = float(base_params[1])
    s0_seed   = float(base_params[2])

    # usa un gain_seed (lo corregimos abajo en 4.2)
    gain_seed = float(base_params[3])

    # número de picos modelados (baseline + PE)
    # si estás probando modelos con diferente cantidad, usa el “n_scales” de cada candidato
    n_scales_seed = max(2, n_peaks_found)  # baseline + al menos 1pe

    # ancho del último pico según el modelo
    s_last = np.sqrt(s0_seed**2 + (float(base_params[4])**2) * (n_scales_seed - 1))

    L, R = 4.0, 4.0
    xmin = mu0_seed - L * s0_seed
    xmax = mu0_seed + (n_scales_seed - 1) * gain_seed + R * s_last

    mask = (data_x >= xmin) & (data_x <= xmax)
    x_fit = data_x[mask]
    y_fit = data_y[mask]

    def _select_fit_window(n_scales: int, L: float = 4.0, R: float = 4.0):
        mu0  = float(base_params[1])
        s0   = float(base_params[2])
        gain = float(base_params[3])
        prop = float(base_params[4])

        # Anchor window to the right-most seed mean, so we don't "ignore" the tail.
        try:
            seed_means = [float(m[0]) for m in gfp.get('mean', [])]
        except Exception:
            seed_means = []
        mu_model_right = mu0 + (n_scales - 1) * gain
        mu_right = max(max(seed_means) if seed_means else mu0, mu_model_right)
        mu_left  = min(min(seed_means) if seed_means else mu0, mu0)
        s_last = np.sqrt(s0**2 + (prop**2) * max(0, n_scales - 1))
        xmin = mu_left - L * s0
        xmax = mu_right + R * max(s_last, s0)

        edges = calibration_histogram.edges
        y = calibration_histogram.counts

        i0 = max(0, np.searchsorted(edges, xmin, side="right") - 1)
        i1 = min(len(y) - 1, np.searchsorted(edges, xmax, side="left") - 1)

        # fallback si quedó muy chico
        if (i1 - i0 + 1) < 20:
            return edges, y

        edges_fit = edges[i0:i1+2]   # +2 porque edges = nbins+1
        y_fit     = y[i0:i1+1]
        return edges_fit, y_fit

    def _run_minuit_fit(pnames, pvals, edges_fit, y_fit):
        if len(pnames) != len(pvals):
            raise RuntimeError("Parameter name/value length mismatch")

        # Bind local model with consistent units
        model = MultiGaussBinnedModel(pnames)
        cost = PoissonDevianceBinned(edges_fit, y_fit, model)
        mm = Minuit(cost, *pvals, name=pnames)
        mm.strategy = 2  # more robust in correlated / degenerate problems

        # ---- Physical / anti-degeneracy bounds (KEY FIX) ----
        # Without a lower bound on std, Poisson binned mixtures often collapse sigma->0,
        # especially when peaks overlap (your 1st figure).
        bw_loc = float(np.median(np.diff(edges_fit)))
        bw_loc = max(bw_loc, 1e-12)

        # Upper bounds for amplitudes/background to prevent mu -> inf (NaNs in deviance).
        # y_fit is counts/bin, while scale_* are counts/x-unit (density) in this model.
        max_y = float(np.max(y_fit)) if len(y_fit) else 1.0
        scale_max = max(1.0, 50.0 * max_y / bw_loc)   # counts/x-unit (very generous)
        bkg_max   = max(1.0, 2.0 * max_y)             # counts/bin

        gain0 = float(base_params[3])
        s0    = float(base_params[2])
        std_min = max(_STD_MIN_BINW_FRAC * bw_loc, _STD_MIN_SEED_FRAC * s0)
        std_max = max(std_min * 1.05, min(_STD_MAX_SEED_MULT * s0, _STD_MAX_GAIN_FRAC * max(gain0, std_min)))
        gain_min = max(std_min * 1.1, _GAIN_MIN_FRAC * gain0)
        gain_max = max(gain_min * 1.05, _GAIN_MAX_MULT * gain0)

        for p in mm.parameters:
            if p == "mean_baseline":
                mm.limits[p] = (None, None)
            elif p == "std_baseline":
                mm.limits[p] = (std_min, std_max)
            elif p == "gain":
                mm.limits[p] = (gain_min, gain_max)
            elif p == "propstd":
                mm.limits[p] = (0.0, std_max)  # keep increments reasonable
            elif p.startswith("scale_"):
                # scale_* are amplitudes (density, counts/x). Cap them to avoid mu overflow.
                mm.limits[p] = (0.0, scale_max)
            elif p in ("bkg", "bkgG"):
                # background parameters are counts/bin-like in this codepath
                mm.limits[p] = (0.0, bkg_max)
            else:
                mm.limits[p] = (0.0, None)
        # keep explicit non-negativity (already enforced above), do not overwrite upper bounds

        # Gaussian background limits (if enabled)
        if "bkgG" in mm.parameters:
            mm.limits["bkgG"] = (0.0, None)  # height counts/bin
        if "bkgG_mean" in mm.parameters:
            mm.limits["bkgG_mean"] = (float(edges_fit[0]), float(edges_fit[-1]))
        if "bkgG_std" in mm.parameters:
            xr = float(edges_fit[-1] - edges_fit[0])
            xr = max(xr, 1e-9)
            mm.limits["bkgG_std"] = (0.15 * xr, 3.0 * xr)
        # ---------------- Stage A (more robust) ----------------
        # Do NOT fix baseline/1pe scales: when peaks overlap, wrong amplitude seeds
        # force Minuit to "cheat" via sigma. Keep only higher-order peaks fixed.
        for p in mm.parameters:
            if p.startswith("scale_"):
                if p in ("scale_baseline", "scale_1pe"):
                    mm.fixed[p] = False
                else:
                    mm.fixed[p] = True

        # Background policy:
        # - If gaussian background is present, keep flat bkg disabled (fixed at 0).
        # - Otherwise allow flat bkg to float (if you still want it).
        if "bkg" in mm.parameters and "bkgG" in mm.parameters:
            mm.values["bkg"] = 0.0
            mm.fixed["bkg"] = True
        elif "bkg" in mm.parameters:
            mm.fixed["bkg"] = False

        # Stage-A: gaussian background: free only the HEIGHT, keep mean/std fixed (stability)
        if "bkgG" in mm.parameters:
            mm.fixed["bkgG"] = False
        if "bkgG_mean" in mm.parameters:
            mm.fixed["bkgG_mean"] = True
        if "bkgG_std" in mm.parameters:
            mm.fixed["bkgG_std"] = True

        # Keep correlated-width parameter fixed initially (helps stability)
        if 'propstd' in mm.parameters:
            mm.fixed['propstd'] = True
        # Allow alignment parameters to move
        if 'mean_baseline' in mm.parameters:
            mm.fixed['mean_baseline'] = False
        if 'gain' in mm.parameters:
            mm.fixed['gain'] = False
        # Optionally allow baseline width to adjust already in Stage A
        if 'std_baseline' in mm.parameters:
            mm.fixed['std_baseline'] = False

        try:
            mm.migrad()
        except Exception:
            # fallback: simplex can rescue bad starting points in degenerate mixtures
            try:
                mm.simplex()
                mm.migrad()
            except Exception:
                return mm, False

        # Stage B
        for p in mm.parameters:
            mm.fixed[p] = False
            # keep previously set bounds (std/gain), do not overwrite them here
            if p == "mean_baseline":
                mm.limits[p] = (None, None)

        # Keep the "no flat offset" policy also in Stage B
        if "bkg" in mm.parameters and "bkgG" in mm.parameters:
            mm.values["bkg"] = 0.0
            mm.fixed["bkg"] = True

        # Optional: only now allow bkgG_mean/std to float (if you really want)
        # If you want max stability, leave them fixed always.
        if "bkgG_mean" in mm.parameters:
            mm.fixed["bkgG_mean"] = False
        if "bkgG_std" in mm.parameters:
            mm.fixed["bkgG_std"] = False

        # keep background non-negative (explicit) — upper bounds already set above
        if 'bkg' in mm.parameters:
            try:
                if mm.limits['bkg'] is None:
                    mm.limits['bkg'] = (0.0, bkg_max)
            except Exception:
                mm.limits['bkg'] = (0.0,	 bkg_max)

        # Keep bounds for propstd/gain/std_baseline as defined above (avoid sigma collapse)
        if 'propstd' in mm.parameters and mm.limits['propstd'] is None:
            mm.limits['propstd'] = (0.0, std_max)

        try:
            mm.migrad(); mm.migrad()
            mm.hesse()
        except Exception:
            return mm, False

        return mm, bool(mm.fmin.is_valid)

    def _goodness(mm, n_data: int):
        n_free = sum(1 for p in mm.parameters if not mm.fixed[p])
        dof = max(1, n_data - n_free)
        # NOTE: with Poisson deviance, fval is a deviance, not a LS chi2
        dev = float(mm.fval)
        redchi2 = dev / dof
        k = n_free
        AIC = dev + 2.0 * k
        BIC = dev + k * np.log(max(1, n_data))
        return dict(chi2=dev, dof=dof, redchi2=redchi2, AIC=AIC, BIC=BIC)
    
    # Build candidate param sets from the immutable base ONLY
    def _build_params_for(n_scales: int, _min_modeled_peaks: int):
        """
        n_scales = number of total peaks to model (including baseline peak).
        For n_scales=1, only baseline is modeled (no scale_1pe).
        For n_scales>=2, we add scale_1pe .. scale_{n_scales-1}pe.
        """

        # Enforce minimum number of peaks
        n_scales = max(n_scales, _min_modeled_peaks)
        pnames = list(base_paramnames)   # 5 base names
        pvals  = list(base_params)       # 5 base vals
        edges_all = calibration_histogram.edges
        y_all     = calibration_histogram.counts

        mu0_seed  = float(base_params[1])
        std0_seed = float(base_params[2])
        gain_seed = float(base_params[3])
        prop_seed = float(base_params[4])

        # ---- Fix #1 (binning-invariant amplitude seeds) ----
        # Renormalize baseline scale seed using the bin containing mu0_seed
        s0 = _seed_scale_from_window(edges_all, y_all, mu0_seed, std0_seed)
        if s0 is not None:
            pvals[0] = float(s0)

        for i in range(1, n_scales):
            mean_i = mu0_seed + i * gain_seed
            std_i  = np.sqrt(std0_seed**2 + (prop_seed**2) * i)

            si = _seed_scale_from_window(edges_all, y_all, mean_i, std_i)
            if si is None:
                # Fallback: take central bin count (counts/bin) and convert to amplitude density (counts/x)
                idx = int(np.argmin(np.abs(data_x - mean_i)))
                idx = max(0, min(idx, len(y_all) - 1))
                yb = float(y_all[idx])  # counts/bin
                bkg_bin = float(base_params[5])
                yb_eff = max(0.0, yb - max(0.0, bkg_bin))
                bw_i = _bin_width_at(edges_all, mean_i)
                # amplitude density seed (counts/x)
                si = max(1e-6, 0.95 * yb_eff / bw_i)

            pvals.append(float(si))
            pnames.append(f'scale_{i}pe')
        # Final sanity: lengths must match
        assert len(pnames) == len(pvals), (len(pnames), len(pvals))
        return pnames, pvals

    # -------- SWEEP min_modeled_peaks and select best by BIC (with plateau stop) --------
    def _fit_for_min(_min_pe: int):
        """Return (best_mm, best_gof, ok) for a single min-modeled-peaks setting."""
        # Candidate A: heuristic upper size
        _v(2, f"[sweep] Try min_modeled_peaks={_min_pe}")
        pA_names, pA_vals = _build_params_for(n_peaks_to_fit_iminuit, _min_pe)
        nA = max(_min_pe, n_peaks_to_fit_iminuit)
        edgesA, yA = _select_fit_window(nA)
        mmA, okA = _run_minuit_fit(pA_names, pA_vals, edgesA, yA)
        gofA = _goodness(mmA, len(yA)) if okA else None
        if okA and verbosity >= 2:
            _v(2, f"[sweep]  CandA(n={len(pA_names)}): BIC={gofA['BIC']:.4g}")
        # Candidate B: truncated to measured peaks
        trunc_n = max(_min_pe, n_peaks_found)
        pB_names, pB_vals = _build_params_for(trunc_n, _min_pe)
        nB = max(_min_pe, n_peaks_found)
        edgesB, yB = _select_fit_window(nB)
        mmB, okB = _run_minuit_fit(pB_names, pB_vals, edgesB, yB)
        gofB = _goodness(mmB, len(yB)) if okB else None
        if okB and verbosity >= 2:
            _v(2, f"[sweep]  CandB(n={len(pB_names)}): BIC={gofB['BIC']:.4g}")

        # If neither converged, try relaxed pass on A: free gain/scale_1pe
        if not (okA or okB):
            edgesR, yR = _select_fit_window(nA)
            modelR = MultiGaussBinnedModel(pA_names)
            costR = PoissonDevianceBinned(edgesR, yR, modelR)
            mmR = Minuit(costR, *pA_vals, name=pA_names)
            mmR.strategy = 2

            # fully relax (do NOT fix gain or scale_1pe here)
            for p in mmR.parameters:
                mmR.fixed[p] = False

            # --- Keep the SAME anti-degeneracy limits as the main fit ---
            # If we drop std/gain bounds here, Poisson mixtures often collapse sigma->0
            # and "eat" peaks (especially in the right tail).
            bw_loc = float(np.median(np.diff(edgesR)))
            bw_loc = max(bw_loc, 1e-12)

            gain0 = float(base_params[3])
            s0    = float(base_params[2])
            std_min = max(_STD_MIN_BINW_FRAC * bw_loc, _STD_MIN_SEED_FRAC * s0)
            std_max = max(std_min * 1.05, min(_STD_MAX_SEED_MULT * s0,
                                              _STD_MAX_GAIN_FRAC * max(gain0, std_min)))
            gain_min = max(std_min * 1.1, _GAIN_MIN_FRAC * gain0)
            gain_max = max(gain_min * 1.05, _GAIN_MAX_MULT * gain0)

            for p in mmR.parameters:
                if p == "mean_baseline":
                    mmR.limits[p] = (None, None)
                elif p == "std_baseline":
                    mmR.limits[p] = (std_min, std_max)
                elif p == "gain":
                    mmR.limits[p] = (gain_min, gain_max)
                elif p == "propstd":
                    mmR.limits[p] = (0.0, std_max)
                else:
                    mmR.limits[p] = (0.0, None)
            if "bkg" in mmR.parameters:
                mmR.limits["bkg"] = (0.0, None)           
            try:
                mmR.migrad(); mmR.hesse()
                okA = bool(mmR.fmin.is_valid)
                mmA = mmR
                gofA = _goodness(mmA, len(yR)) if okA else None
            except Exception:
                okA = False
                mmA = mmR
                gofA = None
        # Verbose: print relaxed pass result with safe BIC formatting
        if verbosity >= 2:
            _bic_str = "NA"
            if gofA is not None and "BIC" in gofA:
                _bic_str = f"{gofA['BIC']:.4g}"
            _v(2, f"[sweep]  RelaxedA: ok={okA} BIC={_bic_str}")

        # Choose winner for this _min_pe by BIC
        if okA and okB:
            mm_win, gof_win = (mmA, gofA) if gofA['BIC'] <= gofB['BIC'] else (mmB, gofB)
        elif okA:
            mm_win, gof_win = mmA, gofA
        elif okB:
            mm_win, gof_win = mmB, gofB
        else:
            return None, None, False

        # Require at least a 1-pe peak in params
        scales = [p for p in mm_win.parameters if p.startswith("scale_") and p.endswith("pe")]
        if "scale_1pe" not in scales:
            return None, None, False

        return mm_win, gof_win, True

    # Sweep configuration
    sweep_start = 3
    # Cap the sweep to avoid runaway models; can tune if needed
    sweep_stop  = min(max(n_peaks_found + 5, sweep_start + 2), 20)

    # If peaks are very close compared to seed sigma, allow the sweep to start smaller.
    # This reduces failures where the optimizer can't stabilize a too-large model early.
    if n_peaks_found >= 2 and float(base_params[3]) < 2.5 * float(base_params[2]):
        sweep_start = 2

    bic_tol     = 2   # plateau threshold (smaller is stricter)
    # New: require "strong" improvement to accept a larger model (Kass & Raftery)
    # DO NOT gate BIC updates: it makes the algorithm "ignore" right-tail peaks.
    bic_improve_min = 0.0
    patience    = 5     # consecutive non-improvements to stop

    best_mm, best_gof = None, None
    best_bic = np.inf
    stall = 0
    for m in range(sweep_start, sweep_stop + 1):
        mm_try, gof_try, ok_try = _fit_for_min(m)
        if not ok_try:
            continue
        # Only accept as "new best" if BIC improves by a meaningful margin
        # Standard selection: keep the minimum BIC
        if gof_try['BIC'] + 1e-12 < best_bic - bic_improve_min:
            best_bic = gof_try['BIC']
            best_mm, best_gof = mm_try, gof_try
            stall = 0
            _v(1, f"[sweep]  m={m}: new best BIC={best_bic:.4g}")
        else:
            # Plateau detection: improvement smaller than bic_tol
            if (gof_try['BIC'] - best_bic) < bic_tol:
                stall += 1
                _v(2, f"[sweep]  m={m}: plateau step ({stall}/{patience}), BIC={gof_try['BIC']:.4g}")
                if stall >= patience:
                    _v(1, f"[sweep]  Early stop at m={m} (plateau).")
                    break
            else:
                stall = 0

    if best_mm is None:
        _v(1, "[fit_peaks] Sweep failed to produce a valid model.")
        return False

    # --- NEW: if best model has fewer peaks than requested, try forcing max_peaks once ---
    # Rationale: BIC may prefer a smaller model even when a weak but real PE peak exists.
    try:
        best_scale_names = [p for p in best_mm.parameters if p.startswith("scale_") and p.endswith("pe")]
        best_n_scales = 1 + len(best_scale_names)  # baseline + npe
    except Exception:
        best_n_scales = 0

    # allow a small BIC penalty to still accept the forced model
    FORCE_MAXPEAKS_BIC_DELTA = 15.0
    if best_n_scales > 0 and best_n_scales < max_peaks:
        _v(1, f"[fit_peaks] Best model has {best_n_scales} peaks, trying forced {max_peaks}.")
        pF_names, pF_vals = _build_params_for(max_peaks, max_peaks)
        edgesF, yF = _select_fit_window(max_peaks)
        mmF, okF = _run_minuit_fit(pF_names, pF_vals, edgesF, yF)
        if okF:
            gofF = _goodness(mmF, len(yF))
            # Accept if it doesn't worsen "too much"
            if gofF["BIC"] <= (best_bic + FORCE_MAXPEAKS_BIC_DELTA):
                best_mm, best_gof = mmF, gofF
                best_bic = gofF["BIC"]
                _v(1, f"[fit_peaks] Forced model accepted: BIC={best_bic:.4g}")
            else:
                _v(1, f"[fit_peaks] Forced model rejected: BIC={gofF['BIC']:.4g} (best {best_bic:.4g})")
        else:
            _v(1, "[fit_peaks] Forced model failed to converge.")

    mm = best_mm
    fitstatus = True

    # Compute present scale_i parameters BEFORE checking for scale_1pe
    scale_param_names = [p for p in mm.parameters if p.startswith("scale_") and p.endswith("pe")]
    if "scale_1pe" not in scale_param_names:
        p2_names, p2_vals = _build_params_for(2, 2)
        edges2, y2 = _select_fit_window(2)
        mm2, ok2 = _run_minuit_fit(p2_names, p2_vals, edges2, y2)
        if ok2:
            mm = mm2
            scale_param_names = [p for p in mm.parameters if p.startswith("scale_") and p.endswith("pe")]       
    # Final guard: if we still don't have a 1pe scale, fail gracefully
    if "scale_1pe" not in scale_param_names:
        _v(1, "[fit_peaks] No scale_1pe found after fallback → returning False.")
        return False


    # -------- POST-FIT TAIL PRUNING to avoid overfitting --------
    # Heuristics: if trailing peaks have tiny height or area, prune them and refit.
    # Thresholds (tunable or could be arguments):
    min_peak_snr   = 3.0    # min height SNR vs residual noise
    min_rel_area   = 0.02   # min area relative to total modeled area
    rel_height_min   = 0.05   # min height vs tallest fitted peak

    # Build model prediction to compute residuals
    data_y = calibration_histogram.counts
    # Consistent residuals: Poisson (Pearson) residuals on binned expected counts
    model_post = MultiGaussBinnedModel(list(mm.parameters))
    mu = model_post(calibration_histogram.edges, *[float(mm.values[p]) for p in mm.parameters])
    resid = (data_y - mu) / np.sqrt(np.clip(mu, 1.0, None))  # Pearson residuals

    # Robust noise estimate via MAD (on Pearson residuals)
    med = np.median(resid)
    mad = np.median(np.abs(resid - med))
    noise_sigma = max(1e-12, 1.4826 * mad)  # ~= std for normal

    # Gather peak metrics
    std0 = float(mm.values['std_baseline'])
    # ValueView has no .get(); guard with parameter presence
    propstd = float(mm.values['propstd']) if 'propstd' in mm.parameters else 0.0
    peak_names = [p for p in mm.parameters if p.startswith("scale_") and p.endswith("pe")]
    peak_names.sort(key=lambda s: int(s.split('_')[1].replace('pe','')))  # scale_1pe, scale_2pe, ...
    # Use peak "height" in COUNTS PER BIN (expected mu at best bin)
    # and peak "area" as TOTAL COUNTS (sum of mu_i over bins).
    edges = calibration_histogram.edges
    mu_total = mu
    peak_mu_max = []
    peak_mu_tot_at_max = []
    peak_area_counts = []
    for k, p in enumerate(peak_names, start=1):
        scale_i = float(mm.values[p])
        mean_i  = float(mm.values['mean_baseline']) + k * float(mm.values['gain'])
        std_i   = float(np.sqrt(std0**2 + (propstd**2) * k))
        mu_i = wun.gaussian_bin_counts(edges, scale_i, mean_i, std_i)
        peak_area_counts.append(float(np.sum(mu_i)))
        j = int(np.argmax(mu_i))
        peak_mu_max.append(float(mu_i[j]))
        peak_mu_tot_at_max.append(float(mu_total[j]))
    heights = np.array(peak_mu_max, dtype=float)
    areas_counts = np.array(peak_area_counts, dtype=float)

    stds    = np.array([np.sqrt(std0**2 + propstd**2 * i) for i in range(1, 1+len(peak_names))], dtype=float)
    total_area = float(np.sum(areas_counts)) if areas_counts.size else 1.0
    rel_area   = areas_counts / max(total_area, 1e-12)
    # SNR in Poisson units at the peak bin, inflated by extra dispersion (noise_sigma)
    denom = noise_sigma * np.sqrt(np.clip(np.array(peak_mu_tot_at_max), 1.0, None))
    snr        = heights / np.clip(denom, 1e-12, None)
    max_h      = np.max(heights) if heights.size else 1.0
    rel_height = heights / max(max_h, 1e-12)

    # Significance: require BOTH a strong statistic (SNR or area) AND reasonable height,
    # OR pass the gain-alignment test (for faint but well-aligned right-tail peaks).
    strong_stat = ((snr >= min_peak_snr) | (rel_area >= min_rel_area)) & (rel_height >= rel_height_min)
    significant = strong_stat
    if significant.any():
        last_sig_idx = np.where(significant)[0].max()  # 0-based among scales (1pe -> idx 0)
    else:
        last_sig_idx = -1  # no significant peaks; we'll fall back to 1pe check below

    # If we have trailing non-significant peaks, prune to last_sig_idx+1
    desired_n_scales = max(2, last_sig_idx + 2)  # +1 to convert idx->count, +1 for baseline
    current_n_scales = 1 + len(peak_names)
    if desired_n_scales < current_n_scales:
        _v(1, f"[prune] Pruning from {current_n_scales} to {desired_n_scales} modeled peaks "
               f"(min_peak_snr={min_peak_snr}, min_rel_area={min_rel_area}).")
        p_names, p_vals = _build_params_for(desired_n_scales, desired_n_scales)  # force the count as the minimum
        edgesP, yP = _select_fit_window(desired_n_scales)
        mm_pruned, ok_pruned = _run_minuit_fit(p_names, p_vals, edgesP, yP)
        if ok_pruned:
            mm = mm_pruned
            scale_param_names = [p for p in mm.parameters if p.startswith("scale_") and p.endswith("pe")]
            if "scale_1pe" not in scale_param_names:
                _v(1, "[prune] After pruning, lost 1pe — reverting to previous model.")
            else:
                _v(2, f"[prune] Pruned fit converged with {len(scale_param_names)} PE peaks.")
        else:
            _v(1, "[prune] Pruned refit failed — keeping the original model.")    

    # ---- SAVE BINNED FIT CURVE (WHAT MINUIT ACTUALLY FITTED) ----
    # This is the key for plotting: expected counts per bin over the original histogram bins.
    try:
        edges_all = np.asarray(calibration_histogram.edges, dtype=float)
        x_centers = 0.5 * (edges_all[:-1] + edges_all[1:])
        pnames = list(mm.parameters)
        pvals  = [float(mm.values[p]) for p in pnames]

        model_bins = MultiGaussBinnedModel(pnames)
        mu_bins = model_bins(edges_all, *pvals)  # expected counts/bin (len = nbins)

        setattr(calibration_histogram, "fit_x_centers", x_centers)
        setattr(calibration_histogram, "fit_mu_bins",  np.asarray(mu_bins, dtype=float))
        # optional: also store the edges used for this curve
        setattr(calibration_histogram, "fit_edges", edges_all)
    except Exception:
        # don't fail the fit if plotting helpers can't be stored
        pass

    # Resize the gaussian_fits_parameters
    # scale_* fitted by gaussian_bin_counts is a CONTINUOUS amplitude (counts/x-unit).
    # For waffles plotting (which uses gaussian(x)=scale*exp(...)) over a histogram (counts/bin),
    # store scale as "counts/bin height" by multiplying by the representative bin width.
    bw0 = float(np.median(np.diff(calibration_histogram.edges)))
    bw0 = max(bw0, 1e-12)

    calibration_histogram._CalibrationHistogram__reset_gaussian_fit_parameters()
    
    # Store scales as HEIGHT (counts/bin) directly (compatible with waffles plotting)
    amp0  = float(mm.values['scale_baseline']) * bw0
    amp0e = (float(mm.errors['scale_baseline']) if mm.errors['scale_baseline'] is not None else 0.0) * bw0
    mu0 = float(mm.params[1].value)
    mu0e = float(mm.params[1].error) if mm.params[1].error is not None else 0.0
    s0 = float(mm.params[2].value)
    s0e = float(mm.params[2].error) if mm.params[2].error is not None else 0.0

    calibration_histogram._CalibrationHistogram__add_gaussian_fit_parameters(
        amp0, amp0e,
        mu0, mu0e,
        s0,  s0e,
    )

    gain = mm.params[3].value
    errgain = mm.params[3].error
    propstd = mm.params[4].value
    errpropstd = mm.params[4].error
    mean0 = float(mm.params[1].value)

    # Add Gaussian parameters for each detected PE peak actually present in the winning model
    # Determine how many scale_i parameters Minuit ended up with
    scale_param_names = [p for p in mm.parameters if p.startswith("scale_") and p.endswith("pe")]
    # Peaks modeled = baseline (i=0) + len(scale_param_names)
    for i in range(1, 1 + len(scale_param_names)):
        pname = f"scale_{i}pe"
        # area -> height counts/bin
        mean_i_val  = gain * i + mean0
        amp_i  = float(mm.values[pname]) * bw0
        amp_ie = (float(mm.errors[pname]) if mm.errors[pname] is not None else 0.0) * bw0
        # d(mean_i)^2 = (i * d(gain))^2 + d(mean_baseline)^2
        m0e = float(mm.params[1].error) if mm.params[1].error is not None else 0.0
        ge  = float(errgain) if errgain is not None else 0.0
        mean_i_err = np.sqrt((i**2) * ge**2 + m0e**2)
        std_i_val   = np.sqrt(mm.params[2].value ** 2 + (propstd ** 2) * i)
        # For s_i = sqrt(s0^2 + (propstd^2) * i),
        # ds_i = (1/s_i) * sqrt( (s0 * ds0)^2 + (i * propstd * dpropstd)^2 )
        s0e = float(mm.params[2].error) if mm.params[2].error is not None else 0.0
        pe  = float(errpropstd) if errpropstd is not None else 0.0
        std_i_err = np.sqrt(
            (mm.params[2].value * s0e) ** 2
            + (i * propstd * pe) ** 2
        ) / max(1e-12, std_i_val)
        calibration_histogram._CalibrationHistogram__add_gaussian_fit_parameters(
            amp_i, amp_ie,
            mean_i_val,  mean_i_err,
            std_i_val,   std_i_err
        )

    n_peaks_found = len(calibration_histogram.gaussian_fits_parameters['mean'])

    setattr(calibration_histogram, 'n_peaks_found', n_peaks_found)

    # Store background in consistent unit (counts/bin) and a local counts/bin proxy
    try:
        setattr(calibration_histogram, 'bkg_counts_per_bin', float(mm.values['bkg']))
    except Exception:
        pass

    # NEW: store binned fit prediction for plotting (this is what you want to overlay!)
    try:
        edges_all = np.asarray(calibration_histogram.edges, dtype=float)
        x_centers = 0.5 * (edges_all[:-1] + edges_all[1:])
        pnames_ = list(mm.parameters)
        pvals_  = [float(mm.values[p]) for p in pnames_]

        model_full = MultiGaussBinnedModel(pnames_)
        mu_total = np.asarray(model_full(edges_all, *pvals_), dtype=float)  # counts/bin

        # Also build "signal-only" = sum of the fitted gaussian peaks (baseline + PE),
        # i.e. excluding constant bkg and excluding gaussian bkg component.
        vals = {n: float(v) for n, v in zip(pnames_, pvals_)}
        bw = float(np.median(np.diff(edges_all)))
        bw = max(bw, 1e-12)

        mu_sig = wun.gaussian_bin_counts(edges_all,
                                         float(vals["scale_baseline"]),
                                         float(vals["mean_baseline"]),
                                         float(vals["std_baseline"]))
        mu0 = float(vals["mean_baseline"])
        gain = float(vals["gain"])
        prop = float(vals.get("propstd", 0.0))
        s0   = float(vals["std_baseline"])

        for pname in pnames_:
            if pname.startswith("scale_") and pname.endswith("pe"):
                k = int(pname.split("_")[1].replace("pe", ""))
                mu_k = mu0 + k * gain
                s_k  = math.sqrt(max(1e-24, s0*s0 + (prop*prop)*k))
                mu_sig = mu_sig + wun.gaussian_bin_counts(edges_all, float(vals[pname]), mu_k, s_k)

        mu_bkg = mu_total - mu_sig

        setattr(calibration_histogram, "fit_x_centers", x_centers)
        setattr(calibration_histogram, "fit_mu_bins_total",  mu_total)
        setattr(calibration_histogram, "fit_mu_bins_signal", mu_sig)
        setattr(calibration_histogram, "fit_mu_bins_bkg",    mu_bkg)
    except Exception:
        pass

    try:
        setattr(calibration_histogram, 'gof', best_gof)
        setattr(calibration_histogram, 'best_fit_model', 'multigauss_iminuit')
        _v(1, f"[fit_peaks] Done. n_peaks={n_peaks_found}, redχ²={best_gof['redchi2']:.4g}, BIC={best_gof['BIC']:.4g}")
    except Exception:
        pass
    # DO NOT store Minuit object if you want picklability.
    # Store a lightweight, picklable summary instead.
    try:
        setattr(calibration_histogram, 'iminuit_summary', {
            "values": {p: float(mm.values[p]) for p in mm.parameters},
            "errors": {p: (float(mm.errors[p]) if mm.errors[p] is not None else None) for p in mm.parameters},
            "fval": float(mm.fval),
            "valid": bool(mm.fmin.is_valid),
        })
    except Exception:
        pass
        
    return fitstatus

def fit_peaks_of_ChannelWsGrid( 
    channel_ws_grid: ChannelWsGrid,
    max_peaks: int,
    prominence: float,
    initial_percentage: float = 0.1,
    percentage_step: float = 0.1,
    return_last_addition_if_fail: bool = False,
    fit_type: str = 'independent_gaussians',
    half_points_to_fit: int = 2,
    std_increment_seed_fallback: float = 1e+2,
    ch_span_fraction_around_peaks: float = 0.05,
    verbose: bool = False
) -> bool:
    """For each ChannelWs object, say chws, contained in
    the ChWfSets attribute of the given ChannelWsGrid
    object, channel_ws_grid, whose channel is present
    in the ch_map attribute of the channel_ws_grid, this
    function calls the
    
        fit_peaks_of_CalibrationHistogram(chws.calib_histo, ...)

    function. It returns False if at least one 
    of the fit_peaks_of_CalibrationHistogram() calls 
    returns False, and True if every 
    fit_peaks_of_CalibrationHistogram() call returned 
    True. I.e. it returns True if max_peaks peaks 
    were successfully found for each histogram, and
    False if only n peaks were found for at least one 
    of the histograms, where n < max_peaks.

    Parameters
    ----------
    channel_ws_grid: ChannelWsGrid
        The ChannelWsGrid object to fit peaks on
    max_peaks: int
        The maximum number of peaks which will be
        searched for in each calibration histogram.
        It is given to the 'max_peaks' parameter of
        the fit_peaks_of_CalibrationHistogram()
        function for each calibration histogram.    
    prominence: float
        It must be greater than 0.0 and smaller than 
        1.0. It gives the minimal prominence of the 
        peaks to spot. This parameter is passed to the 
        'prominence' parameter of the 
        fit_peaks_of_CalibrationHistogram() function 
        for each calibration histogram. For more 
        information, check the documentation of such 
        function.
    initial_percentage: float
        It must be greater than 0.0 and smaller than 1.0.
        This parameter is passed to the 'initial_percentage' 
        parameter of the fit_peaks_of_CalibrationHistogram()
        function for each calibration histogram. For more 
        information, check the documentation of such function.
    percentage_step: float
        It must be greater than 0.0 and smaller than 1.0.
        This parameter is passed to the 'percentage_step'
        parameter of the fit_peaks_of_CalibrationHistogram()
        function for each calibration histogram. For more 
        information, check the documentation of such function.
    return_last_addition_if_fail: bool
        This parameter is given to the
        return_last_addition_if_fail parameter of the
        fit_peaks_of_CalibrationHistogram() function. It
        makes a difference only if the specified number
        of peaks (max_peaks) is not found. For more
        information, check the documentation of the
        fit_peaks_of_CalibrationHistogram() function.
    fit_type: str
        The only supported values are 'independent_gaussians'
        and 'correlated_gaussians'. If any other value is
        given, the 'independent_gaussians' value will be
        used instead. This parameter is passed to the
        'fit_type' parameter of the fit_peaks_of_CalibrationHistogram()
        function for each calibration histogram. For more 
        information, check the documentation of such function.
    half_points_to_fit: int
        This parameter is only used if the fit_type
        parameter is set to 'independent_gaussians'.
        It must be a positive integer. For each peak in
        each calibration histogram, it gives the number 
        of points to consider on either side of the peak 
        maximum, to fit each gaussian function. It is
        given to the 'half_points_to_fit' parameter of
        the fit_peaks_of_CalibrationHistogram() function 
        for each calibration histogram. For more information, 
        check the documentation of such function.
    std_increment_seed_fallback: float
        This parameter is only used if the fit_type
        parameter is set to 'correlated_gaussians'.
        For more information, check the documentation
        of the fit_peaks_of_CalibrationHistogram()
        function.
    ch_span_fraction_around_peaks: float
        This parameter is only used if the fit_type
        parameter is set to 'correlated_gaussians'.
        For more information, check the documentation
        of the fit_peaks_of_CalibrationHistogram()
        function.
    verbose: bool = False
        Whether to print functioning related messages

    Returns
    ----------
    output: bool
        True if max_peaks peaks were successfully found for 
        each histogram, and False if only n peaks were found 
        for at least one of the histograms, where n < max_peaks.
    """

    output = True

    for i in range(channel_ws_grid.ch_map.rows):
        for j in range(channel_ws_grid.ch_map.columns):

            try:
                channel_ws = channel_ws_grid.ch_wf_sets[
                    channel_ws_grid.ch_map.data[i][j].endpoint][
                        channel_ws_grid.ch_map.data[i][j].channel]

            except KeyError:
                continue

            if channel_ws.calib_histo is not None:
                output *= fit_peaks_of_CalibrationHistogram(
                    channel_ws.calib_histo,
                    max_peaks,
                    prominence,
                    initial_percentage=initial_percentage,
                    percentage_step=percentage_step,
                    return_last_addition_if_fail=return_last_addition_if_fail,
                    fit_type=fit_type,
                    half_points_to_fit=half_points_to_fit,
                    std_increment_seed_fallback=std_increment_seed_fallback,
                    ch_span_fraction_around_peaks=ch_span_fraction_around_peaks
                )
            elif verbose:
                print(
                    f"In function fit_peaks_of_ChannelWsGrid(): "
                    f"Skipping the peak-fitting process for channel "
                    f"{channel_ws_grid.ch_map.data[i][j].endpoint}-"
                    f"{channel_ws_grid.ch_map.data[i][j].channel}, "
                    f"because its calibration histogram is not available"
                )

    return output

def auto_domain_from_grid(grid: ChannelWsGrid, analysis_label: str, variable: str = "integral",
                          q_low: float = 0.001, q_high: float = 0.999,
                          pad_frac: float = 0.05):
    vals = []

    for i in range(grid.ch_map.rows):
        for j in range(grid.ch_map.columns):
            try:
                chws = grid.ch_wf_sets[grid.ch_map.data[i][j].endpoint][grid.ch_map.data[i][j].channel]
            except KeyError:
                continue

            # Extrae valores de forma robusta:
            # - evita llamar get_analysis() dos veces por waveform
            # - tolera que falte el analysis o que no exista la key 'variable'
            # - filtra NaN/inf de forma segura
            v_list = []
            for wf in getattr(chws, "waveforms", []):
                try:
                    ana = wf.get_analysis(analysis_label)
                    if ana is None:
                        continue
                    res = getattr(ana, "result", None)
                    if res is None:
                        continue
                    # res suele ser dict-like: res[variable]
                    if isinstance(res, dict):
                        val = res.get(variable, np.nan)
                    else:
                        # fallback por si result es objeto con atributo
                        val = getattr(res, variable, np.nan)

                    # normaliza a float si es posible
                    if val is None:
                        continue
                    val = float(val)
                except Exception:
                    continue

                if np.isfinite(val):
                    v_list.append(val)

            if not v_list:
                continue
            vals.append(np.asarray(v_list, dtype=float))

    if not vals:
        raise RuntimeError("No se encontraron valores para estimar el dominio.")

    x = np.concatenate(vals)
    lo, hi = np.quantile(x, [q_low, q_high])

    # padding para no recortar colas
    span = max(hi - lo, 1e-12)
    pad = pad_frac * span
    return float(lo - pad), float(hi + pad)
