from waffles.data_classes.CalibrationHistogram import CalibrationHistogram
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid

import waffles.utils.fit_peaks.fit_peaks_utils as wuff

from waffles.Exceptions import GenerateExceptionMessage

from iminuit import Minuit

import numpy as np
import waffles.utils.numerical_utils as wun
from typing import Optional

class PoissonDeviance:
    """Pickle-friendly Poisson deviance for binned counts."""
    def __init__(self, x, y, model):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.model = model

    def __call__(self, *p):
        mu = self.model(self.x, *p)
        mu = np.clip(mu, 1e-12, None)
        y = self.y
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
        return 2.0 * np.sum(mu - y + term)
    
class PoissonDevianceBinned:
    """Pickle-friendly Poisson deviance for histogram bins."""
    def __init__(self, edges, y, model_binned):
        self.edges = np.asarray(edges)
        self.y = np.asarray(y)
        self.model = model_binned

    def __call__(self, *p):
        mu = self.model(self.edges, *p)
        mu = np.clip(mu, 1e-12, None)
        y  = self.y
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
        return 2.0 * np.sum(mu - y + term)

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
    verbosity: int = 1
) -> bool:
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
    # This is needed because histogram heights are counts/bin but the binned Gaussian
    # model scale behaves like a density amplitude (counts per x-unit).
    binw = float(np.median(np.diff(calibration_histogram.edges)))
    binw = max(binw, 1e-12)

    def _bin_width_at(edges: np.ndarray, x: float) -> float:
        """Return width of the bin that contains x (robust for non-uniform binning)."""
        edges = np.asarray(edges)
        j = int(np.searchsorted(edges, x, side="right") - 1)
        j = max(0, min(j, len(edges) - 2))
        w = float(edges[j + 1] - edges[j])
        return max(w, 1e-12)

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
    base_paramnames = ['scale_baseline', 'mean_baseline', 'std_baseline', 'gain', 'propstd','bkg']
    base_params = [
        float(gfp['scale'][0][0]) / binw,  # scale_baseline as density seed
        float(gfp['mean'][0][0]),   # mean_baseline
        float(gfp['std'][0][0]),    # std_baseline (según tu parametrización)
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

    # background seed: percentile of counts/bin converted to density (counts/x)
    # (works ok even before windowing)
    bkg_seed_bin = float(np.percentile(calibration_histogram.counts, 5))
    bkg_seed_den = max(0.0, bkg_seed_bin / binw)
    base_params.append(bkg_seed_den)   

    # Estimates the number of peaks that should be fitted based on the histogram
    # Starts with a huge number of peaks, and then reduces it
    # Decide an upper bound for candidate model size; do not mutate base lists
    n_peaks_to_fit_iminuit = n_peaks_found + 15

    data_x = (calibration_histogram.edges[:-1] + calibration_histogram.edges[1:]) * 0.5
    data_y = calibration_histogram.counts


    def _seed_scale_from_bin(edges_all, y_all, mean, std, eps=1e-12):
        """Convert bin-count seed -> continuous Gaussian amplitude using exact bin integral."""
        j = int(np.searchsorted(edges_all, mean, side="right") - 1)
        if j < 0 or j >= len(y_all):
            return None
        eL = float(edges_all[j]); eR = float(edges_all[j + 1])
        yb = float(y_all[j])
        # subtract background expectation in that bin (if available)
        try:
            bkg_den = float(base_params[5])  # counts/x
        except Exception:
            bkg_den = 0.0
        yb_eff = yb - max(0.0, bkg_den) * (eR - eL)
        if not np.isfinite(yb_eff):
            return None
        yb_eff = max(0.0, yb_eff)
        fac = float(wun.gaussian_bin_counts(np.array([eL, eR]), 1.0, mean, std)[0])
        if (not np.isfinite(fac)) or fac <= eps:
            return None
        return yb_eff / max(fac, eps)

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

        s_last = np.sqrt(s0**2 + (prop**2) * max(0, n_scales - 1))
        xmin = mu0 - L * s0
        xmax = mu0 + (n_scales - 1) * gain + R * s_last

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

        cost = PoissonDevianceBinned(edges_fit, y_fit, wun.multigaussfit_binned)
        mm = Minuit(cost, *pvals, name=pnames)

        # Positivity limits MUST exist already in Stage A (avoid negative mu tricks)
        for p in mm.parameters:
            if p != "mean_baseline":
                mm.limits[p] = (0.0, None)
            else:
                mm.limits[p] = (None, None)
        if 'bkg' in mm.parameters:
            mm.limits['bkg'] = (0.0, None)

        # ---------------- Stage A (more robust) ----------------
        # Fix amplitudes and background to reduce degeneracy, but
        # LET mean_baseline and gain move early to avoid "stuck" misalignment.
        for p in mm.parameters:
            if p.startswith("scale_"):
                mm.fixed[p] = True
        if 'bkg' in mm.parameters:
            mm.fixed['bkg'] = True
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
            return mm, False

        # Stage B
        for p in mm.parameters:
            mm.fixed[p] = False
            if p != "mean_baseline":
                mm.limits[p] = (0.0, None)
            else:
                mm.limits[p] = (None, None)

        # keep background non-negative (explicit)
        if 'bkg' in mm.parameters:
            mm.limits['bkg'] = (0.0, None)

        # explícitos útiles (por si acaso)
        if 'propstd' in mm.parameters:      mm.limits['propstd'] = (0.0, None)
        if 'gain' in mm.parameters:         mm.limits['gain']    = (0.0, None)
        if 'std_baseline' in mm.parameters: mm.limits['std_baseline'] = (0.0, None)
        if 'bkg' in mm.parameters:          mm.limits['bkg'] = (0.0, None)

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
    
    def _mu_binned(edges, mm):
        """Expected counts per bin for current Minuit state."""
        return wun.multigaussfit_binned(edges, *[mm.values[p] for p in mm.parameters])
    
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
        s0 = _seed_scale_from_bin(edges_all, y_all, mu0_seed, std0_seed)
        if s0 is not None:
            pvals[0] = float(s0)

        for i in range(1, n_scales):
            mean_i = mu0_seed + i * gain_seed
            std_i  = np.sqrt(std0_seed**2 + (prop_seed**2) * i)

            si = _seed_scale_from_bin(edges_all, y_all, mean_i, std_i)
            if si is None:
                # Fallback: convert counts/bin -> counts/x (density) consistently
                idx = int(np.argmin(np.abs(data_x - mean_i)))
                idx = max(0, min(idx, len(y_all) - 1))
                eL = float(edges_all[idx]); eR = float(edges_all[idx + 1])
                bw = max(eR - eL, 1e-12)
                yb = float(y_all[idx])  # counts/bin
                try:
                    bkg_den = float(base_params[5])  # counts/x
                except Exception:
                    bkg_den = 0.0
                yb_eff = max(0.0, yb - max(0.0, bkg_den) * bw)  # counts/bin after bkg subtraction
                si = 0.95 * (yb_eff / bw)  # -> counts/x

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
            costR = PoissonDevianceBinned(edgesR, yR, wun.multigaussfit_binned)
            mmR = Minuit(costR, *pA_vals, name=pA_names)

            # fully relax (do NOT fix gain or scale_1pe here)
            for p in mmR.parameters:
                mmR.fixed[p] = False

            # positivity constraints for Poisson model
            for p in mmR.parameters:
                if p != "mean_baseline":
                    mmR.limits[p] = (0.0, None)
            if 'bkg' in mmR.parameters:
                mmR.limits['bkg'] = (0.0, None)            
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
    bic_tol     = 0.5   # plateau threshold (smaller is stricter)
    # New: require "strong" improvement to accept a larger model (Kass & Raftery)
    bic_improve_min = 15.0
    patience    = 2     # consecutive non-improvements to stop

    best_mm, best_gof = None, None
    best_bic = np.inf
    stall = 0
    for m in range(sweep_start, sweep_stop + 1):
        mm_try, gof_try, ok_try = _fit_for_min(m)
        if not ok_try:
            continue
        # Only accept as "new best" if BIC improves by a meaningful margin
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
    min_peak_snr   = 8.0    # min height SNR vs residual noise
    min_rel_area   = 0.07   # min area relative to total modeled area
    rel_height_min   = 0.10   # min height vs tallest fitted peak
    gain_tolerance_frac = 0.25  # |μ_i - (μ0+i*gain)| <= frac * σ0

    # Build model prediction to compute residuals
    data_y = calibration_histogram.counts
    # Consistent residuals: Poisson (Pearson) residuals on binned expected counts
    mu = _mu_binned(calibration_histogram.edges, mm)   # expected counts per bin
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

    # Resize the gaussian_fits_parameters
    calibration_histogram._CalibrationHistogram__reset_gaussian_fit_parameters()
    
    # Convert fitted density scales back to counts/bin using the LOCAL bin width at each peak mean.
    # This is robust if edges are not perfectly uniform.
    bw0 = _bin_width_at(calibration_histogram.edges, float(mm.params[1].value))
    sb_val = mm.params[0].value * bw0
    sb_err = mm.params[0].error * bw0

    calibration_histogram._CalibrationHistogram__add_gaussian_fit_parameters(
        sb_val, sb_err,
        mm.params[1].value,mm.params[1].error,
        mm.params[2].value,mm.params[2].error,
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
        # density -> counts/bin (use local bin width at each mean)
        mean_i_val  = gain * i + mean0
        bwi = _bin_width_at(calibration_histogram.edges, float(mean_i_val))
        scale_i_val = mm.values[pname] * bwi
        scale_i_err = mm.errors[pname] * bwi
        # d(mean_i)^2 = (i * d(gain))^2 + d(mean_baseline)^2
        mean_i_err  = np.sqrt((i**2) * (errgain ** 2) + (mm.params[1].error ** 2))
        std_i_val   = np.sqrt(mm.params[2].value ** 2 + (propstd ** 2) * i)
        # For s_i = sqrt(s0^2 + (propstd^2) * i),
        # ds_i = (1/s_i) * sqrt( (s0 * ds0)^2 + (i * propstd * dpropstd)^2 )
        std_i_err   = np.sqrt(
            (mm.params[2].value * mm.params[2].error) ** 2
            + ( (i * propstd * errpropstd) ** 2 )
        ) / max(1e-12, std_i_val)  # safe divide
        calibration_histogram._CalibrationHistogram__add_gaussian_fit_parameters(
            scale_i_val, scale_i_err,
            mean_i_val,  mean_i_err,
            std_i_val,   std_i_err
        )

    n_peaks_found = len(calibration_histogram.gaussian_fits_parameters['mean'])

    setattr(calibration_histogram, 'n_peaks_found', n_peaks_found)

    # Store background in consistent unit (density counts/x) and a local counts/bin proxy
    try:
        bkg_den = float(mm.values['bkg'])
        setattr(calibration_histogram, 'bkg_density', bkg_den)
        setattr(calibration_histogram, 'bkg_counts_per_bin', bkg_den * bw0)
    except Exception:
        pass

    try:
        setattr(calibration_histogram, 'gof', best_gof)
        setattr(calibration_histogram, 'best_fit_model', 'multigauss_iminuit')
        _v(1, f"[fit_peaks] Done. n_peaks={n_peaks_found}, redχ²={best_gof['redchi2']:.4g}, BIC={best_gof['BIC']:.4g}")
    except Exception:
        pass
    setattr(calibration_histogram, 'iminuit', mm)
        
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
