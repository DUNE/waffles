#!/usr/bin/env python3
"""Study the Gaussian width of D_AB for manually defined adjacent PDS pairs.

Each APA pair is evaluated at 1, 2, 3, 5, and 7 GeV/c.  The 1 GeV/c sample
uses APA-local-valid triggers because its beam muon fraction is assumed to be
zero.  At 2--7 GeV/c, the selection uses the nominal Langauss--Gaussian
intersection on the APA 1 average response.  The nominal, minus-one-sigma,
and plus-one-sigma threshold selections can be evaluated in one run.

The principal output is a multi-page PDF.  Each page contains five D_AB
histograms with Gaussian core fits and a plot of their fitted sigma values as a
function of the effective kinetic energy.  It is a differential local-response
study, not an absolute calorimetric energy-resolution measurement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import dataclass, asdict
import json
import math
import sys
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

# The driver lives in scripts/review, while the shared channel geometry remains
# in scripts/utils.py.  Make the parent scripts directory importable regardless
# of the current working directory used to launch the program.
SCRIPTS_DIRECTORY = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from pds_differential_resolution import (
    Channel,
    Pair,
    extract_pair_events,
    gaussian_count_model,
    load_merged_json,
    measure_pair,
    normalized_difference,
    poisson_width,
    selected_event_indices,
)
from utils import adjacent_channel_info


MOMENTA = (1, 2, 3, 5, 7)
MASS_GEV = {
    "e": 0.00051099895,
    "k": 0.493677,
    "p": 0.938272088,
    "pi": 0.13957039,
}
DATA_COLOR = "#0072B2"
TWO_TERM_FIT_COLOR = "#009E73"
THREE_TERM_FIT_COLOR = "#D55E00"
LOW_COVERAGE_EDGE = "#4D4D4D"
RESOLUTION_MODELS = ("two_term", "three_term")
DEFAULT_EXPORT_PAIRS = {
    1: ("apa1_end105_ch26__end105_ch24",),
    2: (),
}


@dataclass
class ResolutionFit:
    """Fit of the differential relative-response width."""

    model: str
    status: str
    message: str
    n_points: int
    momenta: str
    low_coverage_momenta: str
    constant_a: float
    constant_a_error: float
    stochastic_b_sqrt_GeV: float
    stochastic_b_error_sqrt_GeV: float
    noise_c_GeV: float
    noise_c_error_GeV: float
    chi2: float
    ndf: int
    r_squared: float

    @property
    def chi2_ndf(self) -> float:
        return self.chi2 / self.ndf if self.ndf > 0 else float("nan")

    @classmethod
    def failed(
        cls,
        model: str,
        message: str,
        n_points: int = 0,
        momenta: str = "",
        low_coverage_momenta: str = "",
    ) -> "ResolutionFit":
        return cls(
            model=model, status="failed", message=message, n_points=n_points,
            momenta=momenta, low_coverage_momenta=low_coverage_momenta,
            constant_a=float("nan"), constant_a_error=float("nan"),
            stochastic_b_sqrt_GeV=float("nan"), stochastic_b_error_sqrt_GeV=float("nan"),
            noise_c_GeV=float("nan"), noise_c_error_GeV=float("nan"),
            chi2=float("nan"), ndf=0, r_squared=float("nan"),
        )

    def as_flat_dict(self, prefix: str = "resolution_fit") -> dict[str, float | int | str]:
        result = asdict(self)
        result["chi2_ndf"] = self.chi2_ndf
        return {f"{prefix}_{key}": value for key, value in result.items()}


def resolution_model(
    kinetic_energy: np.ndarray | float,
    constant_a: float,
    stochastic_b: float,
    noise_c: float = 0.0,
) -> np.ndarray:
    """Two- or three-term differential-resolution model evaluated at K_eff."""

    energy = np.asarray(kinetic_energy, dtype=float)
    return np.sqrt(
        constant_a**2
        + (stochastic_b / np.sqrt(energy)) ** 2
        + (noise_c / energy) ** 2
    )


def resolution_derivative(
    kinetic_energy: np.ndarray,
    response: np.ndarray,
    stochastic_b: float,
    noise_c: float = 0.0,
) -> np.ndarray:
    """Derivative of the resolution model with respect to K_eff."""

    return -(
        stochastic_b**2 / kinetic_energy**2
        + 2.0 * noise_c**2 / kinetic_energy**3
    ) / (2.0 * response)


def fit_resolution(records: list[dict], model: str) -> ResolutionFit:
    """Fit sigma_D(K_eff), including the horizontal and vertical uncertainties.

    The residual uncertainty is evaluated as ``sqrt(sigma_y^2 +
    (d sigma_D / d K_eff * sigma_K)^2)``.  This is the standard effective-
    variance treatment of the K_eff uncertainty in a bounded nonlinear fit.
    """

    if model not in RESOLUTION_MODELS:
        raise ValueError(f"Unknown resolution model: {model}")
    records = sorted(records, key=lambda row: row["kinetic_mean_GeV"])
    momenta = ";".join(str(int(row["momentum_GeV_c"])) for row in records)
    low_coverage = ";".join(
        str(int(row["momentum_GeV_c"]))
        for row in records
        if row["coverage_status"] == "low_coverage"
    )
    if len(records) < 4:
        return ResolutionFit.failed(
            model,
            "At least four successful Gaussian widths are required.",
            len(records), momenta, low_coverage,
        )

    x = np.asarray([row["kinetic_mean_GeV"] for row in records], dtype=float)
    sx = np.asarray([row["effective_spread_GeV"] for row in records], dtype=float)
    y = np.asarray([row["d_gaussian_sigma"] for row in records], dtype=float)
    sy = np.asarray([row["d_gaussian_sigma_error"] for row in records], dtype=float)
    if (
        not np.all(np.isfinite(x))
        or not np.all(np.isfinite(sx))
        or not np.all(np.isfinite(y))
        or not np.all(np.isfinite(sy))
        or np.any(x <= 0)
        or np.any(sy <= 0)
        or np.any(sx < 0)
    ):
        return ResolutionFit.failed(model, "Non-finite or invalid fit input.", len(records), momenta, low_coverage)

    def residual(parameters: np.ndarray) -> np.ndarray:
        noise_c = parameters[2] if model == "three_term" else 0.0
        expected = resolution_model(x, parameters[0], parameters[1], noise_c)
        derivative = resolution_derivative(x, expected, parameters[1], noise_c)
        uncertainty = np.hypot(sy, derivative * sx)
        return (y - expected) / np.maximum(uncertainty, 1.0e-12)

    minimum = max(float(np.min(y)), 1.0e-4)
    initial = np.asarray((0.65 * minimum, 0.18, 0.10) if model == "three_term" else (0.65 * minimum, 0.18), dtype=float)
    try:
        solution = least_squares(
            residual,
            initial,
            bounds=(np.full(len(initial), 1.0e-8), np.full(len(initial), np.inf)),
            x_scale="jac",
            max_nfev=20_000,
            ftol=1.0e-10,
            xtol=1.0e-10,
            gtol=1.0e-10,
        )
    except (RuntimeError, ValueError, FloatingPointError) as error:
        return ResolutionFit.failed(model, f"Resolution fit failed: {error}", len(records), momenta, low_coverage)
    if not solution.success:
        return ResolutionFit.failed(model, f"Resolution fit failed: {solution.message}", len(records), momenta, low_coverage)

    parameters = np.asarray(solution.x, dtype=float)
    noise_c = parameters[2] if model == "three_term" else 0.0
    prediction = resolution_model(x, parameters[0], parameters[1], noise_c)
    chi2 = float(np.sum(residual(parameters) ** 2))
    ndf = len(x) - len(parameters)
    centered = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = float(1.0 - np.sum((y - prediction) ** 2) / centered) if centered > 0 else float("nan")
    try:
        covariance = np.linalg.pinv(solution.jac.T @ solution.jac)
        scale = max(1.0, chi2 / ndf) if ndf > 0 else 1.0
        errors = np.sqrt(np.diag(covariance) * scale)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        errors = np.full(len(parameters), np.nan)
    message = ""
    if np.any(parameters <= 1.0e-6):
        message = "One or more fit terms are compatible with the lower bound."
    if not np.all(np.isfinite(errors)):
        message = (message + " " if message else "") + "Parameter covariance is not finite."

    return ResolutionFit(
        model=model, status="success", message=message, n_points=len(records),
        momenta=momenta, low_coverage_momenta=low_coverage,
        constant_a=float(parameters[0]), constant_a_error=float(errors[0]),
        stochastic_b_sqrt_GeV=float(parameters[1]), stochastic_b_error_sqrt_GeV=float(errors[1]),
        noise_c_GeV=float(noise_c),
        noise_c_error_GeV=float(errors[2]) if model == "three_term" else float("nan"),
        chi2=chi2, ndf=ndf, r_squared=r_squared,
    )


def exact_integer(value: object) -> int:
    number = Decimal(str(value).strip())
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError("Expected a finite integer.")
    return int(number)


def finite_float(value: object) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Expected a finite number.")
    return number


def parse_flag(value: object) -> int:
    text = str(value).strip()
    if text not in {"0", "1"}:
        raise ValueError("Expected a 0/1 validity flag.")
    return int(text)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def file_info(path: Path) -> dict[str, str | int]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def load_trigger_contexts(path: Path) -> dict[int, dict[tuple[str, int], dict]]:
    required = {
        "momentum_GeV_c", "block", "trigger_time", "apa1_mean",
        "apa1_valid", "apa2_valid",
    }
    contexts = {momentum: {} for momentum in MOMENTA}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, strict=True)
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Missing columns in trigger data: {missing}")
        for row in reader:
            momentum = exact_integer(row["momentum_GeV_c"])
            if momentum not in contexts:
                continue
            key = (row["block"].strip(), exact_integer(row["trigger_time"]))
            if key in contexts[momentum]:
                raise ValueError(f"Duplicate trigger in CSV: {momentum} GeV/c, {key}")
            apa1_valid = parse_flag(row["apa1_valid"])
            contexts[momentum][key] = {
                "apa1_valid": apa1_valid,
                "apa2_valid": parse_flag(row["apa2_valid"]),
                "apa1_mean": finite_float(row["apa1_mean"]) if apa1_valid else math.nan,
            }
    return contexts


def load_nominal_thresholds(path: Path) -> tuple[dict[int, tuple[float, float]], list[dict]]:
    """Read one successful nominal Langauss--Gaussian threshold per momentum."""

    required = {"momentum_GeV_c", "model", "status", "intersection", "intersection_error"}
    rows_by_momentum: dict[int, dict] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, strict=True)
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Missing columns in population-fit results: {missing}")
        for row in reader:
            momentum = exact_integer(row["momentum_GeV_c"])
            if momentum in MOMENTA and momentum != 1 and row["model"] == "langauss_plus_gaussian" and row["status"] == "success":
                rows_by_momentum[momentum] = row

    thresholds: dict[int, tuple[float, float]] = {}
    output_rows = [{
        "momentum_GeV_c": 1,
        "selection_kind": "apa_local_valid_no_muon_selection",
        "threshold_nominal_PE": math.nan,
        "threshold_error_PE": math.nan,
        "threshold_applied_PE": math.nan,
    }]
    for momentum in MOMENTA[1:]:
        row = rows_by_momentum.get(momentum)
        if row is None:
            raise ValueError(f"No successful Langauss--Gaussian intersection at {momentum} GeV/c.")
        threshold = finite_float(row["intersection"])
        error = finite_float(row["intersection_error"])
        if error <= 0:
            raise ValueError(f"Non-positive intersection uncertainty at {momentum} GeV/c.")
        thresholds[momentum] = (threshold, error)
        output_rows.append({
            "momentum_GeV_c": momentum,
            "selection_kind": "apa1_mean_greater_than_nominal_threshold",
            "threshold_nominal_PE": threshold,
            "threshold_error_PE": error,
            "threshold_applied_PE": threshold,
        })
    return thresholds, output_rows


def scenario_name(multiplier: float) -> str:
    """Return a stable label for one threshold-selection scenario."""

    if math.isclose(multiplier, 0.0):
        return "nominal"
    sign = "plus" if multiplier > 0.0 else "minus"
    return f"{sign}_{abs(multiplier):g}".replace(".", "p") + "sigma"


def build_threshold_rows(
    thresholds: dict[int, tuple[float, float]], multipliers: list[float],
) -> list[dict]:
    """Record each applied threshold, including the unchanged 1 GeV/c sample."""

    rows: list[dict] = []
    for multiplier in multipliers:
        scenario = scenario_name(multiplier)
        for momentum in MOMENTA:
            threshold, uncertainty = thresholds.get(momentum, (math.nan, math.nan))
            rows.append({
                "threshold_scenario": scenario,
                "threshold_sigma_multiplier": multiplier,
                "momentum_GeV_c": momentum,
                "selection_kind": "apa_local_valid_no_muon_selection" if momentum == 1 else "apa1_mean_greater_than_threshold",
                "threshold_nominal_PE": threshold,
                "threshold_error_PE": uncertainty,
                "threshold_applied_PE": math.nan if momentum == 1 else threshold + multiplier * uncertainty,
            })
    return rows


def load_kinetic_energies(path: Path, relative_momentum_error: float) -> dict[int, dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, strict=True)
        required = {"Momentum [GeV/c]"} | {f"{species} [Hz]" for species in MASS_GEV}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Missing columns in beam-composition file: {missing}")
        composition = {exact_integer(row["Momentum [GeV/c]"]): row for row in reader}

    masses = np.asarray(list(MASS_GEV.values()), dtype=float)
    result: dict[int, dict] = {}
    for momentum in MOMENTA:
        row = composition.get(momentum)
        if row is None:
            raise ValueError(f"Beam composition missing at {momentum} GeV/c.")
        rates = np.asarray([finite_float(row[f"{species} [Hz]"]) for species in MASS_GEV])
        if np.any(rates < 0) or rates.sum() <= 0:
            raise ValueError(f"Invalid beam rates at {momentum} GeV/c.")
        weights = rates / rates.sum()
        kinetic = np.sqrt(momentum**2 + masses**2) - masses
        mean = float(weights @ kinetic)
        mixture_rms = float(math.sqrt(weights @ (kinetic - mean) ** 2))
        derivative = float(weights @ (momentum / np.sqrt(momentum**2 + masses**2)))
        momentum_error = abs(derivative * momentum * relative_momentum_error)
        result[momentum] = {
            "kinetic_mean_GeV": mean,
            "momentum_error_GeV": momentum_error,
            "mixture_rms_GeV": mixture_rms,
            "effective_spread_GeV": math.hypot(momentum_error, mixture_rms),
        }
    return result


def default_pairs(apa: int) -> list[Pair]:
    """Return every manually defined adjacent pair for the requested APA."""

    apa1_pairs, apa2_pairs, _ = adjacent_channel_info()
    raw_pairs = apa1_pairs if apa == 1 else apa2_pairs
    pairs: list[Pair] = []
    for raw_pair in raw_pairs:
        first = Channel.from_mapping(raw_pair[0])
        second = Channel.from_mapping(raw_pair[1])
        label = f"apa{apa}_end{first.endpoint}_ch{first.channel}__end{second.endpoint}_ch{second.channel}"
        pairs.append(Pair(first, second, label=label))
    return pairs


def add_panel_brand(axis: plt.Axes, location: str, standalone: bool = False) -> None:
    """Add the preliminary-status label inside one plot panel."""

    positions = {
        "left": (0.025, 0.965, "left", "top"),
        "right": (0.975, 0.965, "right", "top"),
        "lower_right": (0.975, 0.035, "right", "bottom"),
        "upper_center": (0.500, 0.965, "center", "top"),
    }
    try:
        x_position, y_position, horizontal_alignment, vertical_alignment = positions[location]
    except KeyError as error:
        raise ValueError(f"Unknown brand location: {location}") from error
    axis.text(
        x_position,
        y_position,
        r"$\mathbf{ProtoDUNE\!-\!HD}$" + "\nWork in Progress",
        transform=axis.transAxes,
        ha=horizontal_alignment,
        va=vertical_alignment,
        fontsize=9.2 if standalone else 5.3,
        linespacing=0.93,
        zorder=10,
    )


def format_panel_axis(axis: plt.Axes, standalone: bool) -> None:
    """Use a readable but compact style for PDF panels and standalone PNGs."""

    if standalone:
        axis.tick_params(labelsize=11.5)
        axis.xaxis.label.set_size(13.0)
        axis.yaxis.label.set_size(13.0)
    else:
        axis.tick_params(labelsize=7.0)


def display_limits(values: np.ndarray, fit_low: float, fit_high: float) -> tuple[float, float]:
    """Choose a central display range without changing the fit sample."""

    finite = np.asarray(values, dtype=float)[np.isfinite(values)]
    if len(finite) == 0:
        return -1.0, 1.0
    low_quantile, high_quantile = np.percentile(finite, [0.5, 99.5])
    span = max(fit_high - fit_low, 1.0e-6)
    low = min(low_quantile, fit_low - 0.35 * span)
    high = max(high_quantile, fit_high + 0.35 * span)
    if not high > low:
        return fit_low - span, fit_high + span
    padding = 0.03 * (high - low)
    return low - padding, high + padding


def response_limits(values_a: np.ndarray, values_b: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return robust display limits for the raw channel-response correlation."""

    def limits(values: np.ndarray) -> tuple[float, float]:
        low, high = np.percentile(values, [0.5, 99.5])
        span = max(high - low, 1.0)
        return max(0.0, low - 0.04 * span), high + 0.04 * span

    return limits(values_a), limits(values_b)


def fit_correlation_line(values_a: np.ndarray, values_b: np.ndarray) -> dict[str, float | int | str]:
    """Fit ``N_PE^B = m N_PE^A + q`` for the raw pair-correlation panel.

    The per-trigger vertical weights use the Poisson reference uncertainty
    ``sqrt(max(N_PE^B, 1))``.  The covariance is inflated by the reduced
    chi-square, so the reported parameter uncertainties reflect the observed
    event-to-event spread rather than a pure counting-statistics idealisation.
    The fit is a diagnostic of the raw channel correlation and is not used in
    the differential-resolution result.
    """

    finite = np.isfinite(values_a) & np.isfinite(values_b)
    x = np.asarray(values_a[finite], dtype=float)
    y = np.asarray(values_b[finite], dtype=float)
    if len(x) < 3 or np.ptp(x) <= 0.0:
        return {
            "status": "failed", "message": "Insufficient variation for linear fit.",
            "slope": math.nan, "slope_error": math.nan,
            "intercept": math.nan, "intercept_error": math.nan,
            "chi2": math.nan, "ndf": 0, "chi2_ndf": math.nan,
            "pearson": math.nan,
        }
    sigma_y = np.sqrt(np.maximum(y, 1.0))
    design = np.column_stack((x, np.ones_like(x)))
    weighted_design = design / sigma_y[:, np.newaxis]
    weighted_y = y / sigma_y
    try:
        parameters, _, rank, _ = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)
        if rank < 2:
            raise np.linalg.LinAlgError("Rank-deficient weighted design matrix.")
        covariance = np.linalg.inv(weighted_design.T @ weighted_design)
    except np.linalg.LinAlgError as error:
        return {
            "status": "failed", "message": f"Linear fit failed: {error}",
            "slope": math.nan, "slope_error": math.nan,
            "intercept": math.nan, "intercept_error": math.nan,
            "chi2": math.nan, "ndf": 0, "chi2_ndf": math.nan,
            "pearson": math.nan,
        }
    slope, intercept = (float(parameters[0]), float(parameters[1]))
    residuals = (y - (slope * x + intercept)) / sigma_y
    chi2 = float(np.sum(residuals**2))
    ndf = len(x) - 2
    chi2_ndf = chi2 / ndf if ndf > 0 else math.nan
    scale = max(1.0, chi2_ndf) if math.isfinite(chi2_ndf) else 1.0
    errors = np.sqrt(np.diag(covariance) * scale)
    pearson = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else math.nan
    return {
        "status": "success", "message": "", "slope": slope,
        "slope_error": float(errors[0]), "intercept": intercept,
        "intercept_error": float(errors[1]), "chi2": chi2, "ndf": ndf,
        "chi2_ndf": chi2_ndf, "pearson": pearson,
    }


def plot_correlation_panel(
    axis: plt.Axes,
    panel: dict,
    momentum: int,
    show_momentum_title: bool = True,
    standalone: bool = False,
) -> None:
    """Draw raw $N_{\rm PE}^{A}$ versus $N_{\rm PE}^{B}$ with diagnostic fits."""

    events = panel.get("events")
    if show_momentum_title:
        axis.set_title(rf"$p_{{\rm beam}} = {momentum:g}$ GeV/c", loc="center", fontsize=8.6)
    add_panel_brand(axis, "lower_right", standalone=standalone)
    axis.grid(alpha=0.22)
    if events is None or events.common_events == 0:
        axis.text(
            0.5, 0.5, panel["message"], transform=axis.transAxes, ha="center", va="center",
            fontsize=11.0 if standalone else 7.2, wrap=True,
        )
        axis.set(xlabel=r"$N_{\rm PE}^{A}$", ylabel=r"$N_{\rm PE}^{B}$")
        format_panel_axis(axis, standalone)
        return

    values_a = events.values_a
    values_b = events.values_b
    (x_low, x_high), (y_low, y_high) = response_limits(values_a, values_b)
    axis.scatter(
        values_a, values_b, s=5.0, color=DATA_COLOR, alpha=0.22,
        edgecolors="none", rasterized=True, zorder=2,
        label=f"Data @ {momentum:g} GeV/c ({events.common_events} triggers)",
    )
    reference_low = min(x_low, y_low)
    reference_high = max(x_high, y_high)
    reference_x = np.asarray((reference_low, reference_high))
    axis.plot(
        reference_x, reference_x, "--", color=THREE_TERM_FIT_COLOR, lw=1.25,
        label=r"Expected fit $y=x$", zorder=3,
    )
    fit = fit_correlation_line(values_a, values_b)
    if fit["status"] == "success":
        x_line = np.asarray((x_low, x_high))
        axis.plot(
            x_line, float(fit["slope"]) * x_line + float(fit["intercept"]),
            color=TWO_TERM_FIT_COLOR, lw=1.45, zorder=4,
            label=(
                "Linear fit\n"
                + rf"$m = ({float(fit['slope']):.3f} \pm {float(fit['slope_error']):.3f})$" + "\n"
                + rf"$q = ({float(fit['intercept']):.1f} \pm {float(fit['intercept_error']):.1f})\,{{\rm PE}}$" + "\n"
                + rf"$\chi^2/{{\rm ndf}} = {float(fit['chi2']):.1f}/{int(fit['ndf'])} = {float(fit['chi2_ndf']):.2f}$" + "\n"
                + rf"$\rho_{{\rm Pearson}} = {float(fit['pearson']):.3f}$"
            ),
        )
    else:
        axis.plot([], [], color=TWO_TERM_FIT_COLOR, lw=1.45, label="Linear fit unavailable")
    axis.set(
        xlim=(x_low, x_high), ylim=(y_low, y_high),
        xlabel=r"$N_{\rm PE}^{A}$", ylabel=r"$N_{\rm PE}^{B}$",
    )
    axis.legend(
        frameon=True, facecolor="white", fontsize=8.6 if standalone else 5.5,
        loc="upper left",
    )
    format_panel_axis(axis, standalone)


def plot_distribution_panel(
    axis: plt.Axes,
    panel: dict,
    momentum: int,
    show_momentum_title: bool = True,
    standalone: bool = False,
) -> None:
    """Draw one D_AB distribution and its Gaussian fit."""

    record = panel.get("record")
    values = panel.get("d_values")
    if show_momentum_title:
        axis.set_title(rf"$p_{{\rm beam}} = {momentum:g}$ GeV/c", loc="center", fontsize=8.6)
    add_panel_brand(axis, "right", standalone=standalone)
    axis.grid(alpha=0.22)
    if values is None or len(values) == 0:
        axis.text(
            0.5, 0.5, panel["message"], transform=axis.transAxes, ha="center", va="center",
            fontsize=11.0 if standalone else 7.2, wrap=True,
        )
        axis.set(xlabel=r"$D_{AB}$ [AU]", ylabel="Counts")
        format_panel_axis(axis, standalone)
        return

    if record is None:
        counts, _, _ = axis.hist(
            values, bins="fd", color="#D0D0D0", alpha=0.82, edgecolor="#333333", lw=0.55,
            label=f"Data @ {momentum:g} GeV/c ({len(values)} triggers)",
        )
        axis.text(
            0.97, 0.82, panel["message"], transform=axis.transAxes, ha="right", va="top",
            fontsize=9.0 if standalone else 6.3,
            bbox={"facecolor": "white", "edgecolor": "#999999", "alpha": 0.92},
        )
        ymax = (1.15 if standalone else 1.42) * max(float(np.max(counts)), 1.0)
        axis.set(xlabel=r"$D_{AB}$ [AU]", ylabel="Counts", ylim=(0.0, ymax))
        axis.legend(frameon=True, facecolor="white", fontsize=9.2 if standalone else 6.0, loc="upper left")
        format_panel_axis(axis, standalone)
        return

    fit_low = record["d_gaussian_fit_low"]
    fit_high = record["d_gaussian_fit_high"]
    bin_width = record["d_gaussian_bin_width"]
    low, high = display_limits(values, fit_low, fit_high)
    bin_edges = np.arange(low, high + bin_width, bin_width)
    if len(bin_edges) < 2:
        bin_edges = 30
    counts, _, _ = axis.hist(
        values, bins=bin_edges, color="#D0D0D0", edgecolor="#333333", lw=0.55,
        label=f"Data @ {momentum:g} GeV/c ({int(record['common_events'])} triggers)",
    )
    maximum = max(float(np.max(counts)), 1.0)
    if record["d_gaussian_status"] == "success":
        x_fit = np.linspace(fit_low, fit_high, 500)
        y_fit = gaussian_count_model(
            x_fit, record["d_gaussian_amplitude"], record["d_gaussian_mean"], record["d_gaussian_sigma"]
        )
        maximum = max(maximum, float(np.max(y_fit)))
        label = (
            "Gaussian fit\n"
            + rf"$\mu = ({record['d_gaussian_mean']:.3f} \pm {record['d_gaussian_mean_error']:.3f})$" + "\n"
            + rf"$\sigma_D = ({record['d_gaussian_sigma']:.3f} \pm {record['d_gaussian_sigma_error']:.3f})$" + "\n"
            + rf"$\chi^2/{{\rm ndf}} = {record['d_gaussian_chi2']:.1f}/{int(record['d_gaussian_ndf'])} = {record['d_gaussian_chi2_ndf']:.2f}$"
        )
        axis.plot(x_fit, y_fit, color=DATA_COLOR, lw=1.55, label=label)
    else:
        axis.plot([], [], color=DATA_COLOR, lw=1.55, label="Gaussian fit failed")
    ymax = (1.15 if standalone else 1.42) * maximum
    axis.set(xlim=(low, high), ylim=(0.0, ymax), xlabel=r"$D_{AB}$ [AU]", ylabel="Counts")
    axis.legend(frameon=True, facecolor="white", fontsize=9.2 if standalone else 6.0, loc="upper left")
    format_panel_axis(axis, standalone)


def plot_pair_resolution_panel(
    axis: plt.Axes,
    panels: dict[int, dict],
    resolution_fits: dict[str, ResolutionFit],
    standalone: bool = False,
) -> None:
    """Draw sigma_D(K_eff) and every selected resolution model."""

    records = []
    for momentum in MOMENTA:
        record = panels.get(momentum, {}).get("record")
        if record is not None and record["d_gaussian_status"] == "success":
            records.append(record)
    add_panel_brand(axis, "upper_center", standalone=standalone)
    if not records:
        axis.text(0.5, 0.5, "No successful Gaussian fit", transform=axis.transAxes, ha="center", va="center")
        axis.set(xlabel=r"$K_{\rm eff}$ [GeV]", ylabel=r"Gaussian $\sigma_D$")
        format_panel_axis(axis, standalone)
        return

    records.sort(key=lambda row: row["kinetic_mean_GeV"])
    for index, row in enumerate(records):
        axis.errorbar(
            row["kinetic_mean_GeV"], row["d_gaussian_sigma"],
            xerr=row["effective_spread_GeV"], yerr=row["d_gaussian_sigma_error"],
            fmt="o", ms=7.0 if standalone else 5.8, capsize=2.8 if standalone else 2.2, color=DATA_COLOR, mfc=DATA_COLOR,
            mec=DATA_COLOR, mew=1.0, zorder=3,
            label=r"Gaussian-fit $\sigma_D$" if index == 0 else None,
        )

    x_min = max(0.05, min(row["kinetic_mean_GeV"] - row["effective_spread_GeV"] for row in records))
    x_max = max(row["kinetic_mean_GeV"] + row["effective_spread_GeV"] for row in records)
    x_curve = np.linspace(x_min, 1.04 * x_max, 400)
    for model in RESOLUTION_MODELS:
        resolution_fit = resolution_fits.get(model)
        if resolution_fit is None:
            continue
        color = TWO_TERM_FIT_COLOR if model == "two_term" else THREE_TERM_FIT_COLOR
        model_label = "Two-term fit" if model == "two_term" else "Three-term fit"
        if resolution_fit.status != "success":
            axis.plot([], [], color=color, lw=1.9, label=f"{model_label} unavailable")
            continue
        noise_c = resolution_fit.noise_c_GeV if model == "three_term" else 0.0
        y_curve = resolution_model(
            x_curve, resolution_fit.constant_a, resolution_fit.stochastic_b_sqrt_GeV, noise_c,
        )
        formula = r"$y = \sqrt{a^2 + (b/\sqrt{x})^2}$" if model == "two_term" else r"$y = \sqrt{a^2 + (b/\sqrt{x})^2 + (c/x)^2}$"
        label_lines = [
            model_label,
            formula,
            rf"$a = ({resolution_fit.constant_a:.3f} \pm {resolution_fit.constant_a_error:.3f})$",
            rf"$b = ({resolution_fit.stochastic_b_sqrt_GeV:.3f} \pm {resolution_fit.stochastic_b_error_sqrt_GeV:.3f})\sqrt{{\rm GeV}}$",
        ]
        if model == "three_term":
            label_lines.append(
                rf"$c = ({resolution_fit.noise_c_GeV:.3f} \pm {resolution_fit.noise_c_error_GeV:.3f})\,{{\rm GeV}}$"
            )
        label_lines.extend((
            rf"$\chi^2/{{\rm ndf}} = {resolution_fit.chi2:.2f}/{resolution_fit.ndf:d} = {resolution_fit.chi2_ndf:.2f}$",
            rf"$R^2 = {resolution_fit.r_squared:.3f}$",
        ))
        axis.plot(
            x_curve, y_curve, color=color, lw=2.3 if standalone else 1.9,
            label="\n".join(label_lines),
        )
    axis.set(xlabel=r"$K_{\rm eff}$ [GeV]", ylabel=r"Gaussian $\sigma_D$")
    axis.grid(alpha=0.24)
    fit_count = sum(fit.status == "success" for fit in resolution_fits.values())
    axis.legend(
        frameon=True, facecolor="white", fontsize=(7.8 if fit_count > 1 else 10.0) if standalone else (5.4 if fit_count > 1 else 6.9),
        loc="upper right",
    )
    format_panel_axis(axis, standalone)


def export_individual_pair_plots(
    pairs: list[Pair],
    panels_by_pair: dict[str, dict[int, dict]],
    resolution_fits: dict[str, dict[str, ResolutionFit]],
    output_directory: Path,
) -> int:
    """Export the five correlations, five D_AB distributions, and fit panel as PNGs."""

    exported = 0
    for pair in pairs:
        pair_directory = output_directory / pair.identifier
        pair_directory.mkdir(parents=True, exist_ok=True)
        for momentum in MOMENTA:
            for suffix, plotter, figsize in (
                ("npe_a_vs_npe_b", plot_correlation_panel, (6.8, 5.3)),
                ("dab_distribution", plot_distribution_panel, (6.8, 5.3)),
            ):
                figure, axis = plt.subplots(figsize=(7.6, 5.9))
                plotter(
                    axis, panels_by_pair[pair.identifier][momentum], momentum,
                    show_momentum_title=False, standalone=True,
                )
                figure.tight_layout()
                figure.savefig(pair_directory / f"{pair.identifier}_{momentum}gev_{suffix}.png", dpi=300)
                plt.close(figure)
                exported += 1
        figure, axis = plt.subplots(figsize=(8.6, 6.1))
        plot_pair_resolution_panel(
            axis, panels_by_pair[pair.identifier], resolution_fits[pair.identifier], standalone=True,
        )
        figure.tight_layout()
        figure.savefig(pair_directory / f"{pair.identifier}_resolution_vs_keff.png", dpi=300)
        plt.close(figure)
        exported += 1
    return exported


def make_pair_pdf(
    pairs: list[Pair],
    panels_by_pair: dict[str, dict[int, dict]],
    pair_summary: list[dict],
    resolution_fits: dict[str, dict[str, ResolutionFit]],
    minimum_momenta_for_pdf: int,
    output: Path,
) -> int:
    """Write one A3 landscape page per pair with correlation and D_AB panels."""

    summaries = {row["pair"]: row for row in pair_summary}
    pages = 0
    with PdfPages(output) as pdf:
        for pair in pairs:
            summary = summaries[pair.identifier]
            if int(summary["gaussian_fit_successes"]) < minimum_momenta_for_pdf:
                continue
            figure = plt.figure(figsize=(16.54, 11.69))
            grid = figure.add_gridspec(3, 4, wspace=0.34, hspace=0.43)
            figure.suptitle(
                f"APA {pair.first.apa}: END {pair.first.endpoint} - CH {pair.first.channel}  and  "
                f"END {pair.second.endpoint} - CH {pair.second.channel}",
                x=0.02, y=0.987, ha="left", fontsize=14,
            )
            positions = {
                1: (grid[0, 0], grid[0, 1]),
                2: (grid[0, 2], grid[0, 3]),
                3: (grid[1, 0], grid[1, 1]),
                5: (grid[1, 2], grid[1, 3]),
                7: (grid[2, 0], grid[2, 1]),
            }
            for momentum, (scatter_slot, distribution_slot) in positions.items():
                plot_correlation_panel(figure.add_subplot(scatter_slot), panels_by_pair[pair.identifier][momentum], momentum)
                plot_distribution_panel(figure.add_subplot(distribution_slot), panels_by_pair[pair.identifier][momentum], momentum)
            plot_pair_resolution_panel(
                figure.add_subplot(grid[2, 2:4]), panels_by_pair[pair.identifier], resolution_fits[pair.identifier]
            )
            figure.tight_layout(rect=(0.005, 0.01, 0.995, 0.955))
            pdf.savefig(figure)
            plt.close(figure)
            pages += 1
    return pages


def build_pair_summary(
    pairs: list[Pair],
    table: pd.DataFrame,
    resolution_fits: dict[str, dict[str, ResolutionFit]],
    min_momenta: int,
    primary_model: str,
) -> list[dict]:
    rows: list[dict] = []
    for pair in pairs:
        group = table.loc[table["pair"] == pair.identifier]
        fit_success = group.loc[group["d_gaussian_status"] == "success"]
        low_coverage = fit_success.loc[fit_success["coverage_status"] == "low_coverage"]
        pair_fits = resolution_fits[pair.identifier]
        primary_fit = pair_fits[primary_model]
        row = {
            "pair": pair.identifier,
            "kind": pair.kind,
            "apa": pair.first.apa,
            "first_endpoint": pair.first.endpoint,
            "first_channel": pair.first.channel,
            "second_endpoint": pair.second.endpoint,
            "second_channel": pair.second.channel,
            "successful_measurements": len(group),
            "gaussian_fit_successes": len(fit_success),
            "gaussian_fit_momentum_values_GeV_c": ";".join(str(int(value)) for value in sorted(fit_success["momentum_GeV_c"].unique())),
            "low_coverage_momenta_GeV_c": ";".join(str(int(value)) for value in sorted(low_coverage["momentum_GeV_c"].unique())),
            "included_in_pair_pdf": int(len(fit_success) >= min_momenta),
            "resolution_fit_primary_model": primary_model,
            **primary_fit.as_flat_dict(),
        }
        for model, resolution_fit in pair_fits.items():
            row.update(resolution_fit.as_flat_dict(prefix=f"resolution_fit_{model}"))
        rows.append(row)
    return rows


def _finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _latex_value_error(value: object, error: object, digits: int = 3) -> str:
    """Format one statistical value for a generated LaTeX table."""

    if not _finite_number(value):
        return r"\text{--}"
    if not _finite_number(error):
        return rf"${float(value):.{digits}f}$"
    return rf"${float(value):.{digits}f} \pm {float(error):.{digits}f}$"


def _format_systematic(value: float, minimum_digits: int = 4) -> str:
    """Keep enough decimals to avoid displaying a non-zero systematic as zero."""

    digits = minimum_digits
    while value != 0.0 and round(value, digits) == 0.0 and digits < 8:
        digits += 1
    return f"{value:.{digits}f}"


def _latex_statistical_systematic(
    value: object,
    statistical_error: object,
    systematic_error: object,
    digits: int = 3,
) -> str:
    """Format nominal, statistical, and threshold-systematic uncertainties."""

    if not _finite_number(value):
        return r"\text{--}"
    if not _finite_number(statistical_error):
        return rf"${float(value):.{digits}f}$"
    if not _finite_number(systematic_error):
        return rf"${float(value):.{digits}f} \pm {float(statistical_error):.{digits}f}$"
    return (
        rf"${float(value):.{digits}f} \pm {float(statistical_error):.{digits}f}"
        rf" \pm {_format_systematic(float(systematic_error))}$"
    )


def _pair_channel_latex(endpoint: object, channel: object) -> str:
    return rf"END~{int(endpoint)} -- CH~{int(channel)}"


def build_pair_threshold_systematics(
    pairs: list[Pair],
    fits_by_scenario: dict[str, dict[str, dict[str, ResolutionFit]]],
    scenario_order: list[str],
    primary_model: str,
) -> list[dict]:
    """Envelope threshold-selection variations around the nominal fit result.

    The same logic used in the channel-linearity analysis is applied here: the
    systematic uncertainty is the largest absolute change from the nominal
    parameter across all requested non-nominal threshold scenarios.
    """

    nominal_fits = fits_by_scenario["nominal"]
    variation_names = [name for name in scenario_order if name != "nominal"]
    rows: list[dict] = []
    parameter_names = (
        "constant_a",
        "stochastic_b_sqrt_GeV",
        "noise_c_GeV",
    )
    for pair in pairs:
        nominal = nominal_fits[pair.identifier][primary_model]
        row: dict[str, object] = {
            "pair": pair.identifier,
            "kind": pair.kind,
            "apa": pair.first.apa,
            "first_endpoint": pair.first.endpoint,
            "first_channel": pair.first.channel,
            "second_endpoint": pair.second.endpoint,
            "second_channel": pair.second.channel,
            "resolution_model": primary_model,
            "nominal_status": nominal.status,
            "threshold_variation_scenarios": ";".join(variation_names),
        }
        for parameter in parameter_names:
            nominal_value = getattr(nominal, parameter)
            row[f"nominal_{parameter}"] = nominal_value
            variations: list[float] = []
            for scenario in variation_names:
                varied = fits_by_scenario[scenario][pair.identifier][primary_model]
                value = getattr(varied, parameter)
                row[f"{scenario}_{parameter}"] = value if varied.status == "success" else math.nan
                if nominal.status == "success" and varied.status == "success" and _finite_number(value):
                    variations.append(abs(float(value) - float(nominal_value)))
            row[f"{parameter}_threshold_systematic"] = max(variations) if variations else math.nan
        rows.append(row)
    return rows


def write_thesis_tables(
    output_dir: Path,
    apa: int,
    pairs: list[Pair],
    nominal_measurements: pd.DataFrame,
    nominal_resolution_rows: list[dict],
    systematic_rows: list[dict],
    primary_model: str,
) -> list[Path]:
    """Write reusable appendix tables without modifying the thesis repository."""

    table_directory = output_dir / "thesis_tables"
    table_directory.mkdir(parents=True, exist_ok=True)
    measurement_lookup = {
        (str(row["pair"]), int(row["momentum_GeV_c"])): row
        for row in nominal_measurements.to_dict("records")
        if row.get("d_gaussian_status") == "success"
    }
    width_lines = [
        "% Requires the booktabs and adjustbox packages.",
        r"\begin{table}[p]",
        r"\centering",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Channel $A$ & Channel $B$ & \gls{sigma_D} at \SI{1}{\GeV/c} & \gls{sigma_D} at \SI{2}{\GeV/c} & \gls{sigma_D} at \SI{3}{\GeV/c} & \gls{sigma_D} at \SI{5}{\GeV/c} & \gls{sigma_D} at \SI{7}{\GeV/c} \\",
        r"\midrule",
    ]
    for pair in pairs:
        cells = [
            _pair_channel_latex(pair.first.endpoint, pair.first.channel),
            _pair_channel_latex(pair.second.endpoint, pair.second.channel),
        ]
        for momentum in MOMENTA:
            record = measurement_lookup.get((pair.identifier, momentum))
            if record is None:
                cells.append(r"\text{--}")
            else:
                cells.append(_latex_value_error(
                    record.get("d_gaussian_sigma"), record.get("d_gaussian_sigma_error"),
                ))
        width_lines.append(" & ".join(cells) + r" \\")
    width_lines.extend((
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        rf"\caption{{Gaussian widths of the normalized differential response $D_{{AB}}$ for the manually defined adjacent channel pairs on \gls{{apa}}~{apa}. The nominal trigger selection is used. The quoted uncertainties are statistical uncertainties from the Gaussian fits; -- denotes an unavailable fit.}}",
        rf"\label{{tab:apa{apa}_adjacent_pair_gaussian_widths}}",
        r"\end{table}",
        "",
    ))
    width_path = table_directory / f"apa{apa}_adjacent_pair_gaussian_widths.tex"
    width_path.write_text("\n".join(width_lines), encoding="utf-8")

    systematic_lookup = {str(row["pair"]): row for row in systematic_rows}
    fit_lookup = {
        str(row["pair"]): row
        for row in nominal_resolution_rows
        if row.get("resolution_model") == primary_model and row.get("resolution_fit_status") == "success"
    }
    model_formula = (
        r"\gls{sigma_D}~$=\sqrt{a^2+b^2/\gls{k_eff}}$"
        if primary_model == "two_term"
        else r"\gls{sigma_D}~$=\sqrt{a^2+b^2/\gls{k_eff}+c^2/\gls{k_eff}^2}$"
    )
    include_noise_term = primary_model == "three_term"
    fit_lines = [
        "% Requires the booktabs and adjustbox packages.",
        r"\begin{table}[p]",
        r"\centering",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{llccccc}" if include_noise_term else r"\begin{tabular}{llcccc}",
        r"\toprule",
        (
            r"Channel $A$ & Channel $B$ & $a$ & $b~[\sqrt{\si{\GeV}}]$ & $c~[\si{\GeV}]$ & $\chi^2/\mathrm{ndf}$ & $R^2$ \\")
            if include_noise_term
            else r"Channel $A$ & Channel $B$ & $a$ & $b~[\sqrt{\si{\GeV}}]$ & $\chi^2/\mathrm{ndf}$ & $R^2$ \\",
        r"\midrule",
    ]
    for pair in pairs:
        fit = fit_lookup.get(pair.identifier)
        systematic = systematic_lookup.get(pair.identifier, {})
        if fit is None:
            continue
        a_text = _latex_statistical_systematic(
            fit.get("resolution_fit_constant_a"),
            fit.get("resolution_fit_constant_a_error"),
            systematic.get("constant_a_threshold_systematic"),
        )
        b_text = _latex_statistical_systematic(
            fit.get("resolution_fit_stochastic_b_sqrt_GeV"),
            fit.get("resolution_fit_stochastic_b_error_sqrt_GeV"),
            systematic.get("stochastic_b_sqrt_GeV_threshold_systematic"),
        )
        chi2_ndf = fit.get("resolution_fit_chi2_ndf")
        r_squared = fit.get("resolution_fit_r_squared")
        cells = [
            _pair_channel_latex(pair.first.endpoint, pair.first.channel),
            _pair_channel_latex(pair.second.endpoint, pair.second.channel),
            a_text,
            b_text,
        ]
        if include_noise_term:
            cells.append(_latex_statistical_systematic(
                fit.get("resolution_fit_noise_c_GeV"),
                fit.get("resolution_fit_noise_c_error_GeV"),
                systematic.get("noise_c_GeV_threshold_systematic"),
            ))
        cells.extend((
            f"${float(chi2_ndf):.2f}$" if _finite_number(chi2_ndf) else r"\text{--}",
            f"${float(r_squared):.3f}$" if _finite_number(r_squared) else r"\text{--}",
        ))
        fit_lines.append(" & ".join(cells) + r" \\")
    fit_lines.extend((
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        rf"\caption{{Results of the {model_formula} fits for the adjacent channel pairs on \gls{{apa}}~{apa}. For $a$ and $b$, the first uncertainty is statistical and the second is the threshold-selection systematic uncertainty, evaluated from the envelope of the requested threshold variations.}}",
        rf"\label{{tab:apa{apa}_adjacent_pair_resolution_results}}",
        r"\end{table}",
        "",
    ))
    fit_path = table_directory / f"apa{apa}_adjacent_pair_resolution_results.tex"
    fit_path.write_text("\n".join(fit_lines), encoding="utf-8")
    return [width_path, fit_path]


def clear_outputs(output_dir: Path) -> None:
    for name in (
        "adjacent_pair_differential_resolution.csv", "pair_availability.csv",
        "pair_configuration.csv", "selection_thresholds.csv", "selected_trigger_counts.csv",
        "pair_analysis_summary.csv", "pair_resolution_fit_results.csv",
        "apa1_adjacent_pair_gaussian_resolution.pdf", "apa2_adjacent_pair_gaussian_resolution.pdf",
        "report.txt", "manifest.json", "pair_threshold_systematics.csv",
        "pair_resolution_threshold_systematics.csv",
        "differential_gaussian_sigma_vs_neff.png", "differential_width_vs_neff.png",
    ):
        path = output_dir / name
        if path.is_file():
            path.unlink()
    diagnostics = output_dir / "diagnostics"
    if diagnostics.is_dir():
        for path in diagnostics.glob("*.png"):
            path.unlink()
    individual = output_dir / "individual_plots"
    if individual.is_dir():
        for path in individual.rglob("*.png"):
            path.unlink()
    thesis_tables = output_dir / "thesis_tables"
    if thesis_tables.is_dir():
        for path in thesis_tables.glob("*.tex"):
            path.unlink()


def parse_arguments() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    analysis_dir = here.parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=analysis_dir / "output/apa1_vs_apa2")
    parser.add_argument("--trigger-data", type=Path, default=analysis_dir / "output/review/apa12_trigger_data_01/apa12_trigger_data.csv")
    parser.add_argument("--population-fit-results", type=Path, default=analysis_dir / "output/review/apa12_trigger_distributions_01/apa1_population_fit_results.csv")
    parser.add_argument("--composition", type=Path, default=analysis_dir / "data/np04_beam_particle_content.csv")
    parser.add_argument("--output-dir", type=Path, default=analysis_dir / "output/review/pds_differential_resolution_01")
    parser.add_argument("--apa", type=int, choices=(1, 2), default=1)
    parser.add_argument("--minimum-events", type=int, default=150)
    parser.add_argument("--low-coverage-threshold", type=float, default=0.80)
    parser.add_argument("--minimum-momenta-for-pdf", type=int, choices=(4, 5), default=4)
    parser.add_argument("--relative-momentum-error", type=float, default=0.05)
    parser.add_argument(
        "--threshold-sigma-multipliers", nargs="+", type=float,
        default=[-1.0, 0.0, 1.0],
        help="Threshold variations in units of the fitted intersection uncertainty.",
    )
    parser.add_argument(
        "--resolution-model",
        choices=("two-term", "three-term", "both"),
        default="two-term",
        help="Resolution model to fit and draw; default: two-term.",
    )
    parser.add_argument(
        "--export-pair", action="append", default=[], metavar="PAIR",
        help="Export individual PNG panels for this pair identifier; may be repeated.",
    )
    arguments = parser.parse_args()
    for attribute in ("input_dir", "trigger_data", "population_fit_results", "composition", "output_dir"):
        setattr(arguments, attribute, getattr(arguments, attribute).expanduser().resolve())
    if arguments.minimum_events < 10:
        parser.error("--minimum-events must be at least 10")
    if not 0.0 < arguments.low_coverage_threshold <= 1.0:
        parser.error("--low-coverage-threshold must be in (0, 1]")
    if arguments.relative_momentum_error < 0:
        parser.error("--relative-momentum-error must be non-negative")
    if not arguments.threshold_sigma_multipliers:
        parser.error("--threshold-sigma-multipliers must not be empty")
    if len(set(arguments.threshold_sigma_multipliers)) != len(arguments.threshold_sigma_multipliers):
        parser.error("--threshold-sigma-multipliers must be distinct")
    if not any(math.isclose(value, 0.0) for value in arguments.threshold_sigma_multipliers):
        parser.error("--threshold-sigma-multipliers must include 0 for the nominal selection")
    for path in (arguments.input_dir, arguments.trigger_data, arguments.population_fit_results, arguments.composition):
        if not path.exists():
            parser.error(f"Input does not exist: {path}")
    if arguments.output_dir == arguments.input_dir:
        parser.error("--output-dir must differ from --input-dir")
    return arguments


def main() -> int:
    arguments = parse_arguments()
    selected_models = (
        RESOLUTION_MODELS
        if arguments.resolution_model == "both"
        else (("two_term",) if arguments.resolution_model == "two-term" else ("three_term",))
    )
    primary_model = "two_term" if "two_term" in selected_models else "three_term"
    nominal_multiplier = next(value for value in arguments.threshold_sigma_multipliers if math.isclose(value, 0.0))
    scenario_multipliers = [nominal_multiplier] + [
        value for value in arguments.threshold_sigma_multipliers if not math.isclose(value, 0.0)
    ]
    scenarios = [(scenario_name(value), value) for value in scenario_multipliers]
    scenario_order = [name for name, _ in scenarios]

    contexts = load_trigger_contexts(arguments.trigger_data)
    thresholds, _ = load_nominal_thresholds(arguments.population_fit_results)
    threshold_rows = build_threshold_rows(thresholds, scenario_multipliers)
    kinetic = load_kinetic_energies(arguments.composition, arguments.relative_momentum_error)
    data_by_momentum = {
        momentum: load_merged_json(momentum, arguments.input_dir / f"{momentum}GeV")
        for momentum in MOMENTA
    }
    pairs = default_pairs(arguments.apa)
    if not pairs:
        raise ValueError(f"No manually defined adjacent pairs found for APA {arguments.apa}.")
    pairs_by_identifier = {pair.identifier: pair for pair in pairs}
    requested_export_identifiers = list(DEFAULT_EXPORT_PAIRS.get(arguments.apa, ())) + list(arguments.export_pair)
    unknown_export_pairs = sorted(set(requested_export_identifiers) - set(pairs_by_identifier))
    if unknown_export_pairs:
        available = ", ".join(sorted(pairs_by_identifier))
        raise ValueError(f"Unknown --export-pair identifier(s): {unknown_export_pairs}. Available identifiers: {available}")
    export_pairs = [pairs_by_identifier[identifier] for identifier in dict.fromkeys(requested_export_identifiers)]

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    clear_outputs(arguments.output_dir)
    measurements: list[dict] = []
    availability_rows: list[dict] = []
    selected_rows: list[dict] = []
    panels_by_pair = {
        pair.identifier: {momentum: {"pair": pair, "message": "No measurement available."} for momentum in MOMENTA}
        for pair in pairs
    }
    input_paths = [arguments.trigger_data, arguments.population_fit_results, arguments.composition]
    for momentum, blocks in data_by_momentum.items():
        input_paths.extend(arguments.input_dir / f"{momentum}GeV" / block / f"photoelectron_dic_{momentum}GeV.json" for block in blocks)

    for threshold_scenario, multiplier in scenarios:
        for momentum in MOMENTA:
            threshold, threshold_error = thresholds.get(momentum, (math.nan, math.nan))
            applied_threshold = math.nan if momentum == 1 else threshold + multiplier * threshold_error
            selected = selected_event_indices(
                data_by_momentum[momentum], contexts[momentum], momentum, arguments.apa,
                None if momentum == 1 else applied_threshold,
            )
            selection_kind = "apa_local_valid_no_muon_selection" if momentum == 1 else "apa1_mean_greater_than_threshold"
            selected_rows.append({
                "threshold_scenario": threshold_scenario,
                "threshold_sigma_multiplier": multiplier,
                "momentum_GeV_c": momentum,
                "selection_kind": selection_kind,
                "threshold_nominal_PE": threshold,
                "threshold_error_PE": threshold_error,
                "threshold_applied_PE": applied_threshold,
                "selected_triggers": int(sum(len(indices) for indices in selected.values())),
            })
            for pair in pairs:
                events = extract_pair_events(data_by_momentum[momentum], pair, selected)
                base = {
                    "threshold_scenario": threshold_scenario,
                    "threshold_sigma_multiplier": multiplier,
                    "selection_kind": selection_kind,
                    "threshold_nominal_PE": threshold,
                    "threshold_error_PE": threshold_error,
                    "threshold_applied_PE": applied_threshold,
                    "kinetic_mean_GeV": kinetic[momentum]["kinetic_mean_GeV"],
                    "effective_spread_GeV": kinetic[momentum]["effective_spread_GeV"],
                }
                availability = {
                    **base,
                    "pair": pair.identifier,
                    "kind": pair.kind,
                    "momentum_GeV_c": momentum,
                    "selected_triggers": events.selected_triggers,
                    "common_events": events.common_events,
                    "common_event_fraction": events.common_fraction,
                    "first_missing": events.first_missing,
                    "second_missing": events.second_missing,
                    "both_missing": events.both_missing,
                    "measurement_status": "",
                    "coverage_status": "",
                    "analysis_status": "",
                    "message": "",
                }
                panel = panels_by_pair[pair.identifier][momentum]
                if threshold_scenario == "nominal":
                    panel.update(pair=pair, events=events, record=None, d_values=None, message="")
                    if events.common_events > 0:
                        try:
                            panel["d_values"], _, _ = normalized_difference(events.values_a, events.values_b)
                        except ValueError as error:
                            panel["message"] = str(error)
                try:
                    measurement = measure_pair(pair, momentum, events, min_events=arguments.minimum_events)
                except ValueError as error:
                    message = str(error)
                    availability.update(
                        measurement_status="not_measured", coverage_status="not_available",
                        analysis_status="not_available", message=message,
                    )
                    availability_rows.append(availability)
                    if threshold_scenario == "nominal":
                        panel["message"] = message
                    continue

                coverage_status = "usable" if events.common_fraction >= arguments.low_coverage_threshold else "low_coverage"
                analysis_status = "usable" if coverage_status == "usable" and measurement.gaussian_d.status == "success" else (
                    "low_coverage" if coverage_status == "low_coverage" else "gaussian_fit_failed"
                )
                row = {
                    **base,
                    **measurement.as_flat_dict(),
                    "measurement_status": "success",
                    "coverage_status": coverage_status,
                    "analysis_status": analysis_status,
                    "message": measurement.gaussian_d.message,
                }
                measurements.append(row)
                availability.update(
                    measurement_status="success", coverage_status=coverage_status,
                    analysis_status=analysis_status, message=measurement.gaussian_d.message,
                )
                availability_rows.append(availability)
                if threshold_scenario == "nominal":
                    panel["record"] = row
                    panel["message"] = measurement.gaussian_d.message

    if not measurements:
        raise RuntimeError("No pair has the required number of common selected triggers.")
    table = pd.DataFrame(measurements)
    resolution_fits_by_scenario: dict[str, dict[str, dict[str, ResolutionFit]]] = {}
    resolution_rows: list[dict] = []
    for threshold_scenario, multiplier in scenarios:
        scenario_table = table.loc[table["threshold_scenario"] == threshold_scenario]
        scenario_fits: dict[str, dict[str, ResolutionFit]] = {}
        for pair in pairs:
            pair_records = scenario_table.loc[
                (scenario_table["pair"] == pair.identifier)
                & (scenario_table["d_gaussian_status"] == "success")
            ].to_dict("records")
            pair_fits: dict[str, ResolutionFit] = {}
            for model in selected_models:
                resolution_fit = fit_resolution(pair_records, model)
                pair_fits[model] = resolution_fit
                resolution_rows.append({
                    "threshold_scenario": threshold_scenario,
                    "threshold_sigma_multiplier": multiplier,
                    "pair": pair.identifier,
                    "kind": pair.kind,
                    "apa": pair.first.apa,
                    "first_endpoint": pair.first.endpoint,
                    "first_channel": pair.first.channel,
                    "second_endpoint": pair.second.endpoint,
                    "second_channel": pair.second.channel,
                    "resolution_model": model,
                    **resolution_fit.as_flat_dict(),
                })
            scenario_fits[pair.identifier] = pair_fits
        resolution_fits_by_scenario[threshold_scenario] = scenario_fits

    nominal_table = table.loc[table["threshold_scenario"] == "nominal"].copy()
    nominal_resolution_fits = resolution_fits_by_scenario["nominal"]
    nominal_resolution_rows = [row for row in resolution_rows if row["threshold_scenario"] == "nominal"]
    pair_summary = build_pair_summary(
        pairs, nominal_table, nominal_resolution_fits, arguments.minimum_momenta_for_pdf, primary_model,
    )
    threshold_systematics = build_pair_threshold_systematics(
        pairs, resolution_fits_by_scenario, scenario_order, primary_model,
    )
    pdf_path = arguments.output_dir / f"apa{arguments.apa}_adjacent_pair_gaussian_resolution.pdf"
    page_count = make_pair_pdf(
        pairs, panels_by_pair, pair_summary, nominal_resolution_fits,
        arguments.minimum_momenta_for_pdf, pdf_path,
    )
    exported_plot_count = export_individual_pair_plots(
        export_pairs, panels_by_pair, nominal_resolution_fits,
        arguments.output_dir / "individual_plots",
    )
    thesis_table_paths = write_thesis_tables(
        arguments.output_dir, arguments.apa, pairs, nominal_table,
        nominal_resolution_rows, threshold_systematics, primary_model,
    )

    write_csv(arguments.output_dir / "adjacent_pair_differential_resolution.csv", measurements)
    write_csv(arguments.output_dir / "pair_availability.csv", availability_rows)
    write_csv(arguments.output_dir / "selection_thresholds.csv", threshold_rows)
    write_csv(arguments.output_dir / "selected_trigger_counts.csv", selected_rows)
    write_csv(arguments.output_dir / "pair_analysis_summary.csv", pair_summary)
    write_csv(arguments.output_dir / "pair_resolution_fit_results.csv", resolution_rows)
    write_csv(arguments.output_dir / "pair_resolution_threshold_systematics.csv", threshold_systematics)
    write_csv(arguments.output_dir / "pair_configuration.csv", [
        {
            "pair": pair.identifier, "kind": pair.kind, "apa": pair.first.apa,
            "first_endpoint": pair.first.endpoint, "first_channel": pair.first.channel,
            "second_endpoint": pair.second.endpoint, "second_channel": pair.second.channel,
        }
        for pair in pairs
    ])

    gaussian_count = int(np.count_nonzero(nominal_table["d_gaussian_status"] == "success"))
    low_coverage_count = int(np.count_nonzero(nominal_table["coverage_status"] == "low_coverage"))
    resolution_successes = {
        model: int(sum(pair_fits[model].status == "success" for pair_fits in nominal_resolution_fits.values()))
        for model in selected_models
    }
    report = [
        "PDS ADJACENT-CHANNEL DIFFERENTIAL RESPONSE STUDY",
        f"APA: {arguments.apa}",
        f"Momenta [GeV/c]: {list(MOMENTA)}",
        f"Minimum common events: {arguments.minimum_events}",
        f"Low-coverage flag threshold: {arguments.low_coverage_threshold:.3f}",
        f"Minimum Gaussian-fit momenta per PDF pair: {arguments.minimum_momenta_for_pdf}",
        f"Threshold scenarios: {', '.join(scenario_order)}.",
        "Bootstrap resampling: not used.",
        "",
        "SELECTION",
        "At 1 GeV/c: APA-local-valid triggers; no muon-selection threshold.",
        "At 2--7 GeV/c: APA1-valid triggers with APA1 mean above the Langauss--Gaussian intersection.",
        "The nominal figures use the nominal intersection; threshold variations are used only for systematic uncertainties.",
        "Missing channel values are never replaced by zero.",
        "",
        "PRIMARY OBSERVABLE",
        "D_AB is fit with a Gaussian in a fixed robust central core: median +/- 2 times the empirical central-68% half-width.",
        "The fitted Gaussian sigma_D is the differential relative-response width.",
        "For comparable channel means, D_AB is the first-order equivalent of the beta asymmetry.",
        "",
        "DIFFERENTIAL-RESOLUTION FIT",
        f"Nominal model(s): {', '.join(selected_models)}.",
        "Two-term model: sqrt(a^2 + b^2/K_eff).",
        "Three-term model: sqrt(a^2 + b^2/K_eff + c^2/K_eff^2).",
        "Both the K_eff uncertainty and the Gaussian sigma_D uncertainty enter the effective-variance fit.",
        "The fit is differential and is not an absolute calorimetric energy-resolution measurement.",
        "",
        "QUALITY (NOMINAL SELECTION)",
        f"Successful pair/momentum measurements: {len(nominal_table)}.",
        f"Successful Gaussian core fits: {gaussian_count}.",
        f"Measurements flagged for coverage below {arguments.low_coverage_threshold:.2f}: {low_coverage_count}.",
        "Low coverage is retained as a quality flag in the CSV outputs and does not alter the PDF point style or the fit sample.",
        *[
            f"Successful {model.replace('_', '-')} differential-resolution fits: {resolution_successes[model]}."
            for model in selected_models
        ],
        f"Pairs included in the PDF: {page_count}.",
        f"Individual PNG panels exported: {exported_plot_count}.",
        "",
        "OUTPUTS",
        "adjacent_pair_differential_resolution.csv: all threshold scenarios, measured pairs, Gaussian-fit parameters, and empirical cross-checks.",
        "pair_availability.csv: availability and missing-value counts for every pair, momentum, and threshold scenario.",
        "pair_resolution_fit_results.csv: one row per pair, threshold scenario, and selected resolution model.",
        "pair_resolution_threshold_systematics.csv: nominal final-fit parameters and threshold-selection systematic envelopes.",
        "pair_analysis_summary.csv: nominal pair availability plus the selected-model fit results.",
        "selected_trigger_counts.csv: selected-trigger count per momentum and threshold scenario.",
        f"{pdf_path.name}: nominal five N_PE,A versus N_PE,B correlations, five D_AB distributions, and sigma_D(K_eff) for each eligible pair.",
        "individual_plots/<pair>/: requested standalone nominal PNG panels for that pair.",
        *(f"{path.relative_to(arguments.output_dir)}: ready-to-input LaTeX appendix table." for path in thesis_table_paths),
    ]
    (arguments.output_dir / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": sys.version,
        "configuration": {
            "apa": arguments.apa,
            "momenta_GeV_c": list(MOMENTA),
            "minimum_events": arguments.minimum_events,
            "low_coverage_threshold": arguments.low_coverage_threshold,
            "minimum_momenta_for_pdf": arguments.minimum_momenta_for_pdf,
            "relative_momentum_error": arguments.relative_momentum_error,
            "threshold_sigma_multipliers": scenario_multipliers,
            "threshold_scenarios": scenario_order,
            "bootstrap_resampling": False,
            "resolution_model_selection": arguments.resolution_model,
            "resolution_models": list(selected_models),
            "primary_resolution_model": primary_model,
            "two_term_resolution_model": "sqrt(a^2 + b^2/K_eff)",
            "three_term_resolution_model": "sqrt(a^2 + b^2/K_eff + c^2/K_eff^2)",
            "export_pair_identifiers": [pair.identifier for pair in export_pairs],
        },
        "inputs": [file_info(path) for path in input_paths if path.is_file()],
    }
    (arguments.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(arguments.output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, csv.Error, json.JSONDecodeError, InvalidOperation, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)
