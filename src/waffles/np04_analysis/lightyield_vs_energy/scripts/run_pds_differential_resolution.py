#!/usr/bin/env python3
"""Study the Gaussian width of D_AB for manually defined adjacent PDS pairs.

Each APA pair is evaluated at 1, 2, 3, 5, and 7 GeV/c.  The 1 GeV/c sample
uses APA-local-valid triggers because its beam muon fraction is assumed to be
zero.  At 2--7 GeV/c, the selection uses the nominal Langauss--Gaussian
intersection on the APA 1 average response.  No threshold systematic is run in
this version.

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
FIT_COLOR = "#D55E00"
LOW_COVERAGE_EDGE = "#4D4D4D"


@dataclass
class ResolutionFit:
    """Three-term fit of the differential relative-response width."""

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
    def failed(cls, message: str, n_points: int = 0, momenta: str = "", low_coverage_momenta: str = "") -> "ResolutionFit":
        return cls(
            status="failed", message=message, n_points=n_points,
            momenta=momenta, low_coverage_momenta=low_coverage_momenta,
            constant_a=float("nan"), constant_a_error=float("nan"),
            stochastic_b_sqrt_GeV=float("nan"), stochastic_b_error_sqrt_GeV=float("nan"),
            noise_c_GeV=float("nan"), noise_c_error_GeV=float("nan"),
            chi2=float("nan"), ndf=0, r_squared=float("nan"),
        )

    def as_flat_dict(self) -> dict[str, float | int | str]:
        result = asdict(self)
        result["chi2_ndf"] = self.chi2_ndf
        return {f"resolution_fit_{key}": value for key, value in result.items()}


def resolution_model(kinetic_energy: np.ndarray | float, constant_a: float, stochastic_b: float, noise_c: float) -> np.ndarray:
    """Three-term differential-resolution model evaluated at K_eff."""

    energy = np.asarray(kinetic_energy, dtype=float)
    return np.sqrt(
        constant_a**2
        + (stochastic_b / np.sqrt(energy)) ** 2
        + (noise_c / energy) ** 2
    )


def resolution_derivative(kinetic_energy: np.ndarray, response: np.ndarray, stochastic_b: float, noise_c: float) -> np.ndarray:
    """Derivative of the resolution model with respect to K_eff."""

    return -(
        stochastic_b**2 / kinetic_energy**2
        + 2.0 * noise_c**2 / kinetic_energy**3
    ) / (2.0 * response)


def fit_resolution(records: list[dict]) -> ResolutionFit:
    """Fit sigma_D(K_eff), including the horizontal and vertical uncertainties.

    The residual uncertainty is evaluated as ``sqrt(sigma_y^2 +
    (d sigma_D / d K_eff * sigma_K)^2)``.  This is the standard effective-
    variance treatment of the K_eff uncertainty in a bounded nonlinear fit.
    """

    records = sorted(records, key=lambda row: row["kinetic_mean_GeV"])
    momenta = ";".join(str(int(row["momentum_GeV_c"])) for row in records)
    low_coverage = ";".join(
        str(int(row["momentum_GeV_c"]))
        for row in records
        if row["coverage_status"] == "low_coverage"
    )
    if len(records) < 4:
        return ResolutionFit.failed(
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
        return ResolutionFit.failed("Non-finite or invalid fit input.", len(records), momenta, low_coverage)

    def residual(parameters: np.ndarray) -> np.ndarray:
        expected = resolution_model(x, *parameters)
        derivative = resolution_derivative(x, expected, parameters[1], parameters[2])
        uncertainty = np.hypot(sy, derivative * sx)
        return (y - expected) / np.maximum(uncertainty, 1.0e-12)

    minimum = max(float(np.min(y)), 1.0e-4)
    initial = np.asarray((0.65 * minimum, 0.18, 0.10), dtype=float)
    try:
        solution = least_squares(
            residual,
            initial,
            bounds=(np.full(3, 1.0e-8), np.full(3, np.inf)),
            x_scale="jac",
            max_nfev=20_000,
            ftol=1.0e-10,
            xtol=1.0e-10,
            gtol=1.0e-10,
        )
    except (RuntimeError, ValueError, FloatingPointError) as error:
        return ResolutionFit.failed(f"Resolution fit failed: {error}", len(records), momenta, low_coverage)
    if not solution.success:
        return ResolutionFit.failed(f"Resolution fit failed: {solution.message}", len(records), momenta, low_coverage)

    parameters = np.asarray(solution.x, dtype=float)
    prediction = resolution_model(x, *parameters)
    chi2 = float(np.sum(residual(parameters) ** 2))
    ndf = len(x) - len(parameters)
    centered = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = float(1.0 - np.sum((y - prediction) ** 2) / centered) if centered > 0 else float("nan")
    try:
        covariance = np.linalg.pinv(solution.jac.T @ solution.jac)
        scale = max(1.0, chi2 / ndf) if ndf > 0 else 1.0
        errors = np.sqrt(np.diag(covariance) * scale)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        errors = np.full(3, np.nan)
    message = ""
    if np.any(parameters <= 1.0e-6):
        message = "One or more fit terms are compatible with the lower bound."
    if not np.all(np.isfinite(errors)):
        message = (message + " " if message else "") + "Parameter covariance is not finite."

    return ResolutionFit(
        status="success", message=message, n_points=len(records),
        momenta=momenta, low_coverage_momenta=low_coverage,
        constant_a=float(parameters[0]), constant_a_error=float(errors[0]),
        stochastic_b_sqrt_GeV=float(parameters[1]), stochastic_b_error_sqrt_GeV=float(errors[1]),
        noise_c_GeV=float(parameters[2]), noise_c_error_GeV=float(errors[2]),
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
    """Add the preliminary-status label inside an individual panel."""

    horizontal_alignment = "right" if location == "right" else "left"
    x_position = 0.975 if location == "right" else 0.025
    axis.text(
        x_position,
        0.965,
        r"$\mathbf{ProtoDUNE\!-\!HD}$" + "\nWork in Progress",
        transform=axis.transAxes,
        ha=horizontal_alignment,
        va="top",
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


def plot_correlation_panel(
    axis: plt.Axes,
    panel: dict,
    momentum: int,
    show_momentum_title: bool = True,
    standalone: bool = False,
) -> None:
    """Draw raw N_PE,A versus N_PE,B on common selected triggers."""

    events = panel.get("events")
    record = panel.get("record")
    if show_momentum_title:
        axis.set_title(rf"$p_{{\rm beam}} = {momentum:g}$ GeV/c", loc="center", fontsize=8.6)
    add_panel_brand(axis, "right", standalone=standalone)
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
    axis.scatter(values_a, values_b, s=5.0, color=DATA_COLOR, alpha=0.22, edgecolors="none", rasterized=True)
    if record is not None and record["mean_a_PE"] > 0:
        x_line = np.asarray((x_low, x_high))
        axis.plot(
            x_line, record["mean_b_PE"] / record["mean_a_PE"] * x_line,
            "--", color=FIT_COLOR, lw=1.25,
            label=r"$N_B = (\mu_B/\mu_A)N_A$",
        )
        text = rf"$\rho_{{AB}} = {record['pearson_correlation']:.3f}$" + "\n" + rf"{int(record['common_events'])} common triggers"
        axis.text(
            0.97, 0.06, text, transform=axis.transAxes, ha="right", va="bottom",
            fontsize=9.0 if standalone else 6.3,
            bbox={"facecolor": "white", "edgecolor": "#999999", "alpha": 0.92, "pad": 1.4},
        )
        axis.legend(frameon=True, facecolor="white", fontsize=9.2 if standalone else 6.0, loc="upper left")
    axis.set(xlim=(x_low, x_high), ylim=(y_low, y_high), xlabel=r"$N_{\rm PE}^{A}$", ylabel=r"$N_{\rm PE}^{B}$")
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
        axis.set(xlabel=r"$D_{AB}$", ylabel="Counts")
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
        axis.set(xlabel=r"$D_{AB}$", ylabel="Counts", ylim=(0.0, ymax))
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
    axis.set(xlim=(low, high), ylim=(0.0, ymax), xlabel=r"$D_{AB}$", ylabel="Counts")
    axis.legend(frameon=True, facecolor="white", fontsize=9.2 if standalone else 6.0, loc="upper left")
    format_panel_axis(axis, standalone)


def plot_pair_resolution_panel(
    axis: plt.Axes,
    panels: dict[int, dict],
    resolution_fit: ResolutionFit,
    standalone: bool = False,
) -> None:
    """Draw the dimensionless sigma_D(K_eff) and the three-term fit."""

    records = []
    for momentum in MOMENTA:
        record = panels.get(momentum, {}).get("record")
        if record is not None and record["d_gaussian_status"] == "success":
            records.append(record)
    add_panel_brand(axis, "center", standalone=standalone)
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

    if resolution_fit.status == "success":
        x_min = max(0.05, min(row["kinetic_mean_GeV"] - row["effective_spread_GeV"] for row in records))
        x_max = max(row["kinetic_mean_GeV"] + row["effective_spread_GeV"] for row in records)
        x_curve = np.linspace(x_min, 1.04 * x_max, 400)
        y_curve = resolution_model(
            x_curve, resolution_fit.constant_a, resolution_fit.stochastic_b_sqrt_GeV, resolution_fit.noise_c_GeV
        )
        label = (
            "Fit function\n"
            + r"$y = \sqrt{a^2 + b^2/x + c^2/x^2}$" + "\n"
            + rf"$a = ({resolution_fit.constant_a:.3f} \pm {resolution_fit.constant_a_error:.3f})$" + "\n"
            + rf"$b = ({resolution_fit.stochastic_b_sqrt_GeV:.3f} \pm {resolution_fit.stochastic_b_error_sqrt_GeV:.3f})\sqrt{{\rm GeV}}$" + "\n"
            + rf"$c = ({resolution_fit.noise_c_GeV:.3f} \pm {resolution_fit.noise_c_error_GeV:.3f})\,{{\rm GeV}}$" + "\n"
            + rf"$\chi^2/{{\rm ndf}} = {resolution_fit.chi2:.2f}/{resolution_fit.ndf:d} = {resolution_fit.chi2_ndf:.2f}$" + "\n"
            + rf"$R^2 = {resolution_fit.r_squared:.3f}$"
        )
        axis.plot(x_curve, y_curve, color=FIT_COLOR, lw=2.3 if standalone else 1.9, label=label)
    else:
        axis.plot([], [], color=FIT_COLOR, lw=1.9, label="Fit function unavailable")
    axis.set(xlabel=r"$K_{\rm eff}$ [GeV]", ylabel=r"Gaussian $\sigma_D$")
    axis.grid(alpha=0.24)
    axis.legend(frameon=True, facecolor="white", fontsize=10.0 if standalone else 6.9, loc="upper right")
    format_panel_axis(axis, standalone)


def export_individual_pair_plots(
    pairs: list[Pair],
    panels_by_pair: dict[str, dict[int, dict]],
    resolution_fits: dict[str, ResolutionFit],
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
    resolution_fits: dict[str, ResolutionFit],
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
    resolution_fits: dict[str, ResolutionFit],
    min_momenta: int,
) -> list[dict]:
    rows: list[dict] = []
    for pair in pairs:
        group = table.loc[table["pair"] == pair.identifier]
        fit_success = group.loc[group["d_gaussian_status"] == "success"]
        low_coverage = fit_success.loc[fit_success["coverage_status"] == "low_coverage"]
        resolution_fit = resolution_fits[pair.identifier]
        rows.append({
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
            **resolution_fit.as_flat_dict(),
        })
    return rows


def clear_outputs(output_dir: Path) -> None:
    for name in (
        "adjacent_pair_differential_resolution.csv", "pair_availability.csv",
        "pair_configuration.csv", "selection_thresholds.csv", "selected_trigger_counts.csv",
        "pair_analysis_summary.csv", "pair_resolution_fit_results.csv",
        "apa1_adjacent_pair_gaussian_resolution.pdf", "apa2_adjacent_pair_gaussian_resolution.pdf",
        "report.txt", "manifest.json", "pair_threshold_systematics.csv",
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


def parse_arguments() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    analysis_dir = here.parent
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
    for path in (arguments.input_dir, arguments.trigger_data, arguments.population_fit_results, arguments.composition):
        if not path.exists():
            parser.error(f"Input does not exist: {path}")
    if arguments.output_dir == arguments.input_dir:
        parser.error("--output-dir must differ from --input-dir")
    return arguments


def main() -> int:
    arguments = parse_arguments()
    contexts = load_trigger_contexts(arguments.trigger_data)
    thresholds, threshold_rows = load_nominal_thresholds(arguments.population_fit_results)
    kinetic = load_kinetic_energies(arguments.composition, arguments.relative_momentum_error)
    data_by_momentum = {
        momentum: load_merged_json(momentum, arguments.input_dir / f"{momentum}GeV")
        for momentum in MOMENTA
    }
    pairs = default_pairs(arguments.apa)
    if not pairs:
        raise ValueError(f"No manually defined adjacent pairs found for APA {arguments.apa}.")
    pairs_by_identifier = {pair.identifier: pair for pair in pairs}
    unknown_export_pairs = sorted(set(arguments.export_pair) - set(pairs_by_identifier))
    if unknown_export_pairs:
        available = ", ".join(sorted(pairs_by_identifier))
        raise ValueError(f"Unknown --export-pair identifier(s): {unknown_export_pairs}. Available identifiers: {available}")
    export_pairs = [pairs_by_identifier[identifier] for identifier in dict.fromkeys(arguments.export_pair)]

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

    for momentum in MOMENTA:
        threshold, threshold_error = thresholds.get(momentum, (math.nan, math.nan))
        applied_threshold = math.nan if momentum == 1 else threshold
        selected = selected_event_indices(
            data_by_momentum[momentum], contexts[momentum], momentum, arguments.apa,
            None if momentum == 1 else applied_threshold,
        )
        selection_kind = "apa_local_valid_no_muon_selection" if momentum == 1 else "apa1_mean_greater_than_nominal_threshold"
        selected_rows.append({
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
                availability.update(measurement_status="not_measured", coverage_status="not_available", analysis_status="not_available", message=message)
                availability_rows.append(availability)
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
                measurement_status="success",
                coverage_status=coverage_status,
                analysis_status=analysis_status,
                message=measurement.gaussian_d.message,
            )
            availability_rows.append(availability)
            panel["record"] = row
            panel["message"] = measurement.gaussian_d.message

    if not measurements:
        raise RuntimeError("No pair has the required number of common selected triggers.")
    table = pd.DataFrame(measurements)
    resolution_fits: dict[str, ResolutionFit] = {}
    resolution_rows: list[dict] = []
    for pair in pairs:
        pair_records = table.loc[
            (table["pair"] == pair.identifier)
            & (table["d_gaussian_status"] == "success")
        ].to_dict("records")
        resolution_fit = fit_resolution(pair_records)
        resolution_fits[pair.identifier] = resolution_fit
        resolution_rows.append({
            "pair": pair.identifier,
            "kind": pair.kind,
            "apa": pair.first.apa,
            "first_endpoint": pair.first.endpoint,
            "first_channel": pair.first.channel,
            "second_endpoint": pair.second.endpoint,
            "second_channel": pair.second.channel,
            **resolution_fit.as_flat_dict(),
        })

    pair_summary = build_pair_summary(
        pairs, table, resolution_fits, arguments.minimum_momenta_for_pdf,
    )
    pdf_path = arguments.output_dir / f"apa{arguments.apa}_adjacent_pair_gaussian_resolution.pdf"
    page_count = make_pair_pdf(
        pairs, panels_by_pair, pair_summary, resolution_fits,
        arguments.minimum_momenta_for_pdf, pdf_path,
    )
    exported_plot_count = export_individual_pair_plots(
        export_pairs, panels_by_pair, resolution_fits,
        arguments.output_dir / "individual_plots",
    )

    write_csv(arguments.output_dir / "adjacent_pair_differential_resolution.csv", measurements)
    write_csv(arguments.output_dir / "pair_availability.csv", availability_rows)
    write_csv(arguments.output_dir / "selection_thresholds.csv", threshold_rows)
    write_csv(arguments.output_dir / "selected_trigger_counts.csv", selected_rows)
    write_csv(arguments.output_dir / "pair_analysis_summary.csv", pair_summary)
    write_csv(arguments.output_dir / "pair_resolution_fit_results.csv", resolution_rows)
    write_csv(arguments.output_dir / "pair_configuration.csv", [
        {
            "pair": pair.identifier, "kind": pair.kind, "apa": pair.first.apa,
            "first_endpoint": pair.first.endpoint, "first_channel": pair.first.channel,
            "second_endpoint": pair.second.endpoint, "second_channel": pair.second.channel,
        }
        for pair in pairs
    ])

    gaussian_count = int(np.count_nonzero(table["d_gaussian_status"] == "success"))
    low_coverage_count = int(np.count_nonzero(table["coverage_status"] == "low_coverage"))
    resolution_successes = int(sum(fit.status == "success" for fit in resolution_fits.values()))
    report = [
        "PDS ADJACENT-CHANNEL DIFFERENTIAL RESPONSE STUDY",
        f"APA: {arguments.apa}",
        f"Momenta [GeV/c]: {list(MOMENTA)}",
        f"Minimum common events: {arguments.minimum_events}",
        f"Low-coverage flag threshold: {arguments.low_coverage_threshold:.3f}",
        f"Minimum Gaussian-fit momenta per PDF pair: {arguments.minimum_momenta_for_pdf}",
        "Threshold scenario: nominal only.",
        "",
        "SELECTION",
        "At 1 GeV/c: APA-local-valid triggers; no muon-selection threshold.",
        "At 2--7 GeV/c: APA1-valid triggers with APA1 mean above the nominal Langauss--Gaussian intersection.",
        "Missing channel values are never replaced by zero.",
        "",
        "PRIMARY OBSERVABLE",
        "D_AB is fit with a Gaussian in a fixed robust central core: median +/- 2 times the empirical central-68% half-width.",
        "The fitted Gaussian sigma_D is the differential relative-response width.",
        "For comparable channel means, D_AB is the first-order equivalent of the beta asymmetry.",
        "",
        "DIFFERENTIAL-RESOLUTION FIT",
        "For pairs with at least four successful Gaussian widths, sigma_D(K_eff) is fit with",
        "sqrt(a^2 + b^2/K_eff + c^2/K_eff^2).",
        "Both the K_eff uncertainty and the Gaussian sigma_D uncertainty enter the effective-variance fit.",
        "The fit is differential and is not an absolute calorimetric energy-resolution measurement.",
        "",
        "QUALITY",
        f"Successful pair/momentum measurements: {len(table)}.",
        f"Successful Gaussian core fits: {gaussian_count}.",
        f"Measurements flagged for coverage below {arguments.low_coverage_threshold:.2f}: {low_coverage_count}.",
        "Low coverage is retained as a quality flag in the CSV outputs and does not alter the PDF point style or the fit sample.",
        f"Successful three-term differential-resolution fits: {resolution_successes}.",
        f"Pairs included in the PDF: {page_count}.",
        f"Individual PNG panels exported: {exported_plot_count}.",
        "",
        "OUTPUTS",
        "adjacent_pair_differential_resolution.csv: all measured pairs, Gaussian fit parameters, and empirical cross-checks.",
        "pair_availability.csv: availability and missing-value counts for every pair and momentum.",
        "pair_resolution_fit_results.csv: three-term differential-resolution fit parameters and uncertainties.",
        "pair_analysis_summary.csv: pair availability, PDF inclusion, and resolution-fit status.",
        "selected_trigger_counts.csv: selected-trigger count per momentum.",
        f"{pdf_path.name}: five N_PE,A versus N_PE,B correlations, five D_AB distributions, and sigma_D(K_eff) for each eligible pair.",
        "individual_plots/<pair>/: requested standalone PNG panels for that pair.",
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
            "threshold_scenario": "nominal",
            "resolution_model": "sqrt(a^2 + b^2/K_eff + c^2/K_eff^2)",
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
