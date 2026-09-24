"""Differential light-response resolution for ProtoDUNE-HD PDS channels.

This module deliberately separates the statistical observable from its
interpretation.  For a pair of channels A and B it measures

    D_AB = (N_A / <N_A> - N_B / <N_B>) / sqrt(2),

on the same selected triggers.  The width of D_AB is a *differential*
relative light-response width.  It becomes an estimator of a single-channel
relative resolution only under extra assumptions about the two channels and
their residual correlation.

The functions accept the JSON structure written by the light-yield analysis.
They do not make missing channels equivalent to zero signal: an event is used
only when both requested channels are present and finite.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import json
import math

import numpy as np


SQRT2 = math.sqrt(2.0)


@dataclass(frozen=True)
class Channel:
    """One PDS channel in the merged photoelectron JSON."""

    apa: int
    endpoint: int
    channel: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Channel":
        return cls(int(value["apa"]), int(value["end"]), int(value["ch"]))

    @property
    def label(self) -> str:
        return f"APA{self.apa}: end{self.endpoint} ch{self.channel}"


@dataclass(frozen=True)
class Pair:
    """A labelled pair.  ``kind`` can distinguish adjacent and control pairs."""

    first: Channel
    second: Channel
    label: str = ""
    kind: str = "adjacent"

    @property
    def identifier(self) -> str:
        if self.label:
            return self.label
        return f"e{self.first.endpoint}c{self.first.channel}_e{self.second.endpoint}c{self.second.channel}"


@dataclass
class WidthEstimate:
    """Width estimators for one distribution, including bootstrap uncertainty."""

    n_events: int
    mean: float
    standard_deviation: float
    central68_half_width: float
    bootstrap_standard_deviation: float
    bootstrap_central68_half_width: float


@dataclass
class PairMeasurement:
    """All quantities needed for one pair and one beam setting."""

    pair: str
    kind: str
    momentum_gev: float
    n_events: int
    mean_a: float
    mean_b: float
    mean_ratio_a_over_b: float
    n_eff: float
    pearson_correlation: float
    width_d: WidthEstimate
    width_beta: WidthEstimate

    def as_flat_dict(self) -> dict[str, float | int | str]:
        row: dict[str, float | int | str] = {
            "pair": self.pair,
            "kind": self.kind,
            "momentum_gev": self.momentum_gev,
            "n_events": self.n_events,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "mean_ratio_a_over_b": self.mean_ratio_a_over_b,
            "n_eff": self.n_eff,
            "pearson_correlation": self.pearson_correlation,
        }
        for name, estimate in (("d", self.width_d), ("beta", self.width_beta)):
            for field, value in asdict(estimate).items():
                if field != "n_events":
                    row[f"{name}_{field}"] = value
        return row


def selected_event_indices(
    merged_data: Mapping[str, Any],
    threshold: float,
    selection_apa: int = 1,
) -> dict[str, np.ndarray]:
    """Return selected trigger indices, keeping the JSON block boundaries.

    The threshold is applied to the APA average NPE used by the existing
    non-muon-like selection.  A non-finite average never passes the cut.
    """

    indices: dict[str, np.ndarray] = {}
    for block, payload in merged_data.items():
        averages = np.asarray(payload.get(str(selection_apa), {}).get("mean", []), dtype=float)
        indices[block] = np.flatnonzero(np.isfinite(averages) & (averages > threshold))
    return indices


def _read_channel(event: Mapping[str, Any], channel: Channel) -> float | None:
    """Read NPE without treating zero as a missing value."""

    data = event.get(str(channel.endpoint), {}).get(str(channel.channel))
    if not isinstance(data, Mapping):
        return None
    value = data.get("n_pe")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def extract_pair_events(
    merged_data: Mapping[str, Any], pair: Pair, selected_indices: Mapping[str, Iterable[int]]
) -> tuple[np.ndarray, np.ndarray]:
    """Extract matched channel values from the common selected event sample."""

    values_a: list[float] = []
    values_b: list[float] = []
    if pair.first.apa != pair.second.apa:
        raise ValueError("A differential pair must belong to one APA.")

    apa_key = str(pair.first.apa)
    for block, indices in selected_indices.items():
        events = merged_data.get(block, {}).get(apa_key, {}).get("channel_dic", [])
        for index in indices:
            if index >= len(events):
                continue
            first = _read_channel(events[index], pair.first)
            second = _read_channel(events[index], pair.second)
            if first is None or second is None:
                continue
            values_a.append(first)
            values_b.append(second)
    return np.asarray(values_a, dtype=float), np.asarray(values_b, dtype=float)


def normalized_difference(values_a: np.ndarray, values_b: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Compute D_AB with means evaluated on the common event sample."""

    mean_a = float(np.mean(values_a))
    mean_b = float(np.mean(values_b))
    if mean_a <= 0 or mean_b <= 0:
        raise ValueError("The normalized difference requires positive channel means.")
    return (values_a / mean_a - values_b / mean_b) / SQRT2, mean_a, mean_b


def beta_asymmetry(values_a: np.ndarray, values_b: np.ndarray) -> np.ndarray:
    """Return the article's original beta observable for a cross-check."""

    denominator = values_a + values_b
    valid = np.isfinite(denominator) & (denominator != 0)
    return SQRT2 * (values_a[valid] - values_b[valid]) / denominator[valid]


def central68_half_width(values: np.ndarray) -> float:
    """Half-width of the central 68% interval; robust against asymmetric tails."""

    low, high = np.percentile(values, [16.0, 84.0])
    return float((high - low) / 2.0)


def _widths(values: np.ndarray) -> tuple[float, float, float]:
    return (
        float(np.mean(values)),
        float(np.std(values, ddof=1)),
        central68_half_width(values),
    )


def bootstrap_widths(
    values_a: np.ndarray,
    values_b: np.ndarray,
    observable: str,
    repetitions: int = 1000,
    seed: int | None = 12345,
) -> tuple[float, float]:
    """Bootstrap width uncertainties, resampling complete A--B event pairs.

    The means in D_AB are recomputed in every resample, as required by the
    observable definition.
    """

    if len(values_a) != len(values_b) or len(values_a) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    standard_deviations = np.empty(repetitions)
    central_widths = np.empty(repetitions)
    for repetition in range(repetitions):
        indices = rng.integers(0, len(values_a), len(values_a))
        if observable == "d":
            sample, _, _ = normalized_difference(values_a[indices], values_b[indices])
        elif observable == "beta":
            sample = beta_asymmetry(values_a[indices], values_b[indices])
        else:
            raise ValueError(f"Unknown observable: {observable}")
        _, standard_deviations[repetition], central_widths[repetition] = _widths(sample)
    return float(np.std(standard_deviations, ddof=1)), float(np.std(central_widths, ddof=1))


def estimate_width(
    values_a: np.ndarray,
    values_b: np.ndarray,
    observable: str,
    bootstrap_repetitions: int = 1000,
    seed: int | None = 12345,
) -> WidthEstimate:
    if observable == "d":
        values, _, _ = normalized_difference(values_a, values_b)
    elif observable == "beta":
        values = beta_asymmetry(values_a, values_b)
    else:
        raise ValueError(f"Unknown observable: {observable}")
    mean, standard_deviation, central_width = _widths(values)
    bootstrap_std, bootstrap_central = bootstrap_widths(
        values_a, values_b, observable, bootstrap_repetitions, seed
    )
    return WidthEstimate(
        n_events=len(values),
        mean=mean,
        standard_deviation=standard_deviation,
        central68_half_width=central_width,
        bootstrap_standard_deviation=bootstrap_std,
        bootstrap_central68_half_width=bootstrap_central,
    )


def measure_pair(
    merged_data: Mapping[str, Any],
    pair: Pair,
    momentum_gev: float,
    threshold: float,
    min_events: int = 100,
    bootstrap_repetitions: int = 1000,
    seed: int | None = 12345,
) -> PairMeasurement:
    """Measure D_AB and beta for one pair and one momentum setting."""

    indices = selected_event_indices(merged_data, threshold)
    values_a, values_b = extract_pair_events(merged_data, pair, indices)
    if len(values_a) < min_events:
        raise ValueError(
            f"{pair.identifier} at {momentum_gev:g} GeV/c has only {len(values_a)} common selected events "
            f"(< {min_events})."
        )
    _, mean_a, mean_b = normalized_difference(values_a, values_b)
    n_eff = 2.0 * mean_a * mean_b / (mean_a + mean_b)
    correlation = float(np.corrcoef(values_a, values_b)[0, 1])
    return PairMeasurement(
        pair=pair.identifier,
        kind=pair.kind,
        momentum_gev=momentum_gev,
        n_events=len(values_a),
        mean_a=mean_a,
        mean_b=mean_b,
        mean_ratio_a_over_b=mean_a / mean_b,
        n_eff=n_eff,
        pearson_correlation=correlation,
        width_d=estimate_width(values_a, values_b, "d", bootstrap_repetitions, seed),
        width_beta=estimate_width(values_a, values_b, "beta", bootstrap_repetitions, seed),
    )


def measurements_table(measurements: Sequence[PairMeasurement]) -> "pd.DataFrame":
    """Make a CSV-ready table with one row per pair and momentum."""

    import pandas as pd

    return pd.DataFrame([measurement.as_flat_dict() for measurement in measurements])


def poisson_width(n_eff: np.ndarray | float) -> np.ndarray:
    """Independent-photoelectron prediction for the D_AB width."""

    return 1.0 / np.sqrt(np.asarray(n_eff, dtype=float))


def fit_two_term_energy_model(
    energy: Sequence[float], width: Sequence[float], width_error: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Fit sqrt(a^2 + b^2/E), avoiding an underconstrained three-term fit."""

    from scipy.optimize import curve_fit

    energy = np.asarray(energy, dtype=float)
    width = np.asarray(width, dtype=float)
    width_error = np.asarray(width_error, dtype=float)
    if len(energy) < 3:
        raise ValueError("At least three energy points are required for the two-term fit.")
    if np.any(energy <= 0) or np.any(width <= 0):
        raise ValueError("Energy and width must be positive.")
    positive_errors = width_error[np.isfinite(width_error) & (width_error > 0)]
    fallback = float(np.median(positive_errors)) if len(positive_errors) else 1.0
    sigma = np.where(np.isfinite(width_error) & (width_error > 0), width_error, fallback)

    def model(x: np.ndarray, constant: float, stochastic: float) -> np.ndarray:
        return np.sqrt(constant**2 + stochastic**2 / x)

    parameters, covariance = curve_fit(
        model,
        energy,
        width,
        sigma=sigma,
        absolute_sigma=True,
        bounds=(0.0, np.inf),
        p0=(float(np.min(width)), float(np.min(width) * np.sqrt(np.min(energy)))),
    )
    return parameters, covariance


def plot_pair_diagnostics(
    values_a: np.ndarray,
    values_b: np.ndarray,
    pair: Pair,
    momentum_gev: float,
    output: str | Path,
) -> None:
    """Save the scatter plot and D_AB distribution for one pair and momentum."""

    import matplotlib.pyplot as plt

    d_values, _, _ = normalized_difference(values_a, values_b)
    beta_values = beta_asymmetry(values_a, values_b)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].scatter(values_a, values_b, s=8, alpha=0.35)
    limit_min = min(np.min(values_a), np.min(values_b))
    limit_max = max(np.max(values_a), np.max(values_b))
    axes[0].plot([limit_min, limit_max], [limit_min, limit_max], "k--", lw=1)
    axes[0].set(xlabel=rf"$N_{{\rm PE}}$ ({pair.first.label})", ylabel=rf"$N_{{\rm PE}}$ ({pair.second.label})")
    axes[0].set_title(f"{pair.identifier}, {momentum_gev:g} GeV/c")
    axes[0].grid(alpha=0.3)

    axes[1].hist(d_values, bins="fd", density=True, alpha=0.65, label=r"$D_{AB}$")
    axes[1].hist(beta_values, bins="fd", density=True, histtype="step", lw=1.5, label=r"$\beta$")
    axes[1].set(xlabel="Normalized differential observable", ylabel="Density")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def load_merged_json(energy: int, energy_directory: str | Path) -> dict[str, Any]:
    """Read all final per-range JSON files without writing auxiliary files."""

    directory = Path(energy_directory)
    merged: dict[str, Any] = {}
    for subdirectory in sorted(directory.glob("*_to_*")):
        json_file = subdirectory / f"photoelectron_dic_{energy}GeV.json"
        if json_file.exists():
            with json_file.open() as stream:
                merged[subdirectory.name] = json.load(stream)
    if not merged:
        raise FileNotFoundError(f"No photoelectron JSON files found in {directory}")
    return merged


if __name__ == "__main__":
    # This file is intentionally a library.  A small run-specific driver should
    # define input paths, threshold variations, and the approved pair list.
    print("Import pds_differential_resolution from a run-specific driver.")
