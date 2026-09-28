r"""Core tools for the adjacent-channel differential PDS response study.

For an adjacent channel pair A--B and a common selected trigger sample, the
primary observable is

.. math::

   D_{AB} = \frac{1}{\sqrt{2}}\left(
   \frac{N_{\rm PE}^{A}}{\mu_A} - \frac{N_{\rm PE}^{B}}{\mu_B}\right).

The program fits the central part of the D_AB distribution with a Gaussian and
uses its sigma as the primary differential-response width.  The empirical
central-68% half-width is retained as a robustness cross-check.  Neither
quantity is an absolute calorimetric energy resolution by itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import json
import math

import numpy as np
from scipy.optimize import curve_fit


SQRT2 = math.sqrt(2.0)


@dataclass(frozen=True)
class Channel:
    """One PDS channel in the photoelectron JSON."""

    apa: int
    endpoint: int
    channel: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Channel":
        return cls(int(value["apa"]), int(value["end"]), int(value["ch"]))

    @property
    def label(self) -> str:
        return f"APA {self.apa}, END {self.endpoint} - CH {self.channel}"


@dataclass(frozen=True)
class Pair:
    """A manually defined geometrically adjacent channel pair."""

    first: Channel
    second: Channel
    label: str = ""
    kind: str = "manual_adjacent_pair"

    @property
    def identifier(self) -> str:
        if self.label:
            return self.label
        return (
            f"apa{self.first.apa}_end{self.first.endpoint}_ch{self.first.channel}"
            f"__end{self.second.endpoint}_ch{self.second.channel}"
        )


@dataclass
class PairEvents:
    """Matched values and availability counters for a channel pair."""

    values_a: np.ndarray
    values_b: np.ndarray
    selected_triggers: int
    common_events: int
    first_missing: int
    second_missing: int
    both_missing: int

    @property
    def common_fraction(self) -> float:
        return self.common_events / self.selected_triggers if self.selected_triggers else float("nan")


@dataclass
class EmpiricalWidth:
    """Non-parametric width diagnostics for one distribution."""

    n_events: int
    mean: float
    standard_deviation: float
    central68_half_width: float


@dataclass
class GaussianFit:
    """Gaussian core-fit result for the D_AB distribution."""

    status: str
    message: str
    fit_events: int
    amplitude: float
    amplitude_error: float
    mean: float
    mean_error: float
    sigma: float
    sigma_error: float
    fit_low: float
    fit_high: float
    bin_width: float
    chi2: float
    ndf: int

    @property
    def chi2_ndf(self) -> float:
        return self.chi2 / self.ndf if self.ndf > 0 else float("nan")

    @classmethod
    def failed(cls, message: str, fit_low: float = float("nan"), fit_high: float = float("nan")) -> "GaussianFit":
        return cls(
            status="failed", message=message, fit_events=0,
            amplitude=float("nan"), amplitude_error=float("nan"),
            mean=float("nan"), mean_error=float("nan"),
            sigma=float("nan"), sigma_error=float("nan"),
            fit_low=fit_low, fit_high=fit_high, bin_width=float("nan"),
            chi2=float("nan"), ndf=0,
        )


@dataclass
class PairMeasurement:
    """Differential response quantities for one pair and one beam setting."""

    pair: str
    kind: str
    momentum_gev: float
    selected_triggers: int
    common_events: int
    common_event_fraction: float
    first_missing: int
    second_missing: int
    both_missing: int
    mean_a: float
    mean_b: float
    mean_ratio_a_over_b: float
    n_eff: float
    pearson_correlation: float
    empirical_d: EmpiricalWidth
    empirical_beta: EmpiricalWidth
    gaussian_d: GaussianFit

    def as_flat_dict(self) -> dict[str, float | int | str]:
        row: dict[str, float | int | str] = {
            "pair": self.pair,
            "kind": self.kind,
            "momentum_GeV_c": self.momentum_gev,
            "selected_triggers": self.selected_triggers,
            "common_events": self.common_events,
            "common_event_fraction": self.common_event_fraction,
            "first_missing": self.first_missing,
            "second_missing": self.second_missing,
            "both_missing": self.both_missing,
            "mean_a_PE": self.mean_a,
            "mean_b_PE": self.mean_b,
            "mean_ratio_a_over_b": self.mean_ratio_a_over_b,
            "n_eff_PE": self.n_eff,
            "pearson_correlation": self.pearson_correlation,
        }
        for prefix, value in (("d", self.empirical_d), ("beta", self.empirical_beta)):
            for field, field_value in asdict(value).items():
                row[f"{prefix}_{field}"] = field_value
        for field, field_value in asdict(self.gaussian_d).items():
            row[f"d_gaussian_{field}"] = field_value
        row["d_gaussian_chi2_ndf"] = self.gaussian_d.chi2_ndf
        return row


def selected_event_indices(
    merged_data: Mapping[str, Any],
    trigger_contexts: Mapping[tuple[str, int], Mapping[str, Any]],
    momentum_gev: int,
    apa: int,
    threshold: float | None,
) -> dict[str, np.ndarray]:
    """Return selected JSON indices using the channel-linearity selection.

    At 1 GeV/c the beam muon fraction is assumed to be zero and only the local
    APA-valid flag is required.  At 2--7 GeV/c, the non-muon-like tag is always
    defined by the APA 1 average response.
    """

    if momentum_gev != 1 and threshold is None:
        raise ValueError("A threshold is required at 2--7 GeV/c.")
    indices: dict[str, np.ndarray] = {}
    for block, payload in merged_data.items():
        chosen: list[int] = []
        for index, raw_time in enumerate(payload.get("trigger time", [])):
            try:
                context = trigger_contexts.get((block, int(raw_time)))
            except (TypeError, ValueError):
                context = None
            if context is None:
                continue
            if momentum_gev == 1:
                keep = bool(context.get(f"apa{apa}_valid", False))
            else:
                mean = context.get("apa1_mean", float("nan"))
                keep = bool(context.get("apa1_valid", False)) and np.isfinite(mean) and mean > threshold
            if keep:
                chosen.append(index)
        indices[block] = np.asarray(chosen, dtype=int)
    return indices


def _read_channel(event: Mapping[str, Any], channel: Channel) -> float | None:
    """Read a finite NPE value, preserving valid zero values."""

    data = event.get(str(channel.endpoint), {}).get(str(channel.channel))
    if not isinstance(data, Mapping):
        return None
    try:
        value = float(data["n_pe"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def extract_pair_events(
    merged_data: Mapping[str, Any], pair: Pair, selected_indices: Mapping[str, Iterable[int]]
) -> PairEvents:
    """Extract values from common selected triggers and record missing values."""

    if pair.first.apa != pair.second.apa:
        raise ValueError("A differential pair must belong to one APA.")
    values_a: list[float] = []
    values_b: list[float] = []
    selected_triggers = common = first_missing = second_missing = both_missing = 0
    apa_key = str(pair.first.apa)
    for block, indices in selected_indices.items():
        events = merged_data.get(block, {}).get(apa_key, {}).get("channel_dic", [])
        for index in indices:
            selected_triggers += 1
            if index < 0 or index >= len(events) or not isinstance(events[index], Mapping):
                both_missing += 1
                continue
            first = _read_channel(events[index], pair.first)
            second = _read_channel(events[index], pair.second)
            if first is None and second is None:
                both_missing += 1
            elif first is None:
                first_missing += 1
            elif second is None:
                second_missing += 1
            else:
                values_a.append(first)
                values_b.append(second)
                common += 1
    return PairEvents(
        values_a=np.asarray(values_a, dtype=float),
        values_b=np.asarray(values_b, dtype=float),
        selected_triggers=selected_triggers,
        common_events=common,
        first_missing=first_missing,
        second_missing=second_missing,
        both_missing=both_missing,
    )


def normalized_difference(values_a: np.ndarray, values_b: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return D_AB and the two sample means for the common trigger sample."""

    mean_a = float(np.mean(values_a))
    mean_b = float(np.mean(values_b))
    if mean_a <= 0 or mean_b <= 0:
        raise ValueError("The normalized difference requires positive channel means.")
    return (values_a / mean_a - values_b / mean_b) / SQRT2, mean_a, mean_b


def beta_asymmetry(values_a: np.ndarray, values_b: np.ndarray) -> np.ndarray:
    """Return the unnormalised beta asymmetry as a diagnostic cross-check."""

    denominator = values_a + values_b
    valid = np.isfinite(denominator) & (denominator != 0)
    return SQRT2 * (values_a[valid] - values_b[valid]) / denominator[valid]


def central68_half_width(values: np.ndarray) -> float:
    """Return half the width between the 16th and 84th percentiles."""

    low, high = np.percentile(values, [16.0, 84.0])
    return float((high - low) / 2.0)


def empirical_width(values: np.ndarray) -> EmpiricalWidth:
    """Return empirical distribution diagnostics."""

    finite = np.asarray(values, dtype=float)[np.isfinite(values)]
    if len(finite) < 2:
        return EmpiricalWidth(0, float("nan"), float("nan"), float("nan"))
    return EmpiricalWidth(
        n_events=len(finite),
        mean=float(np.mean(finite)),
        standard_deviation=float(np.std(finite, ddof=1)),
        central68_half_width=central68_half_width(finite),
    )


def gaussian_count_model(x: np.ndarray, amplitude: float, mean: float, sigma: float) -> np.ndarray:
    """Gaussian model in histogram-count units."""

    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def fit_gaussian_core(values: np.ndarray, core_width_multiplier: float = 2.0) -> GaussianFit:
    """Fit a Gaussian to a fixed robust central core of D_AB.

    The initial core is centered at the sample median and spans plus/minus two
    empirical central-68% half-widths.  This prescription is fixed for every
    pair and momentum, so the fit range is not tuned visually.  Histogram-bin
    Poisson uncertainties are used only for the fit; the plotted binning uses
    the same bin width.
    """

    finite = np.asarray(values, dtype=float)[np.isfinite(values)]
    if len(finite) < 30:
        return GaussianFit.failed("Fewer than 30 finite D_AB values.")
    center = float(np.median(finite))
    robust_width = central68_half_width(finite)
    if not np.isfinite(robust_width) or robust_width <= 0:
        return GaussianFit.failed("The empirical central-68% width is not positive.")
    fit_low = center - core_width_multiplier * robust_width
    fit_high = center + core_width_multiplier * robust_width
    core = finite[(finite >= fit_low) & (finite <= fit_high)]
    minimum_core_events = max(30, int(math.ceil(0.25 * len(finite))))
    if len(core) < minimum_core_events:
        return GaussianFit.failed(
            f"Only {len(core)} values are in the fixed Gaussian core.", fit_low, fit_high
        )

    bin_count = int(np.clip(round(1.1 * math.sqrt(len(core))), 20, 60))
    edges = np.linspace(fit_low, fit_high, bin_count + 1)
    counts, _ = np.histogram(core, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = float(edges[1] - edges[0])
    errors = np.sqrt(np.maximum(counts, 1.0))
    initial_sigma = max(float(np.std(core, ddof=1)), 0.5 * bin_width)
    initial = (float(np.max(counts)), float(np.mean(core)), initial_sigma)
    lower_bounds = (0.0, fit_low, 0.25 * bin_width)
    upper_bounds = (np.inf, fit_high, fit_high - fit_low)
    try:
        parameters, covariance = curve_fit(
            gaussian_count_model,
            centers,
            counts,
            p0=initial,
            sigma=errors,
            absolute_sigma=True,
            bounds=(lower_bounds, upper_bounds),
            method="trf",
            maxfev=20_000,
        )
    except (RuntimeError, ValueError, FloatingPointError) as error:
        return GaussianFit.failed(f"Gaussian core fit failed: {error}", fit_low, fit_high)

    amplitude, mean, sigma = (float(value) for value in parameters)
    if not np.isfinite(sigma) or sigma <= 0:
        return GaussianFit.failed("Gaussian core fit returned a non-positive sigma.", fit_low, fit_high)
    try:
        parameter_errors = np.sqrt(np.diag(covariance))
    except (TypeError, ValueError, FloatingPointError):
        parameter_errors = np.full(3, np.nan)
    if not np.all(np.isfinite(parameter_errors)):
        parameter_errors = np.asarray([
            float("nan"),
            float("nan"),
            sigma / math.sqrt(2.0 * max(len(core) - 1, 1)),
        ])
        message = "Gaussian core fit converged; sigma error uses the normal-sample approximation."
    else:
        message = ""
    prediction = gaussian_count_model(centers, amplitude, mean, sigma)
    chi2 = float(np.sum(((counts - prediction) / errors) ** 2))
    return GaussianFit(
        status="success",
        message=message,
        fit_events=len(core),
        amplitude=amplitude,
        amplitude_error=float(parameter_errors[0]),
        mean=mean,
        mean_error=float(parameter_errors[1]),
        sigma=sigma,
        sigma_error=float(parameter_errors[2]),
        fit_low=fit_low,
        fit_high=fit_high,
        bin_width=bin_width,
        chi2=chi2,
        ndf=max(len(counts) - 3, 0),
    )


def measure_pair(
    pair: Pair,
    momentum_gev: float,
    pair_events: PairEvents,
    min_events: int = 150,
) -> PairMeasurement:
    """Measure one pair with no hard cut on the common-event fraction."""

    if pair_events.common_events < min_events:
        raise ValueError(
            f"{pair.identifier} at {momentum_gev:g} GeV/c has only "
            f"{pair_events.common_events} common selected triggers (< {min_events})."
        )
    d_values, mean_a, mean_b = normalized_difference(pair_events.values_a, pair_events.values_b)
    n_eff = 2.0 * mean_a * mean_b / (mean_a + mean_b)
    correlation = float(np.corrcoef(pair_events.values_a, pair_events.values_b)[0, 1])
    return PairMeasurement(
        pair=pair.identifier,
        kind=pair.kind,
        momentum_gev=momentum_gev,
        selected_triggers=pair_events.selected_triggers,
        common_events=pair_events.common_events,
        common_event_fraction=pair_events.common_fraction,
        first_missing=pair_events.first_missing,
        second_missing=pair_events.second_missing,
        both_missing=pair_events.both_missing,
        mean_a=mean_a,
        mean_b=mean_b,
        mean_ratio_a_over_b=mean_a / mean_b,
        n_eff=n_eff,
        pearson_correlation=correlation,
        empirical_d=empirical_width(d_values),
        empirical_beta=empirical_width(beta_asymmetry(pair_events.values_a, pair_events.values_b)),
        gaussian_d=fit_gaussian_core(d_values),
    )


def poisson_width(n_eff: np.ndarray | float) -> np.ndarray:
    """Independent-photoelectron reference for the D_AB Gaussian sigma."""

    return 1.0 / np.sqrt(np.asarray(n_eff, dtype=float))


def load_merged_json(energy: int, energy_directory: str | Path) -> dict[str, Any]:
    """Read final per-range JSON files without making auxiliary input files."""

    directory = Path(energy_directory)
    blocks: dict[str, Any] = {}
    for path in sorted(directory.glob(f"*/photoelectron_dic_{energy}GeV.json")):
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        block = path.parent.name
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path} does not contain a JSON object.")
        blocks[block] = payload
    if not blocks:
        raise FileNotFoundError(f"No photoelectron JSON files found in {directory}.")
    return blocks
