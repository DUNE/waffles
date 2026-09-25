r"""Differential light-response study for ProtoDUNE-HD PDS channel pairs.

For two channels measured on the same selected triggers, the primary
observable is

.. math::

   D_{AB} = \frac{1}{\sqrt{2}}\left(
   \frac{N_{\rm PE}^A}{\mu_A} - \frac{N_{\rm PE}^B}{\mu_B}\right).

Its width is a differential relative light-response width.  It estimates the
relative resolution of one channel only if the channels have comparable
residual resolutions and their residual fluctuations are uncorrelated.

Missing channels are never replaced by zero.  The analysis keeps their counts
and only measures D_AB on the common available sample.
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
    """A candidate geometrically adjacent channel pair."""

    first: Channel
    second: Channel
    label: str = ""
    kind: str = "adjacent_candidate"

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
class WidthEstimate:
    """Width estimators and their trigger-bootstrap uncertainties."""

    n_events: int
    mean: float
    standard_deviation: float
    central68_half_width: float
    bootstrap_standard_deviation: float
    bootstrap_central68_half_width: float


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
    width_d: WidthEstimate
    width_beta: WidthEstimate

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
        for name, estimate in (("d", self.width_d), ("beta", self.width_beta)):
            for field, value in asdict(estimate).items():
                row[f"{name}_{field}"] = value
        return row


def selected_event_indices(
    merged_data: Mapping[str, Any],
    trigger_contexts: Mapping[tuple[str, int], Mapping[str, Any]],
    momentum_gev: int,
    apa: int,
    threshold: float | None,
) -> dict[str, np.ndarray]:
    """Select JSON indices using the current channel-linearity definition.

    At 2--7 GeV/c the tag is always defined by APA 1.  At 1 GeV/c no
    population threshold exists, and the local APA-valid flag is used.
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
    """Return D_AB, mu_A and mu_B for the common event sample."""

    mean_a = float(np.mean(values_a))
    mean_b = float(np.mean(values_b))
    if mean_a <= 0 or mean_b <= 0:
        raise ValueError("The normalized difference requires positive channel means.")
    return (values_a / mean_a - values_b / mean_b) / SQRT2, mean_a, mean_b


def beta_asymmetry(values_a: np.ndarray, values_b: np.ndarray) -> np.ndarray:
    """Return the uncorrected beta observable as a diagnostic cross-check."""

    denominator = values_a + values_b
    valid = np.isfinite(denominator) & (denominator != 0)
    return SQRT2 * (values_a[valid] - values_b[valid]) / denominator[valid]


def central68_half_width(values: np.ndarray) -> float:
    """Half-width of the central 68% interval."""

    low, high = np.percentile(values, [16.0, 84.0])
    return float((high - low) / 2.0)


def _widths(values: np.ndarray) -> tuple[float, float, float]:
    if len(values) < 2:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values, ddof=1)), central68_half_width(values)


def bootstrap_widths(
    values_a: np.ndarray,
    values_b: np.ndarray,
    observable: str,
    repetitions: int = 500,
    seed: int | None = 12345,
    batch_size: int = 128,
) -> tuple[float, float]:
    """Bootstrap widths by resampling complete A--B trigger pairs.

    The calculation is vectorized in batches.  The two means defining D_AB are
    recalculated in every resample.
    """

    if len(values_a) != len(values_b) or len(values_a) < 2 or repetitions < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    deviations: list[np.ndarray] = []
    central_widths: list[np.ndarray] = []
    for first in range(0, repetitions, batch_size):
        count = min(batch_size, repetitions - first)
        indices = rng.integers(0, len(values_a), size=(count, len(values_a)))
        sample_a = values_a[indices]
        sample_b = values_b[indices]
        if observable == "d":
            mean_a = np.mean(sample_a, axis=1, keepdims=True)
            mean_b = np.mean(sample_b, axis=1, keepdims=True)
            sample = (sample_a / mean_a - sample_b / mean_b) / SQRT2
        elif observable == "beta":
            denominator = sample_a + sample_b
            sample = np.full_like(denominator, np.nan, dtype=float)
            np.divide(
                SQRT2 * (sample_a - sample_b), denominator, out=sample,
                where=denominator != 0,
            )
        else:
            raise ValueError(f"Unknown observable: {observable}")
        deviations.append(np.nanstd(sample, axis=1, ddof=1))
        interval = np.nanpercentile(sample, [16.0, 84.0], axis=1)
        central_widths.append(0.5 * (interval[1] - interval[0]))
    return (
        float(np.nanstd(np.concatenate(deviations), ddof=1)),
        float(np.nanstd(np.concatenate(central_widths), ddof=1)),
    )


def estimate_width(
    values_a: np.ndarray,
    values_b: np.ndarray,
    observable: str,
    bootstrap_repetitions: int = 500,
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
    pair: Pair,
    momentum_gev: float,
    pair_events: PairEvents,
    min_events: int = 150,
    minimum_common_fraction: float = 0.90,
    bootstrap_repetitions: int = 500,
    seed: int | None = 12345,
) -> PairMeasurement:
    """Measure D_AB and beta after common-sample quality checks."""

    if pair_events.common_events < min_events:
        raise ValueError(
            f"{pair.identifier} at {momentum_gev:g} GeV/c has only "
            f"{pair_events.common_events} common selected triggers (< {min_events})."
        )
    if not np.isfinite(pair_events.common_fraction) or pair_events.common_fraction < minimum_common_fraction:
        raise ValueError(
            f"{pair.identifier} at {momentum_gev:g} GeV/c has common-event fraction "
            f"{pair_events.common_fraction:.3f} (< {minimum_common_fraction:.3f})."
        )
    _, mean_a, mean_b = normalized_difference(pair_events.values_a, pair_events.values_b)
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
        width_d=estimate_width(pair_events.values_a, pair_events.values_b, "d", bootstrap_repetitions, seed),
        width_beta=estimate_width(pair_events.values_a, pair_events.values_b, "beta", bootstrap_repetitions, seed),
    )


def poisson_width(n_eff: np.ndarray | float) -> np.ndarray:
    """Independent-photoelectron prediction for the D_AB width."""

    return 1.0 / np.sqrt(np.asarray(n_eff, dtype=float))


def _watermark(axis) -> None:
    axis.text(
        0.98, 0.97, r"$\mathbf{ProtoDUNE\!-\!HD}$" + "\nWork in Progress",
        transform=axis.transAxes, ha="right", va="top", fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
    )


def plot_pair_diagnostics(pair_events: PairEvents, pair: Pair, momentum_gev: float, output: str | Path) -> None:
    """Save scatter and D_AB-distribution diagnostics for one pair."""

    import matplotlib.pyplot as plt

    d_values, mean_a, mean_b = normalized_difference(pair_events.values_a, pair_events.values_b)
    beta_values = beta_asymmetry(pair_events.values_a, pair_events.values_b)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    axes[0].scatter(pair_events.values_a, pair_events.values_b, s=8, alpha=0.35, color="#0072B2", edgecolors="none")
    maximum = max(float(np.max(pair_events.values_a)), float(np.max(pair_events.values_b)))
    reference = np.array([0.0, maximum])
    axes[0].plot(reference, reference * mean_b / mean_a, "--", color="#D55E00", lw=1.5, label=r"$N_B=(\mu_B/\mu_A)N_A$")
    axes[0].set(xlabel=rf"$N_{{\rm PE}}$ [{pair.first.label}]", ylabel=rf"$N_{{\rm PE}}$ [{pair.second.label}]")
    axes[0].legend(frameon=True, facecolor="white", fontsize=8)
    axes[0].grid(alpha=0.28)
    _watermark(axes[0])

    axes[1].hist(d_values, bins="fd", density=True, color="#0072B2", alpha=0.55, label=r"$D_{AB}$")
    axes[1].hist(beta_values, bins="fd", density=True, histtype="step", lw=1.4, color="#D55E00", label=r"$\beta$ (cross-check)")
    low, high = np.percentile(d_values, [16.0, 84.0])
    axes[1].axvspan(low, high, color="#0072B2", alpha=0.12, label="central 68% interval")
    axes[1].axvline(0.0, color="#333333", lw=1.0)
    axes[1].set(xlabel=r"Normalized differential response $D_{AB}$", ylabel="Density")
    axes[1].legend(frameon=True, facecolor="white", fontsize=8)
    axes[1].grid(alpha=0.28)
    _watermark(axes[1])
    figure.suptitle(f"{pair.identifier} at {momentum_gev:g} GeV/c", fontsize=12)
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def load_merged_json(energy: int, energy_directory: str | Path) -> dict[str, Any]:
    """Read final per-range JSON files without making auxiliary input files."""

    directory = Path(energy_directory)
    merged: dict[str, Any] = {}
    for subdirectory in sorted(directory.glob("*_to_*")):
        json_file = subdirectory / f"photoelectron_dic_{energy}GeV.json"
        if json_file.exists():
            with json_file.open(encoding="utf-8") as stream:
                payload = json.load(stream)
            if "trigger time" not in payload or "1" not in payload or "2" not in payload:
                raise ValueError(f"Invalid JSON structure in {json_file}")
            trigger_count = len(payload["trigger time"])
            for apa in ("1", "2"):
                channel_dictionaries = payload[apa].get("channel_dic")
                if not isinstance(channel_dictionaries, list) or len(channel_dictionaries) != trigger_count:
                    raise ValueError(f"Invalid channel_dic length for APA {apa} in {json_file}")
            merged[subdirectory.name] = payload
    if not merged:
        raise FileNotFoundError(f"No photoelectron JSON files found in {directory}")
    return merged


if __name__ == "__main__":
    print("Import pds_differential_resolution from run_pds_differential_resolution.py.")
