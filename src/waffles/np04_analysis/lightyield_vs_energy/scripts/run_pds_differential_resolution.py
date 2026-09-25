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
MOMENTUM_COLORS = {
    1: "#D55E00",
    2: "#0072B2",
    3: "#009E73",
    5: "#CC79A7",
    7: "#56B4E9",
}
USABLE_COLOR = "#0072B2"
LOW_COVERAGE_COLOR = "#D55E00"


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


def watermark(figure: plt.Figure) -> None:
    figure.text(
        0.98, 0.985,
        r"$\mathbf{ProtoDUNE\!-!HD}$" + "\nWork in Progress",
        ha="right", va="top", fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5},
    )


def display_limits(values: np.ndarray, fit_low: float, fit_high: float) -> tuple[float, float]:
    """Choose a stable central display window without changing the fit sample."""

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


def plot_distribution_panel(axis: plt.Axes, panel: dict, momentum: int) -> None:
    """Draw one D_AB distribution and its Gaussian core fit."""

    pair: Pair = panel["pair"]
    record = panel.get("record")
    values = panel.get("d_values")
    axis.set_title(rf"$p_{{\rm beam}} = {momentum:g}$ GeV/c", fontsize=10)
    axis.grid(alpha=0.25)
    if values is None or len(values) == 0:
        axis.text(0.5, 0.5, panel["message"], transform=axis.transAxes, ha="center", va="center", fontsize=9, wrap=True)
        axis.set(xlabel=r"$D_{AB}$", ylabel="Trigger counts")
        return

    if record is None:
        axis.hist(values, bins="fd", color="#BDBDBD", alpha=0.75, edgecolor="#333333", label=f"Data ({len(values)} triggers)")
        axis.text(0.97, 0.93, panel["message"], transform=axis.transAxes, ha="right", va="top", fontsize=8, bbox={"facecolor": "white", "edgecolor": "#999999", "alpha": 0.92})
        axis.set(xlabel=r"$D_{AB}$", ylabel="Trigger counts")
        axis.legend(frameon=True, facecolor="white", fontsize=8)
        return

    fit_low = record["d_gaussian_fit_low"]
    fit_high = record["d_gaussian_fit_high"]
    bin_width = record["d_gaussian_bin_width"]
    low, high = display_limits(values, fit_low, fit_high)
    bin_edges = np.arange(low, high + bin_width, bin_width)
    if len(bin_edges) < 2:
        bin_edges = 30
    color = USABLE_COLOR if record["coverage_status"] == "usable" else LOW_COVERAGE_COLOR
    axis.hist(
        values,
        bins=bin_edges,
        color="#D0D0D0",
        edgecolor="#333333",
        lw=0.7,
        label=f"Data @ {momentum:g} GeV/c ({int(record['common_events'])} triggers)",
    )
    if record["d_gaussian_status"] == "success":
        x_fit = np.linspace(fit_low, fit_high, 500)
        axis.plot(
            x_fit,
            gaussian_count_model(
                x_fit,
                record["d_gaussian_amplitude"],
                record["d_gaussian_mean"],
                record["d_gaussian_sigma"],
            ),
            color=color,
            lw=2.0,
            label="Gaussian core fit",
        )
        text = (
            "Gaussian fit\n"
            rf"$\mu = ({record['d_gaussian_mean']:.3f} \pm {record['d_gaussian_mean_error']:.3f})$\n"
            rf"$\sigma = ({record['d_gaussian_sigma']:.3f} \pm {record['d_gaussian_sigma_error']:.3f})$\n"
            rf"common fraction = {record['common_event_fraction']:.3f}"
        )
    else:
        text = "Gaussian fit failed\n" + str(record["d_gaussian_message"])
    if record["coverage_status"] == "low_coverage":
        text += "\nlow coverage: excluded from trend"
    axis.text(
        0.97, 0.93, text, transform=axis.transAxes, ha="right", va="top", fontsize=7.4,
        bbox={"facecolor": "white", "edgecolor": "#999999", "alpha": 0.93, "pad": 2.0},
    )
    axis.set(xlim=(low, high), xlabel=r"$D_{AB}$", ylabel="Trigger counts")
    axis.legend(frameon=True, facecolor="white", fontsize=7.3, loc="upper left")


def plot_pair_sigma_panel(axis: plt.Axes, panels: dict[int, dict]) -> None:
    """Draw Gaussian sigma against K_eff for one pair."""

    usable = []
    low_coverage = []
    for momentum in MOMENTA:
        record = panels.get(momentum, {}).get("record")
        if record is None or record["d_gaussian_status"] != "success":
            continue
        (usable if record["coverage_status"] == "usable" else low_coverage).append(record)

    if not usable and not low_coverage:
        axis.text(0.5, 0.5, "No successful Gaussian fit", transform=axis.transAxes, ha="center", va="center")
        axis.set(xlabel=r"$K_{\rm eff}$ [GeV]", ylabel=r"Gaussian $\sigma_D$")
        return

    def draw(records: list[dict], color: str, marker: str, label: str, filled: bool) -> None:
        if not records:
            return
        records.sort(key=lambda item: item["kinetic_mean_GeV"])
        x = np.asarray([item["kinetic_mean_GeV"] for item in records])
        xerr = np.asarray([item["effective_spread_GeV"] for item in records])
        y = np.asarray([item["d_gaussian_sigma"] for item in records])
        yerr = np.asarray([item["d_gaussian_sigma_error"] for item in records])
        axis.errorbar(
            x, y, xerr=xerr, yerr=yerr, fmt=marker, color=color, ms=6.5, capsize=2.5,
            mfc=color if filled else "white", mew=1.4, label=label,
        )

    draw(usable, USABLE_COLOR, "o", "Gaussian core fit", True)
    draw(low_coverage, LOW_COVERAGE_COLOR, "o", "Low coverage", False)
    reference = sorted(usable + low_coverage, key=lambda item: item["kinetic_mean_GeV"])
    x = np.asarray([item["kinetic_mean_GeV"] for item in reference])
    poisson = poisson_width(np.asarray([item["n_eff_PE"] for item in reference]))
    axis.plot(x, poisson, "--s", color="#4D4D4D", ms=4.3, mfc="white", label=r"Independent Poisson: $1/\sqrt{N_{\rm eff}}$")
    axis.set(xlabel=r"$K_{\rm eff}$ [GeV]", ylabel=r"Gaussian $\sigma_D$")
    axis.grid(alpha=0.25)
    axis.legend(frameon=True, facecolor="white", fontsize=7.2, loc="best")


def make_pair_pdf(
    pairs: list[Pair],
    panels_by_pair: dict[str, dict[int, dict]],
    pair_summary: list[dict],
    minimum_momenta_for_pdf: int,
    output: Path,
) -> int:
    """Write one six-panel PDF page for every pair with enough usable points."""

    summaries = {row["pair"]: row for row in pair_summary}
    pages = 0
    with PdfPages(output) as pdf:
        for pair in pairs:
            summary = summaries[pair.identifier]
            if int(summary["usable_momenta"]) < minimum_momenta_for_pdf:
                continue
            figure, axes = plt.subplots(3, 2, figsize=(13.0, 15.5))
            figure.suptitle(
                f"APA {pair.first.apa}: END {pair.first.endpoint} - CH {pair.first.channel}  and  "
                f"END {pair.second.endpoint} - CH {pair.second.channel}",
                x=0.04, y=0.985, ha="left", fontsize=15,
            )
            watermark(figure)
            for axis, momentum in zip(axes.flat[:5], MOMENTA):
                plot_distribution_panel(axis, panels_by_pair[pair.identifier][momentum], momentum)
            plot_pair_sigma_panel(axes.flat[5], panels_by_pair[pair.identifier])
            figure.tight_layout(rect=(0.02, 0.02, 0.98, 0.955))
            pdf.savefig(figure)
            plt.close(figure)
            pages += 1
    return pages


def make_summary_plot(table: pd.DataFrame, output: Path) -> None:
    """Plot fitted Gaussian sigma against the pair effective PE scale."""

    valid = table.loc[
        (table["measurement_status"] == "success")
        & (table["coverage_status"] == "usable")
        & (table["d_gaussian_status"] == "success")
    ]
    if valid.empty:
        return
    figure, axis = plt.subplots(figsize=(8.0, 5.8))
    for momentum, group in valid.groupby("momentum_GeV_c"):
        axis.errorbar(
            group["n_eff_PE"], group["d_gaussian_sigma"],
            yerr=group["d_gaussian_sigma_error"], fmt="o", ms=5.8, capsize=2.5,
            color=MOMENTUM_COLORS[int(momentum)], label=rf"{momentum:g} GeV/c",
        )
    low = max(float(valid["n_eff_PE"].min()) * 0.85, 1.0e-3)
    high = float(valid["n_eff_PE"].max()) * 1.20
    x_values = np.geomspace(low, high, 300)
    axis.plot(x_values, poisson_width(x_values), "--", color="#333333", lw=1.8, label=r"Independent Poisson: $1/\sqrt{N_{\rm eff}}$")
    axis.set(
        xscale="log", yscale="log", xlabel=r"$N_{\rm eff}$ [PE]",
        ylabel=r"Gaussian core-fit $\sigma_D$",
    )
    axis.grid(which="both", alpha=0.28)
    axis.legend(frameon=True, facecolor="white", fontsize=8)
    watermark(figure)
    figure.tight_layout()
    figure.savefig(output, dpi=300)
    plt.close(figure)


def build_pair_summary(pairs: list[Pair], table: pd.DataFrame, min_momenta: int) -> list[dict]:
    rows: list[dict] = []
    for pair in pairs:
        group = table.loc[table["pair"] == pair.identifier]
        fit_success = group.loc[group["d_gaussian_status"] == "success"]
        usable = fit_success.loc[fit_success["coverage_status"] == "usable"]
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
            "usable_momenta": len(usable),
            "usable_momentum_values_GeV_c": ";".join(str(int(value)) for value in sorted(usable["momentum_GeV_c"].unique())),
            "included_in_pair_pdf": int(len(usable) >= min_momenta),
        })
    return rows


def clear_outputs(output_dir: Path) -> None:
    for name in (
        "adjacent_pair_differential_resolution.csv", "pair_availability.csv",
        "pair_configuration.csv", "selection_thresholds.csv", "selected_trigger_counts.csv",
        "pair_analysis_summary.csv", "differential_gaussian_sigma_vs_neff.png",
        "apa1_adjacent_pair_gaussian_resolution.pdf", "apa2_adjacent_pair_gaussian_resolution.pdf",
        "report.txt", "manifest.json", "pair_threshold_systematics.csv",
        "differential_width_vs_neff.png",
    ):
        path = output_dir / name
        if path.is_file():
            path.unlink()
    diagnostics = output_dir / "diagnostics"
    if diagnostics.is_dir():
        for path in diagnostics.glob("*.png"):
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
    pair_summary = build_pair_summary(pairs, table, arguments.minimum_momenta_for_pdf)
    pdf_path = arguments.output_dir / f"apa{arguments.apa}_adjacent_pair_gaussian_resolution.pdf"
    page_count = make_pair_pdf(
        pairs, panels_by_pair, pair_summary, arguments.minimum_momenta_for_pdf, pdf_path,
    )

    write_csv(arguments.output_dir / "adjacent_pair_differential_resolution.csv", measurements)
    write_csv(arguments.output_dir / "pair_availability.csv", availability_rows)
    write_csv(arguments.output_dir / "selection_thresholds.csv", threshold_rows)
    write_csv(arguments.output_dir / "selected_trigger_counts.csv", selected_rows)
    write_csv(arguments.output_dir / "pair_analysis_summary.csv", pair_summary)
    write_csv(arguments.output_dir / "pair_configuration.csv", [
        {
            "pair": pair.identifier, "kind": pair.kind, "apa": pair.first.apa,
            "first_endpoint": pair.first.endpoint, "first_channel": pair.first.channel,
            "second_endpoint": pair.second.endpoint, "second_channel": pair.second.channel,
        }
        for pair in pairs
    ])
    make_summary_plot(table, arguments.output_dir / "differential_gaussian_sigma_vs_neff.png")

    usable_count = int(np.count_nonzero(table["analysis_status"] == "usable"))
    report = [
        "PDS ADJACENT-CHANNEL DIFFERENTIAL RESPONSE STUDY",
        f"APA: {arguments.apa}",
        f"Momenta [GeV/c]: {list(MOMENTA)}",
        f"Minimum common events: {arguments.minimum_events}",
        f"Low-coverage threshold: {arguments.low_coverage_threshold:.3f}",
        f"Minimum usable momenta per PDF pair: {arguments.minimum_momenta_for_pdf}",
        "Threshold scenario: nominal only.",
        "",
        "SELECTION",
        "At 1 GeV/c: APA-local-valid triggers; no muon-selection threshold.",
        "At 2--7 GeV/c: APA1-valid triggers with APA1 mean above the nominal Langauss--Gaussian intersection.",
        "Missing channel values are never replaced by zero.",
        "",
        "PRIMARY RESULT",
        "D_AB is fit with a Gaussian in a fixed robust central core: median +/- 2 times the empirical central-68% half-width.",
        "The fitted Gaussian sigma is the primary differential-response width.",
        "The central-68% half-width and beta observable are retained as cross-checks in the CSV.",
        "This is not an absolute calorimetric energy-resolution measurement.",
        "",
        "QUALITY",
        f"Successful pair/momentum measurements: {len(table)}.",
        f"Usable Gaussian-fit pair/momentum measurements: {usable_count}.",
        f"Pairs included in the PDF: {page_count}.",
        "",
        "OUTPUTS",
        "adjacent_pair_differential_resolution.csv: all measured pairs, Gaussian fit parameters, and empirical cross-checks.",
        "pair_availability.csv: availability and missing-value counts for every pair and momentum.",
        "pair_analysis_summary.csv: number of usable momenta and PDF inclusion per pair.",
        "selected_trigger_counts.csv: selected-trigger count per momentum.",
        f"{pdf_path.name}: five D_AB distributions and sigma versus K_eff for each eligible pair.",
        "differential_gaussian_sigma_vs_neff.png: all usable Gaussian sigma values compared with independent Poisson counting.",
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
