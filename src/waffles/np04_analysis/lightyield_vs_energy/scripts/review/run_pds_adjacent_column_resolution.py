#!/usr/bin/env python3
r"""Study the differential PDS response of adjacent APA 1 columns.

Columns 1--2, 2--3, and 3--4 use the geometry in scripts/utils.py.  A column
value is the sum of one fixed set of channels on one selected trigger.  A
trigger enters a column pair only when every required channel in both columns
has a finite value; missing entries are never interpreted as zero.  The
trigger selection, central Gaussian fit, K_eff values, and two-term resolution
fit match run_pds_differential_resolution.py.

The reference sqrt((1/mu_A + 1/mu_B)/2) is the ideal independent-photoelectron
counting width for the normalized difference.  It is a diagnostic, not a
complete prediction of the measured differential width or an absolute energy
resolution.  No electronic-noise term is inferred without an independent
integrated-noise measurement.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

SCRIPTS_DIRECTORY = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from pds_differential_resolution import (
    Channel, Pair, PairEvents, load_merged_json, measure_pair,
    normalized_difference, poisson_width, selected_event_indices,
)
from run_pds_differential_resolution import (
    DATA_COLOR, MOMENTA, ResolutionFit, add_panel_brand, fit_resolution,
    format_panel_axis, load_kinetic_energies, load_nominal_thresholds,
    load_trigger_contexts, plot_correlation_panel, plot_distribution_panel,
    resolution_model, scenario_name,
)
from utils import apa1_columns_channels


IDEAL_COLOR = "#D55E00"
EXCLUDED_COLOR = "#CC79A7"
FIT_COLOR = "#17365D"
COLUMN_PAIRS = ((1, 2), (2, 3), (3, 4))


def channel_value(event: Mapping[str, Any], channel: Channel) -> float | None:
    data = event.get(str(channel.endpoint), {}).get(str(channel.channel))
    if not isinstance(data, Mapping):
        return None
    try:
        value = float(data["n_pe"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def event_at(payload: Mapping[str, Any], block: str, index: int) -> Mapping[str, Any] | None:
    events = payload.get(block, {}).get("1", {}).get("channel_dic", [])
    if 0 <= index < len(events) and isinstance(events[index], Mapping):
        return events[index]
    return None


def column_sum(event: Mapping[str, Any] | None, channels: list[Channel]) -> float | None:
    if event is None:
        return None
    values = [channel_value(event, channel) for channel in channels]
    return float(sum(values)) if all(value is not None for value in values) else None


def selected_channel_coverage(
    data_by_momentum: dict[int, dict],
    selected_by_momentum: dict[int, dict],
    full_columns: dict[int, list[Channel]],
    channels_per_column: int,
) -> tuple[dict[int, list[Channel]], list[dict], list[dict]]:
    """Select one fixed channel set per column, using nominal availability.

    With eight channels (the default), all physical-column channels are kept.
    A smaller requested subset is ranked by the mean of the five per-momentum
    availability fractions, giving each beam setting equal weight.  The same
    selected channels are then used at every momentum and threshold scenario.
    """

    coverage_rows: list[dict] = []
    scores: dict[tuple[int, int, int], list[float]] = {}
    for momentum in MOMENTA:
        selected = selected_by_momentum[momentum]
        total = sum(len(indices) for indices in selected.values())
        for column, channels in full_columns.items():
            counts = {channel: 0 for channel in channels}
            for block, indices in selected.items():
                for index in indices:
                    event = event_at(data_by_momentum[momentum], block, int(index))
                    if event is None:
                        continue
                    for channel in channels:
                        if channel_value(event, channel) is not None:
                            counts[channel] += 1
            for channel in channels:
                fraction = counts[channel] / total if total else 0.0
                scores.setdefault((column, channel.endpoint, channel.channel), []).append(fraction)
                coverage_rows.append({
                    "column": column, "endpoint": channel.endpoint,
                    "channel": channel.channel, "momentum_GeV_c": momentum,
                    "selected_triggers": total, "valid_channel_triggers": counts[channel],
                    "valid_fraction": fraction,
                })

    chosen: dict[int, list[Channel]] = {}
    configuration: list[dict] = []
    for column, channels in full_columns.items():
        ranked = sorted(
            channels,
            key=lambda ch: -float(np.mean(scores[(column, ch.endpoint, ch.channel)])),
        )
        selected_set = set(ranked[:channels_per_column])
        chosen[column] = [channel for channel in channels if channel in selected_set]
        for channel in channels:
            configuration.append({
                "column": column, "endpoint": channel.endpoint,
                "channel": channel.channel,
                "included_in_sum": int(channel in selected_set),
                "mean_valid_fraction_over_momenta": float(np.mean(
                    scores[(column, channel.endpoint, channel.channel)]
                )),
            })
    return chosen, coverage_rows, configuration


def extract_column_pair_events(
    data: dict, selected: dict, first: list[Channel], second: list[Channel],
) -> PairEvents:
    values_a: list[float] = []
    values_b: list[float] = []
    selected_triggers = first_missing = second_missing = both_missing = 0
    for block, indices in selected.items():
        for index in indices:
            selected_triggers += 1
            event = event_at(data, block, int(index))
            sum_a = column_sum(event, first)
            sum_b = column_sum(event, second)
            if sum_a is None and sum_b is None:
                both_missing += 1
            elif sum_a is None:
                first_missing += 1
            elif sum_b is None:
                second_missing += 1
            else:
                values_a.append(sum_a)
                values_b.append(sum_b)
    return PairEvents(
        values_a=np.asarray(values_a, dtype=float),
        values_b=np.asarray(values_b, dtype=float),
        selected_triggers=selected_triggers,
        common_events=len(values_a),
        first_missing=first_missing,
        second_missing=second_missing,
        both_missing=both_missing,
    )


def save_csv(path: Path, rows: list[dict], empty_columns: list[str] | None = None) -> None:
    if rows:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
    else:
        pd.DataFrame(columns=empty_columns or []).to_csv(path, index=False)


def make_column_fit_panel(axis: plt.Axes, panels: dict[int, dict], fits: dict[str, ResolutionFit]) -> None:
    records = [
        panels[momentum]["record"] for momentum in MOMENTA
        if panels[momentum].get("record") is not None
        and panels[momentum]["record"]["d_gaussian_status"] == "success"
    ]
    add_panel_brand(axis, "upper_center")
    axis.grid(alpha=0.22)
    axis.set(xlabel=r"$K_{\rm eff}$ [GeV]", ylabel=r"Gaussian $\sigma_D$ [AU]")
    if not records:
        axis.text(0.5, 0.5, "No successful Gaussian widths", transform=axis.transAxes,
                  ha="center", va="center", fontsize=9)
        format_panel_axis(axis, False)
        return

    records.sort(key=lambda row: row["kinetic_mean_GeV"])
    x = np.asarray([row["kinetic_mean_GeV"] for row in records], dtype=float)
    sx = np.asarray([row["effective_spread_GeV"] for row in records], dtype=float)
    y = np.asarray([row["d_gaussian_sigma"] for row in records], dtype=float)
    sy = np.asarray([row["d_gaussian_sigma_error"] for row in records], dtype=float)
    ideal = np.asarray([row["ideal_independent_PE_sigma_D"] for row in records], dtype=float)
    axis.errorbar(x, y, xerr=sx, yerr=sy, fmt="o", color=DATA_COLOR, ms=5,
                  capsize=2.1, label=r"Gaussian $\sigma_D$", zorder=4)
    axis.plot(x, ideal, "s--", color=IDEAL_COLOR, ms=3.7, lw=1.4,
              label="Independent PE reference", zorder=3)
    x_curve = np.linspace(max(0.05, float(min(x - sx))), float(max(x + sx)) * 1.03, 300)
    full = fits["all"]
    if full.status == "success":
        axis.plot(x_curve, resolution_model(x_curve, full.constant_a,
                  full.stochastic_b_sqrt_GeV), color=FIT_COLOR, lw=1.8,
                  label=(r"Two-term fit" + "\n"
                         + rf"$a=({full.constant_a:.3f}\pm{full.constant_a_error:.3f})$" + "\n"
                         + rf"$b=({full.stochastic_b_sqrt_GeV:.3f}\pm{full.stochastic_b_error_sqrt_GeV:.3f})\sqrt{{\rm GeV}}$" + "\n"
                         + rf"$\chi^2/{{\rm ndf}}={full.chi2_ndf:.2f},\ R^2={full.r_squared:.3f}$"))
    excluded = fits["exclude_1gev"]
    if excluded.status == "success":
        axis.plot(x_curve, resolution_model(x_curve, excluded.constant_a,
                  excluded.stochastic_b_sqrt_GeV), "--", color=EXCLUDED_COLOR, lw=1.5,
                  label="Fit excluding 1 GeV/c")
    axis.set_xlim(max(0.0, float(min(x - sx)) - 0.1), float(max(x + sx)) + 0.1)
    axis.set_ylim(0.0, max(float(max(y + sy)), float(max(ideal))) * 1.37)
    axis.legend(loc="upper right", fontsize=5.7, facecolor="white", frameon=True)
    format_panel_axis(axis, False)


def write_pair_pdf(output: Path, pair_defs: list[tuple[int, int, Pair]],
                   panels: dict[str, dict[int, dict]], fits: dict[str, dict[str, ResolutionFit]],
                   channels_per_column: int) -> int:
    pages = 0
    with PdfPages(output) as pdf:
        for first, second, pair in pair_defs:
            figure = plt.figure(figsize=(16.54, 11.69))
            grid = figure.add_gridspec(3, 4, wspace=0.35, hspace=0.43)
            subset_note = "" if channels_per_column == 8 else f" ({channels_per_column} fixed channels per column)"
            figure.suptitle(f"APA 1: adjacent columns {first} and {second}{subset_note}",
                            x=0.02, y=0.987, ha="left", fontsize=14)
            positions = {
                1: (grid[0, 0], grid[0, 1]),
                2: (grid[0, 2], grid[0, 3]),
                3: (grid[1, 0], grid[1, 1]),
                5: (grid[1, 2], grid[1, 3]),
                7: (grid[2, 0], grid[2, 1]),
            }
            for momentum, (scatter_slot, histogram_slot) in positions.items():
                scatter = figure.add_subplot(scatter_slot)
                plot_correlation_panel(scatter, panels[pair.identifier][momentum], momentum)
                scatter.set(xlabel=rf"$N_{{\rm PE}}^{{C_{{{first}}}}}$ [PE]",
                            ylabel=rf"$N_{{\rm PE}}^{{C_{{{second}}}}}$ [PE]")
                plot_distribution_panel(figure.add_subplot(histogram_slot),
                                        panels[pair.identifier][momentum], momentum)
            make_column_fit_panel(figure.add_subplot(grid[2, 2:4]),
                                  panels[pair.identifier], fits[pair.identifier])
            figure.tight_layout(rect=(0.005, 0.01, 0.995, 0.955))
            pdf.savefig(figure)
            plt.close(figure)
            pages += 1
    return pages


def write_summary_plots(output_dir: Path, pair_defs: list[tuple[int, int, Pair]],
                        panels: dict[str, dict[int, dict]],
                        fits: dict[str, dict[str, ResolutionFit]],
                        systematics: list[dict], channels_per_column: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True, sharey=True)
    for axis, (first, second, pair) in zip(axes, pair_defs):
        make_column_fit_panel(axis, panels[pair.identifier], fits[pair.identifier])
        subset_note = "" if channels_per_column == 8 else f" ({channels_per_column}/8 channels)"
        axis.set_title(f"Columns {first}–{second}{subset_note}")
    fig.tight_layout()
    fig.savefig(output_dir / "apa1_adjacent_column_sigma_D_vs_keff.png", dpi=240)
    plt.close(fig)

    sys_lookup = {row["pair"]: row for row in systematics}
    names = [f"{first}–{second}" for first, second, _ in pair_defs]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for axis, field, error, syst_field, label in (
        (axes[0], "constant_a", "constant_a_error", "constant_a_threshold_systematic",
         r"Constant parameter $a$ [AU]"),
        (axes[1], "stochastic_b_sqrt_GeV", "stochastic_b_error_sqrt_GeV",
         "stochastic_b_threshold_systematic", r"Stochastic parameter $b$ [AU $\sqrt{\rm GeV}$]"),
    ):
        for index, (_, _, pair) in enumerate(pair_defs):
            fit = fits[pair.identifier]["all"]
            if fit.status != "success":
                continue
            value = getattr(fit, field)
            axis.errorbar(index, value, yerr=getattr(fit, error), fmt="o", color=DATA_COLOR,
                          capsize=4, ms=7, label="Fit ± statistical" if index == 0 else None)
            systematic = sys_lookup[pair.identifier][syst_field]
            if math.isfinite(systematic) and systematic > 0:
                axis.errorbar(index + 0.10, value, yerr=systematic, fmt="none",
                              ecolor=IDEAL_COLOR, capsize=4,
                              label="Threshold variation" if index == 0 else None)
        axis.set_xticks(range(3), names)
        axis.set(xlabel="Adjacent APA 1 columns", ylabel=label)
        axis.grid(axis="y", alpha=0.23)
        add_panel_brand(axis, "right", standalone=True)
        axis.legend(loc="upper left", frameon=True, facecolor="white", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "apa1_adjacent_column_fit_parameters.png", dpi=240)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9.2, 5.8))
    colors = (DATA_COLOR, IDEAL_COLOR, "#009E73")
    for color, (first, second, pair) in zip(colors, pair_defs):
        rows = [panels[pair.identifier][momentum].get("record") for momentum in MOMENTA]
        rows = [row for row in rows if row is not None and row["d_gaussian_status"] == "success"]
        if not rows:
            continue
        x = [row["kinetic_mean_GeV"] for row in rows]
        ratios = [row["sigma_D_over_ideal_independent_PE"] for row in rows]
        axis.errorbar(x, ratios, xerr=[row["effective_spread_GeV"] for row in rows],
                      fmt="o-", color=color, capsize=3,
                      label=f"Columns {first}–{second}")
    axis.axhline(1.0, color="#444444", lw=1.4, ls="--", label="Independent PE reference")
    axis.set(xlabel=r"$K_{\rm eff}$ [GeV]",
             ylabel=r"Measured $\sigma_D$ / independent PE reference")
    axis.grid(alpha=0.22)
    add_panel_brand(axis, "right", standalone=True)
    axis.legend(loc="upper left", frameon=True, facecolor="white", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "apa1_adjacent_column_counting_comparison.png", dpi=240)
    plt.close(fig)


def parse_arguments() -> argparse.Namespace:
    analysis_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=analysis_dir / "output/apa1_vs_apa2")
    parser.add_argument("--trigger-data", type=Path,
                        default=analysis_dir / "output/review/apa12_trigger_data_01/apa12_trigger_data.csv")
    parser.add_argument("--population-fit-results", type=Path,
                        default=analysis_dir / "output/review/apa12_trigger_distributions_01/apa1_population_fit_results.csv")
    parser.add_argument("--composition", type=Path,
                        default=analysis_dir / "data/np04_beam_particle_content.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=analysis_dir / "output/review/pds_adjacent_column_resolution_01")
    parser.add_argument("--channels-per-column", type=int, default=8, choices=range(2, 9),
                        help="Fixed channels summed in every column; 8 uses each complete column.")
    parser.add_argument("--minimum-events", type=int, default=150)
    parser.add_argument("--relative-momentum-error", type=float, default=0.05)
    parser.add_argument("--threshold-sigma-multipliers", nargs="+", type=float,
                        default=[-1.0, 0.0, 1.0])
    args = parser.parse_args()
    for attribute in ("input_dir", "trigger_data", "population_fit_results", "composition", "output_dir"):
        setattr(args, attribute, getattr(args, attribute).expanduser().resolve())
    if args.minimum_events < 10:
        parser.error("--minimum-events must be at least 10")
    if args.relative_momentum_error < 0:
        parser.error("--relative-momentum-error must be non-negative")
    if not args.threshold_sigma_multipliers or len(set(args.threshold_sigma_multipliers)) != len(args.threshold_sigma_multipliers):
        parser.error("Threshold multipliers must be nonempty and distinct")
    if not any(math.isclose(value, 0.0) for value in args.threshold_sigma_multipliers):
        parser.error("Threshold multipliers must include 0 for the nominal selection")
    for path in (args.input_dir, args.trigger_data, args.population_fit_results, args.composition):
        if not path.exists():
            parser.error(f"Input does not exist: {path}")
    if args.output_dir == args.input_dir:
        parser.error("--output-dir must differ from --input-dir")
    return args


def main() -> int:
    args = parse_arguments()
    contexts = load_trigger_contexts(args.trigger_data)
    thresholds, _ = load_nominal_thresholds(args.population_fit_results)
    kinetic = load_kinetic_energies(args.composition, args.relative_momentum_error)
    data = {momentum: load_merged_json(momentum, args.input_dir / f"{momentum}GeV")
            for momentum in MOMENTA}
    raw_columns = apa1_columns_channels()
    full_columns = {
        column: [Channel.from_mapping(value) for value in raw_columns[column]]
        for column in range(1, 5)
    }
    if any(len(channels) != 8 for channels in full_columns.values()):
        raise ValueError("The APA 1 column map must contain eight channels per column.")
    flattened = [channel for channels in full_columns.values() for channel in channels]
    if len(set(flattened)) != 32 or any(channel.apa != 1 for channel in flattened):
        raise ValueError("The APA 1 column map has duplicate or non-APA-1 channels.")

    nominal_selected = {
        momentum: selected_event_indices(data[momentum], contexts[momentum], momentum, 1,
                                         None if momentum == 1 else thresholds[momentum][0])
        for momentum in MOMENTA
    }
    columns, coverage_rows, column_config = selected_channel_coverage(
        data, nominal_selected, full_columns, args.channels_per_column,
    )
    pair_defs = [
        (first, second, Pair(columns[first][0], columns[second][0],
                             label=f"apa1_columns_{first}_{second}", kind="adjacent_column_pair"))
        for first, second in COLUMN_PAIRS
    ]

    scenarios = [(scenario_name(value), value) for value in args.threshold_sigma_multipliers]
    scenarios.sort(key=lambda item: (item[0] != "nominal", item[1]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    availability_rows: list[dict] = []
    measurement_rows: list[dict] = []
    selected_rows: list[dict] = []
    panels = {
        pair.identifier: {momentum: {"message": "No measurement available."}
                          for momentum in MOMENTA}
        for _, _, pair in pair_defs
    }

    for scenario, multiplier in scenarios:
        for momentum in MOMENTA:
            threshold, threshold_error = thresholds.get(momentum, (math.nan, math.nan))
            applied = None if momentum == 1 else threshold + multiplier * threshold_error
            selected = nominal_selected[momentum] if scenario == "nominal" else selected_event_indices(
                data[momentum], contexts[momentum], momentum, 1, applied,
            )
            selected_rows.append({
                "threshold_scenario": scenario, "momentum_GeV_c": momentum,
                "threshold_applied_PE": applied if applied is not None else math.nan,
                "selected_triggers": sum(len(indices) for indices in selected.values()),
            })
            for first, second, pair in pair_defs:
                events = extract_column_pair_events(data[momentum], selected,
                                                    columns[first], columns[second])
                base = {
                    "threshold_scenario": scenario, "threshold_sigma_multiplier": multiplier,
                    "momentum_GeV_c": momentum,
                    "threshold_applied_PE": applied if applied is not None else math.nan,
                    "pair": pair.identifier, "first_column": first,
                    "second_column": second,
                    "channels_per_column": args.channels_per_column,
                    "kinetic_mean_GeV": kinetic[momentum]["kinetic_mean_GeV"],
                    "effective_spread_GeV": kinetic[momentum]["effective_spread_GeV"],
                }
                availability = {
                    **base, "selected_triggers": events.selected_triggers,
                    "common_events": events.common_events,
                    "common_event_fraction": events.common_fraction,
                    "first_column_incomplete": events.first_missing,
                    "second_column_incomplete": events.second_missing,
                    "both_columns_incomplete": events.both_missing,
                    "measurement_status": "not_measured", "message": "",
                }
                if scenario == "nominal":
                    panel = panels[pair.identifier][momentum]
                    panel.update(events=events, record=None, d_values=None, message="")
                    if events.common_events:
                        try:
                            panel["d_values"], _, _ = normalized_difference(events.values_a, events.values_b)
                        except ValueError as error:
                            panel["message"] = str(error)
                try:
                    measurement = measure_pair(pair, momentum, events, args.minimum_events)
                except ValueError as error:
                    availability["message"] = str(error)
                    if scenario == "nominal":
                        panel["message"] = str(error)
                    availability_rows.append(availability)
                    continue

                ideal = float(poisson_width(measurement.n_eff))
                gaussian = measurement.gaussian_d
                row = {
                    **base, **measurement.as_flat_dict(),
                    "ideal_independent_PE_sigma_D": ideal,
                    "sigma_D_over_ideal_independent_PE": gaussian.sigma / ideal
                    if gaussian.status == "success" else math.nan,
                    "central68_over_gaussian_sigma_D": measurement.empirical_d.central68_half_width / gaussian.sigma
                    if gaussian.status == "success" else math.nan,
                    "measurement_status": "success",
                    "coverage_status": "usable" if events.common_fraction >= 0.80 else "low_coverage",
                }
                measurement_rows.append(row)
                availability.update(measurement_status="success", message=gaussian.message)
                availability_rows.append(availability)
                if scenario == "nominal":
                    panel["record"] = row
                    panel["message"] = gaussian.message

    fit_rows: list[dict] = []
    fits_by_scenario: dict[str, dict[str, dict[str, ResolutionFit]]] = {}
    for scenario, multiplier in scenarios:
        fits_by_scenario[scenario] = {}
        for first, second, pair in pair_defs:
            fits_by_scenario[scenario][pair.identifier] = {}
            for scope in ("all", "exclude_1gev"):
                selected_records = [
                    row for row in measurement_rows
                    if row["threshold_scenario"] == scenario
                    and row["pair"] == pair.identifier
                    and row["d_gaussian_status"] == "success"
                    and (scope == "all" or row["momentum_GeV_c"] != 1)
                ]
                fit = fit_resolution(selected_records, "two_term")
                fits_by_scenario[scenario][pair.identifier][scope] = fit
                ideal_b = float(np.median([
                    row["ideal_independent_PE_sigma_D"] * math.sqrt(row["kinetic_mean_GeV"])
                    for row in selected_records
                ])) if selected_records else math.nan
                fit_rows.append({
                    "threshold_scenario": scenario, "threshold_sigma_multiplier": multiplier,
                    "pair": pair.identifier, "first_column": first, "second_column": second,
                    "channels_per_column": args.channels_per_column, "fit_scope": scope,
                    **fit.as_flat_dict(prefix="fit"),
                    "ideal_counting_b_reference_sqrt_GeV": ideal_b,
                    "ideal_b_method": "median(ideal_sigma_D * sqrt(K_eff))",
                })

    nominal_fits = fits_by_scenario["nominal"]
    systematics: list[dict] = []
    for first, second, pair in pair_defs:
        for scope in ("all", "exclude_1gev"):
            nominal = nominal_fits[pair.identifier][scope]
            row = {
                "pair": pair.identifier, "first_column": first, "second_column": second,
                "fit_scope": scope, "nominal_status": nominal.status,
                "constant_a": nominal.constant_a,
                "constant_a_stat_error": nominal.constant_a_error,
                "stochastic_b_sqrt_GeV": nominal.stochastic_b_sqrt_GeV,
                "stochastic_b_stat_error_sqrt_GeV": nominal.stochastic_b_error_sqrt_GeV,
            }
            for parameter in ("constant_a", "stochastic_b_sqrt_GeV"):
                shifts = [
                    abs(getattr(fits_by_scenario[scenario][pair.identifier][scope], parameter)
                        - getattr(nominal, parameter))
                    for scenario, _ in scenarios if scenario != "nominal"
                    and nominal.status == "success"
                    and fits_by_scenario[scenario][pair.identifier][scope].status == "success"
                ]
                row[f"{parameter}_threshold_systematic"] = max(shifts) if shifts else math.nan
                row[f"{parameter}_successful_variations"] = len(shifts)
            systematics.append(row)

    save_csv(args.output_dir / "column_configuration.csv", column_config)
    save_csv(args.output_dir / "column_channel_coverage.csv", coverage_rows)
    save_csv(args.output_dir / "selected_trigger_counts.csv", selected_rows)
    save_csv(args.output_dir / "column_pair_availability.csv", availability_rows)
    save_csv(args.output_dir / "column_pair_measurements.csv", measurement_rows,
             ["threshold_scenario", "momentum_GeV_c", "pair", "d_gaussian_sigma"])
    save_csv(args.output_dir / "column_pair_resolution_fits.csv", fit_rows)
    save_csv(args.output_dir / "column_pair_threshold_systematics.csv", systematics)

    pdf_path = args.output_dir / "apa1_adjacent_column_resolution.pdf"
    pages = write_pair_pdf(pdf_path, pair_defs, panels, nominal_fits,
                           args.channels_per_column)
    write_summary_plots(args.output_dir, pair_defs, panels, nominal_fits,
                        [row for row in systematics if row["fit_scope"] == "all"],
                        args.channels_per_column)
    report = [
        "APA 1 ADJACENT-COLUMN DIFFERENTIAL RESPONSE",
        f"Channels per column: {args.channels_per_column} of 8.",
        "Full physical columns are used only when channels per column = 8.",
        "If fewer are requested, one high-availability subset is fixed across all beam settings and threshold scenarios.",
        f"Minimum complete common triggers per pair and momentum: {args.minimum_events}.",
        f"Threshold scenarios: {', '.join(scenario for scenario, _ in scenarios)}.",
        "At 1 GeV/c: APA1-valid triggers, no muon threshold; at 2--7 GeV/c: APA1 mean above the Langauss--Gaussian intersection.",
        "A column sum is recorded only when every included channel has a finite value; valid zero values are retained.",
        "The Gaussian core and two-term sigma_D(K_eff) fit match the adjacent-channel study.",
        "The reference sigma_D = sqrt((1/mu_A + 1/mu_B)/2) assumes independent ideal PE counts.",
        "This reference is not an independent electronic-noise measurement or a full prediction.",
        "The observed/reference ratio is descriptive; its reference uncertainty is not shown as a vertical error bar.",
        "No absolute energy resolution or separate physical noise/geometry contribution is inferred.",
        "",
        "NOMINAL RESULTS",
    ]
    for first, second, pair in pair_defs:
        counts = [
            next(row["common_events"] for row in availability_rows
                 if row["threshold_scenario"] == "nominal" and row["pair"] == pair.identifier
                 and row["momentum_GeV_c"] == momentum)
            for momentum in MOMENTA
        ]
        fit = nominal_fits[pair.identifier]["all"]
        report.append(
            f"Columns {first}-{second}: complete triggers at 1,2,3,5,7 GeV/c = {counts}; "
            f"two-term fit {fit.status} ({fit.n_points} Gaussian widths)."
        )
    report += [
        "", "OUTPUTS",
        "column_configuration.csv: fixed channel membership and nominal availability score.",
        "column_channel_coverage.csv: nominal per-channel availability at every momentum.",
        "column_pair_availability.csv: complete-trigger counts for every pair, momentum, and threshold scenario.",
        "column_pair_measurements.csv: Gaussian widths, means, correlation, empirical width, and ideal PE reference.",
        "column_pair_resolution_fits.csv: two-term fits with all momenta and excluding 1 GeV/c.",
        "column_pair_threshold_systematics.csv: threshold-variation envelope for a and b.",
        f"{pdf_path.name}: {pages} pages with correlations, D_AB distributions, and sigma_D(K_eff).",
        "apa1_adjacent_column_sigma_D_vs_keff.png: three adjacent-column comparisons.",
        "apa1_adjacent_column_fit_parameters.png: a and b by column pair.",
        "apa1_adjacent_column_counting_comparison.png: observed/ideal counting-width ratios.",
    ]
    (args.output_dir / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, csv.Error, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)
