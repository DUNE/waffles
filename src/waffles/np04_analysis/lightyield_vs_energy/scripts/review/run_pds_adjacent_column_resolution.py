#!/usr/bin/env python3
r"""Study the differential PDS response of adjacent APA 1 columns.

Columns 1--2, 2--3, and 3--4 use the geometry in scripts/utils.py.  A column
value is the sum of finite values among the same first seven physical rows on
one selected trigger.  A trigger enters a column pair when at least one channel
is available in each column; missing entries are never interpreted as zero.
The number of contributing channels is recorded for every selected trigger. The
trigger selection, central Gaussian fit, K_eff values, and two-term resolution
fit match run_pds_differential_resolution.py. The fit always includes 1 GeV/c.
Coverage subsets and Gaussian residuals are diagnostics; they do not change the
nominal trigger selection or provide an absolute energy resolution.
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
    Channel, Pair, PairEvents, central68_half_width, fit_gaussian_core,
    gaussian_count_model, load_merged_json, measure_pair,
    normalized_difference, selected_event_indices,
)
from run_pds_differential_resolution import (
    DATA_COLOR, MOMENTA, ResolutionFit, add_panel_brand, fit_resolution,
    format_panel_axis, load_kinetic_energies, load_nominal_thresholds,
    load_trigger_contexts, plot_correlation_panel, plot_distribution_panel,
    resolution_model, scenario_name,
)
from waffles.np04_data.ProtoDUNE_HD_APA_maps import APA_map


SYSTEMATIC_COLOR = "#D55E00"
DIAGNOSTIC_COLOR = "#CC79A7"
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


def column_sum(event: Mapping[str, Any] | None, channels: list[Channel]) -> tuple[float | None, int]:
    if event is None:
        return None, 0
    values = [channel_value(event, channel) for channel in channels]
    valid = [value for value in values if value is not None]
    return (float(sum(valid)), len(valid)) if valid else (None, 0)


def selected_channel_coverage(
    data_by_momentum: dict[int, dict],
    selected_by_momentum: dict[int, dict],
    full_columns: dict[int, list[Channel]],
    channels_per_column: int,
) -> tuple[dict[int, list[Channel]], list[dict], list[dict]]:
    """Use the same physical rows in every column and record all 40 channels.

    Each column list follows the ten-row APA map from top to bottom.  Keeping
    the first N rows ensures that the summed channels cover the same vertical
    positions in every column at every momentum and threshold scenario.
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
            for physical_row, channel in enumerate(channels, start=1):
                fraction = counts[channel] / total if total else 0.0
                scores.setdefault((column, channel.endpoint, channel.channel), []).append(fraction)
                coverage_rows.append({
                    "column": column, "physical_row": physical_row,
                    "endpoint": channel.endpoint,
                    "channel": channel.channel, "momentum_GeV_c": momentum,
                    "selected_triggers": total, "valid_channel_triggers": counts[channel],
                    "valid_fraction": fraction,
                })

    chosen: dict[int, list[Channel]] = {}
    configuration: list[dict] = []
    for column, channels in full_columns.items():
        chosen[column] = channels[:channels_per_column]
        for physical_row, channel in enumerate(channels, start=1):
            configuration.append({
                "column": column, "physical_row": physical_row,
                "endpoint": channel.endpoint,
                "channel": channel.channel,
                "included_in_sum": int(physical_row <= channels_per_column),
                "mean_valid_fraction_over_momenta": float(np.mean(
                    scores[(column, channel.endpoint, channel.channel)]
                )),
            })
    return chosen, coverage_rows, configuration


def extract_column_pair_events(
    data: dict, selected: dict, first: list[Channel], second: list[Channel],
) -> tuple[PairEvents, list[dict]]:
    values_a: list[float] = []
    values_b: list[float] = []
    multiplicity: list[dict] = []
    selected_triggers = first_missing = second_missing = both_missing = 0
    for block, indices in selected.items():
        for index in indices:
            selected_triggers += 1
            event = event_at(data, block, int(index))
            sum_a, n_a = column_sum(event, first)
            sum_b, n_b = column_sum(event, second)
            multiplicity.append({
                "block": block, "json_trigger_index": int(index),
                "valid_channels_first_column": n_a,
                "valid_channels_second_column": n_b,
                "first_column_sum_PE": sum_a if sum_a is not None else math.nan,
                "second_column_sum_PE": sum_b if sum_b is not None else math.nan,
                "used_for_pair": int(sum_a is not None and sum_b is not None),
            })
            if sum_a is None and sum_b is None:
                both_missing += 1
            elif sum_a is None:
                first_missing += 1
            elif sum_b is None:
                second_missing += 1
            else:
                values_a.append(sum_a)
                values_b.append(sum_b)
    events = PairEvents(
        values_a=np.asarray(values_a, dtype=float),
        values_b=np.asarray(values_b, dtype=float),
        selected_triggers=selected_triggers,
        common_events=len(values_a),
        first_missing=first_missing,
        second_missing=second_missing,
        both_missing=both_missing,
    )
    return events, multiplicity


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
    axis.errorbar(x, y, xerr=sx, yerr=sy, fmt="o", color=DATA_COLOR, ms=5,
                  capsize=2.1, label=r"Gaussian $\sigma_D$", zorder=4)
    x_curve = np.linspace(max(0.05, float(min(x - sx))), float(max(x + sx)) * 1.03, 300)
    full = fits["all"]
    if full.status == "success":
        axis.plot(x_curve, resolution_model(x_curve, full.constant_a,
                  full.stochastic_b_sqrt_GeV), color=FIT_COLOR, lw=1.8,
                  label=(r"Two-term fit" + "\n"
                         + rf"$a=({full.constant_a:.3f}\pm{full.constant_a_error:.3f})$" + "\n"
                         + rf"$b=({full.stochastic_b_sqrt_GeV:.3f}\pm{full.stochastic_b_error_sqrt_GeV:.3f})\sqrt{{\rm GeV}}$" + "\n"
                         + rf"$\chi^2/{{\rm ndf}}={full.chi2_ndf:.2f},\ R^2={full.r_squared:.3f}$"))
    axis.set_xlim(max(0.0, float(min(x - sx)) - 0.1), float(max(x + sx)) + 0.1)
    axis.set_ylim(0.0, float(max(y + sy)) * 1.37)
    axis.legend(loc="upper right", fontsize=5.7, facecolor="white", frameon=True)
    format_panel_axis(axis, False)


def write_pair_pdf(output: Path, pair_defs: list[tuple[int, int, Pair]],
                   panels: dict[str, dict[int, dict]], fits: dict[str, dict[str, ResolutionFit]],
                   channels_per_column: int) -> int:
    pages = 0
    with PdfPages(output) as pdf:
        for first, second, pair in pair_defs:
            figure = plt.figure(figsize=(16.54, 11.69))
            grid = figure.add_gridspec(3, 4)
            subset_note = ("" if channels_per_column == 10
                           else f" (physical rows 1–{channels_per_column})")
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
            figure.subplots_adjust(left=0.055, right=0.985, bottom=0.065,
                                   top=0.92, wspace=0.42, hspace=0.48)
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
        subset_note = f" (rows 1–{channels_per_column}/10)"
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
         "stochastic_b_sqrt_GeV_threshold_systematic",
         r"Stochastic parameter $b$ [AU $\sqrt{\rm GeV}$]"),
    ):
        statistical_label_added = False
        systematic_label_added = False
        for index, (_, _, pair) in enumerate(pair_defs):
            fit = fits[pair.identifier]["all"]
            if fit.status != "success":
                continue
            value = getattr(fit, field)
            axis.errorbar(index, value, yerr=getattr(fit, error), fmt="o", color=DATA_COLOR,
                          capsize=4, ms=7,
                          label="Fit ± statistical" if not statistical_label_added else None)
            statistical_label_added = True
            systematic = sys_lookup[pair.identifier][syst_field]
            if math.isfinite(systematic) and systematic > 0:
                axis.errorbar(index + 0.10, value, yerr=systematic, fmt="none",
                              ecolor=SYSTEMATIC_COLOR, capsize=4,
                              label="Threshold variation" if not systematic_label_added else None)
                systematic_label_added = True
        axis.set_xticks(range(3), names)
        axis.set(xlabel="APA 1 column pair (unequal separations)", ylabel=label)
        axis.grid(axis="y", alpha=0.23)
        add_panel_brand(axis, "right", standalone=True)
        if statistical_label_added:
            axis.legend(loc="upper left", frameon=True, facecolor="white", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "apa1_adjacent_column_fit_parameters.png", dpi=240)
    plt.close(fig)



def coverage_check_rows(
    base: dict, d_values: np.ndarray, multiplicity: list[dict],
    nominal_gaussian: Any, channels_per_column: int, minimum_events: int,
) -> list[dict]:
    """Compare coverage subsets without applying a cut to the nominal result."""

    used = [item for item in multiplicity if item["used_for_pair"]]
    if len(used) != len(d_values):
        raise ValueError("Column multiplicities are not aligned with D_AB values.")
    n_min = np.asarray([
        min(item["valid_channels_first_column"], item["valid_channels_second_column"])
        for item in used
    ], dtype=int)
    groups = [("all", np.ones(len(used), dtype=bool)),
              ("complete", n_min == channels_per_column)]
    groups.extend((f"min_valid_{count}", n_min == count)
                  for count in range(1, channels_per_column + 1))
    rows = []
    for name, mask in groups:
        subset = d_values[mask]
        fit = (nominal_gaussian if name == "all" else
               fit_gaussian_core(subset) if name == "complete" and len(subset) >= minimum_events
               else None)
        rows.append({
            **base, "coverage_group": name, "events": len(subset),
            "fraction_of_common_events": len(subset) / len(d_values),
            "d_mean": float(np.mean(subset)) if len(subset) else math.nan,
            "d_central68_half_width": central68_half_width(subset)
            if len(subset) >= 30 else math.nan,
            "gaussian_fit_status": fit.status if fit is not None else "not_fit",
            "gaussian_sigma_D": fit.sigma if fit is not None else math.nan,
            "gaussian_sigma_D_error": fit.sigma_error if fit is not None else math.nan,
            "gaussian_chi2_ndf": fit.chi2_ndf if fit is not None else math.nan,
        })
    return rows


def write_coverage_check_plot(output_dir: Path, pair_defs: list[tuple[int, int, Pair]],
                              diagnostics: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharex=True, sharey=True)
    n_channels = diagnostics[0]["channels_per_column"] if diagnostics else 7
    for axis, (first, second, pair) in zip(axes, pair_defs):
        for group, label, color, marker in (
            ("all", "All common triggers", DATA_COLOR, "o"),
            ("complete", f"{n_channels} + {n_channels} valid channels (check)", DIAGNOSTIC_COLOR, "s"),
        ):
            rows = sorted((row for row in diagnostics if row["pair"] == pair.identifier
                           and row["coverage_group"] == group
                           and row["gaussian_fit_status"] == "success"),
                          key=lambda row: row["kinetic_mean_GeV"])
            if rows:
                axis.errorbar([row["kinetic_mean_GeV"] for row in rows],
                              [row["gaussian_sigma_D"] for row in rows],
                              xerr=[row["effective_spread_GeV"] for row in rows],
                              yerr=[row["gaussian_sigma_D_error"] for row in rows],
                              fmt=f"{marker}-", color=color, capsize=2.5, label=label)
        axis.set(xlabel=r"$K_{\rm eff}$ [GeV]", ylabel=r"Gaussian $\sigma_D$ [AU]")
        axis.set_title(f"Columns {first}–{second}")
        axis.grid(alpha=0.22)
        add_panel_brand(axis, "upper_center")
        if axis.get_legend_handles_labels()[0]:
            axis.legend(loc="upper right", frameon=True, facecolor="white", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "apa1_adjacent_column_coverage_check.png", dpi=240)
    plt.close(fig)


def write_gaussian_residual_checks(output_dir: Path,
                                   pair_defs: list[tuple[int, int, Pair]],
                                   panels: dict[str, dict[int, dict]]) -> None:
    fig, axes = plt.subplots(3, 5, figsize=(16.5, 9), sharey=False)
    for row_index, (first, second, pair) in enumerate(pair_defs):
        for column_index, momentum in enumerate(MOMENTA):
            axis = axes[row_index, column_index]
            panel = panels[pair.identifier][momentum]
            record = panel.get("record")
            values = panel.get("d_values")
            axis.axhline(0.0, color="#444444", lw=0.9)
            axis.set_title(f"Columns {first}–{second}, {momentum} GeV/c", fontsize=10)
            if record is None or values is None or record["d_gaussian_status"] != "success":
                axis.text(0.5, 0.5, "No Gaussian fit", transform=axis.transAxes,
                          ha="center", va="center")
                continue
            low, high = record["d_gaussian_fit_low"], record["d_gaussian_fit_high"]
            bins = int(round((high - low) / record["d_gaussian_bin_width"]))
            counts, edges = np.histogram(values[(values >= low) & (values <= high)],
                                         bins=np.linspace(low, high, bins + 1))
            centers = (edges[:-1] + edges[1:]) / 2.0
            prediction = gaussian_count_model(
                centers, record["d_gaussian_amplitude"], record["d_gaussian_mean"],
                record["d_gaussian_sigma"],
            )
            residual = (counts - prediction) / np.sqrt(np.maximum(counts, 1.0))
            axis.plot(centers, residual, "o", color=DATA_COLOR, ms=3)
            axis.text(0.03, 0.96,
                      rf"$\chi^2/{{\rm ndf}}={record['d_gaussian_chi2_ndf']:.2f}$",
                      transform=axis.transAxes, ha="left", va="top", fontsize=9,
                      bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
            axis.set_xlabel(r"$D_{AB}$ [AU]", fontsize=9)
            axis.set_ylabel("Poisson residual", fontsize=9)
            axis.grid(alpha=0.2)
    fig.text(0.985, 0.995, "ProtoDUNE-HD Work in Progress", ha="right", va="top",
             fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_dir / "apa1_adjacent_column_gaussian_residuals.png", dpi=240)
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
    parser.add_argument("--channels-per-column", type=int, default=7, choices=range(2, 11),
                        help="Same top N physical rows summed in every column (default: rows 1–7).")
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
    raw_rows = APA_map[1].data
    if len(raw_rows) != 10 or any(len(row) != 4 for row in raw_rows):
        raise ValueError("The APA 1 geometry must contain ten rows of four channels.")
    full_columns = {
        column: [Channel.from_mapping({"apa": 1, "end": row[column - 1].endpoint,
                                       "ch": row[column - 1].channel})
                 for row in raw_rows]
        for column in range(1, 5)
    }
    flattened = [channel for channels in full_columns.values() for channel in channels]
    if len(set(flattened)) != 40 or any(channel.apa != 1 for channel in flattened):
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
    multiplicity_rows: list[dict] = []
    coverage_checks: list[dict] = []
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
                events, multiplicity = extract_column_pair_events(
                    data[momentum], selected, columns[first], columns[second],
                )
                multiplicity_rows.extend({
                    "threshold_scenario": scenario, "momentum_GeV_c": momentum,
                    "pair": pair.identifier, "first_column": first,
                    "second_column": second, **item,
                } for item in multiplicity)
                used = [item for item in multiplicity if item["used_for_pair"]]
                coverage_stats = {
                    "fully_observed_common_events": sum(
                        item["valid_channels_first_column"] == args.channels_per_column
                        and item["valid_channels_second_column"] == args.channels_per_column
                        for item in used
                    ),
                    "mean_valid_channels_first_column": float(np.mean([
                        item["valid_channels_first_column"] for item in used
                    ])) if used else math.nan,
                    "mean_valid_channels_second_column": float(np.mean([
                        item["valid_channels_second_column"] for item in used
                    ])) if used else math.nan,
                }
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
                    **base, **coverage_stats,
                    "selected_triggers": events.selected_triggers,
                    "common_events": events.common_events,
                    "common_event_fraction": events.common_fraction,
                    "first_column_no_valid_channel": events.first_missing,
                    "second_column_no_valid_channel": events.second_missing,
                    "both_columns_no_valid_channel": events.both_missing,
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

                gaussian = measurement.gaussian_d
                row = {
                    **base, **coverage_stats, **measurement.as_flat_dict(),
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
                    if panel["d_values"] is not None:
                        coverage_checks.extend(coverage_check_rows(
                            base, panel["d_values"], multiplicity, gaussian,
                            args.channels_per_column, args.minimum_events,
                        ))

    fit_rows: list[dict] = []
    fits_by_scenario: dict[str, dict[str, dict[str, ResolutionFit]]] = {}
    for scenario, multiplier in scenarios:
        fits_by_scenario[scenario] = {}
        for first, second, pair in pair_defs:
            fits_by_scenario[scenario][pair.identifier] = {}
            selected_records = [
                row for row in measurement_rows
                if row["threshold_scenario"] == scenario
                and row["pair"] == pair.identifier
                and row["d_gaussian_status"] == "success"
            ]
            fit = (fit_resolution(selected_records, "two_term")
                   if any(row["momentum_GeV_c"] == 1 for row in selected_records)
                   else ResolutionFit.failed(
                       "two_term", "The required Gaussian width at 1 GeV/c is unavailable.",
                       len(selected_records),
                       ";".join(str(row["momentum_GeV_c"]) for row in selected_records),
                   ))
            fits_by_scenario[scenario][pair.identifier]["all"] = fit
            fit_rows.append({
                "threshold_scenario": scenario, "threshold_sigma_multiplier": multiplier,
                "pair": pair.identifier, "first_column": first, "second_column": second,
                "channels_per_column": args.channels_per_column, "fit_scope": "all",
                **fit.as_flat_dict(prefix="fit"),
            })

    nominal_fits = fits_by_scenario["nominal"]
    systematics: list[dict] = []
    for first, second, pair in pair_defs:
        nominal = nominal_fits[pair.identifier]["all"]
        row = {
            "pair": pair.identifier, "first_column": first, "second_column": second,
            "fit_scope": "all", "nominal_status": nominal.status,
            "constant_a": nominal.constant_a,
            "constant_a_stat_error": nominal.constant_a_error,
            "stochastic_b_sqrt_GeV": nominal.stochastic_b_sqrt_GeV,
            "stochastic_b_stat_error_sqrt_GeV": nominal.stochastic_b_error_sqrt_GeV,
        }
        for parameter in ("constant_a", "stochastic_b_sqrt_GeV"):
            shifts = [
                abs(getattr(fits_by_scenario[scenario][pair.identifier]["all"], parameter)
                    - getattr(nominal, parameter))
                for scenario, _ in scenarios if scenario != "nominal"
                and nominal.status == "success"
                and fits_by_scenario[scenario][pair.identifier]["all"].status == "success"
            ]
            row[f"{parameter}_threshold_systematic"] = max(shifts) if shifts else math.nan
            row[f"{parameter}_successful_variations"] = len(shifts)
        systematics.append(row)

    save_csv(args.output_dir / "column_configuration.csv", column_config)
    save_csv(args.output_dir / "column_channel_coverage.csv", coverage_rows)
    save_csv(args.output_dir / "selected_trigger_counts.csv", selected_rows)
    save_csv(args.output_dir / "column_pair_availability.csv", availability_rows)
    save_csv(args.output_dir / "column_pair_trigger_multiplicity.csv", multiplicity_rows,
             ["threshold_scenario", "momentum_GeV_c", "pair",
              "valid_channels_first_column", "valid_channels_second_column"])
    save_csv(args.output_dir / "column_pair_measurements.csv", measurement_rows,
             ["threshold_scenario", "momentum_GeV_c", "pair", "d_gaussian_sigma"])
    save_csv(args.output_dir / "column_pair_resolution_fits.csv", fit_rows)
    save_csv(args.output_dir / "column_pair_threshold_systematics.csv", systematics)
    save_csv(args.output_dir / "column_pair_coverage_checks.csv", coverage_checks,
             ["momentum_GeV_c", "pair", "coverage_group", "events",
              "d_central68_half_width", "gaussian_sigma_D"])

    pdf_path = args.output_dir / "apa1_adjacent_column_resolution.pdf"
    pages = write_pair_pdf(pdf_path, pair_defs, panels, nominal_fits,
                           args.channels_per_column)
    write_summary_plots(args.output_dir, pair_defs, panels, nominal_fits,
                        systematics, args.channels_per_column)
    write_coverage_check_plot(args.output_dir, pair_defs, coverage_checks)
    write_gaussian_residual_checks(args.output_dir, pair_defs, panels)
    (args.output_dir / "apa1_adjacent_column_counting_comparison.png").unlink(missing_ok=True)
    report = [
        "APA 1 ADJACENT-COLUMN DIFFERENTIAL RESPONSE",
        "APA 1 geometry: 40 channels in 10 rows and 4 columns.",
        f"Each column sum uses the same physical rows 1--{args.channels_per_column} "
        f"({args.channels_per_column} of 10 channels per column).",
        "The included rows are fixed across all beam settings and threshold scenarios.",
        "Excluded physical cells remain listed in the channel configuration and coverage CSVs.",
        "The default seven-row selection omits row 8, which contains END 105 - CH 12, "
        "and the two lowest rows (9 and 10).",
        f"Minimum common triggers per pair and momentum: {args.minimum_events}.",
        f"Threshold scenarios: {', '.join(scenario for scenario, _ in scenarios)}.",
        "At 1 GeV/c: APA1-valid triggers, no muon threshold; at 2--7 GeV/c: APA1 mean above the Langauss--Gaussian intersection.",
        "For each selected trigger, a column sum uses its finite channel values in the fixed included rows.",
        "At least one valid channel in each column is required; valid zero values are retained.",
        "Absent or invalid values are excluded from the sum, never replaced by zero.",
        "Changing channel multiplicity can broaden D_AB; inspect the per-trigger multiplicity CSV before interpreting the widths.",
        "The Gaussian core and two-term sigma_D(K_eff) fit match the adjacent-channel study.",
        "The two-term fit always includes 1 GeV/c; no fit excluding 1 GeV/c is produced.",
        "The three adjacent column pairs have unequal physical separations; their fit parameters are pair-specific.",
        "The complete-channel subset is used only to diagnose coverage dependence, never as a nominal trigger cut.",
        "Coverage diagnostics retain the D_AB normalization of the full common-trigger sample for all subsets.",
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
            f"Columns {first}-{second}: common triggers at 1,2,3,5,7 GeV/c = {counts}; "
            f"two-term fit {fit.status} ({fit.n_points} Gaussian widths)."
        )
    report += [
        "", "OUTPUTS",
        "column_configuration.csv: fixed channel membership and nominal availability score.",
        "column_channel_coverage.csv: nominal per-channel availability at every momentum.",
        "column_pair_availability.csv: common-trigger counts and mean contributing channels for every pair, momentum, and threshold scenario.",
        "column_pair_trigger_multiplicity.csv: contributing channel counts and column sums for every selected trigger.",
        "column_pair_measurements.csv: Gaussian widths, means, correlation, and empirical widths.",
        "column_pair_coverage_checks.csv: D_AB width versus contributing-channel count and complete-channel diagnostic fits.",
        "column_pair_resolution_fits.csv: two-term fits including 1 GeV/c.",
        "column_pair_threshold_systematics.csv: threshold-variation envelope for a and b.",
        f"{pdf_path.name}: {pages} pages with correlations, D_AB distributions, and sigma_D(K_eff).",
        "apa1_adjacent_column_sigma_D_vs_keff.png: three adjacent-column comparisons.",
        "apa1_adjacent_column_fit_parameters.png: a and b by column pair.",
        "apa1_adjacent_column_coverage_check.png: nominal and complete-channel diagnostic widths.",
        "apa1_adjacent_column_gaussian_residuals.png: Gaussian fit residuals at all momenta.",
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
