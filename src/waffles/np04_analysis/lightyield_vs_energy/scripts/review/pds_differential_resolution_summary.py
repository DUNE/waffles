#!/usr/bin/env python3
r"""Create summary products for the adjacent-channel differential PDS study.

This post-processing program reads the CSV files written by
run_pds_differential_resolution.py. It does not refit distributions.

The produced maps have distinct meanings:
* sigma_D at a fixed beam momentum is a direct Gaussian-fit measurement;
* a and b are the parameters of the nominal two-term fit,
  sigma_D = sqrt(a^2 + b^2 / K_eff).

The descriptive summary plot shows, at each energy, the median Gaussian
width across pairs and the central 68% interval of the pair values. That
interval is a spatial pair-to-pair spread, not an uncertainty on the median.

Run from scripts/review after the APA analysis, for example:

    python pds_differential_resolution_summary.py \
        --results-dir ../../output/review/pds_differential_resolution_apa1_01 \
        --apa 1
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from matplotlib.patches import Rectangle


SCRIPTS_DIRECTORY = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from waffles.np04_data.ProtoDUNE_HD_APA_maps import APA_map


MOMENTA = (1, 2, 3, 5, 7)
NOMINAL_SCENARIO = "nominal"
PRIMARY_MODEL = "two_term"
TEXT_COLOR = "#222222"
MISSING_FACE = "#E8E8E8"
MISSING_EDGE = "#777777"
DATA_COLOR = "#0072B2"
PAIR_CMAP = "YlOrRd"


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    """Read one analysis CSV and validate its required columns."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, strict=True)
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Missing columns in {path.name}: {', '.join(missing)}")
        return list(reader)


def as_int(value: object) -> int:
    number = float(str(value).strip())
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"Expected a finite integer, received {value!r}")
    return int(number)


def as_finite_float(value: object) -> float:
    number = float(str(value).strip())
    if not math.isfinite(number):
        raise ValueError(f"Expected a finite number, received {value!r}")
    return number


def optional_float(value: object) -> float:
    try:
        return as_finite_float(value)
    except (TypeError, ValueError):
        return float("nan")


def is_nominal(row: dict[str, str]) -> bool:
    """Accept legacy files without a scenario column as nominal-only output."""
    return row.get("threshold_scenario", NOMINAL_SCENARIO).strip() == NOMINAL_SCENARIO


def map_channels(apa: int) -> dict[tuple[int, int], tuple[int, int]]:
    """Map each endpoint-channel to its physical APA-map row and column."""
    positions: dict[tuple[int, int], tuple[int, int]] = {}
    for row_index, row in enumerate(APA_map[apa].data):
        for column_index, unique_channel in enumerate(row):
            positions[(int(unique_channel.endpoint), int(unique_channel.channel))] = (
                row_index,
                column_index,
            )
    return positions


def channel_centre(apa: int, row: int, column: int) -> tuple[float, float]:
    """Return display coordinates consistent with the existing APA maps."""
    x_origin = 10.0 if apa == 1 else 260.0
    return x_origin + 21.5 + 50.0 * column, 570.0 - 57.0 * row


def collect_pairs(config_rows: list[dict[str, str]], apa: int) -> dict[str, dict[str, object]]:
    """Build the fixed manual-pair configuration for one APA."""
    pairs: dict[str, dict[str, object]] = {}
    for row in config_rows:
        if as_int(row["apa"]) != apa:
            continue
        identifier = row["pair"].strip()
        if identifier in pairs:
            raise ValueError(f"Duplicate pair configuration: {identifier}")
        pairs[identifier] = {
            "pair": identifier,
            "apa": apa,
            "first_endpoint": as_int(row["first_endpoint"]),
            "first_channel": as_int(row["first_channel"]),
            "second_endpoint": as_int(row["second_endpoint"]),
            "second_channel": as_int(row["second_channel"]),
        }
    if not pairs:
        raise ValueError(f"No pair configuration is available for APA {apa}.")
    return pairs


def collect_widths(
    rows: list[dict[str, str]], apa: int
) -> dict[tuple[str, int], dict[str, object]]:
    """Select successful nominal Gaussian widths, indexed by pair and momentum."""
    result: dict[tuple[str, int], dict[str, object]] = {}
    for row in rows:
        if not is_nominal(row):
            continue
        identifier = row["pair"].strip()
        if not identifier.startswith(f"apa{apa}_"):
            continue
        momentum = as_int(row["momentum_GeV_c"])
        if momentum not in MOMENTA or row["d_gaussian_status"].strip() != "success":
            continue
        try:
            sigma = as_finite_float(row["d_gaussian_sigma"])
            sigma_error = as_finite_float(row["d_gaussian_sigma_error"])
            kinetic = as_finite_float(row["kinetic_mean_GeV"])
            kinetic_error = as_finite_float(row["effective_spread_GeV"])
        except ValueError:
            continue
        if sigma <= 0.0 or sigma_error <= 0.0 or kinetic <= 0.0 or kinetic_error < 0.0:
            continue
        key = (identifier, momentum)
        if key in result:
            raise ValueError(
                f"Duplicate nominal Gaussian width for {identifier} at {momentum} GeV/c."
            )
        result[key] = {
            "pair": identifier,
            "momentum_GeV_c": momentum,
            "kinetic_mean_GeV": kinetic,
            "effective_spread_GeV": kinetic_error,
            "sigma_D": sigma,
            "sigma_D_statistical_error": sigma_error,
            "coverage_status": row.get("coverage_status", "").strip(),
        }
    return result


def collect_resolution_fits(
    rows: list[dict[str, str]], apa: int, model: str
) -> dict[str, dict[str, float | str]]:
    """Select successful nominal final fits for the requested resolution model."""
    fits: dict[str, dict[str, float | str]] = {}
    for row in rows:
        if not is_nominal(row):
            continue
        if as_int(row["apa"]) != apa or row["resolution_model"].strip() != model:
            continue
        if row["resolution_fit_status"].strip() != "success":
            continue
        identifier = row["pair"].strip()
        if identifier in fits:
            raise ValueError(f"Duplicate nominal {model} fit for {identifier}.")
        try:
            constant_a = as_finite_float(row["resolution_fit_constant_a"])
            constant_a_error = as_finite_float(row["resolution_fit_constant_a_error"])
            stochastic_b = as_finite_float(row["resolution_fit_stochastic_b_sqrt_GeV"])
            stochastic_b_error = as_finite_float(
                row["resolution_fit_stochastic_b_error_sqrt_GeV"]
            )
        except ValueError:
            continue
        if min(constant_a, constant_a_error, stochastic_b, stochastic_b_error) < 0.0:
            continue
        fits[identifier] = {
            "constant_a": constant_a,
            "constant_a_statistical_error": constant_a_error,
            "stochastic_b_sqrt_GeV": stochastic_b,
            "stochastic_b_statistical_error_sqrt_GeV": stochastic_b_error,
            "chi2_ndf": optional_float(row.get("resolution_fit_chi2_ndf", "")),
            "r_squared": optional_float(row.get("resolution_fit_r_squared", "")),
        }
    return fits


def collect_systematics(
    path: Path, apa: int, model: str
) -> dict[str, dict[str, float]]:
    """Read threshold-selection envelopes when the corresponding CSV exists."""
    if not path.is_file():
        return {}
    rows = read_csv(
        path,
        {
            "pair",
            "apa",
            "resolution_model",
            "nominal_status",
            "constant_a_threshold_systematic",
            "stochastic_b_sqrt_GeV_threshold_systematic",
        },
    )
    systematics: dict[str, dict[str, float]] = {}
    for row in rows:
        if as_int(row["apa"]) != apa or row["resolution_model"].strip() != model:
            continue
        if row["nominal_status"].strip() != "success":
            continue
        systematics[row["pair"].strip()] = {
            "constant_a_threshold_systematic": optional_float(
                row["constant_a_threshold_systematic"]
            ),
            "stochastic_b_sqrt_GeV_threshold_systematic": optional_float(
                row["stochastic_b_sqrt_GeV_threshold_systematic"]
            ),
        }
    return systematics


def pair_rectangle(
    pair: dict[str, object],
    positions: dict[tuple[int, int], tuple[int, int]],
) -> tuple[float, float, float, float]:
    """Return a compact rectangle centred between the two physical channels."""
    first_key = (int(pair["first_endpoint"]), int(pair["first_channel"]))
    second_key = (int(pair["second_endpoint"]), int(pair["second_channel"]))
    if first_key not in positions or second_key not in positions:
        raise ValueError(f"Pair {pair['pair']} is absent from the APA geometry map.")
    first = channel_centre(int(pair["apa"]), *positions[first_key])
    second = channel_centre(int(pair["apa"]), *positions[second_key])
    delta_x = abs(second[0] - first[0])
    delta_y = abs(second[1] - first[1])
    centre_x = 0.5 * (first[0] + second[0])
    centre_y = 0.5 * (first[1] + second[1])
    if delta_x >= delta_y:
        width = max(34.0, min(43.0, 0.86 * delta_x))
        height = 17.0
    else:
        width = 43.0
        height = max(17.0, min(43.0, 0.72 * delta_y))
    return centre_x - 0.5 * width, centre_y - 0.5 * height, width, height


def display_limits(apa: int) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return map limits matching the physical layout of the requested APA."""
    if apa == 1:
        return (-12.0, 225.0), (-4.0, 620.0)
    return (238.0, 475.0), (-4.0, 620.0)


def add_work_in_progress(axis: plt.Axes) -> None:
    axis.text(
        0.985,
        0.985,
        r"$\bf{ProtoDUNE\!-\!HD}$" "\nWork in Progress",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        linespacing=0.95,
        color=TEXT_COLOR,
        zorder=10,
    )


def value_text(value: float, error: float) -> str:
    """Use three decimals for the small dimensionless pair observables."""
    if math.isfinite(error):
        return f"{value:.3f} $\\pm$ {error:.3f}"
    return f"{value:.3f}"


def draw_pair_map(
    pairs: dict[str, dict[str, object]],
    values: dict[str, tuple[float, float]],
    positions: dict[tuple[int, int], tuple[int, int]],
    apa: int,
    colour_bar_label: str,
    output_path: Path,
    dpi: int,
) -> None:
    """Draw one physical-layout map for a pair-level observable."""
    finite_values = np.asarray(
        [value for value, _ in values.values() if math.isfinite(value)], dtype=float
    )
    if len(finite_values) == 0:
        raise ValueError(f"No finite values are available for {output_path.name}.")
    lower = float(np.min(finite_values))
    upper = float(np.max(finite_values))
    if math.isclose(lower, upper):
        padding = max(0.05 * abs(lower), 0.01)
        lower -= padding
        upper += padding
    norm = colors.Normalize(vmin=lower, vmax=upper, clip=True)
    colour_map = cm.get_cmap(PAIR_CMAP)

    figure, axis = plt.subplots(figsize=(10.7, 8.2), constrained_layout=True)
    for pair in pairs.values():
        x_left, y_bottom, width, height = pair_rectangle(pair, positions)
        record = values.get(str(pair["pair"]))
        first_channel = int(pair["first_channel"])
        second_channel = int(pair["second_channel"])
        if record is None or not math.isfinite(record[0]):
            patch = Rectangle(
                (x_left, y_bottom),
                width,
                height,
                facecolor=MISSING_FACE,
                edgecolor=MISSING_EDGE,
                linewidth=0.8,
                hatch="//",
                zorder=2,
            )
            axis.add_patch(patch)
            axis.text(
                x_left + 0.5 * width,
                y_bottom + 0.5 * height,
                "not\navailable",
                ha="center",
                va="center",
                fontsize=5.7,
                color="#555555",
                zorder=3,
            )
        else:
            value, error = record
            patch = Rectangle(
                (x_left, y_bottom),
                width,
                height,
                facecolor=colour_map(norm(value)),
                edgecolor=TEXT_COLOR,
                linewidth=0.9,
                zorder=2,
            )
            axis.add_patch(patch)
            axis.text(
                x_left + 0.5 * width,
                y_bottom + 0.5 * height,
                value_text(value, error),
                ha="center",
                va="center",
                fontsize=8.3,
                fontweight="bold",
                color=TEXT_COLOR,
                zorder=3,
            )
        axis.text(
            x_left + 0.5 * width,
            y_bottom - 8.2,
            f"CH {first_channel}--{second_channel}",
            ha="center",
            va="top",
            fontsize=6.2,
            color="#333333",
            zorder=3,
        )

    x_limits, y_limits = display_limits(apa)
    axis.text(
        x_limits[0] + 10.0,
        600.0,
        f"APA {apa}",
        ha="left",
        va="bottom",
        fontsize=18,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    add_work_in_progress(axis)
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.set_xlabel(r"$z$ direction [cm]", fontsize=13)
    axis.set_ylabel(r"$y$ direction [cm]", fontsize=13)
    axis.tick_params(direction="in", top=True, right=True, labelsize=10)
    axis.grid(linestyle="--", linewidth=0.45, alpha=0.28, zorder=0)

    mapper = cm.ScalarMappable(norm=norm, cmap=colour_map)
    mapper.set_array([])
    colour_bar = figure.colorbar(mapper, ax=axis, pad=0.035, fraction=0.046)
    colour_bar.set_label(colour_bar_label, fontsize=12)
    colour_bar.ax.tick_params(labelsize=10)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def draw_width_summary(
    widths: Iterable[dict[str, object]],
    apa: int,
    output_path: Path,
    dpi: int,
) -> list[dict[str, float | int]]:
    """Draw the descriptive median and central 68% interval across pair widths."""
    rows = list(widths)
    summary: list[dict[str, float | int]] = []
    for momentum in MOMENTA:
        group = [row for row in rows if int(row["momentum_GeV_c"]) == momentum]
        if not group:
            continue
        values = np.asarray([float(row["sigma_D"]) for row in group], dtype=float)
        kinetic = np.asarray([float(row["kinetic_mean_GeV"]) for row in group], dtype=float)
        kinetic_error = np.asarray(
            [float(row["effective_spread_GeV"]) for row in group], dtype=float
        )
        lower, median, upper = np.quantile(values, (0.16, 0.50, 0.84))
        summary.append(
            {
                "apa": apa,
                "momentum_GeV_c": momentum,
                "kinetic_mean_GeV": float(np.median(kinetic)),
                "effective_spread_GeV": float(np.median(kinetic_error)),
                "n_pairs": len(group),
                "sigma_D_median": float(median),
                "sigma_D_central68_low": float(lower),
                "sigma_D_central68_high": float(upper),
            }
        )
    if not summary:
        raise ValueError("No successful nominal Gaussian widths are available.")

    x = np.asarray([row["kinetic_mean_GeV"] for row in summary])
    sx = np.asarray([row["effective_spread_GeV"] for row in summary])
    median = np.asarray([row["sigma_D_median"] for row in summary])
    lower = np.asarray([row["sigma_D_central68_low"] for row in summary])
    upper = np.asarray([row["sigma_D_central68_high"] for row in summary])

    figure, axis = plt.subplots(figsize=(9.0, 6.2), constrained_layout=True)
    axis.fill_between(
        x,
        lower,
        upper,
        color=DATA_COLOR,
        alpha=0.20,
        linewidth=0,
        label="Central 68% of pairs",
        zorder=1,
    )
    axis.errorbar(
        x,
        median,
        xerr=sx,
        marker="o",
        linestyle="-",
        color=DATA_COLOR,
        markersize=6.5,
        linewidth=1.6,
        capsize=3.0,
        label="Median",
        zorder=3,
    )
    for row in summary:
        axis.annotate(
            f"$N={row['n_pairs']}$",
            (float(row["kinetic_mean_GeV"]), float(row["sigma_D_central68_high"])),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=TEXT_COLOR,
        )
    add_work_in_progress(axis)
    axis.set_xlabel(r"$K_{\mathrm{eff}}$ [GeV]", fontsize=13)
    axis.set_ylabel(r"Gaussian fit $\sigma_D$ [AU]", fontsize=13)
    axis.tick_params(direction="in", top=True, right=True, labelsize=10)
    axis.grid(linestyle="--", linewidth=0.45, alpha=0.35)
    axis.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#999999")
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory created by run_pds_differential_resolution.py.",
    )
    parser.add_argument("--apa", type=int, choices=(1, 2), required=True)
    parser.add_argument(
        "--fixed-momentum",
        type=int,
        choices=MOMENTA,
        default=5,
        help="Beam momentum for the direct sigma_D map; default: 5 GeV/c.",
    )
    parser.add_argument("--dpi", type=int, default=250)
    arguments = parser.parse_args()
    arguments.results_dir = arguments.results_dir.expanduser().resolve()
    if not arguments.results_dir.is_dir():
        parser.error(f"--results-dir does not exist or is not a directory: {arguments.results_dir}")
    if arguments.dpi <= 0:
        parser.error("--dpi must be positive")
    return arguments


def main() -> int:
    arguments = parse_arguments()
    results_dir = arguments.results_dir
    apa = arguments.apa
    fixed_momentum = arguments.fixed_momentum

    config_rows = read_csv(
        results_dir / "pair_configuration.csv",
        {
            "pair",
            "apa",
            "first_endpoint",
            "first_channel",
            "second_endpoint",
            "second_channel",
        },
    )
    width_rows = read_csv(
        results_dir / "adjacent_pair_differential_resolution.csv",
        {
            "pair",
            "momentum_GeV_c",
            "kinetic_mean_GeV",
            "effective_spread_GeV",
            "d_gaussian_status",
            "d_gaussian_sigma",
            "d_gaussian_sigma_error",
            "coverage_status",
        },
    )
    fit_rows = read_csv(
        results_dir / "pair_resolution_fit_results.csv",
        {
            "pair",
            "apa",
            "resolution_model",
            "resolution_fit_status",
            "resolution_fit_constant_a",
            "resolution_fit_constant_a_error",
            "resolution_fit_stochastic_b_sqrt_GeV",
            "resolution_fit_stochastic_b_error_sqrt_GeV",
        },
    )

    pairs = collect_pairs(config_rows, apa)
    widths = collect_widths(width_rows, apa)
    resolution_fits = collect_resolution_fits(fit_rows, apa, PRIMARY_MODEL)
    systematics = collect_systematics(
        results_dir / "pair_resolution_threshold_systematics.csv", apa, PRIMARY_MODEL
    )
    positions = map_channels(apa)

    fixed_width_values = {
        identifier: (
            float(widths[(identifier, fixed_momentum)]["sigma_D"]),
            float(widths[(identifier, fixed_momentum)]["sigma_D_statistical_error"]),
        )
        for identifier in pairs
        if (identifier, fixed_momentum) in widths
    }
    constant_a_values = {
        identifier: (
            float(record["constant_a"]),
            float(record["constant_a_statistical_error"]),
        )
        for identifier, record in resolution_fits.items()
    }
    stochastic_b_values = {
        identifier: (
            float(record["stochastic_b_sqrt_GeV"]),
            float(record["stochastic_b_statistical_error_sqrt_GeV"]),
        )
        for identifier, record in resolution_fits.items()
    }

    output_paths = {
        "sigma_D_at_fixed_momentum": (
            results_dir / f"apa{apa}_sigma_D_at_{fixed_momentum}GeV_pair_map.png"
        ),
        "constant_a": results_dir / f"apa{apa}_resolution_constant_a_pair_map.png",
        "stochastic_b": (
            results_dir / f"apa{apa}_resolution_stochastic_b_pair_map.png"
        ),
        "width_summary": (
            results_dir / f"apa{apa}_sigma_D_median_central68_vs_keff.png"
        ),
    }

    draw_pair_map(
        pairs,
        fixed_width_values,
        positions,
        apa,
        rf"Differential width $\sigma_D$ at {fixed_momentum} GeV/c [AU]",
        output_paths["sigma_D_at_fixed_momentum"],
        arguments.dpi,
    )
    draw_pair_map(
        pairs,
        constant_a_values,
        positions,
        apa,
        r"Constant term $a$ [AU]",
        output_paths["constant_a"],
        arguments.dpi,
    )
    draw_pair_map(
        pairs,
        stochastic_b_values,
        positions,
        apa,
        r"Stochastic term $b$ [$\sqrt{\mathrm{GeV}}$]",
        output_paths["stochastic_b"],
        arguments.dpi,
    )
    width_summary = draw_width_summary(
        widths.values(), apa, output_paths["width_summary"], arguments.dpi
    )

    map_rows: list[dict[str, object]] = []
    for identifier, pair in sorted(pairs.items()):
        fixed_width = widths.get((identifier, fixed_momentum), {})
        final_fit = resolution_fits.get(identifier, {})
        systematic = systematics.get(identifier, {})
        map_rows.append(
            {
                **pair,
                "fixed_momentum_GeV_c": fixed_momentum,
                "sigma_D_at_fixed_momentum": fixed_width.get("sigma_D", math.nan),
                "sigma_D_at_fixed_momentum_statistical_error": fixed_width.get(
                    "sigma_D_statistical_error", math.nan
                ),
                "sigma_D_at_fixed_momentum_coverage_status": fixed_width.get(
                    "coverage_status", ""
                ),
                "constant_a": final_fit.get("constant_a", math.nan),
                "constant_a_statistical_error": final_fit.get(
                    "constant_a_statistical_error", math.nan
                ),
                "constant_a_threshold_systematic": systematic.get(
                    "constant_a_threshold_systematic", math.nan
                ),
                "stochastic_b_sqrt_GeV": final_fit.get(
                    "stochastic_b_sqrt_GeV", math.nan
                ),
                "stochastic_b_statistical_error_sqrt_GeV": final_fit.get(
                    "stochastic_b_statistical_error_sqrt_GeV", math.nan
                ),
                "stochastic_b_threshold_systematic_sqrt_GeV": systematic.get(
                    "stochastic_b_sqrt_GeV_threshold_systematic", math.nan
                ),
                "resolution_fit_status": (
                    "success" if identifier in resolution_fits else "not_available"
                ),
            }
        )
    write_csv(results_dir / f"apa{apa}_pair_resolution_summary_map_values.csv", map_rows)
    write_csv(
        results_dir / f"apa{apa}_sigma_D_median_central68_vs_keff.csv",
        width_summary,
    )

    report = [
        "PDS DIFFERENTIAL-RESOLUTION SUMMARY PRODUCTS",
        f"APA: {apa}",
        f"Fixed direct-width map momentum: {fixed_momentum} GeV/c.",
        "",
        "MAP DEFINITIONS",
        "The sigma_D map uses the successful nominal Gaussian fit directly at the fixed momentum.",
        "The a and b maps use successful nominal two-term fits: sigma_D = sqrt(a^2 + b^2 / K_eff).",
        "Map uncertainties are statistical fit uncertainties. Threshold-selection systematics remain in the CSV output and the final-fit table.",
        "",
        "DESCRIPTIVE ENERGY SUMMARY",
        "At each momentum, the central marker is the median sigma_D of all successful pair measurements.",
        "The shaded central 68% interval is the 16th--84th percentile range between pairs.",
        "It is a pair-to-pair spatial spread, not a statistical uncertainty on the median and it is not fitted.",
        "",
        "OUTPUTS",
        *(path.name for path in output_paths.values()),
        f"apa{apa}_pair_resolution_summary_map_values.csv: values shown in the maps plus threshold-systematic envelopes.",
        f"apa{apa}_sigma_D_median_central68_vs_keff.csv: median and central-68% pair spread at each energy.",
    ]
    (results_dir / f"apa{apa}_pair_resolution_summary_report.txt").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
