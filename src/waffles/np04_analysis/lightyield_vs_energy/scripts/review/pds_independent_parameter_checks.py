#!/usr/bin/env python3
"""Independent checks of differential-resolution fit parameters for APA 1.

The script compares measured Gaussian widths of D_AB with (i) the expected
independent photoelectron-counting width and (ii) the independently calibrated
electronic-noise width. It does not refit the nominal resolution model.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MOMENTA = (1, 2, 3, 5, 7)
MOMENTUM_COLORS = {
    1: "#d55e00",
    2: "#0072b2",
    3: "#009e73",
    5: "#cc79a7",
    7: "#56b4e9",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def number(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_calibration(path: Path, batch: str, hpk_ov: float,
                     fbk_ov: float) -> dict[tuple[str, str], dict]:
    selected: dict[tuple[str, str], dict] = {}
    duplicates: list[tuple[str, str]] = []
    for row in read_csv(path):
        if row.get("batch") != batch or row.get("APA") != "1":
            continue
        vendor = row.get("vendor", "").strip().upper()
        ov = number(row, "OV_V")
        expected_ov = hpk_ov if vendor == "HPK" else fbk_ov if vendor == "FBK" else math.nan
        if not math.isfinite(expected_ov) or not math.isclose(ov, expected_ov, abs_tol=1e-6):
            continue
        key = (row.get("endpoint", "").strip(), row.get("channel", "").strip())
        if key in selected:
            duplicates.append(key)
        selected[key] = row
    if duplicates:
        raise ValueError(f"More than one calibration row selected for channels: {sorted(set(duplicates))}")
    if not selected:
        raise ValueError("No APA 1 calibration rows match the requested batch and overvoltages.")
    return selected


def calibration_noise_pe(row: dict[str, str]) -> tuple[float, float]:
    """Return pedestal width and propagated error in PE, assuming std_0/gain."""
    std0, std0_error = number(row, "std_0"), number(row, "std_0_error")
    gain, gain_error = number(row, "gain"), number(row, "gain_error")
    if min(std0, gain) <= 0 or not all(map(math.isfinite, (std0_error, gain_error))):
        return math.nan, math.nan
    sigma_pe = std0 / gain
    relative_error = math.hypot(std0_error / std0, gain_error / gain)
    return sigma_pe, sigma_pe * relative_error


def noise_width(noise_a: float, noise_a_error: float, noise_b: float,
                noise_b_error: float, mean_a: float, mean_b: float) -> tuple[float, float]:
    """Electronic-noise contribution to D_AB and calibration-only uncertainty."""
    if min(noise_a, noise_b, mean_a, mean_b) <= 0:
        return math.nan, math.nan
    sigma = math.sqrt(0.5 * ((noise_a / mean_a) ** 2 + (noise_b / mean_b) ** 2))
    error = math.sqrt(
        ((noise_a / mean_a) ** 4 * (noise_a_error / noise_a) ** 2
         + (noise_b / mean_b) ** 4 * (noise_b_error / noise_b) ** 2)
    ) / (2.0 * sigma)
    return sigma, error


def build_results(measurement_path: Path, fit_path: Path,
                  calibration: dict[tuple[str, str], dict]) -> tuple[list[dict], list[dict]]:
    fits = {
        row["pair"]: row for row in read_csv(fit_path)
        if row.get("threshold_scenario") == "nominal"
        and row.get("resolution_fit_model") == "two_term"
        and row.get("resolution_fit_status") == "success"
    }
    measurements = [
        row for row in read_csv(measurement_path)
        if row.get("threshold_scenario") == "nominal"
        and row.get("kind") == "manual_adjacent_pair"
        and row.get("measurement_status") == "success"
        and row.get("d_gaussian_status") == "success"
    ]

    point_rows: list[dict] = []
    pair_points: dict[str, list[dict]] = defaultdict(list)
    pair_fit = {key: row for key, row in fits.items()}
    used_pairs: set[str] = set()
    missing_calibrations: set[tuple[str, str]] = set()
    for row in measurements:
        pair = row["pair"]
        fit = fits.get(pair)
        if fit is None:
            continue
        channel_a = (fit["first_endpoint"], fit["first_channel"])
        channel_b = (fit["second_endpoint"], fit["second_channel"])
        cal_a, cal_b = calibration.get(channel_a), calibration.get(channel_b)
        if cal_a is None or cal_b is None:
            if cal_a is None:
                missing_calibrations.add(channel_a)
            if cal_b is None:
                missing_calibrations.add(channel_b)
            continue

        momentum = int(float(row["momentum_GeV_c"]))
        if momentum not in MOMENTA:
            continue
        mean_a, mean_b = number(row, "mean_a_PE"), number(row, "mean_b_PE")
        sigma_observed = number(row, "d_gaussian_sigma")
        sigma_observed_error = number(row, "d_gaussian_sigma_error")
        kinetic_energy = number(row, "kinetic_mean_GeV")
        if min(mean_a, mean_b, sigma_observed, kinetic_energy) <= 0:
            continue

        noise_a, noise_a_error = calibration_noise_pe(cal_a)
        noise_b, noise_b_error = calibration_noise_pe(cal_b)
        sigma_poisson = math.sqrt(0.5 * (1.0 / mean_a + 1.0 / mean_b))
        sigma_electronic, sigma_electronic_error = noise_width(
            noise_a, noise_a_error, noise_b, noise_b_error, mean_a, mean_b,
        )
        out = {
            "pair": pair,
            "first_endpoint": channel_a[0], "first_channel": channel_a[1],
            "second_endpoint": channel_b[0], "second_channel": channel_b[1],
            "momentum_GeV_c": momentum,
            "kinetic_mean_GeV": kinetic_energy,
            "selected_triggers": row.get("selected_triggers", ""),
            "common_events": row.get("common_events", ""),
            "mean_a_PE": mean_a, "mean_b_PE": mean_b,
            "sigma_D_observed_AU": sigma_observed,
            "sigma_D_observed_error_AU": sigma_observed_error,
            "sigma_D_poisson_AU": sigma_poisson,
            "observed_over_poisson": sigma_observed / sigma_poisson,
            "noise_A_PE": noise_a, "noise_A_error_PE": noise_a_error,
            "noise_B_PE": noise_b, "noise_B_error_PE": noise_b_error,
            "sigma_D_electronic_noise_AU": sigma_electronic,
            "sigma_D_electronic_noise_calibration_error_AU": sigma_electronic_error,
            "electronic_noise_over_observed": sigma_electronic / sigma_observed,
            "electronic_noise_variance_fraction": (sigma_electronic / sigma_observed) ** 2,
            "b_poisson_equivalent_sqrt_GeV": math.sqrt(kinetic_energy) * sigma_poisson,
            "c_noise_equivalent_GeV": kinetic_energy * sigma_electronic,
            "calibration_batch": cal_a["batch"],
            "calibration_date_A": cal_a.get("date", ""),
            "calibration_date_B": cal_b.get("date", ""),
            "first_vendor": cal_a.get("vendor", ""), "first_OV_V": cal_a.get("OV_V", ""),
            "second_vendor": cal_b.get("vendor", ""), "second_OV_V": cal_b.get("OV_V", ""),
        }
        point_rows.append(out)
        pair_points[pair].append(out)
        used_pairs.add(pair)

    if missing_calibrations:
        raise ValueError(
            "Calibration is missing for fit-success channels: "
            + ", ".join(f"END {e} CH {c}" for e, c in sorted(missing_calibrations))
        )
    if not point_rows:
        raise ValueError("No successful nominal pair measurements could be matched to calibrations.")

    summary_rows: list[dict] = []
    for pair in sorted(used_pairs):
        fit = pair_fit[pair]
        points = sorted(pair_points[pair], key=lambda x: x["momentum_GeV_c"])
        b_poisson = [p["b_poisson_equivalent_sqrt_GeV"] for p in points]
        c_noise = [p["c_noise_equivalent_GeV"] for p in points]
        noise_fractions = [p["electronic_noise_over_observed"] for p in points]
        summary_rows.append({
            "pair": pair,
            "first_endpoint": fit["first_endpoint"], "first_channel": fit["first_channel"],
            "second_endpoint": fit["second_endpoint"], "second_channel": fit["second_channel"],
            "n_momenta": len(points),
            "resolution_fit_constant_a_AU": number(fit, "resolution_fit_constant_a"),
            "resolution_fit_constant_a_error_AU": number(fit, "resolution_fit_constant_a_error"),
            "resolution_fit_stochastic_b_sqrt_GeV": number(fit, "resolution_fit_stochastic_b_sqrt_GeV"),
            "resolution_fit_stochastic_b_error_sqrt_GeV": number(fit, "resolution_fit_stochastic_b_error_sqrt_GeV"),
            "b_poisson_equivalent_median_sqrt_GeV": float(np.median(b_poisson)),
            "b_poisson_equivalent_min_sqrt_GeV": min(b_poisson),
            "b_poisson_equivalent_max_sqrt_GeV": max(b_poisson),
            "fit_b_over_poisson_b_median": number(fit, "resolution_fit_stochastic_b_sqrt_GeV") / float(np.median(b_poisson)),
            "c_noise_equivalent_median_GeV": float(np.median(c_noise)),
            "c_noise_equivalent_min_GeV": min(c_noise),
            "c_noise_equivalent_max_GeV": max(c_noise),
            "electronic_noise_over_observed_median": float(np.median(noise_fractions)),
            "resolution_fit_chi2_ndf": number(fit, "resolution_fit_chi2_ndf"),
            "independent_constant_term_check": "not_available_from_current_inputs",
        })
    return point_rows, summary_rows


def make_plot(output_path: Path, point_rows: list[dict], summary_rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for momentum in MOMENTA:
        rows = [r for r in point_rows if r["momentum_GeV_c"] == momentum]
        color = MOMENTUM_COLORS[momentum]
        axes[0].scatter([r["kinetic_mean_GeV"] for r in rows],
                        [r["observed_over_poisson"] for r in rows],
                        s=27, alpha=0.75, color=color, label=f"{momentum} GeV/c")
        axes[1].scatter([r["kinetic_mean_GeV"] for r in rows],
                        [100 * r["electronic_noise_over_observed"] for r in rows],
                        s=27, alpha=0.75, color=color, label=f"{momentum} GeV/c")
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1.1,
                    label="Poisson expectation")
    axes[0].set(xlabel=r"$K_{\rm eff}$ [GeV]", ylabel=r"Measured / Poisson $\sigma_D$")
    axes[0].legend(frameon=True, fontsize=8)
    axes[1].set(xlabel=r"$K_{\rm eff}$ [GeV]", ylabel="Electronic-noise contribution / measured width [%]")
    axes[1].set_ylim(bottom=0)
    axes[1].legend(frameon=True, fontsize=8, title="Beam momentum")

    xs = [r["b_poisson_equivalent_median_sqrt_GeV"] for r in summary_rows]
    ys = [r["resolution_fit_stochastic_b_sqrt_GeV"] for r in summary_rows]
    yerr = [r["resolution_fit_stochastic_b_error_sqrt_GeV"] for r in summary_rows]
    axes[2].errorbar(xs, ys, yerr=yerr, fmt="o", color="#0072b2", capsize=3, alpha=0.8)
    max_axis = max(xs + [y + e for y, e in zip(ys, yerr) if math.isfinite(e)])
    axes[2].plot([0, max_axis * 1.05], [0, max_axis * 1.05], "k--", linewidth=1,
                 label=r"$b_{\rm fit}=b_{\rm Poisson}$")
    axes[2].set(xlim=(0, max_axis * 1.05), ylim=(0, max_axis * 1.05),
                xlabel=r"Median Poisson-equivalent $b$ [AU$\sqrt{\rm GeV}$]",
                ylabel=r"Fitted $b$ [AU$\sqrt{\rm GeV}$]")
    axes[2].legend(frameon=True, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.text(0.98, 0.98, "ProtoDUNE-HD Work in Progress", transform=axis.transAxes,
                  ha="right", va="top", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True,
                        help="adjacent_pair_differential_resolution.csv")
    parser.add_argument("--fits", type=Path, required=True,
                        help="pair_resolution_fit_results.csv")
    parser.add_argument("--calibration", type=Path, required=True,
                        help="calibration_results.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch", default="1")
    parser.add_argument("--hpk-ov", type=float, default=3.0)
    parser.add_argument("--fbk-ov", type=float, default=4.5)
    args = parser.parse_args()
    for path in (args.measurements, args.fits, args.calibration):
        if not path.is_file():
            parser.error(f"input file does not exist: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    calibration = load_calibration(args.calibration, args.batch, args.hpk_ov, args.fbk_ov)
    points, summaries = build_results(args.measurements, args.fits, calibration)
    point_columns = list(points[0])
    summary_columns = list(summaries[0])
    write_csv(args.output_dir / "pds_independent_parameter_checks.csv", points, point_columns)
    write_csv(args.output_dir / "pds_independent_parameter_summary.csv", summaries, summary_columns)
    make_plot(args.output_dir / "pds_independent_parameter_checks.png", points, summaries)
    print(f"Selected calibration channels: {len(calibration)} (batch {args.batch}; "
          f"HPK OV={args.hpk_ov:g} V; FBK OV={args.fbk_ov:g} V).")
    print(f"Matched nominal measurements: {len(points)} across {len(summaries)} channel pairs.")
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
