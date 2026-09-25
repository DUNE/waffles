#!/usr/bin/env python3
"""Run the differential light-response study for adjacent PDS channel pairs.

The trigger selection is the same as the channel-by-channel calorimetric
linearity analysis.  At 2, 3, 5 and 7 GeV/c, the APA 1 average response must
be above the Langauss--Gaussian intersection read from the current
population-fit CSV.  Threshold systematics use T - sigma_T, T and T + sigma_T
separately at each momentum.  The 1 GeV/c sample can only be included as an
unselected diagnostic because its non-muon-like population is not separated.

The primary result is a differential relative light-response width, not an
intrinsic calorimetric energy resolution.
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
import numpy as np
import pandas as pd

from pds_differential_resolution import (
    Channel,
    Pair,
    extract_pair_events,
    load_merged_json,
    measure_pair,
    plot_pair_diagnostics,
    poisson_width,
    selected_event_indices,
)
from utils import adjacent_channel_info


BASE_MOMENTA = (2, 3, 5, 7)
MASS_GEV = {
    "e": 0.00051099895,
    "k": 0.493677,
    "p": 0.938272088,
    "pi": 0.13957039,
}


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


def scenario_name(multiplier: float) -> str:
    if math.isclose(multiplier, 0.0):
        return "nominal"
    sign = "plus" if multiplier > 0 else "minus"
    return f"{sign}_{abs(multiplier):g}".replace(".", "p") + "sigma"


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


def load_trigger_contexts(path: Path, momenta: tuple[int, ...]) -> dict[int, dict[tuple[str, int], dict]]:
    required = {
        "momentum_GeV_c", "block", "trigger_time", "apa1_mean",
        "apa1_valid", "apa2_valid",
    }
    contexts = {momentum: {} for momentum in momenta}
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


def load_thresholds(
    path: Path, momenta: tuple[int, ...], multipliers: list[float]
) -> tuple[dict[int, tuple[float, float]], list[dict]]:
    rows_by_momentum: dict[int, dict] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, strict=True)
        required = {"momentum_GeV_c", "model", "status", "intersection", "intersection_error"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Missing columns in population-fit results: {missing}")
        for row in reader:
            momentum = exact_integer(row["momentum_GeV_c"])
            if momentum in momenta:
                rows_by_momentum[momentum] = row

    thresholds: dict[int, tuple[float, float]] = {}
    output_rows: list[dict] = []
    for momentum in momenta:
        if momentum == 1:
            for multiplier in multipliers:
                if not math.isclose(multiplier, 0.0):
                    continue
                output_rows.append({
                    "threshold_scenario": scenario_name(multiplier),
                    "threshold_sigma_multiplier": multiplier,
                    "momentum_GeV_c": momentum,
                    "selection_kind": "unselected_apa_local_valid",
                    "threshold_nominal_PE": math.nan,
                    "threshold_error_PE": math.nan,
                    "threshold_applied_PE": math.nan,
                })
            continue
        row = rows_by_momentum.get(momentum)
        if row is None or row["status"] != "success" or row["model"] != "langauss_plus_gaussian":
            raise ValueError(f"No valid Langauss--Gaussian intersection at {momentum} GeV/c.")
        threshold = finite_float(row["intersection"])
        error = finite_float(row["intersection_error"])
        if error <= 0:
            raise ValueError(f"Non-positive intersection uncertainty at {momentum} GeV/c.")
        thresholds[momentum] = (threshold, error)
        for multiplier in multipliers:
            output_rows.append({
                "threshold_scenario": scenario_name(multiplier),
                "threshold_sigma_multiplier": multiplier,
                "momentum_GeV_c": momentum,
                "selection_kind": "apa1_mean_greater_than_threshold",
                "threshold_nominal_PE": threshold,
                "threshold_error_PE": error,
                "threshold_applied_PE": threshold + multiplier * error,
            })
    return thresholds, output_rows


def load_kinetic_energies(path: Path, momenta: tuple[int, ...], relative_momentum_error: float) -> dict[int, dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, strict=True)
        required = {"Momentum [GeV/c]"} | {f"{species} [Hz]" for species in MASS_GEV}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Missing columns in beam-composition file: {missing}")
        composition = {exact_integer(row["Momentum [GeV/c]"]): row for row in reader}

    masses = np.asarray(list(MASS_GEV.values()), dtype=float)
    result: dict[int, dict] = {}
    for momentum in momenta:
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
    apa1_pairs, apa2_pairs, _ = adjacent_channel_info()
    raw_pairs = apa1_pairs if apa == 1 else apa2_pairs
    pairs: list[Pair] = []
    for raw_pair in raw_pairs:
        first = Channel.from_mapping(raw_pair[0])
        second = Channel.from_mapping(raw_pair[1])
        label = f"apa{apa}_end{first.endpoint}_ch{first.channel}__end{second.endpoint}_ch{second.channel}"
        pairs.append(Pair(first, second, label=label, kind="adjacent_candidate"))
    return pairs


def watermark(axis) -> None:
    axis.text(
        0.98, 0.97, r"$\mathbf{ProtoDUNE\!-\!HD}$" + "\nWork in Progress",
        transform=axis.transAxes, ha="right", va="top", fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )


def make_summary_plot(table: pd.DataFrame, output: Path) -> None:
    """Plot the central-68% D_AB width against the Poisson prediction."""

    nominal = table.loc[(table["threshold_scenario"] == "nominal") & (table["measurement_status"] == "success")]
    if nominal.empty:
        return
    figure, axis = plt.subplots(figsize=(7.5, 5.4))
    for momentum, group in nominal.groupby("momentum_GeV_c"):
        axis.errorbar(
            group["n_eff_PE"], group["d_central68_half_width"],
            yerr=group["d_bootstrap_central68_half_width"], fmt="o", ms=5.5,
            capsize=2.5, color="#0072B2", label=rf"{momentum:g} GeV/c",
        )
    low = max(float(nominal["n_eff_PE"].min()) * 0.85, 1.0e-3)
    high = float(nominal["n_eff_PE"].max()) * 1.20
    x_values = np.geomspace(low, high, 300)
    axis.plot(x_values, poisson_width(x_values), "--", color="#333333", lw=1.8, label=r"Independent Poisson: $1/\sqrt{N_{\rm eff}}$")
    axis.set(xscale="log", yscale="log", xlabel=r"$N_{\rm eff}$ [PE]", ylabel=r"Central 68% half-width of $D_{AB}$")
    axis.grid(which="both", alpha=0.28)
    axis.legend(frameon=True, facecolor="white", fontsize=8)
    watermark(axis)
    figure.tight_layout()
    figure.savefig(output, dpi=300)
    plt.close(figure)


def build_systematics(table: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for identifiers, group in table.groupby(["pair", "kind", "momentum_GeV_c"], dropna=False):
        central = group.loc[(group["threshold_scenario"] == "nominal") & (group["measurement_status"] == "success")]
        if central.empty:
            continue
        nominal = central.iloc[0]
        row = {
            "pair": identifiers[0], "kind": identifiers[1], "momentum_GeV_c": identifiers[2],
            "nominal_d_central68_half_width": nominal["d_central68_half_width"],
            "nominal_d_standard_deviation": nominal["d_standard_deviation"],
            "minus_d_central68_half_width": math.nan,
            "plus_d_central68_half_width": math.nan,
            "d_central68_threshold_systematic": math.nan,
            "minus_d_standard_deviation": math.nan,
            "plus_d_standard_deviation": math.nan,
            "d_standard_deviation_threshold_systematic": math.nan,
        }
        for direction in ("minus", "plus"):
            shifted = group.loc[(group["threshold_scenario"] == f"{direction}_1sigma") & (group["measurement_status"] == "success")]
            if not shifted.empty:
                value = shifted.iloc[0]
                row[f"{direction}_d_central68_half_width"] = value["d_central68_half_width"]
                row[f"{direction}_d_standard_deviation"] = value["d_standard_deviation"]
        values = [
            abs(row[name] - row["nominal_d_central68_half_width"])
            for name in ("minus_d_central68_half_width", "plus_d_central68_half_width") if math.isfinite(row[name])
        ]
        deviations = [
            abs(row[name] - row["nominal_d_standard_deviation"])
            for name in ("minus_d_standard_deviation", "plus_d_standard_deviation") if math.isfinite(row[name])
        ]
        if values:
            row["d_central68_threshold_systematic"] = max(values)
        if deviations:
            row["d_standard_deviation_threshold_systematic"] = max(deviations)
        rows.append(row)
    return rows


def clear_outputs(output_dir: Path) -> None:
    for name in (
        "adjacent_pair_differential_resolution.csv", "pair_availability.csv",
        "pair_configuration.csv", "selection_thresholds.csv", "selected_trigger_counts.csv",
        "pair_threshold_systematics.csv", "differential_width_vs_neff.png",
        "report.txt", "manifest.json",
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
    parser.add_argument("--include-1gev", action="store_true")
    parser.add_argument("--threshold-sigma-multipliers", nargs="+", type=float, default=[-1.0, 0.0, 1.0])
    parser.add_argument("--relative-momentum-error", type=float, default=0.05)
    parser.add_argument("--minimum-events", type=int, default=150)
    parser.add_argument("--minimum-common-fraction", type=float, default=0.90)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--diagnostic-pairs", type=int, default=4)
    arguments = parser.parse_args()
    for attribute in ("input_dir", "trigger_data", "population_fit_results", "composition", "output_dir"):
        setattr(arguments, attribute, getattr(arguments, attribute).expanduser().resolve())
    if arguments.minimum_events < 10:
        parser.error("--minimum-events must be at least 10")
    if not 0.0 < arguments.minimum_common_fraction <= 1.0:
        parser.error("--minimum-common-fraction must be in (0, 1]")
    if arguments.bootstrap < 2:
        parser.error("--bootstrap must be at least 2")
    if arguments.relative_momentum_error < 0:
        parser.error("--relative-momentum-error must be non-negative")
    if not arguments.threshold_sigma_multipliers or len(set(arguments.threshold_sigma_multipliers)) != len(arguments.threshold_sigma_multipliers):
        parser.error("Specify distinct threshold-sigma multipliers")
    for path in (arguments.input_dir, arguments.trigger_data, arguments.population_fit_results, arguments.composition):
        if not path.exists():
            parser.error(f"Input does not exist: {path}")
    if arguments.output_dir == arguments.input_dir:
        parser.error("--output-dir must differ from --input-dir")
    return arguments


def main() -> int:
    arguments = parse_arguments()
    momenta = ((1,) if arguments.include_1gev else ()) + BASE_MOMENTA
    contexts = load_trigger_contexts(arguments.trigger_data, momenta)
    thresholds, threshold_rows = load_thresholds(arguments.population_fit_results, momenta, arguments.threshold_sigma_multipliers)
    kinetic = load_kinetic_energies(arguments.composition, momenta, arguments.relative_momentum_error)
    data_by_momentum = {
        momentum: load_merged_json(momentum, arguments.input_dir / f"{momentum}GeV")
        for momentum in momenta
    }
    pairs = default_pairs(arguments.apa)
    if not pairs:
        raise ValueError(f"No adjacent candidate pairs defined for APA {arguments.apa}.")

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    clear_outputs(arguments.output_dir)
    diagnostics_dir = arguments.output_dir / "diagnostics"
    diagnostics_dir.mkdir(exist_ok=True)
    diagnostic_ids = {pair.identifier for pair in pairs[:arguments.diagnostic_pairs]}
    scenarios = [(scenario_name(multiplier), multiplier) for multiplier in arguments.threshold_sigma_multipliers]
    scenarios.sort(key=lambda item: (not math.isclose(item[1], 0.0), item[1]))

    measurements: list[dict] = []
    availability_rows: list[dict] = []
    selected_rows: list[dict] = []
    input_paths = [arguments.trigger_data, arguments.population_fit_results, arguments.composition]
    for momentum, blocks in data_by_momentum.items():
        input_paths.extend(arguments.input_dir / f"{momentum}GeV" / block / f"photoelectron_dic_{momentum}GeV.json" for block in blocks)

    for scenario, multiplier in scenarios:
        for momentum in momenta:
            if momentum == 1 and not math.isclose(multiplier, 0.0):
                continue
            threshold, threshold_error = thresholds.get(momentum, (math.nan, math.nan))
            applied = math.nan if momentum == 1 else threshold + multiplier * threshold_error
            selected = selected_event_indices(
                data_by_momentum[momentum], contexts[momentum], momentum, arguments.apa,
                None if momentum == 1 else applied,
            )
            selected_rows.append({
                "threshold_scenario": scenario,
                "threshold_sigma_multiplier": multiplier,
                "momentum_GeV_c": momentum,
                "selection_kind": "unselected_apa_local_valid" if momentum == 1 else "apa1_mean_greater_than_threshold",
                "threshold_nominal_PE": threshold,
                "threshold_error_PE": threshold_error,
                "threshold_applied_PE": applied,
                "selected_triggers": int(sum(len(indices) for indices in selected.values())),
            })
            for pair_number, pair in enumerate(pairs):
                events = extract_pair_events(data_by_momentum[momentum], pair, selected)
                base = {
                    "threshold_scenario": scenario,
                    "threshold_sigma_multiplier": multiplier,
                    "selection_kind": "unselected_apa_local_valid" if momentum == 1 else "apa1_mean_greater_than_threshold",
                    "threshold_nominal_PE": threshold,
                    "threshold_error_PE": threshold_error,
                    "threshold_applied_PE": applied,
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
                    "message": "",
                }
                try:
                    measurement = measure_pair(
                        pair, momentum, events,
                        min_events=arguments.minimum_events,
                        minimum_common_fraction=arguments.minimum_common_fraction,
                        bootstrap_repetitions=arguments.bootstrap,
                        seed=10_000_000 * arguments.apa + 100_000 * momentum + pair_number,
                    )
                except ValueError as error:
                    availability.update(measurement_status="not_measured", message=str(error))
                    availability_rows.append(availability)
                    continue
                row = {**base, **measurement.as_flat_dict(), "measurement_status": "success", "message": ""}
                measurements.append(row)
                availability.update(measurement_status="success")
                availability_rows.append(availability)
                if scenario == "nominal" and pair.identifier in diagnostic_ids:
                    plot_pair_diagnostics(events, pair, momentum, diagnostics_dir / f"apa{arguments.apa}_{momentum}GeV_{pair.identifier}.png")

    if not measurements:
        raise RuntimeError("No pair passed the common-sample requirements.")
    table = pd.DataFrame(measurements)
    write_csv(arguments.output_dir / "adjacent_pair_differential_resolution.csv", measurements)
    write_csv(arguments.output_dir / "pair_availability.csv", availability_rows)
    write_csv(arguments.output_dir / "selection_thresholds.csv", threshold_rows)
    write_csv(arguments.output_dir / "selected_trigger_counts.csv", selected_rows)
    write_csv(arguments.output_dir / "pair_configuration.csv", [
        {
            "pair": pair.identifier, "kind": pair.kind, "apa": pair.first.apa,
            "first_endpoint": pair.first.endpoint, "first_channel": pair.first.channel,
            "second_endpoint": pair.second.endpoint, "second_channel": pair.second.channel,
        }
        for pair in pairs
    ])
    write_csv(arguments.output_dir / "pair_threshold_systematics.csv", build_systematics(table))
    make_summary_plot(table, arguments.output_dir / "differential_width_vs_neff.png")

    nominal_successes = int(np.count_nonzero((table["threshold_scenario"] == "nominal") & (table["measurement_status"] == "success")))
    report = [
        "PDS DIFFERENTIAL LIGHT-RESPONSE STUDY",
        f"APA: {arguments.apa}",
        f"Momenta [GeV/c]: {list(momenta)}",
        f"Minimum common events: {arguments.minimum_events}",
        f"Minimum common-event fraction: {arguments.minimum_common_fraction:.3f}",
        f"Bootstrap replicas: {arguments.bootstrap}",
        f"Threshold scenarios: {', '.join(name for name, _ in scenarios)}",
        "",
        "PRIMARY OBSERVABLE",
        "D_AB = (NPE_A/mu_A - NPE_B/mu_B)/sqrt(2), evaluated on common selected triggers.",
        "The central-68% half-width is the primary robust differential light-response width.",
        "This is not an intrinsic calorimetric energy resolution without further assumptions.",
        "",
        "SELECTION AND QUALITY",
        "At 2--7 GeV/c: APA1 mean > current Langauss--Gaussian intersection.",
        "The +/-1 sigma scenarios use the momentum-specific intersection uncertainty.",
        "At 1 GeV/c: unselected APA-local-valid triggers; diagnostic only.",
        "Missing channel values are not replaced by zero; availability is written separately.",
        f"Nominal successful pair/momentum measurements: {nominal_successes}.",
        "",
        "OUTPUTS",
        "adjacent_pair_differential_resolution.csv: primary measurements and bootstrap errors.",
        "pair_availability.csv: common-sample coverage for every attempted pair.",
        "pair_threshold_systematics.csv: T +/- sigma_T envelope.",
        "differential_width_vs_neff.png: comparison with independent Poisson counting.",
        f"Diagnostics: {diagnostics_dir}",
    ]
    (arguments.output_dir / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": sys.version,
        "configuration": {
            "apa": arguments.apa,
            "momenta_GeV_c": list(momenta),
            "minimum_events": arguments.minimum_events,
            "minimum_common_fraction": arguments.minimum_common_fraction,
            "bootstrap": arguments.bootstrap,
            "relative_momentum_error": arguments.relative_momentum_error,
            "threshold_sigma_multipliers": arguments.threshold_sigma_multipliers,
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
