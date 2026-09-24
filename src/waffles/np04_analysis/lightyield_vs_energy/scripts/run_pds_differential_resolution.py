"""Run the adjacent-channel differential-resolution study on final JSON files.

Example (from the ``scripts`` directory in the CERN waffles environment)::

    python run_pds_differential_resolution.py \
      --input-folder /path/to/output/apa1_vs_apa2 \
      --output-folder /path/to/output/differential_resolution

The default run uses only 2, 3, 5 and 7 GeV/c because the 1 GeV/c sample does
not use the same non-muon-like selection.  It can be included explicitly as a
diagnostic with ``--include-1gev``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pds_differential_resolution import (
    Channel,
    Pair,
    extract_pair_events,
    load_merged_json,
    measure_pair,
    measurements_table,
    plot_pair_diagnostics,
    poisson_width,
    selected_event_indices,
)
from utils import adjacent_channel_info


NOMINAL_THRESHOLDS = {1: 0.0, 2: 118.0, 3: 119.0, 5: 175.0, 7: 253.0}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-folder", type=Path, required=True, help="Directory containing <p>GeV/<range>/ JSON files.")
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--apa", type=int, default=1)
    parser.add_argument("--include-1gev", action="store_true", help="Add the differently selected 1 GeV/c sample as a diagnostic.")
    parser.add_argument("--threshold-shifts", type=float, nargs="*", default=[0.0], help="Absolute NPE shifts used for selection systematic checks.")
    parser.add_argument("--bootstrap", type=int, default=500, help="Number of event-pair bootstrap replicas.")
    parser.add_argument("--min-events", type=int, default=100)
    parser.add_argument("--diagnostic-pairs", type=int, default=3, help="Number of valid pairs for which to save detailed plots per momentum.")
    return parser.parse_args()


def default_pairs(apa: int) -> list[Pair]:
    apa1_pairs, apa2_pairs, _ = adjacent_channel_info()
    raw_pairs = apa1_pairs if apa == 1 else apa2_pairs
    return [
        Pair(Channel.from_mapping(raw_pair[0]), Channel.from_mapping(raw_pair[1]))
        for raw_pair in raw_pairs
    ]


def make_summary_plot(table: pd.DataFrame, output: Path) -> None:
    """Plot the central-68% D width against the Poisson counting prediction."""

    nominal = table.loc[table["threshold_shift_npe"] == 0.0]
    figure, axis = plt.subplots(figsize=(7, 5))
    for momentum, group in nominal.groupby("momentum_gev"):
        axis.errorbar(
            group["n_eff"],
            group["d_central68_half_width"],
            yerr=group["d_bootstrap_central68_half_width"],
            fmt="o",
            capsize=2,
            label=rf"{momentum:g} GeV/c",
        )
    x_values = np.geomspace(nominal["n_eff"].min() * 0.9, nominal["n_eff"].max() * 1.1, 200)
    axis.plot(x_values, poisson_width(x_values), "k--", label=r"independent Poisson: $1/\sqrt{N_{\rm eff}}$")
    axis.set(xscale="log", yscale="log", xlabel=r"$N_{\rm eff}$", ylabel=r"Central 68% half-width of $D_{AB}$")
    axis.grid(which="both", alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    arguments.output_folder.mkdir(parents=True, exist_ok=True)
    diagnostics_directory = arguments.output_folder / "diagnostics"
    diagnostics_directory.mkdir(exist_ok=True)

    momenta = [2, 3, 5, 7]
    if arguments.include_1gev:
        momenta.insert(0, 1)
    pairs = default_pairs(arguments.apa)
    all_rows: list[pd.DataFrame] = []

    for momentum in momenta:
        merged_data = load_merged_json(momentum, arguments.input_folder / f"{momentum}GeV")
        for threshold_shift in arguments.threshold_shifts:
            threshold = NOMINAL_THRESHOLDS[momentum] + threshold_shift
            measurements = []
            diagnostics_saved = 0
            for pair_number, pair in enumerate(pairs):
                try:
                    measurement = measure_pair(
                        merged_data,
                        pair,
                        momentum,
                        threshold,
                        min_events=arguments.min_events,
                        bootstrap_repetitions=arguments.bootstrap,
                        seed=10000 * momentum + pair_number,
                    )
                except ValueError as error:
                    print(f"Skipping {pair.identifier} at {momentum} GeV/c: {error}")
                    continue
                measurements.append(measurement)

                if threshold_shift == 0.0 and diagnostics_saved < arguments.diagnostic_pairs:
                    selected = selected_event_indices(merged_data, threshold)
                    values_a, values_b = extract_pair_events(merged_data, pair, selected)
                    plot_pair_diagnostics(
                        values_a,
                        values_b,
                        pair,
                        momentum,
                        diagnostics_directory / f"{momentum}GeV_{pair.identifier}.png",
                    )
                    diagnostics_saved += 1

            table = measurements_table(measurements)
            table["selection_threshold_npe"] = threshold
            table["threshold_shift_npe"] = threshold_shift
            all_rows.append(table)

    if not all_rows:
        raise RuntimeError("No valid pair measurement was produced.")
    result = pd.concat(all_rows, ignore_index=True)
    result.to_csv(arguments.output_folder / "adjacent_pair_differential_resolution.csv", index=False)
    make_summary_plot(result, arguments.output_folder / "differential_width_vs_neff.png")
    print(f"Wrote {len(result)} pair measurements to {arguments.output_folder}")


if __name__ == "__main__":
    main()
