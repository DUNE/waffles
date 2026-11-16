#!/usr/bin/env python3
"""Replot the 1 GeV/c time-offset distributions from the saved waveform sets.

The historic thesis PNGs came from the 1GeV/0_to_10 block. Its FS_wfset.pkl
and ST_wfset.pkl contain the selected waveforms and the `finding_peaks`
analysis used for those plots. No peak finding or trigger reselection is done
here: the stored time offsets are plotted directly.

Only load trusted pickle files. Pickle loading can execute arbitrary code.
"""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPECTED_OLD_COUNTS = {1: 752281, 2: 527}
BEAM_WINDOW = (-40, -20)


def load_offsets(path: Path, analysis_label: str):
    with path.open("rb") as stream:
        waveform_set = pickle.load(stream)  # Trusted, user-produced input only.
    waveforms = waveform_set.waveforms
    pieces = []
    for index, waveform in enumerate(waveforms):
        try:
            result = waveform.analyses[analysis_label].result
            values = result["time_offset"]
        except (AttributeError, KeyError, TypeError) as exc:
            raise ValueError(
                f"Missing {analysis_label!r} time offsets in {path}, waveform {index}"
            ) from exc
        array = np.asarray(values, dtype=float).ravel()
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite time offset in {path}, waveform {index}")
        if array.size:
            pieces.append(array)
    if not pieces:
        raise ValueError(f"No time offsets found in {path}")
    return np.concatenate(pieces), len(waveforms)


def add_work_in_progress(axis):
    axis.text(
        0.98, 0.97, "ProtoDUNE-HD", transform=axis.transAxes,
        ha="right", va="top", fontsize=10, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9),
    )
    axis.text(
        0.98, 0.91, "Work in Progress", transform=axis.transAxes,
        ha="right", va="top", fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9),
    )


def plot_distribution(offsets: np.ndarray, apa: int, output_path: Path):
    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    })
    fig, full = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
    full_bins = 10000 if apa == 1 else 1000
    full_counts, full_edges = np.histogram(offsets, bins=full_bins)
    full.stairs(
        full_counts, full_edges, fill=True, color="#72B7D2",
        edgecolor="#253746", linewidth=0.35, zorder=1,
    )
    full.axvspan(
        *BEAM_WINDOW, color="#F0E442", alpha=0.28, zorder=2,
        label="Beam selection [-40; -20] ticks",
    )
    # A linear scale would make this 20-tick interval invisible next to the
    # full APA 1 range of about 175,000 ticks. Keep the whole range visible.
    full.set_xscale("symlog", linthresh=100)
    full.set_xlabel("Time ticks")
    full.set_ylabel("Counts")
    full.grid(alpha=0.18)
    full.set_axisbelow(True)
    full.text(
        0.02, 0.04, "Symmetric-log time axis",
        transform=full.transAxes, ha="left", va="bottom", fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9),
    )
    full.legend(loc="upper right", bbox_to_anchor=(0.99, 0.84),
                frameon=True, facecolor="white", framealpha=1)
    add_work_in_progress(full)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="Directory containing 1GeV/0_to_10/FS_wfset.pkl and ST_wfset.pkl",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-label", default="finding_peaks")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "nominal_beam_momentum_GeV_c": 1,
        "input_block": str(args.input_dir.resolve()),
        "analysis_label": args.analysis_label,
        "beam_selection_ticks": list(BEAM_WINDOW),
        "note": "Stored offsets from preselected waveform sets; no new peak selection",
        "samples": {},
    }
    for apa, mode in ((1, "FS"), (2, "ST")):
        source = args.input_dir / f"{mode}_wfset.pkl"
        if not source.is_file():
            parser.error(f"Missing input file: {source}")
        offsets, waveform_count = load_offsets(source, args.analysis_label)
        filename = f"apa{apa}_offsethist_1GeV.png"
        plot_distribution(offsets, apa, args.output_dir / filename)
        metadata["samples"][f"apa{apa}"] = {
            "mode": mode,
            "source": str(source.resolve()),
            "waveforms": waveform_count,
            "peak_offsets": int(offsets.size),
            "expected_in_historic_thesis_figure": EXPECTED_OLD_COUNTS[apa],
            "matches_historic_count": bool(offsets.size == EXPECTED_OLD_COUNTS[apa]),
            "plot": filename,
        }
        print(
            f"APA {apa}: {waveform_count} waveforms, {offsets.size} peak offsets; "
            f"historic figure had {EXPECTED_OLD_COUNTS[apa]}. Saved {filename}"
        )
    (args.output_dir / "time_offset_provenance.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
