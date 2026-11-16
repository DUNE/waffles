#!/usr/bin/env python3
"""Illustrate ST waveform processing from a screenshot of one processed pulse.

The processed trace is approximately digitized from the supplied screenshot.
The unprocessed panel is reconstructed using an ASSUMED ADC baseline; it is
not a recovered raw waveform. Both figures disclose this provenance.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DEFAULT_IMAGE = (Path(__file__).resolve().parents[2] / "data" /
                 "waveform_examples" / "processed_st_screenshot.png")


def digitize_processed_curve(path):
    rgb = np.asarray(Image.open(path).convert("RGB"))
    if rgb.shape[:2] != (538, 568):
        raise ValueError("Screenshot dimensions changed; pixel axes need recalibration")

    # Calibrate against visible ticks in the screenshot.
    # x = 103 at tick 0, x = 516 at tick 400.
    # y = 274 at amplitude 0; 100 amplitude units span 62 pixels.
    x0, x400, y0, pixels_per_100 = 103, 516, 274, 62.0
    observed_x, observed_y = [], []
    for x in range(x0, x400 + 1):
        column = rgb[48:386, x, :].astype(int)
        blue = ((column[:, 2] > 150) & (column[:, 0] < 110) &
                (column[:, 1] < 110) & (column[:, 2] > column[:, 0] + 60))
        ys = np.flatnonzero(blue)
        if len(ys):
            observed_x.append(x)
            observed_y.append(float(np.median(ys) + 48))
    if len(observed_x) < 350:
        raise ValueError("Too few blue trace pixels for reliable digitization")
    ticks = np.arange(401, dtype=float)
    pixel_x = x0 + (x400 - x0) * ticks / 400
    pixel_y = np.interp(pixel_x, observed_x, observed_y)
    processed = (y0 - pixel_y) * 100 / pixels_per_100
    return ticks, processed, len(observed_x)


def add_mark(ax):
    ax.text(0.98, 0.97, "ProtoDUNE-HD", transform=ax.transAxes,
            ha="right", va="top", fontsize=10, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
    ax.text(0.98, 0.91, "Work in Progress", transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))


def save_panel(ticks, values, path, *, reconstructed, assumed_baseline):
    fig, ax = plt.subplots(figsize=(6.4, 6.4), constrained_layout=True)
    ax.plot(ticks, values, color="#0072B2", lw=1.3,
            label="Original ST waveform (reconstructed)" if reconstructed
                  else "Processed waveform\n(baseline-subtracted and inverted; digitized)")
    if not reconstructed:
        ax.axhline(0, color="#D55E00", lw=1, ls="--", label="Zero baseline")
    ax.set(xlabel="Time ticks", ylabel="Amplitude [ADC]", xlim=(0, 400))
    ax.set_ylim(float(np.min(values)) - 25,
                float(np.max(values)) + (150 if not reconstructed else 35))
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left",
              frameon=True, facecolor="white", framealpha=1)
    add_mark(ax)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--assumed-baseline-adc", type=float, default=8220.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path,
                        help="Where to save the two reusable waveform CSVs; "
                             "defaults to the source image directory")
    args = parser.parse_args()
    ticks, processed, n_columns = digitize_processed_curve(args.source_image)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir or args.source_image.parent
    data_dir.mkdir(parents=True, exist_ok=True)
    reconstructed = args.assumed_baseline_adc - processed
    save_panel(ticks, reconstructed, args.output_dir / "st_waveform_before.png",
               reconstructed=True, assumed_baseline=args.assumed_baseline_adc)
    save_panel(ticks, processed, args.output_dir / "st_waveform_after.png",
               reconstructed=False, assumed_baseline=args.assumed_baseline_adc)
    np.savetxt(data_dir / "st_waveform_before_reconstructed.csv",
               np.column_stack([ticks.astype(int), reconstructed]),
               delimiter=",", header="time_tick,amplitude_adc", comments="",
               fmt=["%d", "%.6f"])
    np.savetxt(data_dir / "st_waveform_after_digitized.csv",
               np.column_stack([ticks.astype(int), processed]),
               delimiter=",", header="time_tick,amplitude_adc", comments="",
               fmt=["%d", "%.6f"])
    print(f"Digitized {n_columns} image columns; saved two CSVs to {data_dir.resolve()}")


if __name__ == "__main__":
    main()
