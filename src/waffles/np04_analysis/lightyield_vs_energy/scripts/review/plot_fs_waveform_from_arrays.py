#!/usr/bin/env python3
"""Plot the saved APA 1 full-streaming waveform and its selected 588-sample window.

Input pickles contain NumPy arrays created by the waveform analysis, rather than
WaveformSet objects. The script verifies that the processed and cut arrays are
derived from the supplied original waveform before producing two PNGs.
"""

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class NumpyArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("numpy._core.multiarray", "_reconstruct"),
            ("numpy", "ndarray"),
            ("numpy", "dtype"),
        }
        if (module, name) not in allowed:
            raise ValueError(f"Unexpected object in pickle: {module}.{name}")
        return super().find_class(module, name)


def read_array(path):
    with path.open("rb") as stream:
        values = NumpyArrayUnpickler(stream).load()
    if not isinstance(values, np.ndarray) or values.ndim != 1:
        raise ValueError(f"{path} does not contain a one-dimensional NumPy array")
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite samples")
    return values


def add_mark(ax):
    ax.text(0.98, 0.96, "ProtoDUNE-HD", transform=ax.transAxes,
            ha="right", va="top", fontsize=10, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
    ax.text(0.98, 0.90, "Work in Progress", transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--cut", type=Path, required=True)
    parser.add_argument("--sample-period-ns", type=float, default=16.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    raw = read_array(args.original)
    processed = read_array(args.processed)
    cut = read_array(args.cut)
    if len(raw) != len(processed):
        raise ValueError("Original and processed arrays have different lengths")
    if len(cut) != 588:
        raise ValueError(f"Extracted window has {len(cut)} samples, expected 588")
    if args.sample_period_ns <= 0:
        raise ValueError("Sample period must be positive")

    baseline_values = raw + processed
    if not np.allclose(baseline_values, baseline_values[0], atol=1e-6, rtol=0):
        raise ValueError("Processed waveform is not baseline minus original waveform")
    baseline = float(baseline_values[0])

    # Match all 588 values, since many individual ADC values recur in the trace.
    candidates = np.flatnonzero(np.isclose(processed, cut[0], atol=1e-8, rtol=0))
    matches = [int(i) for i in candidates if i + len(cut) <= len(processed)
               and np.allclose(processed[i:i+len(cut)], cut, atol=1e-8, rtol=0)]
    if len(matches) != 1:
        raise ValueError(f"Expected one matching cut in the processed waveform, found {matches}")
    start = matches[0]
    end = start + len(cut)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": 11, "axes.labelsize": 13, "xtick.labelsize": 11,
                         "ytick.labelsize": 11, "legend.fontsize": 10})
    time_ticks = np.arange(len(raw))
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    ax.plot(time_ticks, raw, color="#0072B2", lw=0.65, rasterized=True,
            label="Original FS waveform")
    ax.axvspan(start, end,
               facecolor="#F0E442", edgecolor="none", linewidth=0,
               alpha=0.3,
               label=f"Selected window [{start}; {end - 1}]", zorder=2)
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=1)
    ax.set(xlabel="Time ticks", ylabel="Amplitude [ADC]")
    ax.grid(alpha=0.2)
    add_mark(ax)
    fig.savefig(args.output_dir / "fs_waveform_before.png", dpi=300)
    plt.close(fig)

    selected_ticks = np.arange(start, end)
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    ax.plot(selected_ticks, cut, color="#0072B2", lw=1.2,
            label="Selected waveform (baseline-subtracted and inverted)")
    ax.axhline(0, color="#D55E00", lw=1.0, ls="--", label="Zero baseline")
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=1)
    ax.set(xlabel="Time ticks", ylabel="Amplitude [ADC]")
    ax.grid(alpha=0.2)
    add_mark(ax)
    fig.savefig(args.output_dir / "fs_waveform_after.png", dpi=300)
    plt.close(fig)

    print(f"Validated baseline = {baseline:.1f} ADC counts")
    print(f"Original trace duration = {len(raw) * args.sample_period_ns / 1e6:.5f} ms")
    print(f"Extracted window = samples [{start}, {end}), {len(cut) * args.sample_period_ns / 1000:.3f} µs")
    print(f"Saved two PNGs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
