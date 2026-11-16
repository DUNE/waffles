#!/usr/bin/env python3
"""Four thesis panels from actual Waffles ST and FS waveform records.

Accepted inputs are Waffles structured HDF5 files and trusted WaveformSet
pickle files. Select one zero-based waveform index from each input file.
"""

import argparse
import pickle
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DT_NS = 16.0
N_BASE = 50
SIGMA_CUT = 3.0
N_WINDOW = 588
PRE_RISE = 60


def read_waveform(path, index):
    if path.suffix.lower() in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as h5:
            if "adcs" not in h5:
                raise ValueError(f"{path}: no 'adcs' dataset")
            n = h5["adcs"].shape[0]
            if not 0 <= index < n:
                raise IndexError(f"{path}: index {index} outside 0..{n-1}")
            adcs = np.asarray(h5["adcs"][index], dtype=float)
            dt_ns = float(h5.attrs.get("time_step_ns", DT_NS))
            info = {key: int(h5[key][index]) for key in
                    ("run_numbers", "record_numbers", "endpoints", "channels")
                    if key in h5}
    elif path.suffix.lower() in {".pkl", ".pickle"}:
        # Pickle loading is intended only for files created by the user's analysis.
        with path.open("rb") as stream:
            wfset = pickle.load(stream)
        waveforms = wfset.waveforms if hasattr(wfset, "waveforms") else wfset
        n = len(waveforms)
        if not 0 <= index < n:
            raise IndexError(f"{path}: index {index} outside 0..{n-1}")
        wf = waveforms[index]
        adcs = np.asarray(wf.adcs, dtype=float).copy()
        dt_ns = float(wf.time_step_ns)
        info = {key: int(getattr(wf, key)) for key in
                ("run_number", "record_number", "endpoint", "channel")
                if hasattr(wf, key)}
    else:
        raise ValueError(f"Unsupported waveform file: {path} (use .hdf5 or .pkl)")
    if adcs.ndim != 1 or len(adcs) < N_BASE or not np.isfinite(adcs).all():
        raise ValueError(f"{path}: selected waveform must be a finite 1D array with at least 50 samples")
    if not np.isfinite(dt_ns) or dt_ns <= 0:
        raise ValueError(f"{path}: invalid sampling interval {dt_ns}")
    return adcs, dt_ns, info


def baseline(raw):
    """Match WindowBaseliner: median, 3-sigma rejection, final mean."""
    first = raw[:N_BASE]
    median = np.median(first)
    std = np.std(first)
    selected = first[np.abs(first - median) <= SIGMA_CUT * std] if std > 0 else first
    if len(selected) == 0:
        raise ValueError("No baseline points remain after rejection")
    return float(np.mean(selected)), float(np.std(selected))


def fs_window(processed, chosen_peak):
    peak = int(np.argmax(processed)) if chosen_peak is None else chosen_peak
    if not 0 <= peak < len(processed) or processed[peak] <= 0:
        raise ValueError("FS peak index does not identify a positive processed pulse")
    level = 0.30 * processed[peak]
    crossings = np.flatnonzero((processed[1:peak+1] >= level) &
                               (processed[:peak] < level)) + 1
    if not len(crossings):
        raise ValueError("No 30% rising-edge crossing before selected FS peak")
    start = int(crossings[-1]) - PRE_RISE
    end = start + N_WINDOW
    if start < 0 or end > len(processed):
        raise ValueError("Selected FS pulse is too close to a record boundary")
    return peak, start, end


def draw(raw, dt_ns, destination, *, mode, processed=False,
         baseline_value=None, window=None, offset=0):
    n = len(raw)
    ticks = np.arange(n)
    time = ticks * dt_ns / 1000.0
    if mode == "FS" and not processed:
        x = time / 1000.0
        xlabel = "Time [ms]"
        stride = max(1, n // 25000)
    else:
        x = time
        xlabel = "Time [µs]"
        stride = 1
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    ax.plot(x[::stride], raw[::stride], color="#0072B2" if processed else "#333333",
            lw=1.15, rasterized=(stride > 1))
    if processed:
        ax.axhline(0, color="#555555", lw=1, ls="--")
    else:
        ax.axhline(baseline_value, color="#D55E00", lw=1.5, ls="--",
                   label="Estimated baseline")
        baseline_end = N_BASE * dt_ns / (1e6 if mode == "FS" else 1000)
        if mode == "ST":
            ax.axvspan(0, baseline_end, color="#E69F00", alpha=0.18,
                       label="First 50 samples")
        if window is not None:
            lo, hi = window
            ax.axvspan(lo * dt_ns / 1e6, hi * dt_ns / 1e6,
                       color="#009E73", alpha=0.25,
                       label="Extracted 588-sample window")
        ax.legend(loc="upper left", facecolor="white", framealpha=1, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Amplitude [ADC counts]", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(alpha=0.24)
    # Separate bold and regular text so that Matplotlib's mathtext is not needed.
    ax.text(0.985, 0.97, "ProtoDUNE-HD", transform=ax.transAxes,
            ha="right", va="top", weight="bold", fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1})
    ax.text(0.985, 0.91, "Work in Progress", transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1})
    fig.savefig(destination, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--st-file", required=True, type=Path)
    parser.add_argument("--st-index", required=True, type=int,
                        help="Zero-based ST waveform index")
    parser.add_argument("--fs-file", required=True, type=Path)
    parser.add_argument("--fs-index", required=True, type=int,
                        help="Zero-based FS waveform index")
    parser.add_argument("--fs-peak-index", type=int,
                        help="FS sample containing the desired peak; default: strongest pulse")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("waveform_processing_examples"))
    args = parser.parse_args()
    for path in (args.st_file, args.fs_file):
        if not path.is_file():
            parser.error(f"Input file not found: {path}")

    st, st_dt, st_info = read_waveform(args.st_file, args.st_index)
    fs, fs_dt, fs_info = read_waveform(args.fs_file, args.fs_index)
    if len(st) != 1024:
        raise ValueError(f"Selected ST waveform has {len(st)} samples, expected 1024")
    st_base, st_noise = baseline(st)
    fs_base, fs_noise = baseline(fs)
    st_clean = st_base - st
    fs_clean = fs_base - fs
    peak, start, end = fs_window(fs_clean, args.fs_peak_index)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw(st, st_dt, args.output_dir / "st_waveform_before.png", mode="ST",
         baseline_value=st_base)
    draw(st_clean, st_dt, args.output_dir / "st_waveform_after.png",
         mode="ST", processed=True)
    draw(fs, fs_dt, args.output_dir / "fs_waveform_before.png", mode="FS",
         baseline_value=fs_base, window=(start, end))
    draw(fs_clean[start:end], fs_dt, args.output_dir / "fs_waveform_after.png",
         mode="FS", processed=True, offset=start)
    print("Created four PNGs in", args.output_dir.resolve())
    print(f"ST {st_info}: baseline={st_base:.2f} ADC, noise={st_noise:.2f} ADC")
    print(f"FS {fs_info}: baseline={fs_base:.2f} ADC, noise={fs_noise:.2f} ADC, "
          f"peak={peak}, window=[{start}, {end})")


if __name__ == "__main__":
    main()
