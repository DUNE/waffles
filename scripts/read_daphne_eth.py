#!/usr/bin/env python3
"""
CLI helper that demonstrates how to use `waffles.input_output.daphne_eth_reader`.

Example
-------
    ./scripts/read_daphne_eth.py /path/to/file.hdf5 --max-waveforms 10

The script prints a short summary of the DAPHNE Ethernet waveforms it finds and
echoes a few example entries so you can double-check the decoding quickly.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import detchannelmaps
from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.input_output.daphne_eth_reader import load_daphne_eth_waveforms
from waffles.input_output.hdf5_structured import save_structured_waveformset
from waffles.utils.daphne_stats import (
    collect_link_inventory,
    collect_slot_channel_usage,
    compute_waveform_stats,
    map_waveforms_to_links,
)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect DAPHNE Ethernet fragments stored in a raw HDF5 file."
    )
    parser.add_argument("hdf5_file", help="Path to the input raw HDF5 file.")
    parser.add_argument(
        "--detector",
        default="HD_PDS",
        help="Detector string understood by detdataformats (default: %(default)s).",
    )
    parser.add_argument(
        "--channel-map",
        default=None,
        help="Optional channel map plugin forwarded to rawdatautils (default: SimplePDSChannelMap).",
    )
    parser.add_argument(
        "--max-waveforms",
        type=int,
        default=None,
        help="Stop after decoding this many waveforms.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Only examine the first N Trigger Records (after skipping).",
    )
    parser.add_argument(
        "--skip-records",
        type=int,
        default=0,
        help="Number of Trigger Records to skip from the beginning.",
    )
    parser.add_argument(
        "--time-step-ns",
        type=float,
        default=16.0,
        help="Sampling period (ns) passed to the Waveform objects (default: %(default)s).",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=5,
        help="Print up to this many example waveforms (default: %(default)s).",
    )
    parser.add_argument(
        "--structured-out",
        default=None,
        help="If set, write the decoded WaveformSet to this HDF5 (structured) file.",
    )
    parser.add_argument(
        "--plots-dir",
        default=None,
        help="If set, produce per-channel persistence/FFT plots into this directory.",
    )
    parser.add_argument(
        "--plots-max-waveforms",
        type=int,
        default=2000,
        help="Maximum number of waveforms per channel to use for plots (default: %(default)s).",
    )
    parser.add_argument(
        "--plots-time-bins",
        type=int,
        default=256,
        help="Number of time bins for persistence histograms (default: %(default)s).",
    )
    parser.add_argument(
        "--plots-amp-bins",
        type=int,
        default=256,
        help="Number of amplitude bins for persistence histograms (default: %(default)s).",
    )
    parser.add_argument(
        "--print-stats",
        action="store_true",
        help="Print endpoint/channel/link percentage tables after decoding.",
    )
    parser.add_argument(
        "--stats-top",
        type=int,
        default=10,
        help="Maximum number of rows per table when --print-stats is enabled (default: %(default)s).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress info-level logging; only warnings/errors will be shown.",
    )
    return parser


def _collect_channel_samples(
    waveforms: List[Waveform],
    limit: Optional[int],
) -> Dict[Tuple[int, int], List[Waveform]]:
    if limit is not None and limit <= 0:
        limit = None
    grouped: Dict[Tuple[int, int], List[Waveform]] = defaultdict(list)
    for wf in waveforms:
        key = (wf.endpoint, wf.channel)
        bucket = grouped[key]
        if limit is not None and len(bucket) >= limit:
            continue
        bucket.append(wf)
    return grouped


def _stack_waveforms(waveforms: List[Waveform]) -> Tuple[np.ndarray, float]:
    min_samples = min(len(wf.adcs) for wf in waveforms)
    data = np.stack([wf.adcs[:min_samples] for wf in waveforms]).astype(np.float32)
    return data, waveforms[0].time_step_ns


def _save_channel_plots(
    key: Tuple[int, int],
    waveforms: List[Waveform],
    out_dir: Path,
    time_bins: int,
    amp_bins: int,
) -> None:
    data, time_step_ns = _stack_waveforms(waveforms)
    n_waveforms, n_samples = data.shape

    amp_min = float(data.min())
    amp_max = float(data.max())
    if amp_min == amp_max:
        amp_max = amp_min + 1.0

    times = np.tile(np.arange(n_samples), n_waveforms)
    adc_vals = data.reshape(-1)
    time_bin_count = max(1, min(time_bins, n_samples))
    amp_bin_count = max(1, amp_bins)
    hist, xedges, yedges = np.histogram2d(
        times,
        adc_vals,
        bins=[time_bin_count, amp_bin_count],
        range=[[0, n_samples], [amp_min, amp_max]],
    )

    window = np.hanning(n_samples)
    windowed = data * window
    fft_vals = np.fft.rfft(windowed, axis=1)
    freqs = np.fft.rfftfreq(n_samples, d=time_step_ns * 1e-9)
    power = np.abs(fft_vals) ** 2
    mean_power = np.maximum(power.mean(axis=0), 1e-18)

    fig, (ax_persist, ax_fft) = plt.subplots(1, 2, figsize=(12, 5))
    im = ax_persist.imshow(
        hist.T + 1e-9,
        origin="lower",
        aspect="auto",
        extent=[0, n_samples, amp_min, amp_max],
        norm=LogNorm(),
        cmap="viridis",
    )
    ax_persist.set_title("Persistence")
    ax_persist.set_xlabel("Sample")
    ax_persist.set_ylabel("ADC")
    cbar = fig.colorbar(im, ax=ax_persist)
    cbar.set_label("Counts")

    ax_fft.plot(freqs / 1e6, mean_power + 1e-12)
    ax_fft.set_yscale("log")
    ax_fft.set_xlim(left=0)
    ax_fft.set_xlabel("Frequency [MHz]")
    ax_fft.set_ylabel("Mean power")
    ax_fft.set_title("Average FFT magnitude")

    endpoint, channel = key
    fig.suptitle(
        f"Endpoint {endpoint} / Channel {channel} "
        f"({n_waveforms} waveforms, {n_samples} samples)"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"endpoint{endpoint}_channel{channel}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _format_percent(count: int, total: int) -> str:
    if total <= 0:
        return "  n/a "
    return f"{100.0 * count / total:6.2f}%"


def _print_counter_table(
    title: str,
    counter: Counter,
    total: int,
    limit: int,
    formatter: Callable[[object], str],
) -> None:
    print(f"\n{title}:")
    if not counter:
        print("  (no entries)")
        return

    header = "  Count   Share   Entry"
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows_printed = 0
    for key, count in counter.most_common():
        if limit is not None and rows_printed >= max(limit, 0):
            remaining = len(counter) - rows_printed
            if remaining > 0:
                print(f"  ... ({remaining} additional entries truncated)")
            break
        print(f"  {count:6d}  {_format_percent(count, total)}  {formatter(key)}")
        rows_printed += 1


def _format_source(key: int) -> str:
    return f"source_id={key}"


def _format_channel(key: tuple[int, int]) -> str:
    slot, channel = key
    return f"slot={slot} channel={channel}"


def _print_stats(
    args: argparse.Namespace,
    wfset: WaveformSet,
    inventory,
    offline_counts: Counter,
) -> None:
    stats = compute_waveform_stats(wfset.waveforms)
    link_waveform_counts, missing_endpoints = map_waveforms_to_links(stats, inventory)

    slot_channel_counts = collect_slot_channel_usage(
        filepath=args.hdf5_file,
        detector=args.detector,
        skip_records=args.skip_records,
        max_records=args.max_records,
    )

    print(
        f"\nStatistics: {stats.total_waveforms} waveforms, "
        f"{len(stats.endpoint_counts)} source IDs, {len(slot_channel_counts)} channels."
    )
    _print_counter_table(
        "Source ID distribution",
        stats.endpoint_counts,
        stats.total_waveforms,
        args.stats_top,
        _format_source,
    )
    channel_total = sum(slot_channel_counts.values())
    _print_counter_table(
        "Channel distribution",
        slot_channel_counts,
        channel_total,
        args.stats_top,
        _format_channel,
    )

    offline_total = sum(offline_counts.values())
    if offline_total:
        print("\nOffline channel distribution")
        _print_counter_table(
            "Offline channels",
            offline_counts,
            offline_total,
            args.stats_top,
            lambda key: f"slot={key[0]} offline_channel={key[1]}",
        )
    _print_counter_table(
        "Link distribution (by waveform count)",
        link_waveform_counts,
        stats.total_waveforms,
        args.stats_top,
        lambda link: str(link),
    )
    if missing_endpoints:
        missing_str = ", ".join(str(eid) for eid in sorted(missing_endpoints))
        print(f"\nWarning: no link metadata found for endpoint(s): {missing_str}")


def _configure_logger(quiet: bool) -> logging.Logger:
    logger = logging.getLogger("daphne_eth_reader")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.WARNING if quiet else logging.INFO)
    return logger


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logger = _configure_logger(args.quiet)
    channel_map_name = args.channel_map or "SimplePDSChannelMap"

    wfset = load_daphne_eth_waveforms(
        filepath=args.hdf5_file,
        detector=args.detector,
        channel_map=channel_map_name,
        max_waveforms=args.max_waveforms,
        max_records=args.max_records,
        skip_records=args.skip_records,
        time_step_ns=args.time_step_ns,
        logger=logger,
    )

    if wfset is None:
        print("No DAPHNE Ethernet waveforms found.", file=sys.stderr)
        return 1

    print(f"Decoded {len(wfset.waveforms)} waveforms from {args.hdf5_file}")

    inventory = collect_link_inventory(
        filepath=args.hdf5_file,
        detector=args.detector,
        channel_map=channel_map_name,
        skip_records=args.skip_records,
        max_records=args.max_records,
    )
    endpoint_to_slot = {eid: link.slot_id for eid, link in inventory.source_to_link.items()}
    try:
        channel_map = detchannelmaps.make_pds_map(channel_map_name)
    except Exception as err:
        logger.warning("Unable to load channel map %s: %s", channel_map_name, err)
        channel_map = None

    offline_counts: Counter = Counter()
    unique_channels = set()
    for wf in wfset.waveforms:
        raw_channel = getattr(wf, "raw_channel", None)
        link = inventory.source_to_link.get(int(wf.endpoint))
        if link is None:
            continue
        slot = link.slot_id
        if channel_map is not None and raw_channel is not None:
            offline_channel = channel_map.get_offline_channel_from_det_crate_slot_stream_chan(
                link.det_id, link.crate_id, link.slot_id, link.stream_id, int(raw_channel)
            )
            wf.offline_channel = int(offline_channel)
        else:
            offline_channel = getattr(wf, "offline_channel", getattr(wf, "channel", None))
        if offline_channel is None:
            continue
        offline_counts[(slot, int(offline_channel))] += 1
        unique_channels.add((slot, int(offline_channel)))

    unique_channels_list = sorted(unique_channels)
    print(f"Observed {len(unique_channels_list)} unique (slot, offline_channel) pairs.")

    to_show = min(args.show, len(wfset.waveforms))
    if to_show > 0:
        print(f"\nDisplaying the first {to_show} waveforms:")
        for wf in wfset.waveforms[:to_show]:
            slot = endpoint_to_slot.get(int(wf.endpoint), wf.endpoint)
            raw_channel = getattr(wf, "raw_channel", None)
            link = inventory.source_to_link.get(int(wf.endpoint))
            if channel_map is not None and raw_channel is not None and link is not None:
                offline_channel = channel_map.get_offline_channel_from_det_crate_slot_stream_chan(
                    link.det_id, link.crate_id, link.slot_id, link.stream_id, int(raw_channel)
                )
            else:
                offline_channel = getattr(wf, "channel", None)
            print(
                f"  run={wf.run_number} record={wf.record_number} "
                f"slot={slot} offline_channel={offline_channel} "
                f"timestamp={wf.timestamp} nsamples={len(wf.adcs)}"
            )

    if args.print_stats:
        _print_stats(args, wfset, inventory, offline_counts)
    if args.structured_out:
        save_structured_waveformset(
            wfset,
            args.structured_out,
        )
    if args.plots_dir:
        channel_samples = _collect_channel_samples(
            wfset.waveforms,
            args.plots_max_waveforms,
        )
        out_dir = Path(args.plots_dir)
        for key, samples in channel_samples.items():
            if not samples:
                continue
            _save_channel_plots(
                key,
                samples,
                out_dir,
                args.plots_time_bins,
                args.plots_amp_bins,
            )
        logger.info("Saved per-channel persistence/FFT plots to %s", out_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
