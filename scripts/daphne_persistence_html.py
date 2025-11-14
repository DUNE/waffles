#!/usr/bin/env python3
"""
Generate interactive persistence plots for DAPHNE Ethernet channels.

The script decodes the requested raw DAQ HDF5 file, samples up to a configurable
number of waveforms per (endpoint, channel) pair, and writes a single HTML file
containing one Plotly heatmap per channel.
"""

from __future__ import annotations

import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go

from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.input_output.daphne_eth_reader import load_daphne_eth_waveforms


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Produce per-channel interactive persistence plots for DAPHNE Ethernet waveforms.",
    )
    parser.add_argument("hdf5_file", help="Path to the input raw HDF5 file.")
    parser.add_argument(
        "--output",
        default="daphne_persistence.html",
        help="Destination HTML file (default: %(default)s).",
    )
    parser.add_argument(
        "--detector",
        default="HD_PDS",
        help="Detector string understood by detdataformats (default: %(default)s).",
    )
    parser.add_argument(
        "--channel-map",
        default=None,
        help="Optional channel map name forwarded to rawdatautils.",
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
        "--per-channel-limit",
        type=int,
        default=200,
        help="Maximum number of waveforms to use per channel (default: %(default)s).",
    )
    parser.add_argument(
        "--max-channels",
        type=int,
        default=None,
        help="Limit the number of channels to plot (largest populations first).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed used when sampling waveforms per channel (default: %(default)s).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress info-level logging; only warnings/errors will be shown.",
    )
    return parser


def _configure_logger(quiet: bool) -> logging.Logger:
    logger = logging.getLogger("daphne_persistence")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.WARNING if quiet else logging.INFO)
    return logger


def _group_waveforms(waveforms: Sequence[Waveform]) -> Dict[Tuple[int, int], List[Waveform]]:
    grouped: Dict[Tuple[int, int], List[Waveform]] = defaultdict(list)
    for wf in waveforms:
        grouped[(int(wf.endpoint), int(wf.channel))].append(wf)
    return grouped


def _sample_waveforms(waveforms: List[Waveform], limit: Optional[int]) -> List[Waveform]:
    if limit is None or len(waveforms) <= limit:
        return list(waveforms)
    return random.sample(waveforms, limit)


def _compute_persistence(
    waveforms: List[Waveform],
    time_bins: int = 256,
    amp_bins: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_samples = min(len(wf.adcs) for wf in waveforms)
    data = np.stack([wf.adcs[:min_samples] for wf in waveforms]).astype(np.float32)
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

    return hist, xedges, yedges


def _build_figure(
    endpoint: int,
    channel: int,
    waveforms: List[Waveform],
    hist: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
) -> go.Figure:
    n_waveforms = len(waveforms)
    n_samples = len(waveforms[0].adcs)
    log_counts = np.log10(hist.T + 1.0)

    heatmap = go.Heatmap(
        x=xedges[:-1],
        y=yedges[:-1],
        z=log_counts,
        colorscale="Viridis",
        colorbar=dict(title="log10(count)"),
    )

    fig = go.Figure(data=[heatmap])
    fig.update_layout(
        title=(
            f"Endpoint {endpoint} / Channel {channel} — "
            f"{n_waveforms} waveform(s), {n_samples} samples"
        ),
        xaxis_title="Sample index",
        yaxis_title="ADC",
    )
    return fig


def _write_html(figures: Sequence[go.Figure], output: Path) -> None:
    pieces = [
        fig.to_html(full_html=False, include_plotlyjs=False, default_width="100%", default_height="500px")
        for fig in figures
    ]
    html = (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        "<meta charset='utf-8'/>\n"
        "<title>DAPHNE persistence plots</title>\n"
        "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>\n"
        "</head>\n<body>\n"
        "<h1>DAPHNE per-channel persistence plots</h1>\n"
        "<p>Each heatmap shows log10(count) for sample index vs ADC.</p>\n"
        f"{''.join(pieces)}\n"
        "</body>\n</html>\n"
    )
    output.write_text(html)


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logger = _configure_logger(args.quiet)
    random.seed(args.seed)

    wfset = load_daphne_eth_waveforms(
        filepath=args.hdf5_file,
        detector=args.detector,
        channel_map=args.channel_map,
        max_waveforms=args.max_waveforms,
        max_records=args.max_records,
        skip_records=args.skip_records,
        time_step_ns=args.time_step_ns,
        logger=logger,
    )
    if wfset is None:
        logger.error("No DAPHNE Ethernet waveforms found in %s", args.hdf5_file)
        return 1

    channel_groups = _group_waveforms(wfset.waveforms)
    logger.info("Found %d channels", len(channel_groups))

    sorted_channels = sorted(
        channel_groups.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    )
    if args.max_channels is not None:
        sorted_channels = sorted_channels[: max(args.max_channels, 0)]

    figures: List[go.Figure] = []
    for (endpoint, channel), waveforms in sorted_channels:
        selected = _sample_waveforms(waveforms, args.per_channel_limit)
        hist, xedges, yedges = _compute_persistence(selected)
        fig = _build_figure(endpoint, channel, selected, hist, xedges, yedges)
        figures.append(fig)

    if not figures:
        logger.error("No channels matched the selection criteria.")
        return 1

    output_path = Path(args.output)
    _write_html(figures, output_path)
    logger.info("Wrote %d interactive plots to %s", len(figures), output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
