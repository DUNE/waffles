#!/usr/bin/env python3
"""
Inventory helper for DAPHNE Ethernet files.

The script decodes the requested raw HDF5 file, lists the available
source IDs / links / channels, prints their relative contributions,
and optionally writes the resulting WaveformSet to disk.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from typing import Callable, Optional

from waffles.input_output.daphne_eth_reader import load_daphne_eth_waveforms
from waffles.input_output.hdf5_structured import save_structured_waveformset
from waffles.utils.daphne_stats import (
    collect_link_inventory,
    compute_waveform_stats,
    map_waveforms_to_links,
    LinkKey,
)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize which DAPHNE Ethernet channels/source IDs/links exist in a raw HDF5 file.",
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
        help="Only inspect the first N Trigger Records after skipping.",
    )
    parser.add_argument(
        "--skip-records",
        type=int,
        default=0,
        help="Number of Trigger Records to skip from the start of the file.",
    )
    parser.add_argument(
        "--time-step-ns",
        type=float,
        default=16.0,
        help="Sampling period (ns) stored alongside the decoded waveforms.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="Maximum number of rows to show per summary table (default: %(default)s).",
    )
    parser.add_argument(
        "--structured-out",
        default=None,
        help="If set, write the decoded WaveformSet to this HDF5 (structured) file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress info-level logging emitted by the decoder.",
    )
    return parser


def _configure_logger(quiet: bool) -> logging.Logger:
    logger = logging.getLogger("daphne_eth_summary")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.WARNING if quiet else logging.INFO)
    return logger


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


def _channel_formatter(key: tuple[int, int]) -> str:
    endpoint, channel = key
    return f"endpoint={endpoint} channel={channel}"


def _link_formatter(key: LinkKey) -> str:
    return str(key)


def _source_formatter(source_id: int) -> str:
    return f"source_id={source_id}"


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logger = _configure_logger(args.quiet)

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
        print("No DAPHNE Ethernet waveforms found.", file=sys.stderr)
        return 1

    stats = compute_waveform_stats(wfset.waveforms)
    inventory = collect_link_inventory(
        filepath=args.hdf5_file,
        detector=args.detector,
        channel_map=args.channel_map,
        skip_records=args.skip_records,
        max_records=args.max_records,
    )
    link_waveform_counts, missing_endpoints = map_waveforms_to_links(stats, inventory)

    print(f"Decoded {stats.total_waveforms} waveforms from {args.hdf5_file}")
    print(
        f"Processed {inventory.records_considered} record(s) "
        f"and {inventory.fragments_considered} DAPHNE fragment(s)."
    )
    channel_totals: Counter = Counter()
    for (_, channel), count in stats.channel_counts.items():
        channel_totals[int(channel)] += count

    print(f"Observed {len(stats.endpoint_counts)} source IDs and {len(channel_totals)} unique channels.")

    _print_counter_table(
        "Source ID distribution",
        stats.endpoint_counts,
        stats.total_waveforms,
        args.top,
        _source_formatter,
    )
    _print_counter_table(
        "Channel distribution",
        channel_totals,
        stats.total_waveforms,
        args.top,
        _channel_formatter,
    )
    _print_counter_table(
        "Link distribution (by waveform count)",
        link_waveform_counts,
        stats.total_waveforms,
        args.top,
        _link_formatter,
    )

    if missing_endpoints:
        missing_str = ", ".join(str(eid) for eid in sorted(missing_endpoints))
        print(f"\nWarning: no link metadata found for endpoint(s): {missing_str}")

    if inventory.source_to_link:
        print("\nEndpoint → link mapping:")
        for endpoint in sorted(inventory.source_to_link.keys()):
            print(f"  {endpoint:8d}  ->  {inventory.source_to_link[endpoint]}")

    if args.structured_out:
        save_structured_waveformset(
            wfset,
            args.structured_out,
        )
        print(f"\nWaveformSet written to {args.structured_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
