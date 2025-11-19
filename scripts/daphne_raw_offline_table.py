#!/usr/bin/env python3
"""
Print a mapping between raw DAPHNE channels and offline channel IDs for a raw HDF5 file.
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Optional

from waffles.input_output.daphne_eth_reader import load_daphne_eth_waveforms
from waffles.utils.daphne_stats import collect_link_inventory


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize raw-versus-offline DAPHNE channels for a raw HDF5 file.",
    )
    parser.add_argument("hdf5_file", help="Path to the input HDF5 file.")
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
        help="Only inspect the first N Trigger Records.",
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
        default=None,
        help="Maximum number of rows to print (default: all).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)

    wfset = load_daphne_eth_waveforms(
        filepath=args.hdf5_file,
        detector=args.detector,
        channel_map=args.channel_map,
        max_waveforms=args.max_waveforms,
        max_records=args.max_records,
        skip_records=args.skip_records,
        time_step_ns=args.time_step_ns,
    )
    if wfset is None:
        print("No DAPHNE Ethernet waveforms found.")
        return 1

    inventory = collect_link_inventory(
        filepath=args.hdf5_file,
        detector=args.detector,
        channel_map=args.channel_map,
        skip_records=args.skip_records,
        max_records=args.max_records,
    )

    mapping_counts: Counter = Counter()
    total_waveforms = 0

    for wf in wfset.waveforms:
        raw_channel = getattr(wf, "raw_channel", None)
        offline_channel = getattr(wf, "offline_channel", None)
        if raw_channel is None or offline_channel is None:
            continue
        link = inventory.source_to_link.get(int(wf.endpoint))
        if link is None:
            continue
        key = (int(link.slot_id), int(raw_channel), int(offline_channel))
        mapping_counts[key] += 1
        total_waveforms += 1

    print(
        f"Processed {total_waveforms} waveforms across "
        f"{len(mapping_counts)} unique slot/raw/offline combinations."
    )
    print("\nslot  raw_ch  offline_ch  count   share")
    print("-------------------------------------------")

    rows_shown = 0
    for (slot, raw_ch, offline_ch), count in mapping_counts.most_common():
        if args.top is not None and rows_shown >= args.top:
            remaining = len(mapping_counts) - rows_shown
            if remaining > 0:
                print(f"... ({remaining} additional entries)")
            break
        share = 100.0 * count / total_waveforms if total_waveforms > 0 else 0.0
        print(f"{slot:4d}  {raw_ch:6d}  {offline_ch:11d}  {count:5d}  {share:6.2f}%")
        rows_shown += 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
