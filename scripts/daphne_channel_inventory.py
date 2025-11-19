#!/usr/bin/env python3
"""
Scan a DAQ HDF5 file and print the set of raw DAPHNEEthStream channels observed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Set, Tuple, Optional

from daqdataformats import FragmentType
from fddetdataformats import DAPHNEEthStreamFrame
from hdf5libs import HDF5RawDataFile


def _iter_daphne_fragments(h5file: HDF5RawDataFile, max_records: int | None = None):
    records = (
        h5file.get_all_trigger_record_ids()
        if h5file.is_trigger_record_type()
        else h5file.get_all_timeslice_ids()
    )
    if max_records is not None:
        records = records[:max_records]

    for record in records:
        for path in h5file.get_fragment_dataset_paths(record):
            fragment = h5file.get_frag(path)
            if fragment.get_fragment_type() != FragmentType.kDAPHNEEthStream:
                continue
            yield record, path, fragment


def _collect_channels(fragment, limit: Optional[int]) -> Tuple[Set[int], Optional[int]]:
    payload = fragment.get_data_bytes()
    frame_size = DAPHNEEthStreamFrame.sizeof()
    n_frames = len(payload) // frame_size
    max_frames = n_frames if limit is None else min(limit, n_frames)

    channels: Set[int] = set()
    slot_id: Optional[int] = None

    for idx in range(max_frames):
        start = idx * frame_size
        frame = DAPHNEEthStreamFrame(payload[start : start + frame_size])
        if slot_id is None:
            slot_id = int(frame.get_daqheader().slot_id)
        header = frame.get_daphneheader()
        for slot in header.channel_words:
            channels.add(int(slot.channel))

    return channels, slot_id


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List the hardware channel IDs seen in a DAPHNE Ethernet HDF5 file.",
    )
    parser.add_argument("hdf5_file", help="Path to the DAQ HDF5 file.")
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="If set, only inspect the first N records.",
    )
    parser.add_argument(
        "--frames-per-fragment",
        type=int,
        default=None,
        help="Limit the number of frames scanned per fragment (default: all).",
    )
    return parser


def main() -> int:
    args = _build_argparser().parse_args()
    h5file = HDF5RawDataFile(args.hdf5_file)

    channel_counts: Counter = Counter()
    slots: Set[int] = set()

    for record, path, fragment in _iter_daphne_fragments(h5file, args.max_records):
        channels, slot_id = _collect_channels(fragment, args.frames_per_fragment)
        if slot_id is not None:
            slots.add(slot_id)
        for channel in channels:
            channel_counts[channel] += 1

    total_entries = sum(channel_counts.values())
    sorted_channels = sorted(channel_counts.items())

    print(f"File           : {args.hdf5_file}")
    print(f"Unique channels: {len(sorted_channels)}")
    print(f"Slots          : {sorted(slots) if slots else 'n/a'}")
    print("Channel usage (fraction of fragments examined):")
    for channel, count in sorted_channels:
        share = 100.0 * count / total_entries if total_entries > 0 else 0.0
        print(f"  channel={channel:2d}  count={count:4d}  share={share:6.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
