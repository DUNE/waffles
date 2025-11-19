#!/usr/bin/env python3
"""
Inspect a DAQ HDF5 file containing DAPHNE Ethernet fragments.

The script performs two actions:
  1) Prints the low-level header/adcs of the first DAPHNEEthStream frame.
  2) Runs the standard Waffles summary to list offline channel/source/link stats.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from typing import Callable, Optional, Tuple

from daqdataformats import FragmentType, fragment_type_to_string
from fddetdataformats import DAPHNEEthStreamFrame
from hdf5libs import HDF5RawDataFile

from waffles.input_output.daphne_eth_reader import load_daphne_eth_waveforms
from waffles.utils.daphne_stats import (
    collect_link_inventory,
    compute_waveform_stats,
    map_waveforms_to_links,
    LinkKey,
)

# mirror constants from DAPHNEEthStreamFrame.hpp
_BITS_PER_ADC = 14
_CHANNELS = 4
_ADCS_PER_CHANNEL = 280
_WORD_BYTES = 8
_ADC_WORDS = _CHANNELS * _ADCS_PER_CHANNEL * _BITS_PER_ADC // (_WORD_BYTES * 8)


def _first_record_and_fragment(h5file: HDF5RawDataFile) -> Tuple[Tuple[int, int], str]:
    """Return the record ID and dataset path of the first fragment."""
    records = (
        h5file.get_all_trigger_record_ids()
        if h5file.is_trigger_record_type()
        else h5file.get_all_timeslice_ids()
    )
    if not records:
        raise RuntimeError("No record IDs present in file")
    record = records[0]
    frag_paths = h5file.get_fragment_dataset_paths(record)
    if not frag_paths:
        raise RuntimeError("First record contains no fragments")
    return record, frag_paths[0]


def _print_header_summary(fragment, dataset_path: str, record_id) -> None:
    fragment_type = fragment.get_fragment_type()
    print(f"Record ID       : ({record_id[0]}, {record_id[1]})")
    print(f"Fragment path   : {dataset_path}")
    print(f"Fragment type   : {fragment_type_to_string(fragment_type)}")
    print(f"Fragment size   : {fragment.get_size()} bytes")
    print(f"Payload bytes   : {fragment.get_data_size()}")


def _dump_frame(frame: DAPHNEEthStreamFrame) -> None:
    header = frame.get_daqheader()
    print("\nDAQEthHeader")
    print(f"  version     : {header.version}")
    print(f"  det_id      : {header.det_id}")
    print(f"  crate_id    : {header.crate_id}")
    print(f"  slot_id     : {header.slot_id}")
    print(f"  stream_id   : {header.stream_id}")
    print(f"  seq_id      : {header.seq_id}")
    print(f"  block_length: {header.block_length}")
    print(f"  timestamp   : {header.timestamp}")

    d_header = frame.get_daphneheader()
    print("\nChannel words")
    for idx, channel_word in enumerate(d_header.channel_words):
        print(f"  channel_words[{idx}]")
        print(f"    channel : {channel_word.channel}")
        print(f"    version : {channel_word.version}")
        print(f"    tbd     : 0x{channel_word.tbd:013x}")

    payload = frame.get_bytes()
    adc_section = payload[-_ADC_WORDS * _WORD_BYTES :]
    print(f"\nADC payload ({_ADC_WORDS} words)")
    for idx in range(_ADC_WORDS):
        start = idx * _WORD_BYTES
        word = int.from_bytes(adc_section[start : start + _WORD_BYTES], byteorder="little", signed=False)
        print(f"  adc_words[{idx:03}] = 0x{word:016x}")


def _format_percent(count: int, total: int) -> str:
    if total <= 0:
        return "   n/a "
    return f"{100.0 * count / total:6.2f}%"


def _print_counter_table(
    title: str,
    counter: Counter,
    total: int,
    limit: int,
    formatter: Callable[[object], str],
) -> None:
    print(f"\n{title}")
    if not counter:
        print("  (no entries)")
        return

    header = "  Count   Share   Entry"
    print(header)
    print("  " + "-" * (len(header) - 2))

    shown = 0
    for key, count in counter.most_common():
        if limit is not None and shown >= max(limit, 0):
            remaining = len(counter) - shown
            if remaining > 0:
                print(f"  ... ({remaining} additional entries)")
            break
        print(f"  {count:6d}  {_format_percent(count, total)}  {formatter(key)}")
        shown += 1


def _channel_formatter(channel: int) -> str:
    return f"channel={channel}"


def _link_formatter(entry: LinkKey) -> str:
    return str(entry)


def _source_formatter(source_id: int) -> str:
    return f"source_id={source_id}"


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dump first DAPHNE frame headers and channel statistics for a raw HDF5 file.",
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
        help="Stop after decoding this many waveforms for the summary.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Only inspect this many records when computing statistics.",
    )
    parser.add_argument(
        "--skip-records",
        type=int,
        default=0,
        help="Number of records to skip from the beginning of the file.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Maximum number of rows to display per summary table.",
    )
    parser.add_argument(
        "--time-step-ns",
        type=float,
        default=16.0,
        help="Sampling period passed to the Waveform objects (default: %(default)s).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress info-level logging from the decoder.",
    )
    return parser


def main() -> int:
    args = _build_argparser().parse_args()

    # ---- Low-level dump
    h5file = HDF5RawDataFile(args.hdf5_file)
    record_id, dataset_path = _first_record_and_fragment(h5file)
    fragment = h5file.get_frag(dataset_path)
    frame_size = DAPHNEEthStreamFrame.sizeof()
    payload = fragment.get_data_bytes()
    if len(payload) < frame_size:
        raise SystemExit("Fragment payload smaller than a DAPHNEEthStreamFrame")

    frame = DAPHNEEthStreamFrame(payload[:frame_size])

    print(f"File            : {args.hdf5_file}")
    _print_header_summary(fragment, dataset_path, record_id)
    _dump_frame(frame)

    # ---- Summary statistics via waffles helpers
    logger = None
    if args.quiet:
        import logging

        logger = logging.getLogger("daphne_eth_dump")
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.ERROR)

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
        print("\nNo DAPHNE Ethernet waveforms found.", file=sys.stderr)
        return 1

    stats = compute_waveform_stats(wfset.waveforms)
    inventory = collect_link_inventory(
        filepath=args.hdf5_file,
        detector=args.detector,
        channel_map=args.channel_map,
        skip_records=args.skip_records,
        max_records=args.max_records,
    )
    link_waveform_counts, missing = map_waveforms_to_links(stats, inventory)

    print(
        f"\nDecoded {stats.total_waveforms} waveforms across "
        f"{inventory.records_considered} record(s) / {inventory.fragments_considered} fragment(s)."
    )

    channel_totals: Counter = Counter()
    for (_, channel), count in stats.channel_counts.items():
        channel_totals[int(channel)] += count

    _print_counter_table(
        "Source ID distribution",
        stats.endpoint_counts,
        stats.total_waveforms,
        args.top,
        _source_formatter,
    )
    _print_counter_table(
        "Channel distribution (offline IDs)",
        channel_totals,
        stats.total_waveforms,
        args.top,
        _channel_formatter,
    )
    _print_counter_table(
        "Link distribution",
        link_waveform_counts,
        stats.total_waveforms,
        args.top,
        _link_formatter,
    )

    if missing:
        missing_str = ", ".join(str(eid) for eid in sorted(missing))
        print(f"\nWarning: missing link metadata for endpoint(s): {missing_str}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
