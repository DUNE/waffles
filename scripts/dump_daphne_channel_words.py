#!/usr/bin/env python3
"""
Print the DAPHNEEthStreamFrame channel words for the first N frames in a DAQ HDF5 file.
"""

from __future__ import annotations

import argparse
import sys

from hdf5libs import HDF5RawDataFile
from daqdataformats import FragmentType
from fddetdataformats import DAPHNEEthStreamFrame


def _extract_first_fragment(h5file: HDF5RawDataFile):
    """Return the first record ID, dataset path, and fragment payload."""
    records = (
        h5file.get_all_trigger_record_ids()
        if h5file.is_trigger_record_type()
        else h5file.get_all_timeslice_ids()
    )
    if not records:
        raise RuntimeError("No records found in file")
    record = records[0]

    fragment_paths = h5file.get_fragment_dataset_paths(record)
    if not fragment_paths:
        raise RuntimeError("First record contains no fragments")

    for path in fragment_paths:
        fragment = h5file.get_frag(path)
        if fragment.get_fragment_type() == FragmentType.kDAPHNEEthStream:
            return record, path, fragment

    raise RuntimeError("No DAPHNEEthStream fragment found in first record")


def _dump_channel_words(fragment, frames: int) -> None:
    payload = fragment.get_data_bytes()
    frame_size = DAPHNEEthStreamFrame.sizeof()
    max_frames = min(frames, len(payload) // frame_size)
    if max_frames <= 0:
        raise RuntimeError("Fragment payload does not contain any full frames")

    for idx in range(max_frames):
        start = idx * frame_size
        frame = DAPHNEEthStreamFrame(payload[start : start + frame_size])
        header = frame.get_daphneheader()
        print(f"Frame {idx}:")
        for chan_idx, channel_word in enumerate(header.channel_words):
            print(
                f"  channel_words[{chan_idx}] -> channel={channel_word.channel} "
                f"version={channel_word.version} tbd=0x{channel_word.tbd:013x}"
            )


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dump channel words for the first N DAPHNE Ethernet frames.",
    )
    parser.add_argument("hdf5_file", help="Path to the DAQ HDF5 file.")
    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Number of frames to inspect (default: %(default)s).",
    )
    return parser


def main() -> int:
    args = _build_argparser().parse_args()

    try:
        h5file = HDF5RawDataFile(args.hdf5_file)
    except Exception as exc:
        raise SystemExit(f"Unable to open {args.hdf5_file}: {exc}") from exc

    record_id, dataset_path, fragment = _extract_first_fragment(h5file)

    print(f"File          : {args.hdf5_file}")
    print(f"Record ID     : ({record_id[0]}, {record_id[1]})")
    print(f"Fragment path : {dataset_path}")
    print(f"Payload bytes : {fragment.get_data_size()}")

    try:
        _dump_channel_words(fragment, max(args.frames, 0))
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
