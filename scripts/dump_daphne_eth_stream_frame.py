#!/usr/bin/env python3
"""
Dump every field contained in a DAPHNEEthStreamFrame for the first fragment of a DAQ HDF5 file.

This relies on the python bindings provided by hdf5libs, daqdataformats, detdataformats,
and fddetdataformats.
"""

from __future__ import annotations

import argparse
import sys

from daqdataformats import FragmentType, fragment_type_to_string
from fddetdataformats import DAPHNEEthStreamFrame
from hdf5libs import HDF5RawDataFile

# The frame layout constants mirror those in DAPHNEEthStreamFrame.hpp
BITS_PER_ADC = 14
NUM_CHANNELS = 4
ADCS_PER_CHANNEL = 280
WORD_SIZE_BYTES = 8
NUM_ADC_WORDS = NUM_CHANNELS * ADCS_PER_CHANNEL * BITS_PER_ADC // (WORD_SIZE_BYTES * 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", help="Path to the DAQ-produced HDF5 file")
    return parser.parse_args()


def first_record_id(hdf5_file: HDF5RawDataFile) -> HDF5RawDataFile.record_id_t:
    records = (
        hdf5_file.get_all_trigger_record_ids()
        if hdf5_file.is_trigger_record_type()
        else hdf5_file.get_all_timeslice_ids()
    )
    if not records:
        raise RuntimeError("No records found in the file")
    return records[0]


def pick_fragment_for_record(hdf5_file: HDF5RawDataFile, record_id) -> tuple:
    fragment_paths = hdf5_file.get_fragment_dataset_paths(record_id)
    if not fragment_paths:
        raise RuntimeError("Record contains no fragments")

    fallback = None
    fallback_path = None
    for path in fragment_paths:
        fragment = hdf5_file.get_frag(path)
        if fragment.get_fragment_type() == FragmentType.kDAPHNEEthStream:
            return fragment, path
        if fallback is None:
            fallback = fragment
            fallback_path = path

    return fallback, fallback_path


def dump_daq_header(header) -> None:
    print("DAQEthHeader")
    print(f"  version     : {header.version}")
    print(f"  det_id      : {header.det_id}")
    print(f"  crate_id    : {header.crate_id}")
    print(f"  slot_id     : {header.slot_id}")
    print(f"  stream_id   : {header.stream_id}")
    print(f"  reserved    : {header.reserved}")
    print(f"  seq_id      : {header.seq_id}")
    print(f"  block_length: {header.block_length}")
    print(f"  timestamp   : {header.timestamp} (raw) / {header.get_timestamp()} (decoded)")


def dump_channel_words(frame: DAPHNEEthStreamFrame) -> None:
    print("Channel words")
    header = frame.get_daphneheader()
    for idx, channel_word in enumerate(header.channel_words):
        print(f"  channel_words[{idx}]")
        print(f"    tbd     : 0x{channel_word.tbd:013x}")
        print(f"    version : {channel_word.version}")
        print(f"    channel : {channel_word.channel}")


def dump_adc_words(frame_bytes: bytes) -> None:
    adc_section = frame_bytes[-NUM_ADC_WORDS * WORD_SIZE_BYTES :]
    print(f"ADC payload ({NUM_ADC_WORDS} words)")
    for idx in range(NUM_ADC_WORDS):
        start = idx * WORD_SIZE_BYTES
        word = int.from_bytes(adc_section[start : start + WORD_SIZE_BYTES], byteorder="little", signed=False)
        print(f"  adc_words[{idx:03}] = 0x{word:016x}")


def main() -> None:
    args = parse_args()

    try:
        h5file = HDF5RawDataFile(args.file)
    except Exception as exc:  # pylint: disable=broad-except
        raise SystemExit(f"Unable to open {args.file}: {exc}") from exc

    record_id = first_record_id(h5file)
    fragment, dataset_path = pick_fragment_for_record(h5file, record_id)
    if fragment is None:
        raise SystemExit("No fragments could be loaded from the first record")

    fragment_type = fragment.get_fragment_type()
    if fragment_type != FragmentType.kDAPHNEEthStream:
        print(
            f"Warning: fragment at {dataset_path} is {fragment_type_to_string(fragment_type)}, "
            "not DAPHNEEthStream",
            file=sys.stderr,
        )

    payload = fragment.get_data_bytes()
    frame_size = DAPHNEEthStreamFrame.sizeof()
    if len(payload) < frame_size:
        raise SystemExit("Fragment payload is smaller than a single DAPHNEEthStreamFrame")

    frame = DAPHNEEthStreamFrame(payload[:frame_size])
    frame_bytes = frame.get_bytes()

    print(f"File            : {args.file}")
    print(f"Record ID       : ({record_id[0]}, {record_id[1]})")
    print(f"Fragment path   : {dataset_path}")
    print(f"Fragment type   : {fragment_type_to_string(fragment_type)}")
    print(f"Fragment size   : {fragment.get_size()} bytes")
    print(f"Payload bytes   : {len(payload)}")
    print(f"Frame bytes     : {frame_size}")

    dump_daq_header(frame.get_daqheader())
    dump_channel_words(frame)
    dump_adc_words(frame_bytes)


if __name__ == "__main__":
    main()
